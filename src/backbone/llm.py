from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class FrozenLLM:
    """Frozen HuggingFace causal LM that extracts hidden states from text."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16"):
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
        ).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device
        self._hidden_size = self.model.config.hidden_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Encode text, return last hidden state of final token. Shape: (hidden_size,)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][0, -1, :]  # last layer, last token

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Batched encode. Returns (batch, hidden_size)."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        # Use last non-pad token for each sequence
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1  # index of last real token
        last_hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)
        batch_size = last_hidden.shape[0]
        return last_hidden[torch.arange(batch_size, device=self.device), seq_lengths]


class HiddenStateCache:
    """Caches LLM hidden states keyed by observation bytes to avoid redundant forward passes."""

    def __init__(self, backbone: FrozenLLM, obs_to_text_fn, max_size: int = 4096):
        self.backbone = backbone
        self.obs_to_text_fn = obs_to_text_fn
        self.max_size = max_size
        self._cache: dict[bytes, torch.Tensor] = {}

    def get(self, obs) -> torch.Tensor:
        """Get hidden state for observation, computing and caching if needed."""
        key = obs.tobytes()
        if key not in self._cache:
            if len(self._cache) >= self.max_size:
                self._cache.clear()
            text = self.obs_to_text_fn(obs)
            self._cache[key] = self.backbone.encode(text)
        return self._cache[key]

    def get_batch(self, obs_list) -> torch.Tensor:
        """Batch version: returns (batch, hidden_size). Uses cache where possible."""
        results = []
        uncached_indices = []
        uncached_texts = []
        for i, obs in enumerate(obs_list):
            key = obs.tobytes()
            if key in self._cache:
                results.append((i, self._cache[key]))
            else:
                uncached_indices.append(i)
                uncached_texts.append(self.obs_to_text_fn(obs))

        if uncached_texts:
            hidden = self.backbone.encode_batch(uncached_texts)
            for j, idx in enumerate(uncached_indices):
                h = hidden[j]
                self._cache[obs_list[idx].tobytes()] = h
                results.append((idx, h))

        results.sort(key=lambda x: x[0])
        return torch.stack([r[1] for r in results])

    def clear(self):
        self._cache.clear()
