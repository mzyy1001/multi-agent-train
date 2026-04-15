from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRALLM(nn.Module):
    """LLM backbone with LoRA adapters for fine-tuning.

    Unlike FrozenLLM, this version:
    1. Wraps the model as an nn.Module so LoRA params appear in parameters()
    2. Enables gradient computation through the LoRA-adapted layers
    3. Returns hidden states with gradient tracking

    The LoRA adapters are applied to the attention projection layers,
    adding only rank*2*hidden_size parameters per layer while enabling
    the backbone to adapt its representations for the communication task.

    Key insight: When paired with a differentiable communication channel,
    gradients from the listener's policy loss flow backward through the
    message, through the speaker's projector, and into the speaker's
    LoRA weights — creating a cross-agent training signal.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "float16",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
    ):
        super().__init__()
        torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype,
        )

        # Auto-detect target modules if not specified
        if lora_target_modules is None:
            lora_target_modules = self._detect_target_modules(base_model)

        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(base_model, lora_config)
        self.model.to(device)
        self.model.print_trainable_parameters()

        self.device = device
        self._hidden_size = base_model.config.hidden_size
        self._lora_rank = lora_rank
        self._dtype = torch_dtype

    @staticmethod
    def _detect_target_modules(model) -> list[str]:
        """Auto-detect attention projection module names for different architectures."""
        # Common patterns across model families
        patterns = [
            # SmolLM2, LLaMA, Qwen, DeepSeek
            ["q_proj", "v_proj"],
            # GPT-2 style
            ["c_attn"],
            # Falcon
            ["query_key_value"],
        ]
        model_modules = {name for name, _ in model.named_modules()}
        for pattern in patterns:
            if any(any(p in name for name in model_modules) for p in pattern):
                return pattern
        # Fallback: target all linear layers (aggressive but works)
        return ["q_proj", "v_proj", "k_proj", "o_proj"]

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def lora_rank(self) -> int:
        return self._lora_rank

    def trainable_params(self) -> int:
        """Count trainable (LoRA) parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text, return last hidden state of final token WITH gradients.

        Unlike FrozenLLM.encode, this returns a tensor connected to the
        computation graph so gradients can flow back into LoRA weights.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        # Last layer, last token — gradient flows through LoRA adapters
        return outputs.hidden_states[-1][0, -1, :]

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Batched encode WITH gradient tracking."""
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)

        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_hidden = outputs.hidden_states[-1]
        batch_size = last_hidden.shape[0]
        return last_hidden[torch.arange(batch_size, device=self.device), seq_lengths]


class LoRAHiddenStateCache:
    """Cache for LoRA-enabled LLM that re-computes when in training mode.

    During training: no caching (need fresh gradients each forward pass).
    During eval: cache hidden states to avoid redundant computation.
    """

    def __init__(self, backbone: LoRALLM, obs_to_text_fn, max_size: int = 4096):
        self.backbone = backbone
        self.obs_to_text_fn = obs_to_text_fn
        self.max_size = max_size
        self._cache: dict[bytes, torch.Tensor] = {}

    def get(self, obs, training: bool = False) -> torch.Tensor:
        """Get hidden state. Bypasses cache during training for fresh gradients."""
        if training:
            text = self.obs_to_text_fn(obs)
            return self.backbone.encode(text)

        key = obs.tobytes()
        if key not in self._cache:
            if len(self._cache) >= self.max_size:
                self._cache.clear()
            text = self.obs_to_text_fn(obs)
            with torch.no_grad():
                self._cache[key] = self.backbone.encode(text).detach()
        return self._cache[key]

    def get_batch(self, obs_list, training: bool = False) -> torch.Tensor:
        """Batch version. Bypasses cache during training."""
        if training:
            texts = [self.obs_to_text_fn(obs) for obs in obs_list]
            return self.backbone.encode_batch(texts)

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
            with torch.no_grad():
                hidden = self.backbone.encode_batch(uncached_texts)
            for j, idx in enumerate(uncached_indices):
                h = hidden[j].detach()
                self._cache[obs_list[idx].tobytes()] = h
                results.append((idx, h))

        results.sort(key=lambda x: x[0])
        return torch.stack([r[1] for r in results])

    def clear(self):
        self._cache.clear()
