"""Cooperative Text-Grounded Task using SNLI (real NLI dataset).

Two agents cooperate to classify textual entailment where the premise and
hypothesis are split between agents:
- Agent A (Encoder): Sees the PREMISE
- Agent B (Classifier): Sees the HYPOTHESIS but NOT the premise

Agent A must communicate relevant information from the premise to Agent B
through a differentiable message vector. Agent B uses the message + hypothesis
to make an entailment/contradiction/neutral decision.

Uses the Stanford Natural Language Inference (SNLI) corpus for evaluation
on real, human-written text rather than synthetic templates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class TextCoopSample:
    context: str       # Premise (what encoder sees)
    question: str      # Hypothesis (what classifier sees)
    label: int         # 0=entailment, 1=neutral, 2=contradiction -> binary: 0=entail, 1=not


class SNLICoopDataset:
    """SNLI-based cooperative text classification dataset.

    Converts 3-way NLI to binary: entailment (1) vs not-entailment (0).
    This is standard practice and avoids the harder 3-way distinction.
    """

    def __init__(self, split: str = "train", max_samples: int = 10000, seed: int = 42):
        self.rng = random.Random(seed)

        # Map split names
        hf_split = {"train": "train", "val": "validation", "test": "test"}[split]

        # Load SNLI
        ds = load_dataset("snli", split=hf_split)

        # Filter out samples with label=-1 (no gold label)
        samples = []
        for item in ds:
            if item["label"] == -1:
                continue
            # Binary: entailment=1, neutral/contradiction=0
            binary_label = 1 if item["label"] == 0 else 0
            samples.append(TextCoopSample(
                context=item["premise"],
                question=item["hypothesis"],
                label=binary_label,
            ))

        # Shuffle and limit
        self.rng.shuffle(samples)
        self.samples = samples[:max_samples]
        self._idx = 0

    def __len__(self):
        return len(self.samples)

    def sample_batch(self, batch_size: int) -> list[TextCoopSample]:
        return [self.rng.choice(self.samples) for _ in range(batch_size)]

    def get_batch(self, batch_size: int) -> list[TextCoopSample]:
        batch = []
        for _ in range(batch_size):
            batch.append(self.samples[self._idx % len(self.samples)])
            self._idx += 1
        return batch

    def reset(self):
        self._idx = 0
