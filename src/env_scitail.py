"""SciTail: Real scientific entailment benchmark.

Uses the SciTail dataset (Allen AI) — scientific premises from
textbooks/web with entailment hypotheses from science exams.

Domain-specific vocabulary (orbital mechanics, biology, chemistry)
that small LLMs may not handle well → LoRA adaptation should help.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class SciTailSample:
    context: str       # Scientific premise
    question: str      # Hypothesis
    label: int         # 1=entails, 0=neutral


class SciTailCoopDataset:

    def __init__(self, split: str = "train", max_samples: int = 10000, seed: int = 42):
        self.rng = random.Random(seed)
        hf_split = {"train": "train", "val": "validation", "test": "test"}[split]
        ds = load_dataset("allenai/scitail", "snli_format", split=hf_split)

        samples = []
        for item in ds:
            if item["gold_label"] == "entailment":
                label = 1
            elif item["gold_label"] == "neutral":
                label = 0
            else:
                continue
            samples.append(SciTailSample(
                context=item["sentence1"],
                question=item["sentence2"],
                label=label,
            ))

        self.rng.shuffle(samples)
        self.samples = samples[:max_samples]
        self._idx = 0

    def __len__(self):
        return len(self.samples)

    def sample_batch(self, batch_size):
        return [self.rng.choice(self.samples) for _ in range(batch_size)]

    def get_batch(self, batch_size):
        batch = []
        for _ in range(batch_size):
            batch.append(self.samples[self._idx % len(self.samples)])
            self._idx += 1
        return batch

    def reset(self):
        self._idx = 0
