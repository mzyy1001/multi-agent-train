"""Cooperative Text-Grounded Task: Split-Context Classification.

Two agents cooperate to classify text where relevant information is split:
- Agent A (Encoder): Sees the CONTEXT paragraph (background information)
- Agent B (Classifier): Sees the QUESTION/CLAIM but NOT the context

Agent A must communicate relevant information from the context to Agent B
through a differentiable message vector. Agent B uses the message + question
to make a binary classification decision.

This task is designed to test whether differentiable communication enables
meaningful cross-agent LLM adaptation:
- The LLM's language understanding GENUINELY MATTERS (unlike Speaker-Listener
  where the observation is just 3 floats)
- LoRA adaptation should help the encoder LLM learn to compress task-relevant
  information into the message
- The gradient highway thesis: classifier loss → message → encoder LLM LoRA
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextCoopSample:
    context: str       # What Agent A (encoder) sees
    question: str      # What Agent B (classifier) sees
    label: int         # 0 or 1 (binary classification)


class TextCoopDataset:
    """Simple cooperative text classification dataset.

    Uses entailment-style examples where:
    - context: a premise statement
    - question: a hypothesis
    - label: 1 if hypothesis follows from premise, 0 otherwise

    We generate simple examples programmatically for the proof-of-concept.
    Can be replaced with real NLI datasets (SNLI, MultiNLI) for the full paper.
    """

    def __init__(self, split: str = "train", seed: int = 42):
        self.rng = random.Random(seed)
        self.samples = self._generate_samples(split)
        self._idx = 0

    def _generate_samples(self, split: str) -> list[TextCoopSample]:
        """Generate programmatic entailment examples."""
        samples = []

        # Template-based generation for reproducibility
        entities = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        locations = ["park", "library", "office", "restaurant", "school", "hospital"]
        actions = ["reading", "working", "eating", "studying", "sleeping", "running"]
        objects = ["book", "laptop", "phone", "coffee", "bag", "umbrella"]
        colors = ["red", "blue", "green", "yellow", "black", "white"]
        sizes = ["large", "small", "medium", "tiny", "huge"]

        for entity in entities:
            for loc in locations:
                for action in actions[:3]:
                    for obj in objects[:3]:
                        color = self.rng.choice(colors)
                        size = self.rng.choice(sizes)

                        context = f"{entity} is at the {loc}. They are {action} with a {color} {obj}. The {obj} is {size}."

                        # Positive: hypothesis follows from context
                        samples.append(TextCoopSample(
                            context=context,
                            question=f"{entity} is at the {loc}.",
                            label=1,
                        ))
                        samples.append(TextCoopSample(
                            context=context,
                            question=f"{entity} has a {color} {obj}.",
                            label=1,
                        ))
                        samples.append(TextCoopSample(
                            context=context,
                            question=f"The {obj} is {size}.",
                            label=1,
                        ))

                        # Negative: hypothesis contradicts context
                        wrong_loc = self.rng.choice([l for l in locations if l != loc])
                        wrong_color = self.rng.choice([c for c in colors if c != color])
                        samples.append(TextCoopSample(
                            context=context,
                            question=f"{entity} is at the {wrong_loc}.",
                            label=0,
                        ))
                        samples.append(TextCoopSample(
                            context=context,
                            question=f"{entity} has a {wrong_color} {obj}.",
                            label=0,
                        ))

        self.rng.shuffle(samples)

        # Split
        n = len(samples)
        if split == "train":
            return samples[:int(0.8 * n)]
        elif split == "val":
            return samples[int(0.8 * n):int(0.9 * n)]
        else:
            return samples[int(0.9 * n):]

    def __len__(self):
        return len(self.samples)

    def sample_batch(self, batch_size: int) -> list[TextCoopSample]:
        """Sample a random batch."""
        return [self.rng.choice(self.samples) for _ in range(batch_size)]

    def get_batch(self, batch_size: int) -> list[TextCoopSample]:
        """Get next sequential batch (for evaluation)."""
        batch = []
        for _ in range(batch_size):
            batch.append(self.samples[self._idx % len(self.samples)])
            self._idx += 1
        return batch

    def reset(self):
        self._idx = 0
