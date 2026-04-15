"""Noisy Multi-Fact Retrieval v2: Fair noise model + harder task.

Changes from v1:
1. Question does NOT contain the answer (harder — no string matching shortcut)
2. Both agents see the same noise model (Gaussian on hidden states)
   rather than different noise types

Setup:
- Encoder sees: "Alice lives in Paris. Bob works as a chef. ..." (5-8 facts)
- Classifier sees: "What is Bob's job?" (question only, NO answer)
- Label: 1 if candidate=true answer, 0 if candidate=wrong answer
- Candidate answer is provided as a separate input to avoid string matching

The noise is applied to the COMMUNICATION CHANNEL (SSR vector or text embedding),
making the comparison fair across modalities.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class RetrievalSample:
    context: str       # Long context with multiple facts
    question: str      # Question WITHOUT answer
    candidate: str     # Candidate answer (may be right or wrong)
    label: int         # 1=candidate is correct, 0=candidate is wrong


ENTITIES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
CITIES = ["Paris", "London", "Tokyo", "Berlin", "Sydney", "Toronto", "Mumbai", "Cairo"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]
JOBS = ["engineer", "doctor", "teacher", "artist", "chef", "pilot", "lawyer", "writer"]
FOODS = ["pizza", "sushi", "pasta", "curry", "tacos", "ramen", "salad", "steak"]
PETS = ["dog", "cat", "bird", "fish", "rabbit", "hamster", "turtle", "parrot"]
HOBBIES = ["reading", "swimming", "painting", "cooking", "hiking", "gaming", "singing", "dancing"]
NUMBERS = list(range(20, 90))

FACT_TEMPLATES = [
    ("{entity} lives in {city}.", "Where does {entity} live?", "city"),
    ("{entity} works as a {job}.", "What is {entity}'s job?", "job"),
    ("{entity} likes {food}.", "What food does {entity} like?", "food"),
    ("{entity} has a {color} {pet}.", "What color is {entity}'s {pet}?", "color"),
    ("{entity} enjoys {hobby} on weekends.", "What does {entity} do on weekends?", "hobby"),
    ("{entity} is {number} years old.", "How old is {entity}?", "number"),
    ("{entity} drives a {color} car.", "What color is {entity}'s car?", "color"),
    ("{entity} studied in {city}.", "Where did {entity} study?", "city"),
]

POOLS = {
    "city": CITIES, "color": COLORS, "job": JOBS,
    "food": FOODS, "pet": PETS, "hobby": HOBBIES,
    "number": [str(x) for x in NUMBERS],
}


class NoisyRetrievalDatasetV2:

    def __init__(self, split: str = "train", n_samples: int = 10000,
                 min_facts: int = 5, max_facts: int = 8, seed: int = 42):
        self.rng = random.Random(seed)
        self.samples = self._generate(n_samples, min_facts, max_facts)
        self._idx = 0

        n = len(self.samples)
        if split == "train":
            self.samples = self.samples[:int(0.8 * n)]
        elif split == "val":
            self.samples = self.samples[int(0.8 * n):int(0.9 * n)]
        else:
            self.samples = self.samples[int(0.9 * n):]

    def _generate(self, n_samples, min_facts, max_facts):
        samples = []
        for _ in range(n_samples):
            n_facts = self.rng.randint(min_facts, max_facts)
            entities_used = self.rng.sample(ENTITIES, min(n_facts, len(ENTITIES)))
            facts = []
            fact_data = []

            for entity in entities_used:
                t_idx = self.rng.randint(0, len(FACT_TEMPLATES) - 1)
                template, q_template, ans_key = FACT_TEMPLATES[t_idx]
                vals = {
                    "entity": entity, "city": self.rng.choice(CITIES),
                    "color": self.rng.choice(COLORS), "job": self.rng.choice(JOBS),
                    "food": self.rng.choice(FOODS), "pet": self.rng.choice(PETS),
                    "hobby": self.rng.choice(HOBBIES), "number": str(self.rng.choice(NUMBERS)),
                }
                fact = template.format(**vals)
                question = q_template.format(**vals)
                answer = vals[ans_key]
                facts.append(fact)
                fact_data.append((question, answer, ans_key))

            target_idx = self.rng.randint(0, len(facts) - 1)
            target_q, target_ans, target_key = fact_data[target_idx]

            self.rng.shuffle(facts)
            context = " ".join(facts)

            # Positive: correct candidate
            samples.append(RetrievalSample(
                context=context, question=target_q,
                candidate=target_ans, label=1,
            ))

            # Negative: wrong candidate
            wrong_pool = [a for a in POOLS[target_key] if a != target_ans]
            if wrong_pool:
                samples.append(RetrievalSample(
                    context=context, question=target_q,
                    candidate=self.rng.choice(wrong_pool), label=0,
                ))

        self.rng.shuffle(samples)
        return samples

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
