"""Noisy Multi-Fact Retrieval Task.

Designed to be HARD for naive text transfer and favor learned compression:

Setup:
- Agent A (Encoder): Sees a LONG context with 5-8 facts, only 1-2 are relevant
- Agent B (Classifier): Sees a question about a specific fact
- Communication: message vector through noisy channel (Gaussian noise added)

Why text transfer struggles here:
1. LONG context (5-8 sentences) — can't just concatenate to classifier LLM
   without hitting token limits or drowning the signal
2. NOISY CHANNEL — Gaussian noise corrupts the message. Discrete tokens are
   fragile (one corrupted token = garbage). Continuous vectors degrade gracefully.
3. SELECTIVE COMPRESSION needed — must learn which facts matter for the question

Why SSR/VQ-SSR should win:
- Learned compression: encoder LLM + SSR learns to extract only relevant facts
- Noise robustness: continuous representations degrade gracefully under noise
- With LoRA: encoder can specialize for this extraction task
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class RetrievalSample:
    context: str       # Long context with multiple facts (encoder sees this)
    question: str      # Question about one specific fact (classifier sees this)
    label: int         # 1=answer is in context, 0=answer contradicts context


# Fact templates for generating diverse contexts
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


class NoisyRetrievalDataset:
    """Multi-fact retrieval dataset with distractor facts.

    Each sample has:
    - 5-8 facts about different entities (context)
    - A question about ONE entity's ONE attribute
    - Label: 1 if the stated answer matches, 0 if it contradicts

    The key difficulty: the question targets a specific entity+attribute,
    and the context has many irrelevant facts as distractors.
    """

    def __init__(self, split: str = "train", n_samples: int = 10000,
                 min_facts: int = 5, max_facts: int = 8, seed: int = 42):
        self.rng = random.Random(seed)
        self.samples = self._generate(n_samples, min_facts, max_facts)
        self._idx = 0

        # Split
        n = len(self.samples)
        if split == "train":
            self.samples = self.samples[:int(0.8 * n)]
        elif split == "val":
            self.samples = self.samples[int(0.8 * n):int(0.9 * n)]
        else:
            self.samples = self.samples[int(0.9 * n):]

    def _generate(self, n_samples: int, min_facts: int, max_facts: int):
        samples = []
        for _ in range(n_samples):
            n_facts = self.rng.randint(min_facts, max_facts)

            # Generate facts about different entities
            entities_used = self.rng.sample(ENTITIES, min(n_facts, len(ENTITIES)))
            facts = []
            fact_data = []

            for entity in entities_used:
                template_idx = self.rng.randint(0, len(FACT_TEMPLATES) - 1)
                template, q_template, ans_key = FACT_TEMPLATES[template_idx]

                # Fill in template
                vals = {
                    "entity": entity,
                    "city": self.rng.choice(CITIES),
                    "color": self.rng.choice(COLORS),
                    "job": self.rng.choice(JOBS),
                    "food": self.rng.choice(FOODS),
                    "pet": self.rng.choice(PETS),
                    "hobby": self.rng.choice(HOBBIES),
                    "number": str(self.rng.choice(NUMBERS)),
                }

                fact = template.format(**vals)
                question = q_template.format(**vals)
                answer = vals[ans_key] if ans_key in vals else str(vals.get(ans_key, ""))

                facts.append(fact)
                fact_data.append((entity, question, answer, template_idx, vals))

            # Pick one fact as the target
            target_idx = self.rng.randint(0, len(facts) - 1)
            target_entity, target_question, target_answer, t_idx, t_vals = fact_data[target_idx]

            # Shuffle facts so target isn't always first
            self.rng.shuffle(facts)
            context = " ".join(facts)

            # Positive: question matches the context
            samples.append(RetrievalSample(
                context=context,
                question=f"{target_question} Answer: {target_answer}",
                label=1,
            ))

            # Negative: question with wrong answer
            template, q_template, ans_key = FACT_TEMPLATES[t_idx]
            pool = {"city": CITIES, "color": COLORS, "job": JOBS,
                    "food": FOODS, "pet": PETS, "hobby": HOBBIES,
                    "number": [str(x) for x in NUMBERS]}
            if ans_key in pool:
                wrong_answers = [a for a in pool[ans_key] if a != target_answer]
                if wrong_answers:
                    wrong = self.rng.choice(wrong_answers)
                    samples.append(RetrievalSample(
                        context=context,
                        question=f"{target_question} Answer: {wrong}",
                        label=0,
                    ))

        self.rng.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def sample_batch(self, batch_size: int) -> list[RetrievalSample]:
        return [self.rng.choice(self.samples) for _ in range(batch_size)]

    def get_batch(self, batch_size: int) -> list[RetrievalSample]:
        batch = []
        for _ in range(batch_size):
            batch.append(self.samples[self._idx % len(self.samples)])
            self._idx += 1
        return batch

    def reset(self):
        self._idx = 0
