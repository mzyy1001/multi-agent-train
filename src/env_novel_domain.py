"""Novel Domain Cooperative Task: Synthetic language the LLM has never seen.

Key insight: LoRA adaptation matters when the LLM's pretrained features are
INSUFFICIENT — not when the task uses language the LLM already understands.

We create a domain with:
1. Made-up entity names (zorplex, kwindari, etc.)
2. Made-up attribute types (fluxion level, chromatic index, etc.)
3. Made-up values (zeta-7, phi-prime, etc.)

The LLM has NEVER seen these tokens in pretraining, so:
- Frozen features are BAD (the LLM can't understand the domain)
- LoRA adaptation MUST help (the encoder LLM learns new representations)
- The gradient highway becomes ESSENTIAL for the encoder to adapt

This is the ideal test case for the gradient highway thesis.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class NovelSample:
    context: str
    question: str
    candidate: str
    label: int


# Made-up vocabulary the LLM has never seen
ENTITIES = [
    "Zorplex", "Kwindari", "Thyxon", "Belvari", "Cryneth",
    "Dulphor", "Exvani", "Fyndarr", "Glyphos", "Hyrvex",
    "Ixalume", "Jytheris", "Kvoldren", "Lumixar", "Myrvanis",
    "Noxiphar", "Orvethis", "Pyxalune", "Qinthari", "Ryvoxen",
]

ATTR_TYPES = [
    ("fluxion level", "fluxion_level"),
    ("chromatic index", "chromatic_index"),
    ("resonance class", "resonance_class"),
    ("vortical phase", "vortical_phase"),
    ("dynametric score", "dynametric_score"),
    ("spectral rank", "spectral_rank"),
]

ATTR_VALUES = {
    "fluxion_level": ["zeta-7", "phi-prime", "omega-null", "kappa-3", "sigma-12",
                       "tau-zero", "delta-9", "gamma-flux", "alpha-peak", "beta-void"],
    "chromatic_index": ["ultraviolex", "infrablue", "neochrome", "hypergold", "nullwhite",
                         "voidblack", "axigreen", "omnired", "psiviolet", "etasilver"],
    "resonance_class": ["harmonic-A", "dissonant-X", "neutral-Q", "amplified-Z", "dampened-W",
                          "resonant-M", "chaotic-P", "stable-E", "volatile-K", "dormant-J"],
    "vortical_phase": ["clockwise-prime", "counter-null", "spiral-zeta", "linear-phi",
                        "orbital-gamma", "radial-sigma", "tangent-omega", "axial-tau",
                        "helical-kappa", "planar-delta"],
    "dynametric_score": ["nexus-42", "vertex-17", "apex-88", "nadir-3", "zenith-56",
                          "pivot-29", "fulcrum-71", "axis-14", "core-93", "shell-6"],
    "spectral_rank": ["tier-alpha", "tier-beta", "tier-gamma", "tier-delta", "tier-epsilon",
                       "tier-zeta", "tier-eta", "tier-theta", "tier-iota", "tier-kappa"],
}

TEMPLATES = [
    ("{entity} has {attr_type} of {value}.", "What is the {attr_type} of {entity}?"),
    ("The {attr_type} for {entity} is {value}.", "What {attr_type} does {entity} have?"),
    ("{entity} registers {value} on {attr_type}.", "What does {entity} register on {attr_type}?"),
]


class NovelDomainDataset:
    """Cooperative task with synthetic vocabulary unknown to the LLM.

    Each sample has 5-8 facts using made-up terminology.
    The encoder must learn (via LoRA) to understand and compress
    this unfamiliar language — frozen LLM features will be poor.
    """

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
                attr_name, attr_key = self.rng.choice(ATTR_TYPES)
                value = self.rng.choice(ATTR_VALUES[attr_key])
                template, q_template = self.rng.choice(TEMPLATES)

                fact = template.format(entity=entity, attr_type=attr_name, value=value)
                question = q_template.format(entity=entity, attr_type=attr_name)
                facts.append(fact)
                fact_data.append((question, value, attr_key))

            target_idx = self.rng.randint(0, len(facts) - 1)
            target_q, target_ans, target_key = fact_data[target_idx]

            self.rng.shuffle(facts)
            context = " ".join(facts)

            # Positive
            samples.append(NovelSample(
                context=context, question=target_q,
                candidate=target_ans, label=1,
            ))

            # Negative
            wrong_pool = [v for v in ATTR_VALUES[target_key] if v != target_ans]
            if wrong_pool:
                samples.append(NovelSample(
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
