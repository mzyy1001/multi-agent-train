# Differentiable Communication for LLM-based Multi-Agent Systems

A feasibility study on whether a **differentiable inter-agent communication interface** can enable effective coordination between language-model-based agents, without collapsing the system into a single monolithic network.

## Motivation

When two LLM agents communicate through sampled natural-language tokens, the communication channel is **discrete**, and ordinary backpropagation cannot pass gradients through it. This project replaces free-form text communication with a compact **SSR (Structured Semantic Representation) message vector** during training: a low-dimensional differentiable bottleneck that forces agents to learn compressed, task-relevant cross-agent semantics.

The research question:

> Can a low-dimensional differentiable SSR channel support effective coordination under partial observability, and does the structured bottleneck outperform simpler alternatives?

## Benchmark

We use **Simple Speaker Listener** from PettingZoo MPE as the testbed:

- **Speaker**: observes the target landmark color (3-dim), cannot move, sends a message
- **Listener**: observes its own velocity and relative landmark positions (11-dim), must navigate to the correct landmark based on the received message

This is an ideal minimal setting because the listener **cannot solve the task without communication** --- it does not know which landmark is the goal.

## Architecture

Each agent is built from a **frozen LLM backbone** (SmolLM2-135M-Instruct) with small trainable modules on top:

```
Speaker side                              Listener side
============                              =============

speaker_obs (3,)                          listener_obs (11,)
     |                                         |
 text prompt                               text prompt
     |                                         |
 FrozenLLM [no grad]                       FrozenLLM [no grad]
     |                                         |
  h_speaker (576,)                          h_listener (576,)
     |                                         |
 ObsProjector [trainable]                  ObsProjector [trainable]
     |                                         |
  z_speaker (128,)                          z_listener (128,)
     |                                         |
 CommChannel [trainable]                        |
     |                                         |
  message (d,) ---------> ---------->  ReceiverAdapter [trainable]
                                        concat(z_listener, message)
                                               |
                                          h_tilde (128,)
                                               |
                                         ActionHead [trainable]
                                          /          \
                                   action_logits    value
                                        |
                                     PPO loss
```

**Frozen parts**: Both LLM backbones. They serve as fixed feature extractors that map text observations to hidden-state vectors.

**Trainable parts** (total ~150K parameters):
- `ObsProjector`: 2-layer MLP projecting LLM hidden states to a fixed-size latent
- `CommChannel`: the communication module (varies by method, see below)
- `ReceiverAdapter`: fuses the listener's own representation with the incoming message
- `ActionHead`: continuous policy head (tanh-squashed Gaussian) + value head

The key design constraint is the **narrow communication bottleneck**. Cross-agent information must pass through a small vector (d = 4, 8, or 16), preventing the system from collapsing into a single model while still allowing end-to-end gradient flow.

### Why continuous actions?

The listener uses continuous actions with the **reparameterization trick** (tanh-squashed Gaussian). This keeps the entire path from the speaker's observation through the message to the listener's action fully differentiable. Discrete actions would require REINFORCE for action sampling and block gradients through the message.

### Gradient flow

During PPO updates, both agents are **re-forwarded with gradient tracking**. The critical gradient path is:

```
PPO loss --> ActionHead --> ReceiverAdapter --> message --> CommChannel --> ObsProjector (speaker)
```

This means the speaker's trainable modules learn **through the listener's policy loss**, mediated by the differentiable message. A single Adam optimizer is used over all trainable parameters from both agents.

## Four Communication Methods

We implement four communication methods to isolate what matters: whether communication helps at all, whether continuity helps, and whether structure helps.

### 1. No Communication (Lower Bound)

```python
class NoChannel:
    def forward(self, z_sender):
        return torch.zeros(batch, d)
```

The message is always a zero vector. The listener receives no information from the speaker and must act based only on its own local observation (velocity + landmark positions). Since the listener does not know which landmark is the target, it **cannot solve the task** --- this establishes the performance floor.

**Purpose**: Confirms that communication is necessary. Any method that does not beat this baseline is broken.

### 2. Discrete Communication (Gumbel-Softmax)

```python
class DiscreteChannel:
    def forward(self, z_sender):
        logits = self.logit_net(z_sender)         # (batch, num_symbols)
        if training:
            return gumbel_softmax(logits, tau)     # soft one-hot, differentiable
        else:
            return one_hot(argmax(logits))          # hard one-hot
```

The speaker selects one of K discrete symbols (default K=8). During training, we use **Gumbel-Softmax relaxation** to make the discrete choice differentiable: the output is a soft probability vector that approximates a one-hot vector, with temperature `tau` annealed from 1.0 to 0.1 over training. During evaluation, the argmax is taken for a hard symbol.

This is the closest baseline to **classic discrete communication in multi-agent RL** (e.g., RIAL/DIAL). The message is effectively a categorical variable.

**Architecture**: 2-layer MLP mapping the projected hidden state to K logits, followed by Gumbel-Softmax sampling.

**Purpose**: Tests whether a continuous channel outperforms the discrete paradigm under the same bandwidth budget.

### 3. Unstructured Continuous Channel (Ablation Control)

```python
class ContinuousChannel:
    def forward(self, z_sender):
        return self.linear(z_sender)              # single linear projection
```

The speaker emits a d-dimensional continuous vector, but through a **single linear projection** with no nonlinearity, no normalization, and no bottleneck MLP. The message dimension matches SSR (d = 4, 8, or 16).

This is the most important baseline because it isolates the contribution of **SSR's structured bottleneck**. Both SSR and this channel transmit the same number of continuous floats --- the difference is that SSR processes the signal through a deeper network with LayerNorm, while this channel is a raw linear map.

**Purpose**: Tests whether the SSR bottleneck structure (nonlinear compression + normalization) adds value beyond merely using a continuous channel.

### 4. SSR --- Structured Semantic Representation (Proposed Method)

```python
class SSRChannel:
    def forward(self, z_sender):
        h = self.mlp(z_sender)                    # 2-layer MLP with GELU
        return self.layer_norm(h)                  # LayerNorm on output
```

The speaker's projected hidden state is mapped through a **2-layer MLP bottleneck** (hidden dim = 4 * d) with GELU activation, followed by **LayerNorm** on the output. The output is a d-dimensional vector (d = 4, 8, or 16).

The architectural choices are deliberate:

- **Nonlinear bottleneck**: The 2-layer MLP with expansion factor 4x forces the network to learn a compressed, nonlinear encoding of the task-relevant information, rather than passing through a raw linear projection.
- **LayerNorm**: Normalizes the message vector to have zero mean and unit variance per dimension. This prevents message collapse (all dimensions converging to the same value) and stabilizes training, especially important when the message is small (d=4).
- **Low dimensionality**: The small d forces each dimension to carry structured, non-redundant information. This is the "structured" part of SSR --- the bottleneck pressure encourages the network to allocate dimensions to distinct semantic roles.

**Purpose**: The primary method under study. Hypothesized to outperform discrete communication (by enabling richer gradient signal) and unstructured continuous communication (by forcing more structured representations).

### Comparison Summary

| Method | Message Type | Dimension | Differentiable? | Structure |
|--------|-------------|-----------|----------------|-----------|
| No Comm | zeros | d | N/A | None |
| Discrete | soft/hard one-hot | K symbols | via Gumbel-Softmax | Categorical |
| Continuous | real vector | d | fully | None (linear) |
| **SSR** | **real vector** | **d** | **fully** | **MLP + LayerNorm** |

## Project Structure

```
multi-agent-train/
|-- configs/
|   |-- default.yaml                # base configuration
|   |-- comm/
|   |   |-- ssr.yaml                # SSR channel overrides
|   |   |-- discrete.yaml           # Gumbel-Softmax discrete
|   |   |-- continuous.yaml         # unstructured continuous
|   |   +-- none.yaml               # no communication
|   +-- model/
|       |-- smol_smol.yaml          # SmolLM2 x 2 (feasibility pair)
|       +-- smol_qwen.yaml          # SmolLM2 + Qwen2.5-0.5B (heterogeneity test)
|
|-- scripts/
|   |-- train.py                    # main training entry point
|   |-- evaluate.py                 # evaluate a saved checkpoint
|   +-- sweep.py                    # launch grid of experiments
|
|-- src/
|   |-- config.py                   # dataclass config + YAML loader
|   |-- env_wrapper.py              # PettingZoo parallel env wrapper
|   |
|   |-- backbone/
|   |   +-- llm.py                  # FrozenLLM: load model, extract hidden states, cache
|   |
|   |-- modules/
|   |   |-- obs_projector.py        # E_i: LLM hidden state -> latent z
|   |   |-- receiver_adapter.py     # D_j: fuse (z_listener, message) -> h_tilde
|   |   +-- action_head.py          # policy (Gaussian) + value heads
|   |
|   |-- comm/
|   |   |-- base.py                 # abstract CommChannel interface
|   |   |-- ssr.py                  # SSR: MLP bottleneck + LayerNorm
|   |   |-- discrete.py             # Gumbel-Softmax discrete symbols
|   |   |-- continuous.py           # unstructured linear projection
|   |   +-- none.py                 # zero vector (no communication)
|   |
|   |-- agents/
|   |   |-- base.py                 # shared encoding logic + caching
|   |   |-- speaker.py              # produces message, no movement
|   |   |-- listener.py             # receives message, produces movement
|   |   +-- centralized.py          # single-model upper bound baseline
|   |
|   |-- training/
|   |   |-- rollout_buffer.py       # episode storage + GAE computation
|   |   +-- ppo.py                  # PPO trainer with re-forward for diff. comm
|   |
|   +-- utils/
|       |-- logging.py              # tensorboard / wandb logging
|       |-- seeding.py              # reproducibility
|       +-- text_prompt.py          # observation -> text prompt formatting
|
+-- tests/
    |-- test_env.py
    |-- test_comm.py
    +-- test_training.py
```

## Configuration

All experiments are driven by YAML configs. The base config (`configs/default.yaml`) can be overridden by layering comm and model configs:

```bash
# SSR with d=8, SmolLM2 x 2 (default)
python scripts/train.py

# Discrete baseline
python scripts/train.py --overrides comm/discrete.yaml

# SSR with d=16
python scripts/train.py --set comm.dim=16

# Heterogeneous agents (SmolLM2 speaker + Qwen2.5-0.5B listener)
python scripts/train.py --overrides model/smol_qwen.yaml

# Combined overrides
python scripts/train.py --overrides comm/ssr.yaml model/smol_qwen.yaml --set comm.dim=4
```

Key config options:

| Key | Default | Description |
|-----|---------|-------------|
| `comm.type` | `ssr` | Communication method: `ssr`, `discrete`, `continuous`, `none` |
| `comm.dim` | `8` | Message vector dimension (SSR/continuous) |
| `comm.num_symbols` | `8` | Number of discrete symbols (discrete only) |
| `training.total_episodes` | `50000` | Total training episodes |
| `training.rollout_episodes` | `32` | Episodes per PPO rollout batch |
| `training.lr` | `3e-4` | Learning rate |

## Running Experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

Requires: PyTorch >= 2.1, Transformers, PettingZoo with MPE, and a GPU with >= 4GB VRAM.

### Single Run

```bash
PYTHONPATH=. python scripts/train.py --config configs/default.yaml
```

### Sweep All Baselines

```bash
# Preview the experiment grid
PYTHONPATH=. python scripts/sweep.py --dry-run

# Run all experiments
PYTHONPATH=. python scripts/sweep.py
```

### Evaluate a Checkpoint

```bash
PYTHONPATH=. python scripts/evaluate.py --checkpoint checkpoints/ssr_d8_s42/checkpoint_final.pt --episodes 100
```

### Run Tests

```bash
PYTHONPATH=. python tests/test_env.py
PYTHONPATH=. python tests/test_comm.py
PYTHONPATH=. python tests/test_training.py
```

## Models

**Feasibility pair** (Phase A): `SmolLM2-135M-Instruct x 2`
- Identical models removes heterogeneity as a confound
- hidden_size = 576, ~270MB in float16

**Heterogeneity pair** (Phase B): `SmolLM2-135M-Instruct + Qwen2.5-0.5B-Instruct`
- Tests whether the SSR channel bridges different model families
- Different hidden sizes (576 vs 896), different tokenizers

Both LLM backbones are **frozen** during training. Only the communication and action modules are trained (~150K parameters total).

## Evaluation Criteria

The SSR approach is considered feasible if:

1. **SSR clearly outperforms no communication** (the channel carries useful information)
2. **SSR matches or outperforms discrete communication** under similar bandwidth
3. **SSR outperforms unstructured continuous communication**, or at least generalizes better on held-out combinations

Metrics tracked:
- Episode reward / success rate
- Sample efficiency (reward vs. training steps)
- Message variance per dimension (checks for message collapse)
- Gradient norms at speaker projector and comm channel (confirms gradient flow)
