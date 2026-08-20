# Robometer Policy Learning

A flexible reinforcement \ imitation learning framework supporting multiple algorithms (SAC, IQL, BC), reward models (Robometer, RoboReward), and distributed training. We will soon include detailed guides for DSRL (Diffusion-steering RL) with Pi0/0.5 on LIBERO and Real World tasks.

>[!WARNING]
>
> This repository is under active development, so some modules and features may change over time. You may encounter issues when using features that are not yet documented in this README. Please feel free to open an issue — we will do our best to help.

## Documentation

| Document | Contents |
| --- | --- |
| [CLAUDE.md](CLAUDE.md) | orientation and repo-wide conventions (also loaded automatically by Claude Code) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | package map, how a training run is wired, extension recipes |
| [docs/WORKFLOWS.md](docs/WORKFLOWS.md) | runbooks for every pipeline: supervised, RL, DSRL, HITL, real robot |
| [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) | observation/action contracts, HDF5 + LeRobot schemas, pi0 format |
| [docs/CONFIGS.md](docs/CONFIGS.md) | Hydra layout and which config belongs to which script |
| [docs/GOTCHAS.md](docs/GOTCHAS.md) | non-obvious constraints — read before your first run |
| [hitl_pi05_openpi_reference.md](hitl_pi05_openpi_reference.md) | the pi0.5 HITL pipeline in detail |
| [docs/LIBERO_PLUS.md](docs/LIBERO_PLUS.md) | LIBERO-plus robustness benchmark |
| [docs/REAL_ROBOT_README.md](docs/REAL_ROBOT_README.md) | real-robot setup |

## Table of Contents

- [Setup](#setup)
- [Training](#training)
  - [Basic Training](#basic-training-ground-truth-rewards)
  - [Training with Reward Model](#training-with-robometer-reward-model)
  - [Example: Online RL in LIBERO](#example-online-rl-in-libero)
- [Real-World Online RL with DSRL + Remote Reward Labeling - Coming Soon]
- [Project Structure](#project-structure)
---

## Setup

### Prerequisites

- Git (with Git LFS)
- Python 3.11 (the project pins `requires-python = "==3.11.*"`)
- NVIDIA Drivers (for GPU support)

### Installation

1. **Install `uv` (if not already installed):**
  ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
2. **Clone and set up submodules (required for openpi/Pi0, LIBERO, robometer):**
  ```bash
   git submodule update --init --recursive
  ```
   Submodules live under `third_party/` (see `.gitmodules`): `dsrl_openpi` (openpi / Pi0·π0.5, tracks
   branch `hitl-work`), `LIBERO`, and `robometer`. Each is wired as an **editable dependency group**
   (`[dependency-groups]` + `[tool.uv.sources]` in `pyproject.toml`).

   `LIBERO-plus` (the robustness benchmark: 10,030 perturbed tasks) is also a submodule but is
   deliberately *not* installed — it ships the same `libero` package as LIBERO and is activated
   per run with `env.libero_plus=true`. Set it up with `bash scripts/setup_libero_plus.sh` and see
   [docs/LIBERO_PLUS.md](docs/LIBERO_PLUS.md).

3. **Create and sync the virtual environment.** A plain `uv sync` installs the default groups —
   **`openpi` + `libero`** — which is the Pi0 / π0.5, LIBERO, and DSRL / HITL path:
  ```bash
   # GIT_LFS_SKIP_SMUDGE=1 avoids Git-LFS smudge errors when pulling LeRobot (an openpi dependency).
   GIT_LFS_SKIP_SMUDGE=1 uv sync

   # Optional: also install the dev tools (pytest, ruff, pre-commit, ...)
   GIT_LFS_SKIP_SMUDGE=1 uv sync --extra dev
  ```

**Activate the environment:**

```bash
 source .venv/bin/activate
```



**Robometer reward-model path:** the `robometer` group is **mutually exclusive** with `openpi` (they pin
conflicting `torch` / `datasets` versions), so install it on its own instead of alongside the default
groups:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --group robometer --no-default-groups
```

If you use your own fork of any submodule, update `.gitmodules` and `[tool.uv.sources]` accordingly.

---

## Training

### Basic Training (Ground Truth Rewards)

Train with ground truth rewards (default, no reward model - Online only):

```bash
uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=config \
  algorithm@online_algorithm=sac \
  alg.online_alg_name=sac \
  env.use_gt_rewards=true
```

Offline pretraining with online fine-tuning:

```bash
uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=config \
  algorithm@offline_algorithm=iql \
  algorithm@online_algorithm=sac \
  alg.offline_alg_name=iql \
  alg.online_alg_name=sac \
  env.use_gt_rewards=true
```

### Training with Robometer Reward Model

Train with Robometer reward model (Online only):

```bash
uv run python scripts/train.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=config \
  reward_model=robometer \
  algorithm@online_algorithm=sac \
  alg.online_alg_name=sac \
  env.use_gt_rewards=false \
  reward_model.model_path=robometer/Robometer-4B
```

Train with Robometer reward model (Offline-to-online):

```bash
uv run python scripts/train.py   \
  --config-path=../robometer_policy_learning/configs   \
  --config-name=config   \
  reward_model=robometer   \
  algorithm@online_algorithm=sac   \
  alg.online_alg_name=sac   \
  algorithm@offline_algorithm=iql   \
  alg.offline_alg_name=iql   \
  env.use_gt_rewards=false   \
  reward_model.model_path=robometer/Robometer-4B
```

### Example: Online RL in LIBERO

Train a SAC policy in LIBERO using ground-truth rewards:

```bash
uv run python scripts/train.py   \
  --config-path=../robometer_policy_learning/configs  \
  --config-name=libero_online_rl  \
  env.env_name=libero_90  \
  env.task_id=28  \
  env.use_gt_rewards=true  \
  algorithm@online_algorithm=sac   \
  alg.online_alg_name=sac   \
  training.num_rollouts=100000  \
  training.seed=100  \
  eval.eval_freq=5000  \
  eval.eval_num_episodes=20  \
  online_algorithm.num_critic_updates_per_actor_update=1  \
  online_algorithm.learning_starts=5000  \
  online_algorithm.critic_optimizer_lr=1e-5  \
  online_algorithm.actor_optimizer_lr=1e-5  \
  logging.wandb_name=libero_online_rl_gt_rewards  \
  logging.wandb_entity=YOUR_WANDB_ENTITY
```

Train a SAC policy in LIBERO using Robometer rewards:

```bash
uv run python scripts/train.py   \
  --config-path=../robometer_policy_learning/configs  \
  --config-name=libero_online_rl  \
  reward_model=robometer  \
  reward_model.model_path=robometer/Robometer-4B  \
  reward_model.add_estimated_reward=true  \
  reward_model.use_success_detection=false  \
  env.env_name=libero_90  \
  env.task_id=28  \
  env.use_gt_rewards=false  \
  algorithm@online_algorithm=sac   \
  alg.online_alg_name=sac   \
  training.num_rollouts=100000  \
  training.seed=100  \
  eval.eval_freq=5000  \
  eval.eval_num_episodes=20  \
  online_algorithm.num_critic_updates_per_actor_update=1  \
  online_algorithm.learning_starts=5000  \
  online_algorithm.critic_optimizer_lr=1e-5  \
  online_algorithm.actor_optimizer_lr=1e-5  \
  logging.wandb_name=libero_online_rl_robometer_rewards  \
  logging.wandb_entity=YOUR_WANDB_ENTITY
```

You should see evaluation curves similar to the example below:

<p align="center">
  <img src="docs/libero_rl.png" alt="LIBERO RL Experiments" width="600"/>
</p>

---

# Real-World Online RL with DSRL + Remote Reward Labeling

Coming soon...
Files are in this repo but need to be cleaned up, should be done by mid June. 

Also coming soon: DSRL+Pi0 sanity check command before running real world online RL. 

---

# Additional Resources

- gRPC service definitions:
  `robometer_policy_learning/distributed/protos/`

- Configuration files:
  `robometer_policy_learning/configs/`

- Algorithm-specific configs:
  `robometer_policy_learning/configs/algorithm/`

## Project Structure

```text
.
├── docs/                         # Documentation assets and figures
├── robometer_policy_learning/    # Main policy learning package
│   ├── algorithms/               # BC, IQL, SAC, and DSRL algorithm code
│   ├── buffers/                  # Replay and offline data buffers
│   ├── configs/                  # Hydra configs for algorithms, envs, and reward models
│   ├── distributed/              # Distributed training and reward relabeling services
│   ├── envs/                     # Environment wrappers and task interfaces
│   ├── loggers/                  # Logging integrations
│   ├── modules/                  # Policy, critic, and value network modules
│   ├── robots/                   # Real-robot interfaces
│   ├── rollouts/                 # Rollout collection utilities
│   ├── runners/                  # Training and evaluation runners
│   └── utils/                    # Shared helpers
├── scripts/                      # Training, evaluation, and server entrypoints
├── tests/                        # Test suite
```

