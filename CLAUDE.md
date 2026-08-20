# CLAUDE.md

Orientation for agents working in this repo. Read the linked docs before acting in an area you
haven't touched — especially [docs/GOTCHAS.md](docs/GOTCHAS.md).

## What this repo is

`robometer-policy-learning` is a research framework for learning robot manipulation policies. It
covers four families of method over a shared component set:

* **Supervised / imitation** — BC, Diffusion Policy, Flow Matching from offline datasets.
* **RL** — SAC / IQL, online or offline→online, with ground-truth or *learned* rewards (Robometer /
  RoboReward reward models served over gRPC).
* **DSRL** — freeze a large VLA (pi0 / pi0.5 via openpi) and train a small RL policy that steers it
  through its noise input.
* **HITL** — a human takes over mid-rollout; corrections train the policy via HG-DAgger or the MILE
  family (MILE / Flow-MILE / Diffusion-MILE).

Environments: LIBERO (+ LIBERO-plus robustness benchmark), robosuite/robomimic, MetaWorld, SIMPLER,
and real robots (YAM via raiden, DROID, WidowX) behind remote-env servers.

**The active line of work is pi0.5 HITL on LIBERO**, including robustness evaluation with
LIBERO-plus.

## Documentation index

| Read this | For |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | package map, how a run is wired, extension recipes |
| [docs/WORKFLOWS.md](docs/WORKFLOWS.md) | runbooks + commands for every pipeline |
| [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) | obs/action contracts, HDF5 + LeRobot schemas, pi0 format |
| [docs/CONFIGS.md](docs/CONFIGS.md) | Hydra layout, groups, which config goes with which script |
| [docs/GOTCHAS.md](docs/GOTCHAS.md) | **the traps** — read first |
| [hitl_pi05_openpi_reference.md](hitl_pi05_openpi_reference.md) | the pi0.5 HITL pipeline in full detail |
| [docs/LIBERO_PLUS.md](docs/LIBERO_PLUS.md) | the robustness benchmark |
| [docs/REAL_ROBOT_README.md](docs/REAL_ROBOT_README.md) | real-robot setup |
| [docs/TESTS.md](docs/TESTS.md) | test/command reference |
| [README.md](README.md) | install |
| `train_commands.sh` | the exact commands run so far (not curated, but real) |

## Repo map

```
robometer_policy_learning/
  algorithms/   losses + update rules      (configuration_X.py / modeling_X.py per method)
  modules/      networks                   (BaseActor/BaseCritic; mlp, rnn, transformer, diffusion, encoders)
  buffers/      transition storage         (+ samplers.py)
  envs/         env construction/wrappers  (setup_libero_env, libero_plus.py, RemoteEnv, ...)
  rollouts/     stepping envs w/ a policy  (RolloutWorker, EvaluationWorker, DSRL variants)
  runners/      training loops             (SerialRunner, async_runner)
  distributed/  gRPC servers + clients     (learner server, reward-relabel server, protos)
  robots/       real-robot host servers
  utils/        glue                       (training_utils.py wiring, pi0_integration.py, hitl_utils.py)
  configs/      Hydra YAML + dataclasses
scripts/        31 python entry points (train*, eval*, collect*, convert*, serve*, hitl_pi0_loop.py)
third_party/    submodules: dsrl_openpi, LIBERO, robometer, LIBERO-plus
tests/          pytest suite
```

`utils/training_utils.py::setup_training(cfg) -> TrainingComponents` is the funnel most training
scripts pass through (envs, models, buffers, algorithms, logging). DSRL and pi0.5-HITL deliberately
bypass parts of it.

## Environment

Python 3.11, managed by `uv`. **Two mutually exclusive environments:**

```bash
uv sync                                        # default: openpi + libero + libero-plus
uv sync --group robometer --no-default-groups  # the reward-model path
```

`robometer` and `openpi` pin conflicting `torch`/`datasets` and can never be installed together.
`uv run` re-syncs *and prunes*, so `uv pip install` does not persist — add deps to `pyproject.toml`.

## Rules to follow

1. **Read [docs/GOTCHAS.md](docs/GOTCHAS.md) first.** Most bugs here are one of those 17 items.
2. **Never import `robometer` directly** — use `utils/logging_compat.py` (logging) and
   `utils/robometer_compat.py` (everything else) so the code still imports without that group.
3. **Keep the JAX/cv2 preamble** at the top of pi0/openpi scripts (env vars → `jax.devices()` → `cv2`
   → torch). Order is load-bearing.
4. **Actions are stored in env space and normalized to `[-1, 1]` by the buffer**; `BaseActor.act()`
   unnormalizes. Don't re-normalize anywhere else.
5. **Follow the file-pair convention** when adding components: `configuration_<x>.py` +
   `modeling_<x>.py`, plus a YAML in the matching config group.
6. **Selecting an algorithm takes two overrides**: `algorithm@online_algorithm=sac` *and*
   `alg.online_alg_name=sac`.
7. **Don't `git add -A`.** Large blobs (datasets, zips) at the repo root are not all gitignored; a
   21 GB zip once had to be gc'd back out of `.git`. Stage explicit paths.
8. **Prefer code over docstrings when they disagree** — a few docstrings have gone stale (gotcha 16).

## Common commands

```bash
# tests
uv run pytest tests/ -q                    # unit
uv run pytest tests/ -m integration        # builds envs/models (slow); Box2D tests need --extra dev

# lint (config in pyproject.toml: line-length 120, ruff select E,W,F,I,B,C4,UP,RUF)
uvx ruff check --no-fix <paths>

# a training run (see docs/WORKFLOWS.md for the rest)
uv run python scripts/train_supervised.py --config-path=../robometer_policy_learning/configs \
  --config-name=robomimic_lowdim_bc env.h5_dataset_path=/path/to/data.hdf5

# pi0.5 HITL round on LIBERO (needs a display for teleop)
uv run python scripts/hitl_pi0_loop.py --rounds 3 --env-name libero_90 --task-ids 57 58 59 \
  --repo-id-prefix <you>/libero_hitl --exp-prefix hitl_t575859 --dry-run
```

Runs land in `outputs/<date>/<time>/` with `.hydra/config.yaml` + `checkpoints/<step>/`; eval and HITL
scripts take `load_dir=<that dir>` to reproduce the exact env and model.

## Code style

* Ruff, line length 120. Match surrounding style — the codebase uses `typing.Optional`/`List`
  (pre-PEP-585) throughout, so don't modernize files you're only lightly touching.
* Comments explain *why*, not what. Existing comments carry a lot of hard-won context — preserve them.
* New pipelines get a module docstring with a runnable `Usage:` block; that is how the existing
  scripts document themselves and it is the fastest way for the next agent to learn them.
