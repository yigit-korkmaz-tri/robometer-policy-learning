# Gotchas

Non-obvious constraints that have already cost time. Read this before your first run and before
debugging anything that "should work". Ordered roughly by how often they bite.

---

## Environment / install

### 1. `robometer` and `openpi` cannot be installed together
They pin conflicting versions (`robometer` → `datasets==4.1.1` and, via `xformers`, `torch==2.8.0`;
`openpi` → `datasets<4` and `torch==2.7.1`). `pyproject.toml` declares them as conflicting uv groups,
so uv locks each combination separately and asking for both fails **by design**.

```bash
uv sync                                        # default: openpi + libero + libero-plus  (pi0/LIBERO/DSRL/HITL)
uv sync --group robometer --no-default-groups  # the reward-model path
```
Anything importing `robometer` (reward-relabel server, `buffers/robometer_replay_buffer.py`,
`rollouts/dsrl_evaluation_worker.py`, `scripts/train_dsrl.py`) needs the second env. Elsewhere,
import logging via `utils/logging_compat.py` and robometer helpers via `utils/robometer_compat.py`,
which degrade gracefully when robometer is absent — **never** `from robometer.utils.logger import ...`.

### 2. `uv run` re-syncs and *prunes* the environment
Every `uv run` brings the venv back in line with the lockfile, removing anything not in the resolved
set. So `uv pip install X` does not survive the next `uv run`. To add a dependency, put it in
`pyproject.toml`; if it must be present for the default workflow, add its group to
`[tool.uv] default-groups` (this is exactly why `libero-plus` is a default group — otherwise
`wand`/`scikit-image` get pruned and `env.libero_plus=true` breaks).

### 3. Import order is load-bearing in the pi0 scripts
The JAX/openpi entry points set env vars **before** importing JAX and import `cv2` **before** torch:

```python
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # don't grab all VRAM
os.environ["MUJOCO_GL"] = "egl"                                   # headless rendering
import jax; jax.devices(); del jax                                # claim a clean GPU context first
import cv2  # before torch: cv2's HighGUI deadlocks against the pynput listener otherwise
```
Keep this preamble when writing a new script in that family. For openpi training OOMs, raise
`XLA_PYTHON_CLIENT_MEM_FRACTION` (JAX defaults to 0.75; `hitl_pi0_loop.py` passes 0.9).

---

## Data and training semantics

### 4. Actions: env space on disk, `[-1, 1]` in the loss
Buffers store env-space actions and normalize to `[-1, 1]` **at sample time**, using
`min_action`/`max_action`; `BaseActor.act()` unnormalizes at inference. If you build a buffer without
those bounds, normalization silently no-ops and the policy trains in raw action scale. Details in
[DATA_FORMATS.md](DATA_FORMATS.md).

### 5. Offline data must match the online env's observation space
For offline→online RL, the dataset's obs keys and dims must equal what the online env emits. Nothing
checks this at runtime — you get a policy trained on a different state space than it is evaluated in.
Run `scripts/check_state_alignment.py`, and convert with `scripts/convert_robomimic_to_aligned.py`.

### 6. Corrections-only HITL training catastrophically forgets
Fine-tuning on human corrections alone (`hitl.store_only_human=true`, no offline anchor) destroys the
pretrained behaviour. Always mix in demos: set `hitl.offline_mode` to `pretraining`, `warmup` or
`self` (§4 of [WORKFLOWS.md](WORKFLOWS.md)). The same applies on the openpi side — fold base demos
into every round's LeRobot repo (`--libero-base-suite` / `--base-demos` in `hitl_pi0_loop.py`),
because openpi trains one repo per round.

### 7. Two image pipelines ("Mode A" vs "Mode B")
* **Mode A** — a `DinoEmbeddingWrapper` on the env precomputes frozen DINOv2 features; the actor sees
  a `dino_embedding` vector. Configured by `model.dinov2_model` + `env.dino_image_keys`.
* **Mode B** — the actor owns an image featurizer and consumes raw frames. Configured by the nested
  `model.image_encoder` block (or the older `model.image_encoder_type`, e.g. `impala`).

They are not interchangeable at eval time: a checkpoint must be evaluated in the mode it was trained
in (`eval_policy.py` rebuilds the right one from the saved config). **`--vectorization async` is
incompatible with Mode A**, because the DINO encoder cannot live in the subprocess envs.

### 8. `flow_mile` needs a fixed-shape rollout pool
Flow-MILE's frozen baselines are stored per frame as `rollout_samples [P, H, A]`, and **every frame in
an exported LeRobot repo must share one `(P, H, A)`**. The exporter takes the shape from the first
demo that has it and zero-fills the rest, warning when a demo mismatches — zero-filled label-0
baselines silently weaken the objective. Set `hitl.rollout_pool_size >= flow_mile.num_samples` at
collection time (it costs one extra policy forward per human-intervened frame).

---

## LIBERO / LIBERO-plus

### 9. LIBERO and LIBERO-plus task ids are NOT comparable
Under LIBERO-plus, `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` contain **only
perturbed variants** (2402 / 2518 / 2591 / 2519 tasks) in a different order; the original unperturbed
tasks are absent. `libero_90` is byte-identical in both. Pin perturbed tasks by **name**
(`env.task_name` / `env.task_names`), and remember that with `env.perturbation` set, `env.task_id`
means a **base** task id `0..9` in plain-LIBERO ordering.

### 10. LIBERO-plus is never pip-installed, and one process can't mix the two
It ships the same top-level `libero` package as LIBERO. `envs/libero_plus.py::activate()` redirects
`$LIBERO_CONFIG_PATH` and inserts the checkout at `sys.path[0]` **before the first `import libero`** —
so keep `import libero` out of module top-level in `envs/`. Activation raises if `libero` was already
imported from elsewhere.

### 11. Objects-Layout perturbations only apply via init states
`env.init_state_index=null` under LIBERO-plus means robosuite randomizes placements and the Objects
Layout perturbations are **not reproduced**. The default `"auto"` resolves to index 0 there. A warning
is logged, but it is easy to miss.

### 12. Sensor-noise tasks need system ImageMagick
LIBERO-plus imports `wand` unconditionally. Without `libmagickwand-dev`, `activate()` installs a stub
so everything still runs *except* the 336 motion-blur tasks (`_noise_N`, N ≤ 10), which raise. Fix:
`sudo apt-get install -y libmagickwand-dev`.

### 13. Upstream LIBERO quirks already worked around
Don't "fix" these again — they are handled in `envs/libero_plus.py` and `envs/dsrl_env_wrappers.py`:
* `torch.load` on `.pruned_init` files fails on torch ≥ 2.6 (`weights_only=True` default) →
  `load_task_init_states`.
* `Task.language` is derived from the *filename*, so perturbation suffixes leak into the instruction
  → the instruction is read from the loaded bddl instead.
* `Benchmark._make_benchmark` prints 2402+ integers per env build → `make_task_suite` silences it.
* 121 tasks (all `libero_goal` / Light Conditions) have `difficulty_level: null` upstream.

---

## Repo hygiene

### 14. Large local files are not all gitignored
`data/`, `logs/`, `outputs/`, `wandb/` are ignored, but a blob dropped at the repo **root** is not — a
21 GB `project_data.zip` was swept in by a `git add -A` and had to be unstaged and `git gc`'d out
(38 GB → 766 MB `.git`). GitHub rejects files > 100 MB, so this fails the push rather than succeeding
quietly. Check `git status` before staging, and prefer explicit paths over `git add -A`.

### 15. Submodule pointers move
`third_party/dsrl_openpi` tracks branch `hitl-work` on a fork. When the parent repo records a new
submodule commit, **push the submodule first** — otherwise clones cannot resolve the pointer.

### 16. Some docstrings are stale
Example: `scripts/collect_hitl_rollout.py` says the stock `H5ReplayBuffer` ignores per-demo
`intervention` arrays and needs a loader extension. That extension **has since been made** —
`H5ReplayBuffer._get_intervention_array` reads them and falls back to `default_intervention_label`.
Prefer the code over prose when they disagree, and fix the prose when you notice.

### 17. HITL run configs don't carry `env` / `training` / `model` / `policy`
A `train_hitl.py` run adopts those from the *pretraining* run it started from, at runtime — after
Hydra has already written `.hydra/config.yaml`. So a HITL run's saved config lacks them; loaders
follow its `load_dir` back to the pretraining run. Keep that indirection intact when adding tooling
that reads run directories.
