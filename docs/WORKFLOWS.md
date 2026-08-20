# Workflows

Runbooks for every pipeline in the repo. Each section says what it does, how to run it, the knobs
that matter, and where the code lives. Read [GOTCHAS.md](GOTCHAS.md) before the first run of anything.

**Common facts**
* Everything runs through `uv run python ...` (uv re-syncs the env on each invocation).
* All training/eval entry points use **Hydra**. Outputs land in `outputs/<date>/<time>/` containing
  `.hydra/config.yaml` (the exact resolved config) and `checkpoints/<step>/`. Pin it with
  `hydra.run.dir=...`.
* Most eval/HITL scripts take `load_dir=<run output dir>` and read env/model/policy settings from
  that run's saved Hydra config — so a checkpoint is always evaluated with the env it was trained on.
* Set `logging.wandb_entity` / `logging.wandb_name`, or `logging.wandb=false` where supported.

---

## 1. Supervised / imitation learning

Train BC, Diffusion Policy or Flow Matching from an offline dataset. Entry point:
`scripts/train_supervised.py`. Configs: `robomimic_{lowdim,image}_{bc,dp,flow}.yaml`,
`yam_lerobot_flow*.yaml`, `pusht_lerobot_flow.yaml`.

```bash
uv run python scripts/train_supervised.py \
  --config-path=../robometer_policy_learning/configs \
  --config-name=robomimic_lowdim_bc \
  policy=mlp policy.mlp.hidden_dims=[256] policy.mlp.dropout_rate=0.1 \
  env.h5_dataset_path="/path/to/low_dim_converted.hdf5" \
  training.num_offline_steps=10000 eval.eval_freq=1000 \
  logging.wandb_name=lift_bc logging.wandb_entity=<entity>
```

Knobs that matter: `policy=` (`mlp` / `rnn` / `transformer`), `training.chunk_size` +
`training.n_action_steps` (action chunking / open-loop horizon), `training.normalize_lowdim_obs`,
`algorithm@offline_algorithm=` (`bc` / `dp` / `flow`) with `alg.offline_alg_name` matching.
See `train_commands.sh` for the exact per-task commands used so far.

**Evaluate** a run (autonomous, no wandb):
```bash
uv run python scripts/eval_policy.py --load-dir outputs/2026-06-04/16-21-35 \
  --num-episodes 50 --num-envs 25 --vectorization async
```
`--vectorization async` runs one subprocess per env (the real speedup for image obs) but is
**incompatible with Mode A / DINO-embedding envs** — see [GOTCHAS.md](GOTCHAS.md).

---

## 2. Online and offline→online RL

Entry point: `scripts/train.py` (config `config.yaml`, or `libero_online_rl.yaml` for LIBERO).
Online-only, or offline pretraining followed by online fine-tuning.

```bash
# Online SAC with ground-truth rewards
uv run python scripts/train.py --config-path=../robometer_policy_learning/configs \
  --config-name=config algorithm@online_algorithm=sac alg.online_alg_name=sac \
  env.use_gt_rewards=true

# Offline IQL -> online SAC
uv run python scripts/train.py --config-path=../robometer_policy_learning/configs \
  --config-name=config \
  algorithm@offline_algorithm=iql alg.offline_alg_name=iql \
  algorithm@online_algorithm=sac  alg.online_alg_name=sac \
  env.use_gt_rewards=true env.h5_dataset_path=/path/to/data.h5
```

Knobs: `training.num_offline_steps`, `training.num_rollouts` (online env steps),
`online_algorithm.learning_starts`, `buffer.capacity`, `buffer.sample_ratio` (offline:online mix),
`eval.eval_freq`, `eval.eval_num_episodes`.

### With a learned reward model
```bash
uv run python scripts/train.py ... reward_model=robometer \
  reward_model.model_path=robometer/Robometer-4B env.use_gt_rewards=false
```
The reward model can run **in-process** or behind a **gRPC relabeling server** (recommended: it keeps
the VLM off the training GPU and lets rollouts continue while rewards are computed):

```bash
uv run python scripts/start_reward_relabel_server.py   # config: reward_relabel_server.yaml
# then train with:
#   buffer.distributed_reward_relabel.enabled=true
#   buffer.distributed_reward_relabel.server_address=localhost:50052
```
Server: `distributed/servers/reward_relabel_server.py` (both `RewardRelabelService` and
`RoboRewardRelabelService`). Client-side buffering: `buffers/remote_reward_relabel_buffer.py` and
`envs/async_reward_relabel_wrapper.py`. Smoke test with `scripts/test_reward_relabel_server.py`.

> `reward_model=robometer` needs the **robometer** uv group, which cannot coexist with **openpi**.
> See [GOTCHAS.md](GOTCHAS.md).

### Distributed / async
`scripts/train_async.py` (config `config_distributed.yaml`) runs three roles that talk over gRPC:
`learner` (trains, serves weights + ingestion), `rollout` (collects, streams transitions, pulls
weights), `eval` (periodically evaluates the latest policy). Code: `runners/async_runner.py`,
`distributed/servers/learner_server.py`, `distributed/clients/`.

---

## 3. DSRL — RL steering of a frozen pi0 / pi0.5

Freeze the VLA and train a small policy over its **noise** input. Entry point:
`scripts/train_dsrl.py` (config `dsrl_config*.yaml`, e.g. `dsrl_libero_config.yaml`,
`dsrl_droid_config.yaml`, `dsrl_bridge_config.yaml`). It supports the same
serial / learner / rollout / eval modes as `train_async.py`.

Evaluation:
```bash
# pi0 with random noise, or a trained steering policy
uv run python scripts/eval_pi0.py use_random_noise=false policy_checkpoint=./checkpoints/policy.pt
uv run python scripts/eval_trained_dsrl.py     # config: eval_trained_dsrl.yaml
```

Key pieces: `utils/pi0_integration.py` (`Pi0Wrapper`, `pi0_infer_with_noise`, VLM feature
extraction), `rollouts/dsrl_{rollout,evaluation}_worker.py`, `envs/dsrl_env_wrappers.py`
(`DummyDSRLEnv` defines the noise action space), `modules/dsrl/`, `utils/dsrl_utils.py`
(`ActionQueue`, chunk reward discounting).

`pi0.action_exec_len` controls how many actions of each predicted chunk are executed open-loop before
replanning; the effective discount is `gamma ** action_exec_len`.

---

## 4. HITL on robomimic / robosuite (MILE, HG-DAgger)

Human (or simulated expert) takes over during rollouts; corrections train the policy.

**Collect** corrections into an HDF5:
```bash
uv run python scripts/collect_hitl_rollout.py --config-name robomimic_collect_hitl \
  load_dir=/path/to/pretrained_run hitl.human_mode=simulated \
  hitl.expert_load_dir=/path/to/expert_run \
  hitl.collect_num_rollouts=50 hitl.collect_output_path=data/hitl_rollouts.hdf5
```
`hitl.human_mode`: `real` (keyboard / SpaceMouse teleop) or `simulated` (a frozen expert policy plus
an intervention criterion — see `utils/hitl_utils.py::get_intervention_criteria`).

**Train** on them — iteratively (collect ↔ train) or offline from a precollected file:
```bash
uv run python scripts/train_hitl.py --config-name robomimic_hitl \
  algorithm@offline_algorithm=flow_mile alg.offline_alg_name=flow_mile \
  load_dir=/path/to/pretrained_run checkpoint=10000 \
  hitl.precollected_hitl_dataset=data/hitl_rollouts.hdf5 \
  hitl.num_iterations=1 hitl.rollouts_per_iter=0 hitl.train_steps_per_iter=1000 \
  offline_algorithm.intervention_cost=1.0 offline_algorithm.actor_optimizer_lr=1.0e-5 \
  logging.wandb_name=square_flow_mile
```

The two objectives differ in **what gets stored**:
* **HG-DAgger** — `hitl.store_only_human=true`, train on corrections only (BC on label 1).
* **MILE family** (`mile`, `flow_mile`, `diffusion_mile`) — store *all* steps; the loss uses both the
  human and policy labels, with a frozen rollout-policy snapshot as the baseline.

Anti-forgetting is controlled by `hitl.offline_mode`: `null` (none), `"pretraining"` (mix the
pretraining H5), `"warmup"` (mix a fixed user-provided H5 — best for comparing algorithms), or
`"self"` (collect the policy's own successful rollouts as the anchor). **Corrections-only training
without an anchor catastrophically forgets** — see [GOTCHAS.md](GOTCHAS.md).

Sweeps: `scripts/sweep_flow_mile.py`. Real commands used so far: `train_commands.sh` (`## HitL`).

---

## 5. HITL on LIBERO with pi0.5 (openpi)

The active project. A separate pipeline from §4 — it fine-tunes pi0.5 through openpi rather than the
in-house algorithms. **Full detail lives in
[`hitl_pi05_openpi_reference.md`](../hitl_pi05_openpi_reference.md)**; this is the summary.

One round = collect → export → norm-stats → train → eval:

```bash
# 0. Eval the current checkpoint on a suite
uv run python scripts/eval_pi0_libero.py --config-name libero_eval \
  env.env_name=libero_90 'env.task_ids=[57,58,59]' eval.num_episodes=20 \
  pi0.checkpoint=gs://openpi-assets/checkpoints/pi05_libero/

# 1. Collect human corrections (needs a display + keyboard/SpaceMouse)
uv run python scripts/collect_hitl_libero_pi0.py --config-name libero_collect_hitl \
  env.env_name=libero_90 env.task_id=57 teleop.device=spacemouse \
  hitl.collect_num_rollouts=20 hitl.collect_output_path=data/hitl/round0_t57.hdf5

# 2. Export to a LeRobot dataset (+ optional base demos), 3. norm stats, 4. train (openpi env)
uv run python scripts/export_hitl_to_lerobot.py ...
uv run python third_party/dsrl_openpi/scripts/compute_norm_stats.py ...
uv run python third_party/dsrl_openpi/scripts/train.py pi05_libero_hitl_lora ...
```

Or drive all of it:
```bash
uv run python scripts/hitl_pi0_loop.py --rounds 3 --env-name libero_90 --task-ids 57 58 59 \
  --collect-num-rollouts 5 --num-train-steps 3000 --eval-num-episodes 20 \
  --repo-id-prefix <you>/libero_hitl --exp-prefix hitl_t575859 \
  --libero-base-suite libero_90 --libero-base-num-demos 10 --train-config pi05_libero_hitl_lora
```
`--dry-run` prints every command without executing — use it first. Each round runs its steps as
separate subprocesses so JAX training and robosuite teleop never share a process.

Teleop controls: `Tab` take/release, `q` abort episode, `ESC` finish collection early (saves what was
collected). Code: `utils/pi0_hitl_utils.py::Pi0LiberoHitlWorker`.

Training configs on the openpi side: `pi05_libero_hitl_lora` (HG-DAgger) and
`pi05_libero_flow_mile_lora` (Flow-MILE). When collecting with a fine-tuned checkpoint, set
`pi0.config_name` so the LoRA architecture is reconstructed correctly.

---

## 6. LIBERO-plus robustness evaluation

Perturbed variants of LIBERO tasks (10,030 tasks, 7 perturbation dimensions). One-time setup:
`bash scripts/setup_libero_plus.sh` (6.4 GB asset download). Then add `env.libero_plus=true` to the
eval/collect commands above, and optionally `env.perturbation=<family>` to run every episode under a
different variant of that family:

```bash
uv run python scripts/eval_pi0_libero.py --config-name libero_eval \
  env.libero_plus=true env.env_name=libero_spatial 'env.task_ids=[3,7]' \
  env.perturbation=camera eval.num_episodes=20
```

Full documentation: [LIBERO_PLUS.md](LIBERO_PLUS.md).

---

## 7. Real robots

* **Serve a policy** trained by `train_supervised.py` over websockets (same wire protocol as openpi's
  serve_policy, so openpi-client scaffolding works):
  ```bash
  uv run python scripts/serve_policy.py --load-dir outputs/2026-08-10/18-13-47 --port 8000 --selftest
  ```
  Consistency with training is enforced by `<run>/checkpoints/policy_metadata.json` (obs
  normalization stats, image size, camera map, `n_action_steps`).
* **Robot-side runners** for the YAM arm: `scripts/raiden_flow_rollout.py` (robometer-trained
  flow/dp/bc) and `scripts/raiden_pi05_rollout.py` (pi0.5 via openpi).
* **Remote env servers** (the robot exposes a gym-like env over a socket):
  `robometer_policy_learning/robots/{droid,widowx}_remote_server*.py`, consumed by
  `envs/remote_env.py::RemoteEnv` / `envs/dsrl_env_wrappers.py::make_remote_robot_env`.
* Background and setup: [REAL_ROBOT_README.md](REAL_ROBOT_README.md).

---

## 8. Data preparation and inspection

| Script | Purpose |
| --- | --- |
| `convert_robomimic_to_aligned.py` | rewrite a robomimic dataset into the online env's obs format |
| `convert_lerobot_to_h5.py` | LeRobot (real-robot) dataset → robomimic-style HDF5 |
| `regenerate_libero_dataset.py` | replay LIBERO demos to regenerate observations |
| `check_robomimic_dataset.py` | validate a dataset, optionally rebuild the env and step it |
| `check_state_alignment.py` | verify dataset `obs/state` matches the online env's `state` |
| `inspect_obs_stats.py` | per-key statistics, for checking normalization |
| `visualize_libero_demos.py` | render demo HDF5s to mp4 |
| `compare_correction_vs_expert.py` | correction rollout vs LIBERO expert demo, side by side |
| `collect_warmup_demos.py` | autonomous warmup demos from a pretrained policy |

---

## 9. Tests

```bash
uv run pytest tests/ -q                 # unit tests
uv run pytest tests/ -m integration     # integration (builds envs/models; slow)
```
`tests/test_integration_{sac,iql}.py` need Box2D (`uv sync --extra dev`) and will error without it.
More context in [TESTS.md](TESTS.md).
