# Architecture

How the codebase is organised, what each package owns, and where to plug new things in.
For *running* things see [WORKFLOWS.md](WORKFLOWS.md); for the traps see [GOTCHAS.md](GOTCHAS.md).

## Big picture

`robometer_policy_learning` is a research framework for **learning robot manipulation policies**,
covering four families of method that share one set of components:

1. **Supervised / imitation** — BC, Diffusion Policy (DP), Flow Matching, from offline datasets.
2. **Online + offline RL** — SAC, IQL, with ground-truth or *learned* rewards (Robometer / RoboReward
   reward models, served over gRPC).
3. **DSRL** (diffusion steering RL) — freeze a large VLA (pi0 / pi0.5 via openpi) and train a small
   RL policy that steers it by choosing its *noise* input rather than its actions.
4. **HITL** (human-in-the-loop) — a human takes over during rollouts; corrections train the policy via
   HG-DAgger or the MILE family (MILE / Flow-MILE / Diffusion-MILE).

Environments span simulation (LIBERO, robosuite/robomimic, MetaWorld, SIMPLER) and real robots
(YAM arm via raiden, DROID, WidowX) behind a common `gym`/`gymnasium` vector-env interface.

## Three planes

Almost every entry point is an arrangement of the same three planes:

```
  interaction plane          data plane                learning plane
  ─────────────────          ──────────                ──────────────
  envs/       wrappers   →   buffers/    transitions → algorithms/  losses
  rollouts/   workers    ←   samplers                  modules/     nets
  runners/    loops                                    (+ reward models)
```

* **interaction** — `envs/` builds and wraps environments; `rollouts/` steps them with a policy and
  produces transitions; `runners/` sequences offline/online phases.
* **data** — `buffers/` stores transitions (in memory, from HDF5, or streamed from a remote learner)
  and `buffers/samplers.py` decides what a batch looks like.
* **learning** — `algorithms/` owns losses and update rules; `modules/` owns the networks
  (actors, critics, encoders, featurizers) they use.

## Package map

| Package | Owns | Key entry points |
| --- | --- | --- |
| `algorithms/` | Loss functions and update rules, one subpackage per method | `BaseAlgorithm` (`modeling_algorithm.py`); `sac/`, `iql/`, `bc/`, `dp/`, `flow_matching/`, `mile/`, `flow_mile/`, `diffusion_mile/` |
| `modules/` | Networks: actors, critics, encoders, featurizers | `BaseActor`/`BaseCritic` (`modules/base/`); `mlp/`, `rnn/`, `transformer/`, `diffusion/`, `encoders/`, `dsrl/` |
| `buffers/` | Transition storage + sampling | `BaseReplayBuffer`, `ReplayBuffer`, `H5ReplayBuffer`, `MixedReplayBuffer`, `EpisodicReplayBuffer`, `SuccessFailureReplayBuffer`, `AsyncRewardRelabelBuffer`, `samplers.py` |
| `envs/` | Env construction and wrapper stacks | `setup_libero_env`, `setup_robosuite_env`/`setup_robomimic_env`, `make_remote_robot_env`, `make_simpler_env`, `libero_plus.py`, `RemoteEnv` |
| `rollouts/` | Stepping envs with a policy | `RolloutWorker`, `EvaluationWorker`, `DSRLRolloutWorker`, `DSRLEvaluationWorker`, `RobometerRolloutWorker` |
| `runners/` | Training loop orchestration | `SerialRunner`, `async_runner.py` (learner / robot processes) |
| `distributed/` | gRPC services and clients | `servers/learner_server.py`, `servers/reward_relabel_server.py`, `clients/`, `protos/` |
| `robots/` | Real-robot host-side servers | `droid_remote_server.py`, `widowx_remote_server.py`, `remote_server_utils.py` |
| `utils/` | Cross-cutting glue | `training_utils.py` (wiring), `pi0_integration.py`, `pi0_hitl_utils.py`, `hitl_utils.py`, `dsrl_utils.py`, `env_utils.py` |
| `configs/` | Hydra YAML + dataclass schema | `config.yaml`, groups (`algorithm/`, `policy/`, ...), `configs.py` |
| `loggers/` | wandb / stdout logging | `WandbLogger` |

## How a training run is wired

`utils/training_utils.py::setup_training(cfg) -> TrainingComponents` is the single funnel that most
training scripts go through. It reads the Hydra config and returns a dataclass holding *everything*:

```
setup_training(cfg)
  ├─ logging      : loguru + WandbLogger (Hydra's runtime.output_dir is the run dir)
  ├─ encoders     : optional DINOv2 image model + SentenceTransformer language model
  ├─ envs         : env + eval_env (vectorized), plus `remove_obs_keys`
  ├─ models       : build_actor_critic_models(...) -> actor, critic, v_net
  ├─ reward model : optional Robometer / RoboReward (local or gRPC client)
  ├─ buffers      : create_buffer(...) -> offline_buffer, online_buffer
  └─ algorithms   : offline_algo (e.g. IQL/BC) and/or online_algo (e.g. SAC)
```

`TrainingComponents` then feeds `runners/serial_runner.py::SerialRunner`, which runs the offline phase
(`num_offline_steps`), then the online phase (`num_rollouts`), interleaving evaluation every
`eval.eval_freq`. The async/distributed variant lives in `runners/async_runner.py` and splits the same
pieces across a learner process and a robot/rollout process talking over gRPC.

**The DSRL and HITL paths deliberately bypass parts of this.** `scripts/train_dsrl.py` builds
DSRL-specific workers, and the pi0.5 LIBERO HITL path (`scripts/collect_hitl_libero_pi0.py`,
`scripts/hitl_pi0_loop.py`) does not use `setup_training` at all — it drives openpi training as a
subprocess. See [WORKFLOWS.md](WORKFLOWS.md).

## Conventions the code follows

* **HuggingFace-style pairs.** Every algorithm is `configuration_<name>.py` (a dataclass config) plus
  `modeling_<name>.py` (the implementation). Same for modules: `configuration_mlp_actor.py` /
  `modeling_mlp_actor.py`. Follow this when adding one.
* **Config as dataclass + YAML.** `configs/configs.py` defines typed dataclasses; the YAML in
  `configs/` populates them, and `utils/parser.py::dictconfig_to_dataclass` converts. Hydra groups
  (`algorithm@online_algorithm=sac`) select variants.
* **Observations are dicts.** Envs emit `Dict` observation spaces (`state`, image keys, `language`,
  `prompt`, ...). Wrappers add/remove keys; `remove_obs_keys` tells buffers what to drop.
* **Actions are stored in env space, learned in `[-1, 1]`.** See [DATA_FORMATS.md](DATA_FORMATS.md) —
  this one bites people.

## Extension recipes

### Add an algorithm
1. `algorithms/<name>/configuration_<name>.py` — dataclass extending `BaseAlgorithmConfig`.
2. `algorithms/<name>/modeling_<name>.py` — class extending `BaseAlgorithm`; implement the update /
   loss methods the base class declares.
3. `configs/algorithm/<name>.yaml` — the Hydra group entry.
4. Register it where algorithms are constructed (`utils/training_utils.py`) and select it with
   `algorithm@online_algorithm=<name> alg.online_alg_name=<name>`.

### Add a policy or critic architecture
Add `modules/<arch>/{configuration,modeling}_<arch>_{actor,critic}.py` extending `BaseActor` /
`BaseCritic`, plus `configs/policy/<arch>.yaml` and `configs/value_function/<arch>.yaml`. `BaseActor`
owns the action (un)normalization contract — inherit it, don't reimplement it.

### Add an environment
Add a `setup_<env>_env(...)` builder in `envs/` returning `(vector_env, remove_obs_keys)`, matching
`setup_libero_env`'s shape. Keep the observation dict contract; if the env is remote, wrap
`RemoteEnv`. Wire it into `utils/env_utils.py::make_env`.

### Add a buffer or sampler
Extend `BaseReplayBuffer` (`buffers/base_replay_buffer.py`) or `BaseSampler`
(`buffers/samplers.py`), then wire into `utils/training_utils.py::create_buffer`.

### Add a HITL intervention criterion
Criteria are built in `utils/hitl_utils.py::get_intervention_criteria(name, ...)`. Existing names:
`fixed`, `mile`, `mile_window`, `flow_mile`, `flow_mile_paired`, `flow_mile_window`,
`diffusion_mile`, `diffusion_mile_paired`, `diffusion_mile_window`, `episode_len`,
`episode_num_interventions`. The MILE-family factories live next to their algorithms (e.g.
`algorithms/flow_mile/modeling_flow_mile.py::make_flow_mile_intervention_criteria`).

## Third-party submodules

Under `third_party/`, wired as editable uv dependency *groups* (see `pyproject.toml`):

| Path | What | Installed? |
| --- | --- | --- |
| `dsrl_openpi` | openpi (pi0 / pi0.5), fork tracking branch `hitl-work` | yes (`openpi` group) |
| `LIBERO` | LIBERO benchmark | yes (`libero` group) |
| `robometer` | Robometer reward model stack | yes, but **conflicts with openpi** |
| `LIBERO-plus` | robustness benchmark, 10,030 perturbed tasks | **no** — activated at runtime |

See [GOTCHAS.md](GOTCHAS.md) for why `robometer` and `openpi` cannot coexist, and
[LIBERO_PLUS.md](LIBERO_PLUS.md) for why LIBERO-plus is not installed.
