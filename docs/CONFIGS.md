# Configuration reference

Everything is configured with [Hydra](https://hydra.cc). This is the map of what lives where and how
overrides compose.

## Layout

```
robometer_policy_learning/configs/
├── config.yaml                  # the base experiment config (env/training/buffer/logging/alg/eval)
├── <workflow>.yaml              # per-workflow variants (robomimic_*, libero_*, dsrl_*, yam_*, ...)
├── algorithm/                   # bc dp flow flow_mile mile diffusion_mile iql sac + *_dsrl_sac
├── policy/                      # base mlp rnn transformer          (actor architecture)
├── value_function/              # base mlp rnn transformer          (critic architecture)
├── reward_model/                # robometer roboreward
├── env/                         # remote_robot
├── configs.py                   # typed dataclasses mirroring the YAML
└── register.py                  # Hydra ConfigStore registration
```

`configs.py` defines the schema (`EnvironmentConfig`, `TrainingConfig`, `ModelConfig`, `PolicyConfig`,
`ReplayBufferConfig`, `AlgorithmConfig`, `RewardModelConfig`, `EvaluationConfig`, `LoggingConfig`,
`TrainConfig`, `DSRLSectionConfig`, ...). `utils/parser.py::dictconfig_to_dataclass` converts a
resolved `DictConfig` into those dataclasses, so **the dataclass is the source of truth for valid
keys and defaults** — read it when a YAML key is unclear.

## Config groups and the `@` syntax

`config.yaml`'s defaults list selects one entry from each group:

```yaml
defaults:
  - algorithm@offline_algorithm: null   # no offline phase by default
  - algorithm@online_algorithm: sac
  - reward_model: null
  - policy@policy: mlp
  - value_function@value_function: mlp
```

`algorithm@online_algorithm: sac` means "load `algorithm/sac.yaml` **into the `online_algorithm`
node**". That is why selecting an algorithm takes two overrides — one to load its hyperparameters,
one to tell the code which algorithm to build:

```bash
algorithm@online_algorithm=sac  alg.online_alg_name=sac
algorithm@offline_algorithm=iql alg.offline_alg_name=iql
```
Forgetting the `alg.*_alg_name` half is a classic mistake: the config loads but the wrong algorithm
is constructed.

Override nested values with dotted paths, and quote list literals for the shell:
```bash
online_algorithm.critic_optimizer_lr=1e-5 'env.task_ids=[57,58,59]' policy.mlp.hidden_dims=[256]
```

## Top-level configs and their entry points

| Config | Used by | Purpose |
| --- | --- | --- |
| `config.yaml` | `train.py`, `train_supervised.py`, `train_hitl.py`, `collect_hitl_rollout.py`, `rollout_hitl.py` | base experiment config |
| `config_distributed.yaml` | `train_async.py` | learner / rollout / eval roles |
| `libero_online_rl.yaml` | `train.py` | online RL in LIBERO |
| `robosuite.yaml` | `train.py` | online RL in robosuite |
| `robomimic_{lowdim,image}_{bc,dp,flow}.yaml` | `train_supervised.py` | supervised training on robomimic |
| `robomimic_hitl.yaml`, `robomimic_hgdagger.yaml` | `train_hitl.py` | HITL training (MILE family / HG-DAgger) |
| `robomimic_collect_hitl.yaml`, `robomimic_collect_hgdagger.yaml` | `collect_hitl_rollout.py` | HITL collection |
| `yam_lerobot_flow*.yaml`, `pusht_lerobot_flow.yaml` | `train_supervised.py` | flow training on LeRobot datasets |
| `dsrl_libero_config.yaml`, `dsrl_droid_config.yaml`, `dsrl_bridge_config.yaml`, `dsrl_*_remote_*.yaml` | `train_dsrl.py` | DSRL per environment |
| `eval_pi0.yaml` | `eval_pi0.py` | pi0 eval (SIMPLER / remote robot) |
| `libero_eval.yaml` | `eval_pi0_libero.py` | pi0/pi0.5 eval on LIBERO(-plus) |
| `libero_collect_hitl.yaml` | `collect_hitl_libero_pi0.py` | pi0.5 HITL collection on LIBERO |
| `eval_trained_dsrl.yaml` | `eval_trained_dsrl.py` | evaluate a DSRL checkpoint |
| `reward_relabel_server.yaml` | `start_reward_relabel_server.py` | reward-model gRPC server |

> **`train_dsrl.py` has no usable default.** Its decorator names `dsrl_config`, which does not exist —
> always pass `--config-name=dsrl_libero_config` (or another `dsrl_*`).

Note several scripts declare `config_name="config"` in the decorator but are normally invoked with an
explicit `--config-name`; check the script's docstring for the intended one.

## Key sections

| Section | What it controls |
| --- | --- |
| `env` | `env_name`, `task_id(s)`, `h5_dataset_path`, `use_gt_rewards`, `use_full_state`, `dino_image_keys`, `extra_keys_to_drop`, `max_episode_steps` |
| `training` | `num_envs`, `num_rollouts` (online steps), `num_offline_steps`, `chunk_size` / `n_action_steps`, `use_rnn`, `normalize_lowdim_obs`, `save_interval`, `load_dir`, `continue_training` |
| `buffer` | `capacity`, `sample_ratio` (offline:online mix), success/failure buffer options, `distributed_reward_relabel.*` |
| `policy` / `value_function` | architecture + its hyperparameters (`policy.mlp.hidden_dims`, ...) |
| `model` | encoders: `dinov2_model`, `sentence_model`, `image_encoder` (Mode B) / `image_encoder_type` |
| `offline_algorithm` / `online_algorithm` | the selected algorithms' hyperparameters (lrs, batch size, `intervention_cost`, ...) |
| `hitl` | collection + iteration control: `human_mode`, `store_only_human`, `offline_mode`, `num_iterations`, `rollouts_per_iter`, `train_steps_per_iter`, `precollected_hitl_dataset`, `collect_output_path` |
| `pi0` | `checkpoint`, `config_name` (LoRA architecture), `action_exec_len`, `noise_dim` |
| `eval` | `eval_freq`, `eval_num_episodes` / `num_episodes`, `record_video` |
| `logging` | `wandb_project`, `wandb_entity`, `wandb_name`, `wandb_offline`, `log_level` |

## Run outputs

Hydra writes each run to `outputs/<date>/<time>/` (override with `hydra.run.dir=...`):

```
outputs/2026-06-11/16-17-39/
├── .hydra/config.yaml        # fully resolved config -- what actually ran
├── checkpoints/<step>/       # actor.pt, critic.pt, policy_metadata.json, ...
└── <logs, videos, results>
```

`load_dir=<that directory>` is how eval/HITL scripts recover the exact env + model setup, so **treat
run directories as the unit of provenance**. See gotcha 17 in [GOTCHAS.md](GOTCHAS.md) for the one
case where a saved config is deliberately incomplete.
