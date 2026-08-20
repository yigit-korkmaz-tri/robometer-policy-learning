# Data formats and contracts

The conventions every component agrees on. Getting these wrong is the most common source of silent
breakage, so check here before writing a converter, a buffer, or a new env.

## 1. Observations are dicts

Environments emit `gymnasium.spaces.Dict` observations. Which keys are present depends on the env and
its wrapper stack, but the vocabulary is shared:

| Key | Shape | Meaning |
| --- | --- | --- |
| `state` | `(D,)` float32 | concatenated proprioception (robosuite: `robot0_proprio-state`, plus `object-state` when `env.use_full_state=true`) |
| `image`, `agentview_image`, `observation.images.*` | `(H,W,3)` uint8 | camera views; exact names vary by env |
| `observation/image`, `observation/wrist_image`, `observation/state` | see §4 | the **pi0** vocabulary (note the slash) |
| `language` | `(384,)` float32 | sentence-embedding of the instruction (when a `SentenceTransformer` is configured) |
| `prompt` | str | raw language instruction (pi0 consumes this) |
| `dino_embedding` | `(F,)` float32 | precomputed DINOv2 features (when `DinoEmbeddingWrapper` is active) |

`setup_*_env(...)` returns `(env, remove_obs_keys)`. **`remove_obs_keys` is the list of keys the
buffers must drop** — usually raw images that have been replaced by embeddings, plus `prompt` /
`language`. Pass it through; dropping it silently bloats the buffer or breaks the actor's input spec.

## 2. Actions: stored in env space, learned in `[-1, 1]`

This is the single most important contract.

* **Rollout workers store env-space actions** in the buffer (whatever the env's action space is).
* **Buffers normalize on sample**: `BaseReplayBuffer._normalize_action_batch` maps
  `[min_action, max_action] -> [-1, 1]` via `2*(a - min)/span - 1` when `min_action`/`max_action` are
  set. Training therefore always sees normalized actions, matching the actor's `tanh` output range.
* **`BaseActor.act()` unnormalizes at inference**, so the env receives env-space actions again.

Consequences:
* If you construct a buffer **without** `min_action`/`max_action`, normalization silently no-ops and
  the policy trains on raw-scale actions. Check `create_buffer` in `utils/training_utils.py`.
* A BC/MSE objective regresses `tanh(mean)` against the *normalized* action.
* Datasets you convert yourself must store **env-space** actions and let the buffer normalize.

## 3. robomimic-style HDF5

The in-house datasets (offline data, HITL corrections, converted robomimic/LeRobot data) all use this
layout, which `buffers/h5_replay_buffer.py::H5ReplayBuffer` reads:

```
/data/demo_{i}/actions          [N, A]  float32   env-space actions actually executed
/data/demo_{i}/rewards          [N]     float32   optional
/data/demo_{i}/dones            [N]     float32   optional; 1 at the terminal step
/data/demo_{i}/intervention     [N]     int64     optional; HITL label (see §5)
/data/demo_{i}/obs/<key>        [N, ...]          one dataset per observation key
/data/demo_{i}.attrs["num_samples"]
/data/demo_{i}.attrs["prompt"]            language instruction
/data.attrs["total"]                      total transitions
/meta/min_action, /meta/max_action        action bounds (feed the buffer's normalization)
/meta.attrs["info"]                       JSON provenance blob
```

Only `actions` and `obs/` are strictly required; `rewards`, `dones` and `intervention` are read when
present. Converters: `scripts/convert_robomimic_to_aligned.py`, `scripts/convert_lerobot_to_h5.py`.
Inspect with `scripts/check_robomimic_dataset.py` and `scripts/inspect_obs_stats.py`.

## 4. pi0 / pi0.5 observation format

openpi policies consume a fixed vocabulary (`utils/pi0_hitl_utils.py`, `utils/pi0_integration.py`):

```python
PI0_IMAGE_KEY = "observation/image"        # (224, 224, 3) uint8
PI0_WRIST_KEY = "observation/wrist_image"  # (224, 224, 3) uint8
PI0_STATE_KEY = "observation/state"        # (8,) float32
obs["prompt"]                               # str, the task instruction
```

For LIBERO, `utils/pi0_integration.py::preprocess_obs_for_pi0` builds these from raw robosuite obs:

* images are **flipped both horizontally and vertically** (`[::-1, ::-1]`), then resized with padding
  to 224x224 and cast to uint8;
* `state` is `[eef_pos (3), eef_axis_angle (3), gripper_qpos (2)]` = 8 dims.

Observations are stored **raw** in HITL datasets (pi0 normalizes internally), so the same HDF5 can be
exported to LeRobot later without re-deriving anything.

## 5. Intervention labels

HITL datasets tag every stored step with who produced it:

```python
ROLLOUT_LABEL, INTERVENTION_LABEL = 0, 1   # 0 = policy, 1 = human correction
```

HG-DAgger trains on label-1 steps; the MILE family uses both labels (and a masked label-2 in some
variants). `hitl.store_only_human=true` keeps only corrections — which is what makes
[the aggregation gotcha](GOTCHAS.md) bite.

## 6. LeRobot export (openpi training)

`scripts/export_hitl_to_lerobot.py` converts one or more collection HDF5s into a LeRobot dataset with
the standard LIBERO features — `image`, `wrist_image`, `state`, `actions` — plus:

* `intervention` — the per-step label, carried through so the openpi-side loss can use it;
* `rollout_samples` — optional `[P, H, A]` frozen baseline pool for Flow-MILE. **Every frame in the
  repo must share one fixed `(P, H, A)` shape**, so the exporter peeks at the first demo that has it
  and zero-fills the rest; a mismatch degrades label-0 baselines to zero (it warns).

Base (non-HITL) demos can be folded in with a `default_label`, which is how anti-forgetting data is
mixed in. After export, openpi needs norm stats computed
(`third_party/dsrl_openpi/scripts/compute_norm_stats.py`) before training.

## 7. Observation/action spaces from a dataset

`utils/dataset_spaces.py::build_spaces_from_h5` derives gym spaces directly from an HDF5, so
supervised training can build networks without constructing an env. It also widens degenerate bounds
(where min == max) so samplers don't produce constants.

## 8. Alignment between offline data and the online env

For offline→online RL the dataset's observation keys **must match what the online env emits**, key for
key and dim for dim. `scripts/check_state_alignment.py` verifies a dataset's `obs/state` against the
live env's composed `state`, and `scripts/convert_robomimic_to_aligned.py` rewrites a robomimic
dataset into the online format. Run the check before any offline→online experiment — a silent
mismatch trains a policy on a different state space than it is evaluated in.
