# HITL for pi0.5 + openpi — reference (LIBERO today, real robots next)

A working reference for the human-in-the-loop (HITL) fine-tuning pipeline built around **pi0.5**
(openpi, JAX/Flax) with **LIBERO**. It documents the loop, the data formats, and the two training
objectives (**HG-DAgger** and **Flow-MILE**), and — most importantly — calls out **what is
LIBERO-specific vs. embodiment-agnostic** so you can port it to a real robot.

> TL;DR mental model: *collect corrections with a human → bridge to a LeRobot dataset → compute norm
> stats → LoRA-fine-tune pi0.5 in openpi → eval → repeat with the improved policy.* HG-DAgger and
> Flow-MILE differ **only** in the openpi train config; everything upstream is shared.

---

## 1. The iterative loop

Orchestrated by `scripts/hitl_pi0_loop.py` (one subprocess per stage so JAX training and robosuite
teleop never share a process). Per round **R** over a suite of task ids:

```
        ┌─────────────────────────────────────────────────────────────────────┐
        │  round R  (policy = base pi0.5 for R=0, else round R-1's checkpoint)  │
        └─────────────────────────────────────────────────────────────────────┘
 0. EVAL      scripts/eval_pi0_libero.py           success rate on the task suite (curve across rounds)
 1. COLLECT   scripts/collect_hitl_libero_pi0.py   per-task corrections HDF5 (human takeover, 0/1 labels)
 2. EXPORT    scripts/export_hitl_to_lerobot.py    all rounds' corrections (+ base demos) → ONE LeRobot repo
 3. NORMSTATS third_party/dsrl_openpi/scripts/compute_norm_stats.py
 4. TRAIN     third_party/dsrl_openpi/scripts/train.py <config>   LoRA fine-tune → new checkpoint
        └── hand the new checkpoint to round R+1's eval/collect ──┘
```

- **Environments** are the ROOT project's env (`uv run python ...` from the repo root).
- **openpi** stages run from `third_party/dsrl_openpi` with `uv run --frozen python ...`.
- Anti-forgetting: the export folds **base/offline expert demos** into every round's repo (openpi
  trains a single repo). See [[dp-hgdagger-needs-aggregation]] — corrections-only fine-tuning forgets.

Example:
```bash
uv run python scripts/hitl_pi0_loop.py --rounds 3 --env-name libero_90 --task-ids 57 58 59 \
    --collect-num-rollouts 5 --num-train-steps 3000 --eval-num-episodes 20 \
    --repo-id-prefix you/libero_hitl --exp-prefix hitl_t575859 \
    --train-config pi05_libero_flow_mile_lora \
    --libero-base-suite libero_90 --libero-base-num-demos 10
```
Select the objective with `--train-config`: `pi05_libero_hitl_lora` (HG-DAgger) or
`pi05_libero_flow_mile_lora` (Flow-MILE).

---

## 2. Data formats

### 2.1 Collection HDF5 (robomimic-style) — `scripts/collect_hitl_libero_pi0.py`
Raw pi0-format observations (pi0 normalizes internally, so store RAW):

```
/data/demo_i/actions            [N, 7]          env-space actions actually executed
/data/demo_i/rewards            [N]
/data/demo_i/dones              [N]
/data/demo_i/intervention       [N]  int64       0=policy, 1=human
/data/demo_i/rollout_samples    [N, P, H, 7]    Flow-MILE baseline pool (only if hitl.rollout_pool_size>0)
/data/demo_i/obs/image          [N,224,224,3] uint8   pi0 "observation/image"
/data/demo_i/obs/wrist_image    [N,224,224,3] uint8   pi0 "observation/wrist_image"
/data/demo_i/obs/state          [N, 8]  f32          pi0 "observation/state"
/data/demo_i.attrs["prompt"]    language instruction
/meta.attrs["info"]             JSON provenance (checkpoint, task, action_exec_len, labels, ...)
```

### 2.2 LeRobot dataset — `scripts/export_hitl_to_lerobot.py`
The bridge to openpi. Features:

| key | dtype | shape | notes |
|---|---|---|---|
| `image` | image | (224,224,3) | base camera |
| `wrist_image` | image | (224,224,3) | wrist camera |
| `state` | float32 | (8,) | proprio |
| `actions` | float32 | (7,) | env-space |
| `intervention` | int64 | (1,) | 0=policy, 1=human, **2=offline demo** |
| `rollout_samples` | float32 | (P,H,7) | Flow-MILE only; env-space; auto-added if present |

Task/prompt goes in per-frame (`add_frame({... "task": prompt})`) in the current lerobot API.

### 2.3 pi0-format conventions (LIBERO)
- **Images**: `[224,224,3]` uint8, **180° rotated** (`[::-1,::-1]`) to match pi0.5's training
  orientation, then `resize_with_pad`. The stored `obs/image` is already right-side-up.
- **State (8-d)**: `[eef_pos(3), axis_angle(3), gripper_qpos(2)]` = `concat(ee_states, gripper_states)`.
- **Actions (7-d)**: LIBERO delta OSC (`[dpos(3), daxis-angle(3), gripper(1)]`); teleop deltas map
  straight onto the 7-dim action.
- **Normalization**: pi0.5 uses **quantile** norm (q01/q99) on `state`/`actions`, computed AFTER
  collection by `compute_norm_stats.py`. Stored data is raw; the loader normalizes.

---

## 3. Intervention labels (the backbone of both objectives)

| label | meaning | who set it |
|---|---|---|
| 0 | policy (autonomous) step | collection worker |
| 1 | human correction step | collection worker (takeover) |
| 2 | offline/base expert demo | exporter (base-demo streams) |

`hitl.store_only_human` controls whether collection keeps **full trajectories** (0 and 1 — default)
or **corrections only** (1). **Full trajectories are required for Flow-MILE**; HG-DAgger works with
either (it ignores the label). The exporter never filters — it preserves every frame + label.

---

## 4. Two objectives

### 4.1 HG-DAgger (`pi05_libero_hitl_lora`)
- Plain flow-matching BC on the aggregated LeRobot repo (corrections + base demos). **No openpi loss
  changes** — the `intervention` label rides along unused.
- Anti-forgetting = **aggregation** (mix demos + corrections in the one repo).
- Data: full trajectories or corrections-only both fine.

### 4.2 Flow-MILE (`pi05_libero_flow_mile_lora`)
MILE intervention-probit objective on top of the flow-matching loss:

```
total = BC(labels {1,2})  +  λ · BCE_probit(labels {0,1})
```
- **BC term** trains on human corrections (1) + offline demos (2).
- **BCE probit term** models "would a human intervene here?" using the per-sample flow-matching loss
  as a log-prob proxy. Score `ell(a,s)` is pi0.5's own `compute_loss` (MSE, mean over horizon).
- Needs **labels {0,1}** (both policy and human steps) → requires full trajectories.
- Assumes `condition_intervention_on_action=True`, `condition_nonintervention_on_robot=True`; no
  proximal loss, no score-gap normalization.
- **Reference-relative score** (`reference_relative_score=True`): `ell = flow_loss_π0 − flow_loss_θ`,
  comparing the online policy θ to a frozen snapshot π0 (the round's collection policy).

Knobs live in `FlowMileParams` (`third_party/dsrl_openpi/src/openpi/training/config.py`):
`lambda_intervention (λ)`, `probit_scale (β)`, `intervention_cost (c)`, `num_samples (K)`,
`score_mc_samples`, `expected_rollout_score_weight (w)`, `num_sample_steps`,
`reference_relative_score`, `use_stored_rollout_samples`, `anchor_loss_weight`,
`anchor_monte_carlo_samples`.

**Anchor loss** (optional, `anchor_loss_weight > 0`, default off): adds
`anchor_loss_weight · E_{s,a0~π0,t,x0} ‖v_θ(a0_t,t,s) − v_π0(a0_t,t,s)‖²` — it pins the online
velocity field to the frozen field on rollout-sampled actions, countering the drift in the online
policy's score of rollout-like actions that otherwise inflates the intervention probability. It reuses
the probit's rollout chunks (`anchor_monte_carlo_samples`, capped at K) and needs the frozen policy's
velocity `v_π0`, so **it keeps the resident frozen policy** — it does *not* compose with the
`use_stored_rollout_samples` memory saving. Enable with `--flow-mile.anchor-loss-weight=<w>`. Implemented
via `Pi0.predict_velocity` (raw velocity field) in `models/pi0.py`.

---

## 5. Flow-MILE memory: the frozen rollout policy & stored samples

Flow-MILE needs a **frozen rollout policy π0** to (a) sample K baseline action chunks per state and
(b) compute the reference-relative flow-losses. Naively this keeps a **second full copy of the model
resident on GPU** (`TrainState.rollout_params`) → OOM for pi0.5.

**Fix (mirrors the local `use_stored_rollout_samples`):** precompute the baseline chunks **at
collection time** and read them during training instead of keeping/sampling the frozen policy.

- Collection samples `P = hitl.rollout_pool_size` chunks **only at human-intervened (label-1) states**
  — the only frames whose pool feeds the loss (via `observed_probs`); label-0/2 frames are zero-filled.
- They flow through as the `rollout_samples` field ([P,H,7] env-space → normalized by the **aliased
  action stats** → padded 7→32, exactly like `actions`).
- In `_flow_mile_grads` the K-of-P pool is sliced instead of calling `sample_actions`.
- `FlowMileParams.use_stored_rollout_samples=True` turns this on. **Set `P ≥ K`.**

**The memory is only fully reclaimed with `reference_relative_score=False`** — that's the *only other*
consumer of the frozen policy. Then `init_train_state` sets `rollout_params=None` (no second copy).
With `reference_relative_score=True` the frozen copy stays resident (the reference losses still need
it). Run: `--flow-mile.use-stored-rollout-samples=True --flow-mile.reference-relative-score=False`.

Other standard openpi memory levers: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, LoRA (freezes the
backbone; only adapters + Adam state are trainable), EMA off for LoRA, `--fsdp-devices <n>`,
Gemma/SigLIP activation checkpointing (already on).

---

## 6. Key files & roles

| file | role | embodiment-specific? |
|---|---|---|
| `robometer_policy_learning/utils/pi0_hitl_utils.py` | `Pi0LiberoHitlWorker`: autonomous chunks + human takeover, labels, stores raw obs, `_pi0_pool` | **YES** (env + teleop) |
| `scripts/collect_hitl_libero_pi0.py` | driver: load pi0.5, build env, collect, write HDF5 | **YES** (env build) |
| `robometer_policy_learning/configs/libero_collect_hitl.yaml` | collection config | **YES** |
| `scripts/export_hitl_to_lerobot.py` | HDF5 → LeRobot; `_iter_demos` (generic), `_iter_libero_demos` (LIBERO base demos) | partly |
| `scripts/eval_pi0_libero.py` | multi-task eval | **YES** (env) |
| `scripts/hitl_pi0_loop.py` | orchestrator | mostly generic |
| `third_party/dsrl_openpi/src/openpi/policies/libero_hitl_policy.py` | `LiberoHitlInputs`: maps dataset keys → model inputs, forwards `intervention`/`rollout_samples` | **YES** (copy per embodiment) |
| `.../training/config.py` `LeRobotLiberoHitlDataConfig` | repack + norm-stats alias + transforms | **YES** (copy per embodiment) |
| `.../training/config.py` `pi05_libero_hitl_lora` / `pi05_libero_flow_mile_lora` | train configs | **YES** (copy per embodiment) |
| `.../scripts/train.py` `_flow_mile_grads`, `train_step`, `init_train_state` | Flow-MILE loss + rollout-sample consumption + memory gating | **NO** (reuse as-is) |
| `.../src/openpi/transforms.py` `PadStatesAndActions`, `Normalize` | padding + normalization (pads `rollout_samples` too) | **NO** |
| `.../training/data_loader.py` `DataLoaderImpl.__iter__` | yields the 4-tuple `(obs, actions, intervention, rollout_samples)` | **NO** |
| `.../training/utils.py` `TrainState` | `rollout_params` field | **NO** |

---

## 7. Commands (per stage)

```bash
# 0. Eval a checkpoint on a suite
uv run python scripts/eval_pi0_libero.py --config-name libero_eval \
    env.env_name=libero_90 'env.task_ids=[57,58,59]' \
    pi0.checkpoint=gs://openpi-assets/checkpoints/pi05_libero/ eval.num_episodes=20

# 1. Collect corrections (Flow-MILE needs full trajectories + a rollout pool)
uv run python scripts/collect_hitl_libero_pi0.py --config-name libero_collect_hitl \
    env.env_name=libero_90 env.task_id=57 teleop.device=spacemouse \
    hitl.collect_num_rollouts=5 hitl.store_only_human=false hitl.rollout_pool_size=8 \
    hitl.collect_output_path=outputs/round0_t57.hdf5 \
    pi0.checkpoint=gs://openpi-assets/checkpoints/pi05_libero/

# 2. Export (+ base demos) — from the openpi env
cd third_party/dsrl_openpi && uv run --frozen python ../../scripts/export_hitl_to_lerobot.py \
    --inputs ../../outputs/round0_t57.hdf5 --repo-id you/libero_hitl_r0 \
    --libero-base-suite libero_90 --libero-base-task-ids 57 --libero-base-num-demos 10

# 3. Norm stats
cd third_party/dsrl_openpi && uv run --frozen python scripts/compute_norm_stats.py \
    --config-name pi05_libero_flow_mile_lora --repo-id you/libero_hitl_r0

# 4a. Train — HG-DAgger
cd third_party/dsrl_openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --frozen python scripts/train.py \
    pi05_libero_hitl_lora --exp-name=hitl_r0 --data.repo-id=you/libero_hitl_r0 --overwrite

# 4b. Train — Flow-MILE (memory-safe: stored pool + non-reference-relative)
cd third_party/dsrl_openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --frozen python scripts/train.py \
    pi05_libero_flow_mile_lora --exp-name=fm_r0 --data.repo-id=you/libero_hitl_r0 \
    --flow-mile.use-stored-rollout-samples=True --flow-mile.reference-relative-score=False --overwrite
```

openpi LoRA configs use `gemma_2b_lora` + `gemma_300m_lora`, `get_freeze_filter()`, EMA off, and load
round R from round R-1's checkpoint (`--weight-loader.params-path`, first round =
`gs://openpi-assets/checkpoints/pi05_libero/params`).

---

## 8. Porting to a real robot — checklist

**Reuse unchanged (embodiment-agnostic):** the entire Flow-MILE loss and rollout-sample machinery in
`train.py`, `transforms.py` (`PadStatesAndActions`/`Normalize`), `data_loader.py` (the 4-tuple),
`TrainState.rollout_params`, and the intervention-label contract (0/1/2). The HDF5↔LeRobot **schema**
and the generic `_iter_demos` in the exporter are reusable as long as you match the field names.

**Swap / re-implement per embodiment:**

1. **Collection worker** — replace `Pi0LiberoHitlWorker`'s robosuite env + robosuite teleop with your
   robot's control loop and teleop source (SpaceMouse / leader arm / keyboard). Keep the contract:
   - run pi0.5 autonomously in receding-horizon chunks (`action_exec_len`), allow human takeover,
   - label every executed step 0 (policy) / 1 (human),
   - store **raw** pi0-format obs + executed action + prompt,
   - if doing Flow-MILE, sample `rollout_pool_size` policy chunks **at each human-intervened frame**
     (`_pi0_pool`) — the only frames whose pool is consumed; zero-fill the rest.
2. **Obs/action space** — decide your **state** vector and **action** space (delta vs absolute, dim),
   your camera set (base + wrist), and image preprocessing (orientation, resize to 224, uint8). Update
   the HDF5 `state`/`actions` shapes and the LeRobot feature shapes to match. Watch the **180° flip** —
   it exists because LIBERO/openvla training rotated images; verify what YOUR pi0.5 checkpoint expects.
3. **Prompt** — provide the language instruction per task/episode.
4. **Exporter base demos** — `_iter_libero_demos` is LIBERO-only. Replace it with a loader for your
   real-robot expert demos (label them `intervention=2`), or just pass them as collection-schema HDF5s
   via `--base-demos` (they go through the generic `_iter_demos`).
5. **openpi DataConfig** — copy `LeRobotLiberoHitlDataConfig` → your embodiment. Edit the **repack
   mapping** (dataset keys → model keys), the **Inputs transform** (copy `LiberoHitlInputs`: map your
   cameras/state, forward `intervention` + `rollout_samples`), the **delta mask** if your actions are
   deltas, and keep the **norm-stats alias** (`rollout_samples` → `actions` stats).
6. **openpi TrainConfig** — copy `pi05_libero_hitl_lora` / `pi05_libero_flow_mile_lora` → your configs
   pointing at your DataConfig. Keep LoRA variants + freeze filter + EMA off. For Flow-MILE keep
   `use_stored_rollout_samples=True`; choose `reference_relative_score` (False for the memory win).
7. **Norm stats** — recompute per dataset (`compute_norm_stats.py`); `state`/`actions` only.
8. **Eval** — replace the LIBERO eval env; success detection is task-specific.

**Rule of thumb:** everything *before* the LeRobot dataset is embodiment code you own; everything the
openpi training loop does with `intervention` + `rollout_samples` is generic and already done.

---

## 9. Gotchas

- **Full trajectories for Flow-MILE.** `hitl.store_only_human=true` breaks the BCE term (needs label-0
  policy steps). Default is full trajectories — keep it.
- **`P ≥ K`.** Rollout pool size must be ≥ `num_samples`.
- **Memory only drops with `reference_relative_score=False`.** Stored samples alone don't free the
  frozen copy while reference-relative scoring is on.
- **Aggregate base demos** every round (anti-forgetting); openpi trains one repo.
- **Image orientation** is the most common silent bug — corrections and offline demos must match the
  checkpoint's expected orientation. Compare a correction rollout vs an expert demo before pushing
  (`scripts/compare_correction_vs_expert.py` does this for LIBERO).
- **Object/scene randomization** on reset is intentional in LIBERO (varied layouts); decide the
  analogous policy for your setup.
- **openpi env uses `uv run --frozen`**; the ROOT env runs collection/eval flag-free.
- **Early finish**: during collection, press **ESC** in the teleop window to stop collecting and save
  what's been collected so far (the in-progress rollout is discarded; completed ones are kept). The
  orchestrator only feeds HDF5s that actually exist to the exporter, so a round that saved nothing is
  skipped.
- **Empty `store_only_human` datasets are not written**: if `store_only_human=true` and no human
  interventions were collected, no file is created (an empty correction dataset would break the
  exporter). Downstream conversion must tolerate a missing per-round path (the orchestrator does).

---

## 10. Related references
- openpi HITL configs — `third_party/dsrl_openpi/src/openpi/training/config.py`.
- Local (torch) Flow-MILE for comparison — `robometer_policy_learning/algorithms/flow_mile/`.
