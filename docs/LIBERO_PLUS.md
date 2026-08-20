# LIBERO-plus: robustness benchmark for the pi0.5 HITL experiments

[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) expands LIBERO's four 10-task suites into
**10,030 perturbed tasks** across seven perturbation dimensions. It is wired in here so pi0 / pi0.5
HITL collection and evaluation can run on perturbed tasks — where the base policy actually fails —
instead of only on the clean tasks it was tuned on.

| Perturbation dimension | Tasks | Encoded in the task name as |
| --- | --- | --- |
| Background Textures | 1,076 | `_table_N`, `_tb_N` |
| Camera Viewpoints | 1,599 | `_view_h_v_scale_rot_vert` |
| Language Instructions | 1,537 | `_language_N` (LLM-rewritten instruction) |
| Light Conditions | 1,142 | `_light_N` |
| Objects Layout | 1,525 | `_add_N`, `_levelN` (confounding / displaced objects) |
| Robot Initial States | 1,550 | `_initstate_N` (robot base variant `MountedPandaN`) |
| Sensor Noise | 1,601 | `_noise_N` (photometric corruption of `agentview_image`) |

## Setup (one time)

```bash
bash scripts/setup_libero_plus.sh
```

This initialises the `third_party/LIBERO-plus` submodule, downloads `assets.zip` (**6.4 GB**, ~9.5 GB
extracted) from the HF dataset `Sylvest/LIBERO-plus`, and unpacks it to
`third_party/LIBERO-plus/libero/libero/assets`. The zip lands in `data/libero_plus/` (gitignored) and
can be deleted afterwards. Re-running is a no-op unless `FORCE_ASSETS=1`.

One optional system package: `sudo apt-get install -y libmagickwand-dev`. Without it, the 336
motion-blur Sensor Noise tasks (`_noise_N` with N ≤ 10, **3.3%** of the benchmark) raise an actionable
error; everything else — including the other four noise families — works, because `activate()`
installs a stub `wand` module. The python-side extras (`wand`, `scikit-image`) are in the
`libero-plus` dependency group, which is part of `default-groups`, so a plain `uv sync` installs them.

## Why it is not just pip-installed

LIBERO-plus ships the **same top-level `libero` package** as LIBERO (`setup.py` declares
`name="libero"`), so the two cannot be installed side by side. Rather than replacing the install —
which would silently change what every existing `task_id` means — the checkout stays uninstalled and
is selected per run by `env.libero_plus=true`. `robometer_policy_learning/envs/libero_plus.py` then:

1. writes a LIBERO path config into `~/.libero_plus/config.yaml` and points `$LIBERO_CONFIG_PATH` at
   it, redirecting `bddl_files` / `init_files` / `assets` into the LIBERO-plus checkout (your
   `~/.libero/config.yaml` is never touched);
2. inserts the checkout at `sys.path[0]`, which shadows the editable-installed LIBERO — its finder is
   *appended* to `sys.meta_path` and therefore loses to the regular `sys.path` finder.

`setup_libero_env(..., use_libero_plus=True)` calls `activate()` before the first `import libero`, so
scripts need no special import ordering. Because the shadowing is process-wide, **one process cannot
mix the two checkouts** — activation raises if `libero` was already imported from elsewhere.

> ⚠️ **Task ids are not comparable across the two checkouts.** Under LIBERO-plus,
> `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` contain *only* perturbed variants
> (2402 / 2518 / 2591 / 2519 tasks) in a different order, and the original unperturbed tasks are
> absent. `libero_90` is byte-identical in both. Prefer `env.task_name` (or `env.task_names`) over
> ids when pinning a perturbed task.

## Selecting tasks: base task + perturbation family

The main mode mirrors how plain LIBERO is used. Give a **base task id** and a **perturbation family**,
and every episode runs a *different* variant of that family for that base task:

```bash
env.libero_plus=true env.env_name=libero_spatial env.task_ids=[3] env.perturbation=camera
```

`env.task_id` / `env.task_ids` are then **base** ids `0..9` in **plain-LIBERO order**, so `task_id=3`
means exactly what it always did. (This ordering matters: LIBERO-plus itself lists base tasks
alphabetically, a different order in all four suites, so base ids are read from the plain LIBERO
checkout's `libero_task_map`.) Families: `background`, `camera`, `language`, `light`, `layout`,
`robot`, `noise` — canonical names like `"Camera Viewpoints"` work too, as does a list, or `"all"` to
mix every family.

Every (base task, family) cell is populated. Variants per cell for `libero_spatial`:

| base task id | backgr. | camera | language | light | layout | robot | noise | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 29 | 6 | 46 | 42 | 44 | 32 | 32 | 231 |
| 1 | 10 | 37 | 32 | 10 | 37 | 38 | 34 | 198 |
| 2 | 46 | 35 | 28 | 50 | 38 | 35 | 35 | 267 |
| 3 | 31 | 49 | 42 | 24 | 45 | 24 | 31 | 246 |
| 4 | 26 | 48 | 37 | 44 | 45 | 40 | 40 | 280 |
| 5 | 35 | 46 | 32 | 34 | 40 | 32 | 37 | 256 |
| 6 | 20 | 42 | 46 | 31 | 32 | 27 | 34 | 232 |
| 7 | 41 | 43 | 35 | 32 | 34 | 43 | 41 | 269 |
| 8 | 6 | 33 | 45 | **4** | 27 | 37 | 28 | 180 |
| 9 | 14 | 37 | 47 | 21 | 43 | 42 | 39 | 243 |

Cells range from 4 to 50 variants across all suites. When `num_episodes` exceeds the pool the sampler
reshuffles and repeats, and logs a warning — so a run never silently evaluates fewer perturbations
than you asked for.

**Sampling.** `env.variant_sampling=shuffle` (default) walks a seeded permutation of the pool with no
repeats until it is exhausted; `sequential` walks it in id order (adjacent ids are near-identical
perturbations, so a short sequential run samples a narrow slice). Narrow the pool further with
`env.variant_difficulty_levels` and `env.variant_name_contains`.

**Seeding.** `env.variant_seed` seeds the variant order *only*, through a private RNG that never
disturbs env or policy seeding. It defaults to `null`, meaning a **random seed is drawn once per run**
— logged as `selected variant_seed=N` and saved (to `eval_results.json`, or the dataset's
`meta.attrs["info"]`), so any run can be replayed by passing that value back in:

```bash
env.variant_seed=581894657   # replays the exact same perturbation sequence
```

Pin it explicitly when runs need to be comparable — most importantly across HITL rounds, so every
round's eval walks the same perturbations. `scripts/hitl_pi0_loop.py` handles that for you: it draws
one seed at startup (or takes `--variant-seed`) and passes the *same* value to every round, since
leaving it null would otherwise let each round's subprocess draw its own.

**Rebuild cost.** A variant is a distinct MuJoCo scene (different bddl, robot model, or post-render
corruption), so it cannot be swapped in at reset: the env is rebuilt per episode, ~1–2 s. Negligible
against a 300–800-step episode.

To pin **one** specific variant instead, pass its exact name and leave `env.perturbation` null —
`env.task_name=...` for collection, `env.task_names=[...]` for eval. The two are mutually exclusive
and the scripts say so rather than silently preferring one.

### Programmatic access

```python
from robometer_policy_learning.envs.libero_plus import (
    base_task_name, variants_of, VariantCycler, categories_of_suite,
)

base_task_name("libero_spatial", 3)
# 'pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate'

pool = variants_of("libero_spatial", 3, categories="camera")   # 49 TaskInfo records
cycler = VariantCycler(pool, seed=0)
cycler.next().task_id        # 708 -- feed to setup_libero_env(task_id=...)

categories_of_suite("libero_spatial")
# {'Background Textures': 258, 'Camera Viewpoints': 376, ...}
```

`list_tasks(suite, categories=..., difficulty_levels=..., name_contains=..., limit=...)` remains
available for scanning variants across *all* base tasks. Note `difficulty_level` is `None` for 121
tasks (all `libero_goal` / Light Conditions) where upstream left it null; those never match a
difficulty filter.

## Init states

`env.init_state_index` controls which of the benchmark's fixed start states each reset restores:

| value | behaviour |
| --- | --- |
| `"auto"` (default) | `0` under LIBERO-plus, `null` under plain LIBERO |
| `null` | don't restore — robosuite randomizes placements each reset (historical behaviour here) |
| `<int>` | always that state (LIBERO-plus's protocol is index 0 with one trial per task) |
| `"cycle"` | episode *k* uses state *k % N* — varied starts, still on-distribution |
| `"random"` | uniform random per reset |

This matters: the **Objects Layout** perturbations only take effect through their init states, so
`init_state_index=null` under LIBERO-plus does not reproduce them (a warning is logged). After
restoring a state, `env.settle_steps` (default 10) no-op steps let MuJoCo settle, matching LIBERO's
evaluation protocol; they do not count against the episode time limit.

## Evaluation

```bash
# Base tasks 3 and 7, 20 episodes each, every episode a different camera perturbation
uv run python scripts/eval_pi0_libero.py --config-name libero_eval \
    env.libero_plus=true env.env_name=libero_spatial 'env.task_ids=[3,7]' \
    env.perturbation=camera eval.num_episodes=20 \
    pi0.checkpoint=gs://openpi-assets/checkpoints/pi05_libero/
```

One summary row per (base task, family), with how many distinct variants were actually run:

```
   task_id |  success |  succ/eps | avg_steps |  variants
         3 |    35.0% |     7/20  |     412.6 |   20/49
         7 |    45.0% |     9/20  |     388.1 |   20/43
```

plus the per-dimension breakdown (aggregated over episodes, so `env.perturbation=all` works too) and
`eval_results.json`, which keeps a per-episode list — `variant_task_id`, `variant_task_name`,
`difficulty_level`, `success` — so any failure traces back to the exact perturbation that caused it.

## HITL collection

```bash
# Base task 3; each rollout is a different camera perturbation, all into one HDF5
uv run python scripts/collect_hitl_libero_pi0.py --config-name libero_collect_hitl \
    env.libero_plus=true env.env_name=libero_spatial env.task_id=3 env.perturbation=camera \
    teleop.device=spacemouse hitl.collect_num_rollouts=20 \
    hitl.collect_output_path=data/hitl/camera_t3.hdf5
```

The env is rebuilt between rollouts while the worker keeps the SpaceMouse/keyboard, the takeover
toggle and the teleop window open (`Pi0LiberoHitlWorker.rebind_env`), so collection is uninterrupted.
The variant advances **per attempt**, not per kept rollout, so `require_success` /
`keep_only_hitl_rollouts` filtering cannot bias the dataset toward whichever variants happen to be
easy.

Each demo records the variant it ran, as `/data/demo_{i}.attrs`: `variant_task_id`,
`variant_task_name`, `perturbation_category`, `difficulty_level`. `meta.attrs["info"]` adds the whole
pool, the seed and the sampling mode, so a corrections dataset is fully traceable.

`env.init_state_index` still applies *within* each variant. With per-episode variants the default
(`auto` = 0) is usually right — the variant supplies the diversity.

### Collect / eval variant overlap

Collection and evaluation draw from the **same** variant pool, so with the same `variant_seed` they
walk the *same* variant sequence — the eval then largely re-measures the variants that were just
corrected, which flatters the result. (With the default null seed they draw independent random seeds
instead, so the overlap is incidental rather than total.) That is intentional (it makes "did corrections help on exactly
these perturbations?" answerable) but it is not a generalization measurement. To draw a different
sequence from the same pool, give the collect step its own seed: `env.variant_seed=<other>` on the CLI,
or `--collect-variant-seed` in the HITL loop. For a true held-out split you would need disjoint pools,
which is not implemented.

For a full iterative loop, `scripts/hitl_pi0_loop.py` forwards `--perturbation` and `--variant-seed`
to both the eval and collect step of every round, so each round measures and corrects the same
perturbation sequence:

```bash
uv run python scripts/hitl_pi0_loop.py --rounds 3 --env-name libero_spatial --task-ids 3 7 \
    --perturbation camera --variant-seed 0 --collect-num-rollouts 5 --eval-num-episodes 20 \
    --repo-id-prefix yourname/libero_plus_hitl --exp-prefix hitl_plus_camera
```

## Upstream quirks worth knowing

* **Virtual bddl paths.** For `_view_` / `_initstate_` / `_noise_` tasks the bddl path built from the
  task name does not exist on disk — LIBERO-plus's `ControlEnv` parses the perturbation out of the
  string and strips it back to a real file. Same for init-state paths, resolved back to the base task.
* **`task.language` is filename-derived.** For those same tasks the perturbation suffix leaks into
  LIBERO's `Task.language` (`"... place it on the plate view 0 0 100 0 0 initstate 0 noise 12"`).
  `setup_libero_env` therefore takes the instruction from the loaded bddl instead, so the pi0 prompt
  and the recorded metadata are clean. `_language_N` rewrites come through correctly.
* **`torch.load` on init states.** Both checkouts call `torch.load(path)` on `.pruned_init` files,
  which are numpy pickles; torch ≥ 2.6 defaults `weights_only=True` and refuses them.
  `libero_plus.load_task_init_states` allows the pickle for the duration of that call.
* **Noisy suite construction.** `Benchmark._make_benchmark` prints its whole task-order permutation
  (2402+ integers). `libero_plus.make_task_suite` silences it.
* **Base-name parsing traps.** `strip_perturbation` anchors markers to digits because
  `pick_up_the_black_bowl_from_table_center_...` is a real *base* task name containing `_table_`, and
  it strips `_moved` alongside `_level<N>` for `libero_goal`'s `..._moved_level1_sample1` layout
  variants. A test asserts all 10,030 names strip to one of their suite's ten base tasks.
* **Null difficulty levels.** 121 tasks (all `libero_goal` / Light Conditions) have
  `difficulty_level: null` in `task_classification.json`.

## Tests

```bash
uv run pytest tests/test_libero_plus.py                  # metadata / activation (no assets needed)
uv run pytest tests/test_libero_plus.py -m integration   # builds real perturbed envs (needs assets)
```
