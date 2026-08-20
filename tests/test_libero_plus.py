"""Tests for the LIBERO-plus integration (robometer_policy_learning/envs/libero_plus.py).

The metadata tests only need the LIBERO-plus checkout (task_classification.json) and are skipped when
the submodule is not initialised. The env-building test additionally needs the ~6.4 GB assets
download and MuJoCo, so it is marked ``integration``.

    uv run pytest tests/test_libero_plus.py                       # metadata only (fast)
    uv run pytest tests/test_libero_plus.py -m integration        # + build a perturbed env
"""

import os

import numpy as np
import pytest

from robometer_policy_learning.envs import libero_plus

CHECKOUT = os.path.isfile(
    os.path.join(libero_plus.default_root(), "libero", "libero", "benchmark", "task_classification.json")
)
ASSETS = os.path.isdir(os.path.join(libero_plus.default_root(), "libero", "libero", "assets"))

needs_checkout = pytest.mark.skipif(not CHECKOUT, reason="third_party/LIBERO-plus not initialised")
needs_assets = pytest.mark.skipif(not ASSETS, reason="LIBERO-plus assets not downloaded")

# Task counts per suite, as published by LIBERO-plus (10,030 tasks in total).
EXPECTED_SUITE_SIZES = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}


def test_normalize_categories_accepts_aliases_and_canonical_names():
    assert libero_plus.normalize_categories("camera") == ("Camera Viewpoints",)
    assert libero_plus.normalize_categories(["noise", "Light Conditions"]) == ("Sensor Noise", "Light Conditions")
    assert libero_plus.normalize_categories("all") == libero_plus.CATEGORIES
    assert libero_plus.normalize_categories(None) == ()
    # Duplicates (alias + canonical name for the same dimension) collapse.
    assert libero_plus.normalize_categories(["layout", "Objects Layout"]) == ("Objects Layout",)
    with pytest.raises(ValueError):
        libero_plus.normalize_categories("not_a_perturbation")


@needs_checkout
@pytest.mark.parametrize("suite,size", EXPECTED_SUITE_SIZES.items())
def test_suite_sizes_and_category_coverage(suite, size):
    counts = libero_plus.categories_of_suite(suite)
    assert sum(counts.values()) == size
    # Every suite exercises all seven perturbation dimensions.
    assert set(counts) == set(libero_plus.CATEGORIES)


@needs_checkout
def test_list_tasks_filters_are_applied():
    tasks = libero_plus.list_tasks("libero_spatial", categories="camera", difficulty_levels=[1], limit=5)
    assert 0 < len(tasks) <= 5
    assert {t.category for t in tasks} == {"Camera Viewpoints"}
    assert {t.difficulty_level for t in tasks} == {1}
    # Returned in suite order, and task_id round-trips through resolve_task_id / describe_task.
    assert [t.task_id for t in tasks] == sorted(t.task_id for t in tasks)
    for t in tasks:
        assert libero_plus.resolve_task_id("libero_spatial", t.name) == t.task_id
        assert libero_plus.describe_task("libero_spatial", t.task_id)["name"] == t.name

    named = libero_plus.list_tasks("libero_spatial", name_contains="_light_", limit=3)
    assert named and all("_light_" in t.name for t in named)


@needs_checkout
def test_describe_task_is_none_for_unclassified_suites_and_bad_ids():
    # libero_90 is unperturbed in LIBERO-plus, so it carries no classification.
    assert libero_plus.describe_task("libero_90", 57) is None
    assert libero_plus.describe_task("libero_spatial", 10**9) is None
    with pytest.raises(KeyError):
        libero_plus.resolve_task_id("libero_spatial", "no_such_task")


@needs_checkout
def test_activate_is_idempotent_and_rejects_a_second_root(tmp_path):
    root = libero_plus.activate(require_assets=False)
    assert libero_plus.is_active() and libero_plus.active_root() == root
    assert libero_plus.activate(require_assets=False) == root  # idempotent
    # LIBERO-plus shadows `libero` process-wide, so switching roots mid-process must fail loudly.
    with pytest.raises((RuntimeError, FileNotFoundError)):
        libero_plus.activate(str(tmp_path), require_assets=False)
    # Activation redirects LIBERO's path config into the checkout.
    import libero.libero as libero_pkg

    assert os.path.abspath(libero_pkg.get_libero_path("bddl_files")).startswith(root)


@needs_checkout
def test_validate_root_rejects_a_plain_libero_checkout():
    plain = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "LIBERO")
    if not os.path.isdir(plain):
        pytest.skip("third_party/LIBERO not initialised")
    with pytest.raises(FileNotFoundError, match="does not look like LIBERO-plus"):
        libero_plus._validate_root(plain)


@pytest.mark.integration
@needs_checkout
@needs_assets
@pytest.mark.parametrize("category", ["Background Textures", "Camera Viewpoints", "Objects Layout"])
def test_build_perturbed_env(category):
    """Build a real perturbed env per dimension and check the pi0 observation contract holds."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from robometer_policy_learning.envs.dsrl_env_wrappers import setup_libero_env

    task = libero_plus.list_tasks("libero_spatial", categories=category, limit=1)[0]
    env, _ = setup_libero_env(
        task_suite_name="libero_spatial",
        task_id=task.task_id,
        n_envs=1,
        device="cpu",
        seed=0,
        max_episode_steps=20,
        use_libero_plus=True,
        init_state_index=0,
    )
    try:
        info = env.libero_task_info
        assert info["libero_plus"] and info["task_name"] == task.name
        assert info["perturbation"]["category"] == category
        # The instruction must come from the bddl, not from the perturbed FILENAME (upstream derives
        # `task.language` from the name, leaking suffixes like "... view 0 0 100 0 0 initstate 0").
        for marker in ("_view_", "_initstate_", "_noise_", "_table_", "initstate"):
            assert marker not in info["language"]

        obs, _ = env.reset()
        assert np.asarray(obs["observation/image"][0]).shape == (224, 224, 3)
        assert np.asarray(obs["observation/wrist_image"][0]).shape == (224, 224, 3)
        assert np.asarray(obs["observation/state"][0]).shape == (8,)
        obs, reward, terminated, truncated, _ = env.step(np.zeros((1, 7), dtype=np.float32))
        assert np.isfinite(np.asarray(reward)).all()
    finally:
        env.close()


@pytest.mark.integration
@needs_checkout
@needs_assets
def test_init_state_modes_control_the_start_state():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from robometer_policy_learning.envs.dsrl_env_wrappers import setup_libero_env

    def eef(env):
        return np.asarray(env.reset()[0]["observation/state"][0])[:3]

    env, _ = setup_libero_env(
        task_suite_name="libero_spatial", task_id=0, n_envs=1, device="cpu", seed=0,
        max_episode_steps=20, use_libero_plus=True, init_state_index="cycle",
    )
    try:
        positions, indices = [], []
        for _ in range(3):
            positions.append(eef(env))
            indices.append(env.envs[0].last_init_state_index)
        assert indices == [0, 1, 2]  # successive resets walk the benchmark's init states
        assert not np.allclose(positions[0], positions[1])  # ... and land on different start states
    finally:
        env.close()

    env, _ = setup_libero_env(
        task_suite_name="libero_spatial", task_id=0, n_envs=1, device="cpu", seed=0,
        max_episode_steps=20, use_libero_plus=True, init_state_index=3,
    )
    try:
        assert np.allclose(eef(env), eef(env))  # a pinned index is reproducible across resets
    finally:
        env.close()


# --------------------------------------------------------------------------------------------------
# Base tasks + per-episode perturbation sampling
# --------------------------------------------------------------------------------------------------
@needs_checkout
def test_strip_perturbation_handles_the_tricky_names():
    strip = libero_plus.strip_perturbation
    # A BASE task name that itself contains `_table_`: only `_table_<N>` is a marker.
    base = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
    assert strip(base) == base
    assert strip(base + "_table_7") == base
    # libero_goal's Objects Layout form strips `_moved` along with `_level<N>`.
    assert strip("open_the_middle_drawer_of_the_cabinet_moved_level1_sample1") == (
        "open_the_middle_drawer_of_the_cabinet"
    )
    # Camera / robot / noise markers, including the `_language_N` prefix form.
    stem = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    for suffix in (
        "_view_0_0_100_2_352_initstate_0",
        "_view_0_0_100_0_0_initstate_1",
        "_view_0_0_100_0_0_initstate_0_noise_3",
        "_language_1_view_0_0_100_0_0_initstate_0",
        "_light_1",
        "_add_10",
        "_tb_4",
    ):
        assert strip(stem + suffix) == stem, suffix


@needs_checkout
@pytest.mark.parametrize("suite", EXPECTED_SUITE_SIZES)
def test_every_task_strips_to_one_of_the_ten_base_tasks(suite):
    """The strong invariant behind base-task selection, checked over all 10,030 classified tasks."""
    originals = set(libero_plus.base_task_names(suite))
    assert len(originals) == 10
    for entry in libero_plus._classification(suite):
        assert libero_plus.strip_perturbation(entry["name"]) in originals, entry["name"]


@needs_checkout
@pytest.mark.parametrize("suite", EXPECTED_SUITE_SIZES)
def test_base_task_names_follow_plain_libero_order(suite):
    """Base ids must mean what they mean in plain LIBERO -- LIBERO-plus lists bases alphabetically."""
    plain = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "LIBERO", "libero", "libero", "benchmark", "libero_suite_task_map.py",
    )
    if not os.path.isfile(plain):
        pytest.skip("third_party/LIBERO not initialised")
    import ast

    with open(plain) as f:
        task_map = next(
            ast.literal_eval(n.value)
            for n in ast.parse(f.read()).body
            if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "libero_task_map" for t in n.targets)
        )
    assert libero_plus.base_task_names(suite) == task_map[suite]
    # Round-trip through a perturbed name.
    name = libero_plus.base_task_name(suite, 3)
    assert libero_plus.base_task_id_of(suite, name + "_table_7") == 3
    with pytest.raises(IndexError):
        libero_plus.base_task_name(suite, 10)


@needs_checkout
@pytest.mark.parametrize("suite", EXPECTED_SUITE_SIZES)
def test_every_base_task_and_family_cell_has_variants(suite):
    """`task_id + perturbation` must never resolve to an empty pool (4-50 variants per cell)."""
    for base_id in range(10):
        for category in libero_plus.CATEGORIES:
            pool = libero_plus.variants_of(suite, base_id, categories=category)
            assert pool, (suite, base_id, category)
            assert {v.category for v in pool} == {category}
            # Every variant really is a perturbation of the requested base task.
            base = libero_plus.base_task_name(suite, base_id)
            assert all(libero_plus.strip_perturbation(v.name) == base for v in pool)


@needs_checkout
def test_variants_of_rejects_bad_input():
    with pytest.raises(ValueError, match="no LIBERO-plus perturbation variants"):
        libero_plus.variants_of("libero_90", 0, categories="camera")
    with pytest.raises(ValueError, match="exactly one of"):
        libero_plus.variants_of("libero_spatial", categories="camera")
    with pytest.raises(ValueError, match="exactly one of"):
        libero_plus.variants_of("libero_spatial", 0, base_task="x", categories="camera")
    with pytest.raises(IndexError):
        libero_plus.variants_of("libero_spatial", 99, categories="camera")
    # An over-restrictive filter must fail loudly rather than yield "no episodes to run".
    with pytest.raises(ValueError, match="No LIBERO-plus variants"):
        libero_plus.variants_of("libero_spatial", 0, categories="camera", name_contains="nonexistent")


@needs_checkout
def test_nullable_difficulty_level_is_tolerated():
    """Upstream leaves difficulty_level null for 121 libero_goal / Light Conditions tasks."""
    pool = libero_plus.variants_of("libero_goal", 0, categories="light")
    assert any(v.difficulty_level is None for v in pool)
    # A level filter must exclude the nulls rather than crash on them.
    filtered = libero_plus.variants_of("libero_goal", 0, categories="light", difficulty_levels=[1, 2, 3, 4, 5])
    assert 0 < len(filtered) < len(pool)
    assert all(v.difficulty_level is not None for v in filtered)


@needs_checkout
def test_variant_cycler_is_seeded_and_avoids_repeats():
    pool = libero_plus.variants_of("libero_spatial", 3, categories="camera")
    assert len(pool) > 5

    def draw(n, seed=0, sampling="shuffle"):
        cycler = libero_plus.VariantCycler(pool, seed=seed, sampling=sampling)
        return [cycler.next().task_id for _ in range(n)]

    first_epoch = draw(len(pool))
    assert first_epoch == draw(len(pool))                    # reproducible from the seed alone
    assert first_epoch != draw(len(pool), seed=1)            # and the seed actually matters
    assert len(set(first_epoch)) == len(pool)                # no repeats within an epoch
    assert draw(4, sampling="sequential") == [v.task_id for v in pool[:4]]

    # Sparse cells exist (as few as 4 variants), so exhaustion must wrap rather than raise.
    small = libero_plus.variants_of("libero_spatial", 8, categories="light")
    cycler = libero_plus.VariantCycler(small, seed=0)
    drawn = [cycler.next().task_id for _ in range(len(small) + 2)]
    assert set(drawn) == {v.task_id for v in small}
    assert cycler.epoch == 1
    with pytest.raises(ValueError):
        libero_plus.VariantCycler([])
    with pytest.raises(ValueError):
        libero_plus.VariantCycler(pool, sampling="bogus")


@pytest.mark.integration
@needs_checkout
@needs_assets
def test_per_episode_variant_rebuild_changes_the_scene():
    """The eval / HITL inner loop: sample a variant, rebuild the env, and get a genuinely new scene."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from robometer_policy_learning.envs.dsrl_env_wrappers import setup_libero_env

    pool = libero_plus.variants_of("libero_spatial", 0, categories="background")
    cycler = libero_plus.VariantCycler(pool, seed=0)
    seen_ids, images = [], []
    for _ in range(3):
        variant = cycler.next()
        env, _ = setup_libero_env(
            task_suite_name="libero_spatial", task_id=variant.task_id, n_envs=1, device="cpu",
            seed=0, max_episode_steps=20, use_libero_plus=True, init_state_index=0,
        )
        try:
            assert env.libero_task_info["task_name"] == variant.name
            assert env.libero_task_info["perturbation"]["category"] == "Background Textures"
            seen_ids.append(variant.task_id)
            images.append(np.asarray(env.reset()[0]["observation/image"][0], dtype=np.float64))
        finally:
            env.close()

    assert len(set(seen_ids)) == 3, "each episode must run a distinct variant"
    # Same base task and same init state, so any difference is the texture perturbation itself.
    assert not np.allclose(images[0], images[1])
    assert not np.allclose(images[1], images[2])
