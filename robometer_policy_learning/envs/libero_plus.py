"""LIBERO-plus support: activation of the alternate ``libero`` checkout + task selection helpers.

`LIBERO-plus <https://github.com/sylvestf/LIBERO-plus>`_ is a drop-in fork of LIBERO that expands the
four 10-task suites into 10,030 perturbed tasks across seven perturbation dimensions (background
textures, camera viewpoints, language instructions, light conditions, objects layout, robot initial
states, sensor noise). ``libero_90`` is byte-identical in both checkouts.

Why this module exists
----------------------
LIBERO-plus ships the SAME top-level ``libero`` package as LIBERO, so the two cannot be pip-installed
side by side. Instead of replacing the install (which would silently change what every existing
``task_id`` means -- the expanded suites do NOT contain the original unperturbed tasks, and are
ordered differently), LIBERO-plus is kept as an un-installed checkout at ``third_party/LIBERO-plus``
and selected per run:

* ``libero`` resolves paths through ``$LIBERO_CONFIG_PATH/config.yaml`` (see
  ``libero/libero/__init__.py``), so pointing that at a generated config redirects bddl files,
  init states and assets into the LIBERO-plus checkout.
* the editable install of LIBERO *appends* its finder to ``sys.meta_path``, which is consulted after
  the regular ``sys.path``-based finder -- so ``sys.path.insert(0, <LIBERO-plus>)`` shadows it.

:func:`activate` does both, and must run before the first ``import libero`` in the process.
``setup_libero_env(..., use_libero_plus=True)`` calls it for you.

How the perturbations are encoded
---------------------------------
Everything rides on the task NAME; there is no extra env kwarg. A LIBERO-plus task name carries one
or more markers, and LIBERO-plus's ``ControlEnv``/``Benchmark`` decode them:

===========================  =================================================================
marker                       effect
===========================  =================================================================
``_table_N`` / ``_tb_N``     background/table texture swap (real bddl file on disk)
``_light_N``                 light intensity/direction/color/shadow (real bddl file)
``_language_N``              LLM-rewritten instruction (real bddl file, before ``_view_``)
``_add_N`` / ``_levelN``     confounding objects / displaced layout (bddl + libero_newobj inits)
``_view_h_v_s_r_t``          camera pose/FOV -- VIRTUAL path, stripped by ``ControlEnv``
``_initstate_N``             robot base variant (``MountedPandaN``), N=0 means unperturbed
``_noise_N``                 photometric corruption applied to ``agentview_image`` post-render
===========================  =================================================================

Because ``_view_``/``_initstate_``/``_noise_`` names are stripped by ``ControlEnv`` before touching
the filesystem, the bddl path our code builds for them does not exist on disk -- that is expected and
handled upstream. The init-state file is likewise resolved back to the base task by
``Benchmark.get_task_init_states``.

Extra runtime dependencies
--------------------------
LIBERO-plus's ``envs/env_wrapper.py`` imports ``wand`` and ``skimage`` at module import time (they
implement the Sensor Noise corruptions), so both are needed for ANY LIBERO-plus env:

* ``scikit-image`` -- pure pip install; :func:`activate` raises if missing.
* ``wand`` -- a ctypes binding that additionally needs the system ImageMagick library
  (``sudo apt-get install -y libmagickwand-dev``). When it is unimportable, :func:`activate` installs
  a stub module so every non-motion-blur task still runs, and only motion-blur tasks (``_noise_N``
  with N <= 10) fail, with an actionable message.

Both are declared in the ``libero-plus`` dependency group (``uv sync --group libero-plus``).

Usage
-----
    from robometer_policy_learning.envs.libero_plus import activate, list_tasks

    activate()                                       # before any `import libero`
    tasks = list_tasks("libero_spatial", categories="camera", difficulty_levels=[1, 2])
    ids = [t.task_id for t in tasks]
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import yaml
from loguru import logger

__all__ = [
    "CATEGORIES",
    "PERTURBED_SUITES",
    "TaskInfo",
    "VariantCycler",
    "activate",
    "active_root",
    "base_task_id_of",
    "base_task_name",
    "base_task_names",
    "categories_of_suite",
    "default_root",
    "describe_task",
    "is_active",
    "list_tasks",
    "load_task_init_states",
    "make_task_suite",
    "normalize_categories",
    "resolve_task_id",
    "strip_perturbation",
    "variants_of",
]

# Canonical category names, exactly as they appear in benchmark/task_classification.json.
CATEGORIES = (
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
)

# Suites LIBERO-plus expands with perturbed variants (libero_90 is left untouched, and libero_100 is
# a concatenation, so neither appears in task_classification.json).
PERTURBED_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# Convenience aliases so configs can say `categories: [camera, light]` instead of the long names.
_CATEGORY_ALIASES: Dict[str, str] = {
    "background": "Background Textures",
    "background_textures": "Background Textures",
    "texture": "Background Textures",
    "textures": "Background Textures",
    "camera": "Camera Viewpoints",
    "camera_viewpoints": "Camera Viewpoints",
    "view": "Camera Viewpoints",
    "viewpoint": "Camera Viewpoints",
    "language": "Language Instructions",
    "language_instructions": "Language Instructions",
    "instruction": "Language Instructions",
    "light": "Light Conditions",
    "light_conditions": "Light Conditions",
    "lighting": "Light Conditions",
    "layout": "Objects Layout",
    "objects": "Objects Layout",
    "objects_layout": "Objects Layout",
    "robot": "Robot Initial States",
    "robot_initial_states": "Robot Initial States",
    "initstate": "Robot Initial States",
    "noise": "Sensor Noise",
    "sensor": "Sensor Noise",
    "sensor_noise": "Sensor Noise",
}

_ACTIVE_ROOT: Optional[str] = None
_CLASSIFICATION_CACHE: Dict[str, List[dict]] = {}


@dataclass(frozen=True)
class TaskInfo:
    """One LIBERO-plus task: its suite index plus the perturbation it encodes."""

    task_id: int  # index passed to `Benchmark.get_task` / `env.task_id` (0-based)
    name: str  # task name as registered in libero_task_map
    suite: str
    category: str  # one of CATEGORIES
    # 1 (easiest) .. 5 as scored by the LIBERO-plus authors, or None -- upstream leaves it null
    # for 121 tasks (all libero_goal / Light Conditions).
    difficulty_level: Optional[int]

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "suite": self.suite,
            "category": self.category,
            "difficulty_level": self.difficulty_level,
        }


# --------------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------------
def _repo_root() -> str:
    # <repo>/robometer_policy_learning/envs/libero_plus.py -> <repo>
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_root() -> str:
    """Filesystem root of the LIBERO-plus checkout (``$LIBERO_PLUS_ROOT`` or the submodule path)."""
    env_root = os.environ.get("LIBERO_PLUS_ROOT")
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))
    return os.path.join(_repo_root(), "third_party", "LIBERO-plus")


def _validate_root(root: str) -> str:
    root = os.path.abspath(os.path.expanduser(root))
    pkg = os.path.join(root, "libero", "libero")
    if not os.path.isdir(pkg):
        raise FileNotFoundError(
            f"LIBERO-plus checkout not found at {root} (expected {pkg}). "
            "Run `bash scripts/setup_libero_plus.sh` to initialise the submodule and fetch assets, "
            "or set LIBERO_PLUS_ROOT to an existing checkout."
        )
    if not os.path.isfile(os.path.join(pkg, "benchmark", "task_classification.json")):
        raise FileNotFoundError(
            f"{root} does not look like LIBERO-plus (benchmark/task_classification.json is missing). "
            "It may be a plain LIBERO checkout."
        )
    return root


def _path_dict(root: str) -> Dict[str, str]:
    """The five paths ``libero.libero.get_libero_path`` reads, pointed at the LIBERO-plus checkout."""
    pkg = os.path.join(root, "libero", "libero")
    datasets = os.environ.get("LIBERO_PLUS_DATASETS") or os.path.join(root, "libero", "datasets")
    return {
        "benchmark_root": pkg,
        "bddl_files": os.path.join(pkg, "bddl_files"),
        "init_states": os.path.join(pkg, "init_files"),
        "datasets": os.path.abspath(os.path.expanduser(datasets)),
        "assets": os.path.join(pkg, "assets"),
    }


def _write_config(root: str) -> str:
    """Write the LIBERO path config for LIBERO-plus and return the directory holding it.

    Kept outside the submodule (default ``~/.libero_plus``, override with
    ``$LIBERO_PLUS_CONFIG_PATH``) so activation never dirties the checkout and never touches the
    plain-LIBERO ``~/.libero/config.yaml``. Rewritten whenever the content would change, so moving
    the repo self-heals.
    """
    config_dir = os.path.abspath(
        os.path.expanduser(os.environ.get("LIBERO_PLUS_CONFIG_PATH") or "~/.libero_plus")
    )
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "config.yaml")
    wanted = _path_dict(root)
    current = None
    if os.path.isfile(config_file):
        try:
            with open(config_file) as f:
                current = yaml.safe_load(f.read())
        except Exception:  # a corrupt config is simply rewritten
            current = None
    if current != wanted:
        with open(config_file, "w") as f:
            yaml.dump(wanted, f)
        logger.debug(f"Wrote LIBERO-plus path config to {config_file}")
    return config_dir


# --------------------------------------------------------------------------------------------------
# Dependency preflight
# --------------------------------------------------------------------------------------------------
_WAND_HINT = (
    "LIBERO-plus needs the `wand` binding to ImageMagick for motion-blur Sensor Noise tasks "
    "(`_noise_N` with N<=10). Install it with `uv sync --group libero-plus` plus the system library: "
    "`sudo apt-get install -y libmagickwand-dev`."
)


def _install_wand_stub() -> None:
    """Register a stub ``wand`` package so LIBERO-plus imports without a working ImageMagick.

    ``env_wrapper.py`` does ``from wand.api import library``, sets ``library.MagickMotionBlurImage
    .argtypes`` and subclasses ``wand.image.Image`` -- all at import time, unconditionally. The stub
    satisfies those three operations and raises only if a motion-blur task is actually stepped.
    """
    import types

    class _MissingFunc:
        """Accepts attribute writes (``argtypes``); raises when called."""

        def __setattr__(self, name, value):  # allow `.argtypes = (...)`
            object.__setattr__(self, name, value)

        def __call__(self, *args, **kwargs):
            raise RuntimeError(_WAND_HINT)

    class _MissingLibrary:
        def __getattr__(self, name):
            func = _MissingFunc()
            object.__setattr__(self, name, func)
            return func

    class Image:  # mirrors wand.image.Image
        def __init__(self, *args, **kwargs):
            raise RuntimeError(_WAND_HINT)

    wand = types.ModuleType("wand")
    wand.__path__ = []  # mark as a package so `wand.api` / `wand.image` resolve
    api = types.ModuleType("wand.api")
    api.library = _MissingLibrary()
    image = types.ModuleType("wand.image")
    image.Image = Image
    wand.api, wand.image = api, image
    sys.modules.setdefault("wand", wand)
    sys.modules.setdefault("wand.api", api)
    sys.modules.setdefault("wand.image", image)


def _check_env_deps(require: bool) -> None:
    """Verify (or stub) the imports LIBERO-plus's env_wrapper performs at module import time."""
    if importlib.util.find_spec("skimage") is None:
        msg = (
            "LIBERO-plus requires scikit-image (imported by libero/libero/envs/env_wrapper.py). "
            "Install it with `uv sync --group libero-plus` (or `uv pip install scikit-image`)."
        )
        if require:
            raise ImportError(msg)
        logger.warning(msg)

    if "wand" in sys.modules:
        return
    try:
        importlib.import_module("wand.api")  # fails without the system ImageMagick library
        importlib.import_module("wand.image")
    except Exception as e:  # ImportError, OSError, ... all mean "unusable"
        logger.warning(f"`wand` is unusable ({type(e).__name__}: {e}); installing a stub. {_WAND_HINT}")
        _install_wand_stub()


# --------------------------------------------------------------------------------------------------
# Activation
# --------------------------------------------------------------------------------------------------
def is_active() -> bool:
    """True if :func:`activate` has redirected ``libero`` to the LIBERO-plus checkout."""
    return _ACTIVE_ROOT is not None


def active_root() -> Optional[str]:
    """The LIBERO-plus root currently activated, or None."""
    return _ACTIVE_ROOT


def _imported_libero_root() -> Optional[str]:
    """Root of an already-imported ``libero`` package, or None if it has not been imported."""
    mod = sys.modules.get("libero")
    path = getattr(mod, "__file__", None) or next(iter(getattr(mod, "__path__", []) or []), None)
    if not path:
        return None
    # <root>/libero/__init__.py -> <root>
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(path)), ".."))


def activate(root: Optional[str] = None, *, require_assets: bool = True, require_env_deps: bool = True) -> str:
    """Make ``import libero`` resolve to the LIBERO-plus checkout. Returns the activated root.

    Must be called BEFORE the first ``import libero`` in the process; raises if ``libero`` was
    already imported from somewhere else. Idempotent for the same root.

    Args:
        root: LIBERO-plus checkout. Defaults to :func:`default_root`.
        require_assets: raise if ``libero/libero/assets`` is missing (the 6.4 GB HF download). Set
            False for metadata-only use, where no MuJoCo scene is ever built.
        require_env_deps: raise if ``scikit-image`` is missing rather than only warning.
    """
    global _ACTIVE_ROOT

    root = _validate_root(root or default_root())

    if _ACTIVE_ROOT is not None:
        if _ACTIVE_ROOT != root:
            raise RuntimeError(
                f"LIBERO-plus is already activated from {_ACTIVE_ROOT}; cannot switch to {root} "
                "in the same process."
            )
        return _ACTIVE_ROOT

    imported_root = _imported_libero_root()
    if imported_root is not None and imported_root != root:
        raise RuntimeError(
            f"`libero` was already imported from {imported_root}, so LIBERO-plus ({root}) can no "
            "longer shadow it. Call libero_plus.activate() (or build the env via "
            "setup_libero_env(use_libero_plus=True)) before importing libero."
        )

    assets = os.path.join(root, "libero", "libero", "assets")
    if not os.path.isdir(assets):
        msg = (
            f"LIBERO-plus assets are missing at {assets}. They are a separate ~6.4 GB download: "
            "run `bash scripts/setup_libero_plus.sh`."
        )
        if require_assets:
            raise FileNotFoundError(msg)
        logger.warning(msg)

    _check_env_deps(require=require_env_deps)

    # 1. Redirect the path config (read lazily by get_libero_path on every call).
    os.environ["LIBERO_CONFIG_PATH"] = _write_config(root)
    # 2. Shadow the editable-installed LIBERO. Its finder is APPENDED to sys.meta_path, which runs
    #    after the sys.path-based PathFinder, so a front insert wins.
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

    _ACTIVE_ROOT = root
    _CLASSIFICATION_CACHE.clear()
    logger.info(f"LIBERO-plus activated: {root} (LIBERO_CONFIG_PATH={os.environ['LIBERO_CONFIG_PATH']})")
    return root


# --------------------------------------------------------------------------------------------------
# Task selection
# --------------------------------------------------------------------------------------------------
def normalize_categories(spec: Union[None, str, Iterable[str]]) -> tuple:
    """Map a category spec (canonical names, aliases, or ``"all"``) to canonical category names."""
    if spec is None:
        return ()
    if isinstance(spec, str):
        spec = [spec]
    out: List[str] = []
    for raw in spec:
        key = str(raw).strip()
        if key.lower() in ("all", "*"):
            return tuple(CATEGORIES)
        canonical = None
        for cat in CATEGORIES:
            if key.lower() == cat.lower():
                canonical = cat
                break
        if canonical is None:
            canonical = _CATEGORY_ALIASES.get(key.lower().replace(" ", "_").replace("-", "_"))
        if canonical is None:
            raise ValueError(
                f"Unknown LIBERO-plus perturbation category {raw!r}. "
                f"Valid: {list(CATEGORIES)} or aliases {sorted(set(_CATEGORY_ALIASES))}."
            )
        if canonical not in out:
            out.append(canonical)
    return tuple(out)


def _entry_level(entry: dict) -> Optional[int]:
    """``difficulty_level`` of a classification entry, or None (upstream leaves 121 of them null)."""
    level = entry.get("difficulty_level")
    return None if level is None else int(level)


def _entry_to_task_info(entry: dict, suite: str) -> TaskInfo:
    """Build a :class:`TaskInfo` from a classification entry (json ``id`` is 1-based)."""
    return TaskInfo(
        task_id=int(entry["id"]) - 1,
        name=entry["name"],
        suite=suite,
        category=entry["category"],
        difficulty_level=_entry_level(entry),
    )


def _classification(suite: str, root: Optional[str] = None) -> List[dict]:
    """Load ``task_classification.json`` for one suite (cached).

    The file's entries are index-aligned with the suite's task list and use 1-based ``id``, so
    ``task_id = id - 1``.
    """
    suite = str(suite).lower()
    if suite in _CLASSIFICATION_CACHE:
        return _CLASSIFICATION_CACHE[suite]
    root = _validate_root(root or _ACTIVE_ROOT or default_root())
    path = os.path.join(root, "libero", "libero", "benchmark", "task_classification.json")
    with open(path) as f:
        data = json.load(f)
    if suite not in data:
        raise KeyError(
            f"Suite {suite!r} has no LIBERO-plus perturbation classification. "
            f"Classified suites: {list(data)} (libero_90 is unperturbed in LIBERO-plus)."
        )
    _CLASSIFICATION_CACHE[suite] = data[suite]
    return data[suite]


def categories_of_suite(suite: str, root: Optional[str] = None) -> Dict[str, int]:
    """Task count per perturbation category for one suite."""
    counts: Dict[str, int] = {}
    for entry in _classification(suite, root):
        counts[entry["category"]] = counts.get(entry["category"], 0) + 1
    return counts


def list_tasks(
    suite: str,
    *,
    categories: Union[None, str, Iterable[str]] = None,
    difficulty_levels: Optional[Sequence[int]] = None,
    name_contains: Optional[str] = None,
    limit: Optional[int] = None,
    root: Optional[str] = None,
) -> List[TaskInfo]:
    """Select LIBERO-plus tasks from one suite by perturbation category / difficulty / name.

    Args:
        suite: one of :data:`PERTURBED_SUITES`.
        categories: category names or aliases (``"camera"``, ``"noise"``, ...), or ``"all"``.
        difficulty_levels: keep only these author-assigned difficulty levels (1 = easiest). Tasks
            whose level is null upstream never match a level filter.
        name_contains: substring filter on the task name, e.g. the base task it perturbs.
        limit: keep at most this many tasks (after filtering, in suite order).

    Returns:
        Matching :class:`TaskInfo` records in suite order (i.e. ascending ``task_id``).
    """
    wanted = normalize_categories(categories)
    levels = {int(x) for x in difficulty_levels} if difficulty_levels else None
    suite_name = str(suite).lower()

    out: List[TaskInfo] = []
    for entry in _classification(suite_name, root):
        if wanted and entry["category"] not in wanted:
            continue
        if levels is not None and _entry_level(entry) not in levels:
            continue
        if name_contains and name_contains not in entry["name"]:
            continue
        out.append(_entry_to_task_info(entry, suite_name))
        if limit is not None and len(out) >= limit:
            break
    return out


def describe_task(suite: str, task_id: int, root: Optional[str] = None) -> Optional[dict]:
    """Perturbation metadata for one ``task_id``, or None if the suite/id is not classified."""
    try:
        entries = _classification(suite, root)
    except (KeyError, FileNotFoundError):
        return None
    if not 0 <= int(task_id) < len(entries):
        return None
    entry = entries[int(task_id)]
    return {
        "task_id": int(task_id),
        "name": entry["name"],
        "suite": str(suite).lower(),
        "category": entry["category"],
        "difficulty_level": _entry_level(entry),
    }


def make_task_suite(task_suite_name: str, task_order_index: int = 0):
    """Instantiate a ``libero`` benchmark suite without its multi-thousand-element stdout dump.

    ``Benchmark._make_benchmark`` prints the whole task-order permutation, which is 2402+ integers per
    env construction under LIBERO-plus. Requires ``libero`` to be importable (activate first when
    using LIBERO-plus).
    """
    import contextlib
    import io

    from libero.libero import benchmark

    suite_cls = benchmark.get_benchmark_dict()[task_suite_name]
    with contextlib.redirect_stdout(io.StringIO()):
        return suite_cls(task_order_index=task_order_index)


def load_task_init_states(task_suite, task_id: int):
    """Load a task's fixed benchmark init states as a ``[N, D]`` float array.

    Works around an upstream incompatibility present in BOTH LIBERO and LIBERO-plus:
    ``Benchmark.get_task_init_states`` calls ``torch.load(path)`` on ``.pruned_init`` files, which are
    plain numpy pickles. torch >= 2.6 flipped ``weights_only`` to True, so that call raises
    ``UnpicklingError``. These files are local benchmark data from a pinned checkout, so the full
    pickle is allowed for the duration of the call.

    Not thread-safe (it swaps ``torch.load`` while loading); env construction here is single-threaded.
    """
    import numpy as np
    import torch

    original_load = torch.load

    def _load_allowing_pickle(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _load_allowing_pickle
    try:
        states = task_suite.get_task_init_states(int(task_id))
    finally:
        torch.load = original_load

    states = np.asarray(states.cpu() if hasattr(states, "cpu") else states)
    if states.ndim == 1:  # LIBERO-plus reshapes `_add_`/`_level` inits to a single row
        states = states.reshape(1, -1)
    return states


def resolve_task_id(suite: str, task_name: str, root: Optional[str] = None) -> int:
    """Look up the ``task_id`` of a LIBERO-plus task by exact name.

    Uses ``task_classification.json``, which is index-aligned with the suite's task list, so this
    works without importing ``libero``.
    """
    for entry in _classification(suite, root):
        if entry["name"] == task_name:
            return int(entry["id"]) - 1
    raise KeyError(
        f"Task {task_name!r} not found in LIBERO-plus suite {suite!r}. "
        "Use list_tasks(name_contains=...) to search."
    )


# --------------------------------------------------------------------------------------------------
# Base tasks and per-episode perturbation sampling
#
# LIBERO-plus task names are `{base task name}{perturbation markers}`. Selecting a BASE task plus a
# perturbation family (rather than one opaque variant id) is what lets an eval / HITL run put every
# episode under a *different* perturbation of the same kind -- the way the benchmark is meant to be
# used. Every (base task, family) cell is populated, with 4-50 variants (typically 20-50).
# --------------------------------------------------------------------------------------------------

# Digit-anchored so a marker is only recognised when a perturbation INDEX follows it. Two real cases
# depend on this:
#   * `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` is a BASE task name that
#     itself contains `_table_`; `_table_center` must not be mistaken for the `_table_<N>` marker.
#   * libero_goal's Objects Layout variants use `..._moved_level<N>_sample<M>`, so `_moved` has to go
#     with `_level<N>`.
# `_view_<N>_` keeps a trailing underscore because the camera marker is `_view_h_v_scale_rot_vert`.
_PERTURBATION_MARKER = re.compile(r"_(?:table|tb|light|language|add|noise)_\d+|(?:_moved)?_level\d+|_view_\d+_")

# Base task names per suite, in PLAIN-LIBERO order (see _plain_task_map).
_BASE_TASK_CACHE: Dict[str, List[str]] = {}


def strip_perturbation(task_name: str) -> str:
    """Reduce a LIBERO-plus task name to the base task it perturbs.

    Verified against all 10,030 classified tasks: the result is always exactly one of the suite's ten
    original LIBERO task names.
    """
    match = _PERTURBATION_MARKER.search(task_name)
    return task_name[: match.start()] if match else task_name


def _plain_task_map() -> Dict[str, List[str]]:
    """Read plain LIBERO's ``libero_task_map`` from the LIBERO checkout, without importing ``libero``.

    Base task ids must follow PLAIN LIBERO's task order so that ``task_id=3`` keeps the meaning it has
    in every existing config: LIBERO-plus lists base tasks alphabetically instead, which is a different
    order in all four suites. The file is a single dict literal, so it is read with ``ast.literal_eval``
    -- importing ``libero`` is not an option here because LIBERO-plus shadows that package.
    """
    import ast

    path = os.path.join(_repo_root(), "third_party", "LIBERO", "libero", "libero", "benchmark",
                        "libero_suite_task_map.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cannot read plain LIBERO's task map at {path}, which defines the base-task ordering. "
            "Initialise the LIBERO submodule: `git submodule update --init third_party/LIBERO`."
        )
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "libero_task_map" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"No `libero_task_map` assignment found in {path}")


def base_task_names(suite: str) -> List[str]:
    """The suite's original (unperturbed) task names, indexed by plain-LIBERO ``task_id``."""
    suite = str(suite).lower()
    if suite not in _BASE_TASK_CACHE:
        task_map = _plain_task_map()
        if suite not in task_map:
            raise KeyError(f"Unknown LIBERO suite {suite!r}. Available: {sorted(task_map)}.")
        _BASE_TASK_CACHE[suite] = list(task_map[suite])
    return _BASE_TASK_CACHE[suite]


def base_task_name(suite: str, base_task_id: int) -> str:
    """Name of base task ``base_task_id`` (plain-LIBERO ordering)."""
    names = base_task_names(suite)
    idx = int(base_task_id)
    if not 0 <= idx < len(names):
        raise IndexError(
            f"Base task id {idx} out of range for suite {suite!r} ({len(names)} base tasks). "
            "Under LIBERO-plus, `task_id` with a perturbation family selects a BASE task, not a variant."
        )
    return names[idx]


def base_task_id_of(suite: str, task_name: str) -> int:
    """Plain-LIBERO ``task_id`` of a base task name (accepts a perturbed name and strips it first)."""
    base = strip_perturbation(task_name)
    names = base_task_names(suite)
    if base not in names:
        raise KeyError(f"Base task {base!r} (from {task_name!r}) is not in suite {suite!r}.")
    return names.index(base)


def variants_of(
    suite: str,
    base_task_id: Union[int, None] = None,
    *,
    base_task: Optional[str] = None,
    categories: Union[None, str, Iterable[str]] = None,
    difficulty_levels: Optional[Sequence[int]] = None,
    name_contains: Optional[str] = None,
    root: Optional[str] = None,
) -> List[TaskInfo]:
    """All LIBERO-plus variants of ONE base task, optionally restricted to perturbation families.

    Args:
        suite: one of :data:`PERTURBED_SUITES` (``libero_90`` is unperturbed, so it has no variants).
        base_task_id: base task index in PLAIN-LIBERO order (0..9). Mutually exclusive with
            ``base_task``.
        base_task: base task name instead of an index.
        categories: perturbation families to keep (names or aliases, or ``"all"``).
        difficulty_levels / name_contains: further filters, as in :func:`list_tasks`.

    Returns:
        Matching :class:`TaskInfo` records in suite order. Raises if the pool ends up empty, since
        that silently degrades to "no episodes to run".
    """
    suite = str(suite).lower()
    if suite not in PERTURBED_SUITES:
        raise ValueError(
            f"Suite {suite!r} has no LIBERO-plus perturbation variants (it is unperturbed in "
            f"LIBERO-plus). Perturbed suites: {list(PERTURBED_SUITES)}."
        )
    if (base_task_id is None) == (base_task is None):
        raise ValueError("Pass exactly one of base_task_id or base_task.")
    base = base_task if base_task is not None else base_task_name(suite, base_task_id)
    if base not in base_task_names(suite):
        raise KeyError(f"Base task {base!r} is not one of suite {suite!r}'s base tasks.")

    wanted = normalize_categories(categories)
    levels = {int(x) for x in difficulty_levels} if difficulty_levels else None

    out: List[TaskInfo] = []
    for entry in _classification(suite, root):
        if strip_perturbation(entry["name"]) != base:
            continue
        if wanted and entry["category"] not in wanted:
            continue
        if levels is not None and _entry_level(entry) not in levels:
            continue
        if name_contains and name_contains not in entry["name"]:
            continue
        out.append(_entry_to_task_info(entry, suite))
    if not out:
        raise ValueError(
            f"No LIBERO-plus variants of {base!r} in {suite!r} match categories={list(wanted) or 'any'}, "
            f"difficulty_levels={difficulty_levels}, name_contains={name_contains!r}."
        )
    return out


class VariantCycler:
    """Hands out one perturbation variant per episode, without repeats until the pool is exhausted.

    ``sampling="shuffle"`` (default) walks a seeded permutation of the pool, reshuffling each time it
    runs dry, so a run sees maximally different perturbations and the SEQUENCE is reproducible from
    ``seed`` alone -- which is what makes HITL round 0 and round N evals comparable.
    ``sampling="sequential"`` walks the pool in suite order instead (adjacent ids are near-identical
    perturbations, so this samples a narrow slice in a short run).

    Uses a private ``numpy.random.Generator``, never the global RNG, so variant choice cannot disturb
    env/policy seeding.
    """

    def __init__(self, variants: Sequence[TaskInfo], seed: int = 0, sampling: str = "shuffle"):
        import numpy as np

        if not variants:
            raise ValueError("VariantCycler needs a non-empty variant pool.")
        sampling = str(sampling).lower()
        if sampling not in ("shuffle", "sequential"):
            raise ValueError(f"sampling must be 'shuffle' or 'sequential', got {sampling!r}")
        self.variants = list(variants)
        self.sampling = sampling
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.epoch = 0  # how many times the pool has been walked end to end
        self._order: List[int] = []
        self._pos = 0
        self._reorder()

    @property
    def num_variants(self) -> int:
        return len(self.variants)

    def _reorder(self) -> None:
        if self.sampling == "shuffle":
            self._order = list(self._rng.permutation(len(self.variants)))
        else:
            self._order = list(range(len(self.variants)))
        self._pos = 0

    def next(self) -> TaskInfo:
        """The next variant. Wraps around (re-shuffling) once the pool is exhausted."""
        if self._pos >= len(self._order):
            self.epoch += 1
            self._reorder()
        index = self._order[self._pos]
        self._pos += 1
        return self.variants[index]
