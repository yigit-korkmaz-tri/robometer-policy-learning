#!/usr/bin/env python3
"""Convert a LeRobot dataset (real-robot data) to the robomimic-style HDF5 the buffers consume.

LeRobot stores low-dim features in per-episode parquet files and camera streams either as
per-episode MP4s (``dtype: video``) or as per-frame PNGs (``dtype: image``). ``H5ReplayBuffer``
instead expects a single robomimic-style HDF5:

    /data.attrs["total"]                       total number of transitions
    /data/demo_{i}/actions                     [T, A] float32
    /data/demo_{i}/rewards                     [T]    float32
    /data/demo_{i}/dones                       [T]    int64   (1 at the terminal step)
    /data/demo_{i}/obs/state                   [T, S] float32
    /data/demo_{i}/obs/{image_key}             [T, H, W, 3] uint8
    /data/demo_{i}/next_obs/...                only with --write-next-obs (see below)
    /data/demo_{i}/language_instruction        the task string
    /data/demo_{i}.attrs["num_samples"], .attrs["prompt"]
    /meta/min_action, /meta/max_action         [A] float32 action bounds
    /meta.attrs["info"]                        JSON provenance

The image observation keys must contain one of ``image``/``rgb``/``camera``/``cam`` -- that
substring is how ``H5ReplayBuffer`` decides which keys are read lazily per sample instead of being
cached in RAM. ``--camera-map`` controls the naming and defaults to the YAM bimanual layout.

``next_obs`` is *not* written by default: per-sample image reads always go through the ``obs``
group, so the only consumer of a ``next_obs`` group is the frozen-DINO precompute path, which now
falls back to ``obs`` when the group is absent. Writing it would roughly double the file size.

The dataset is read directly (pyarrow + PyAV/cv2) rather than through ``LeRobotDataset``, because
``lerobot`` is only installed transitively via the ``openpi`` extra and that extra is mutually
exclusive with ``robometer``. Pass --use-lerobot-api to go through ``LeRobotDataset`` instead.

Both LeRobot layouts are supported and detected from ``meta/info.json``:

* **v2.x** -- one parquet and one MP4 *per episode*; episode/task metadata in ``meta/*.jsonl``.
* **v3.0** -- episodes are *packed* into shared files: ``data/chunk-{c}/file-{f}.parquet`` holds
  many episodes' rows (selected by the ``episode_index`` column) and
  ``videos/{key}/chunk-{c}/file-{f}.mp4`` concatenates many episodes' frames (selected by the
  ``from_timestamp``/``to_timestamp`` range recorded per episode). Metadata lives in
  ``meta/episodes/**/*.parquet`` and ``meta/tasks.parquet`` instead of JSONL.

Note that ``--use-lerobot-api`` only works for v2.x: the pinned ``lerobot`` is v2.1-era and its
``LeRobotDataset`` cannot open a v3.0 dataset.

Usage:
    uv run python scripts/convert_lerobot_to_h5.py \
        --repo-id ykorkmaz/yam_raiden_test \
        --output data/lerobot_datasets/yam_raiden_test.hdf5 \
        --image-size 224
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from robometer_policy_learning.utils.policy_serving import resize_image

# LeRobot feature-name conventions.
STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
IMAGE_FEATURE_PREFIX = "observation.images."

# Default camera renaming for the YAM bimanual layout. `image` is the repo-wide convention for the
# primary camera (matches the default `dino_image_keys: ["image"]`).
DEFAULT_CAMERA_MAP = {
    "scene_camera": "image",
    "left_wrist_camera": "left_wrist_image",
    "right_wrist_camera": "right_wrist_image",
}

_IMAGE_KEY_KEYWORDS = ("image", "rgb", "camera", "cam")


# --------------------------------------------------------------------------------------
# Dataset discovery
# --------------------------------------------------------------------------------------
def default_lerobot_home() -> Path:
    """Root LeRobot caches datasets under, mirroring lerobot's own HF_LEROBOT_HOME resolution."""
    for var in ("HF_LEROBOT_HOME", "LEROBOT_HOME"):
        if os.environ.get(var):
            return Path(os.environ[var]).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "lerobot"
    return Path("~/.cache/huggingface/lerobot").expanduser()


def resolve_root(repo_id: Optional[str], root: Optional[str]) -> Path:
    if root is not None:
        path = Path(root).expanduser()
    elif repo_id is not None:
        path = default_lerobot_home() / repo_id
    else:
        raise ValueError("Pass either --repo-id or --root.")
    if not (path / "meta" / "info.json").exists():
        raise FileNotFoundError(f"{path} does not look like a LeRobot dataset (no meta/info.json).")
    return path


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_v3(info: dict) -> bool:
    """True for the v3.0 layout (episodes packed into shared parquet/MP4 files).

    Keyed off ``data_path`` rather than ``codebase_version`` because the path template is what
    actually changes how the files are read: v2.x formats with ``episode_chunk``/``episode_index``,
    v3.0 with ``chunk_index``/``file_index``.
    """
    return "{chunk_index" in info.get("data_path", "") or str(info.get("codebase_version", "")).startswith("v3")


def _read_meta_parquet(paths: List[Path], columns: Optional[List[str]] = None) -> List[dict]:
    """Read and concatenate v3 metadata parquet shards as a list of row dicts."""
    rows: List[dict] = []
    for path in sorted(paths):
        table = pq.read_table(path)
        if columns is not None:
            keep = [c for c in table.column_names if c in columns]
            table = table.select(keep)
        rows.extend(table.to_pylist())
    return rows


def load_meta_v3(root: Path, info: dict) -> Tuple[Dict[int, dict], Dict[int, str]]:
    """v3.0 metadata: meta/episodes/**/*.parquet + meta/tasks.parquet."""
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"{root}: no meta/episodes/**/*.parquet (expected for a v3.0 dataset)")

    # The episode shards also carry per-episode `stats/...` columns (hundreds of them, several
    # nesting levels deep). Only the locators are needed, so drop the rest before materializing.
    wanted_prefixes = ("episode_index", "length", "tasks", "data/", "dataset_from_index", "dataset_to_index", "videos/")
    columns = [
        c
        for c in pq.read_schema(episode_files[0]).names
        if c.startswith(wanted_prefixes) and not c.startswith("stats/")
    ]
    episodes = {int(row["episode_index"]): row for row in _read_meta_parquet(episode_files, columns)}

    tasks: Dict[int, str] = {}
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        table = pq.read_table(tasks_path)
        names = table.column_names
        task_strings = table.column("task").to_pylist() if "task" in names else []
        if "task_index" in names:
            indices = [int(i) for i in table.column("task_index").to_pylist()]
        else:
            # Some writers store `task` as the pandas index and drop the explicit column.
            indices = list(range(len(task_strings)))
        tasks = dict(zip(indices, task_strings))
    return episodes, tasks


def load_meta(root: Path) -> Tuple[dict, Dict[int, dict], Dict[int, str]]:
    """Return (info, episode_index -> episode record, task_index -> task string)."""
    with open(root / "meta" / "info.json", "r") as f:
        info = json.load(f)
    if is_v3(info):
        episodes, tasks = load_meta_v3(root, info)
        return info, episodes, tasks
    episodes = {int(e["episode_index"]): e for e in read_jsonl(root / "meta" / "episodes.jsonl")}
    tasks = {int(t["task_index"]): t["task"] for t in read_jsonl(root / "meta" / "tasks.jsonl")}
    return info, episodes, tasks


def image_feature_keys(info: dict) -> List[str]:
    return [k for k in info["features"] if k.startswith(IMAGE_FEATURE_PREFIX)]


def parse_camera_map(raw: Optional[str], info: dict) -> Dict[str, str]:
    """Map LeRobot image feature name -> HDF5 obs key.

    Without --camera-map, fall back to DEFAULT_CAMERA_MAP for known YAM cameras and derive a name
    for anything else, appending `_image` when needed so the buffer's modality heuristic fires.
    """
    overrides: Dict[str, str] = {}
    if raw:
        for pair in raw.split(","):
            if not pair.strip():
                continue
            if "=" not in pair:
                raise ValueError(f"--camera-map entries must be src=dst, got {pair!r}")
            src, dst = pair.split("=", 1)
            overrides[src.strip()] = dst.strip()

    mapping: Dict[str, str] = {}
    for feature in image_feature_keys(info):
        cam = feature[len(IMAGE_FEATURE_PREFIX) :]
        if cam in overrides:
            name = overrides[cam]
        elif feature in overrides:
            name = overrides[feature]
        elif cam in DEFAULT_CAMERA_MAP:
            name = DEFAULT_CAMERA_MAP[cam]
        else:
            name = cam
        if not any(kw in name.lower() for kw in _IMAGE_KEY_KEYWORDS):
            name = f"{name}_image"
        mapping[feature] = name

    duplicates = [n for n in set(mapping.values()) if list(mapping.values()).count(n) > 1]
    if duplicates:
        raise ValueError(f"--camera-map produced duplicate HDF5 obs keys: {duplicates}")
    return mapping


# --------------------------------------------------------------------------------------
# Low-dim (parquet) reading
# --------------------------------------------------------------------------------------
def _column_to_2d(table, name: str) -> np.ndarray:
    """Read a parquet column into [T, D] float32 (LeRobot stores these as fixed-size lists)."""
    values = table.column(name).to_pylist()
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def episode_parquet_path(root: Path, info: dict, episode_index: int, record: Optional[dict] = None) -> Path:
    if is_v3(info):
        if record is None:
            raise ValueError(f"episode {episode_index}: v3.0 datasets need the meta/episodes record to locate data")
        rel = info["data_path"].format(
            chunk_index=int(record["data/chunk_index"]), file_index=int(record["data/file_index"])
        )
        return root / rel
    chunk = episode_index // int(info.get("chunks_size", 1000))
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    return root / rel


# v3.0 packs many episodes into one parquet, so the same file is re-read for every episode in it.
_DATA_TABLE_CACHE: Dict[Path, object] = {}


def _read_data_table(path: Path):
    if path not in _DATA_TABLE_CACHE:
        # Only one file is kept: episodes are visited in file order, so a single slot is enough and
        # a full multi-file dataset never accumulates in RAM.
        _DATA_TABLE_CACHE.clear()
        _DATA_TABLE_CACHE[path] = pq.read_table(path)
    return _DATA_TABLE_CACHE[path]


def load_episode_lowdim(
    root: Path, info: dict, episode_index: int, record: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (state [T,S], action [T,A], task_index) for one episode."""
    path = episode_parquet_path(root, info, episode_index, record)
    if is_v3(info):
        import pyarrow.compute as pc

        full = _read_data_table(path)
        table = full.filter(pc.equal(full.column("episode_index"), episode_index))
        if table.num_rows == 0:
            raise ValueError(f"episode {episode_index}: no rows with episode_index=={episode_index} in {path}")
    else:
        table = pq.read_table(path)
    state = _column_to_2d(table, STATE_FEATURE)
    action = _column_to_2d(table, ACTION_FEATURE)
    if len(state) != len(action):
        raise ValueError(f"episode {episode_index}: {len(state)} states vs {len(action)} actions")
    task_index = 0
    if "task_index" in table.column_names:
        task_index = int(table.column("task_index").to_pylist()[0])
    return state, action, task_index


# --------------------------------------------------------------------------------------
# Camera frame readers
# --------------------------------------------------------------------------------------
def _resize(frame: np.ndarray, image_size: Optional[int]) -> np.ndarray:
    """Shared with inference (utils.policy_serving.resize_image) so the robot's frames are
    preprocessed exactly the way the training frames were. INTER_AREA is the right filter for the
    large downscale (720x1280 -> 224x224) these datasets need."""
    return resize_image(frame, image_size)


def iter_video_frames(path: Path, image_size: Optional[int]):
    """Yield RGB uint8 frames from an MP4, decoding sequentially (no seeking) and resizing eagerly."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield _resize(frame.to_ndarray(format="rgb24"), image_size)


def iter_video_frames_range(path: Path, image_size: Optional[int], from_timestamp: float, to_timestamp: float, fps: float):
    """Yield one episode's frames out of a v3.0 MP4 that concatenates several episodes.

    The episode occupies ``[from_timestamp, to_timestamp)`` of the file. We seek to the keyframe at
    or before the start (``backward=True``, the PyAV default) and then drop the decoded frames that
    precede it, which is the only correct way to land on a non-keyframe boundary. Presentation
    timestamps are compared with a half-frame tolerance so float rounding in the metadata cannot
    shift the episode by one frame.
    """
    import av

    half_frame = 0.5 / float(fps) if fps else 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if from_timestamp > 0:
            container.seek(int(from_timestamp / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if timestamp < from_timestamp - half_frame:
                continue
            if timestamp >= to_timestamp - half_frame:
                return
            yield _resize(frame.to_ndarray(format="rgb24"), image_size)


def iter_image_files(root: Path, info: dict, feature: str, episode_index: int, image_size: Optional[int]):
    """Yield RGB uint8 frames for LeRobot datasets that store cameras as per-frame image files."""
    template = info.get("image_path", "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png")
    chunk = episode_index // int(info.get("chunks_size", 1000))
    frame_index = 0
    while True:
        rel = template.format(
            image_key=feature,
            episode_chunk=chunk,
            episode_index=episode_index,
            frame_index=frame_index,
        )
        path = root / rel
        if not path.exists():
            return
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read frame {path}")
        yield _resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), image_size)
        frame_index += 1


def open_camera_stream(
    root: Path,
    info: dict,
    feature: str,
    episode_index: int,
    image_size: Optional[int],
    record: Optional[dict] = None,
):
    dtype = info["features"][feature].get("dtype")
    if dtype == "video":
        if is_v3(info):
            if record is None:
                raise ValueError(f"episode {episode_index}: v3.0 datasets need the meta/episodes record to locate videos")
            rel = info["video_path"].format(
                video_key=feature,
                chunk_index=int(record[f"videos/{feature}/chunk_index"]),
                file_index=int(record[f"videos/{feature}/file_index"]),
            )
            path = root / rel
            if not path.exists():
                raise FileNotFoundError(f"Missing video for {feature} episode {episode_index}: {path}")
            return iter_video_frames_range(
                path,
                image_size,
                float(record[f"videos/{feature}/from_timestamp"]),
                float(record[f"videos/{feature}/to_timestamp"]),
                float(info.get("fps", 30)),
            )
        chunk = episode_index // int(info.get("chunks_size", 1000))
        rel = info["video_path"].format(episode_chunk=chunk, video_key=feature, episode_index=episode_index)
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing video for {feature} episode {episode_index}: {path}")
        return iter_video_frames(path, image_size)
    if dtype == "image":
        if is_v3(info):
            raise ValueError(
                f"Feature {feature} has dtype 'image' in a v3.0 dataset; only 'video' cameras are supported here."
            )
        return iter_image_files(root, info, feature, episode_index, image_size)
    raise ValueError(f"Feature {feature} has unsupported dtype {dtype!r} (expected 'video' or 'image').")


# --------------------------------------------------------------------------------------
# Action bounds
# --------------------------------------------------------------------------------------
def compute_action_bounds(
    actions: List[np.ndarray], eps: float, min_span_frac: float
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Per-dim min/max over all episodes, with a floor on each dim's span.

    Two distinct problems, one fix:

    * **Divide by zero.** The buffer normalizes as ``(a - min) / (max - min)``, so a joint that
      never moves in the dataset produces inf/NaN for the whole batch. A bimanual robot doing a
      single-arm task has several such dims (this dataset's left arm is completely idle, and
      ``left_joint_0`` has a span of exactly 0.0).
    * **Noise amplification.** A dim whose span is only encoder jitter (1e-3 rad) gets stretched
      across the full [-1, 1] policy output range, so the policy spends capacity regressing noise.

    ``min_span_frac`` sets the floor relative to the largest per-dim span, which keeps it
    meaningful across robots and action scales: at the 0.02 default a dim must span at least 2% of
    the most-active dim's range, so near-static dims land in a narrow band around 0 instead.
    """
    stacked = np.concatenate(actions, axis=0)
    min_a = stacked.min(axis=0).astype(np.float32)
    max_a = stacked.max(axis=0).astype(np.float32)
    span = max_a - min_a
    floor = max(float(eps), float(min_span_frac) * float(span.max()))
    degenerate = np.flatnonzero(span < floor)
    if degenerate.size:
        center = 0.5 * (min_a + max_a)
        min_a[degenerate] = center[degenerate] - floor / 2.0
        max_a[degenerate] = center[degenerate] + floor / 2.0
    return min_a, max_a, degenerate.tolist()


# --------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------
def convert(args: argparse.Namespace) -> dict:
    root = resolve_root(args.repo_id, args.root)
    info, episode_meta, tasks = load_meta(root)
    camera_map = parse_camera_map(args.camera_map, info)

    if args.episodes:
        episode_indices = [int(e) for e in args.episodes.split(",") if e.strip()]
    else:
        episode_indices = sorted(episode_meta) or list(range(int(info.get("total_episodes", 0))))
    if not episode_indices:
        raise RuntimeError(f"No episodes found in {root}")

    print(f"Dataset: {root}")
    print(f"  codebase_version={info.get('codebase_version')} robot_type={info.get('robot_type')} fps={info.get('fps')}")
    print(f"  episodes={episode_indices}")
    for feature, name in camera_map.items():
        print(f"  camera {feature} -> obs/{name} ({info['features'][feature].get('dtype')})")

    # Pass 1: low-dim data for every episode, so action bounds cover the whole dataset before we
    # start writing (the bounds go into /meta and are read back as the synthesized action space).
    lowdim = {}
    for episode_index in tqdm(episode_indices, desc="Reading low-dim", unit="ep"):
        state, action, task_index = load_episode_lowdim(root, info, episode_index, episode_meta.get(episode_index))
        expected = episode_meta.get(episode_index, {}).get("length")
        if expected is not None and int(expected) != len(action):
            raise ValueError(
                f"episode {episode_index}: meta/episodes.jsonl says length={expected} "
                f"but the parquet has {len(action)} rows"
            )
        if args.action_mode == "delta":
            # LeRobot YAM actions are absolute joint targets one control tick ahead -- in this
            # dataset `action[t] == observation.state[t+1]` exactly. A single-step policy
            # conditioned on `state` can therefore hit a near-zero loss by learning the identity
            # map. Storing `action - state` removes that shortcut; the deployment side must add
            # `state` back (action_mode is recorded in /meta.attrs["info"]).
            action = (action - state).astype(np.float32)
        lowdim[episode_index] = (state, action, task_index)

    min_action, max_action, widened = compute_action_bounds(
        [a for _, a, _ in lowdim.values()], args.action_range_eps, args.action_min_span_frac
    )
    if widened:
        print(f"  floored the span of {len(widened)} near-constant action dim(s): {widened}")

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression

    total = 0
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        for demo_idx, episode_index in enumerate(episode_indices):
            state, action, task_index = lowdim[episode_index]
            num_samples = len(action)
            prompt = tasks.get(task_index, "")
            if not prompt:
                ep_tasks = episode_meta.get(episode_index, {}).get("tasks") or []
                prompt = ep_tasks[0] if ep_tasks else ""

            demo = data_grp.create_group(f"demo_{demo_idx}")
            demo.create_dataset("actions", data=action.astype(np.float32))
            rewards = np.zeros(num_samples, dtype=np.float32)
            if args.terminal_reward != 0.0:
                rewards[-1] = float(args.terminal_reward)
            demo.create_dataset("rewards", data=rewards)
            dones = np.zeros(num_samples, dtype=np.int64)
            dones[-1] = 1
            demo.create_dataset("dones", data=dones)
            demo.create_dataset("language_instruction", data=np.bytes_(prompt))
            demo.attrs["num_samples"] = num_samples
            demo.attrs["prompt"] = prompt
            demo.attrs["lerobot_episode_index"] = episode_index

            obs_grp = demo.create_group("obs")
            next_obs_grp = demo.create_group("next_obs") if args.write_next_obs else None

            obs_grp.create_dataset("state", data=state.astype(np.float32))
            if next_obs_grp is not None:
                next_state = np.concatenate([state[1:], state[-1:]], axis=0).astype(np.float32)
                next_obs_grp.create_dataset("state", data=next_state)

            for feature, obs_key in camera_map.items():
                _write_camera(
                    root=root,
                    info=info,
                    feature=feature,
                    obs_key=obs_key,
                    episode_index=episode_index,
                    record=episode_meta.get(episode_index),
                    num_samples=num_samples,
                    image_size=args.image_size,
                    obs_grp=obs_grp,
                    next_obs_grp=next_obs_grp,
                    compression=compression,
                )

            total += num_samples
            print(f"  demo_{demo_idx} (episode {episode_index}): {num_samples} steps, prompt={prompt!r}")

        data_grp.attrs["total"] = total
        # Robomimic env reconstruction is not applicable to real-robot data; the offline-only
        # training path synthesizes gym spaces from this file instead.
        data_grp.attrs["env_args"] = ""

        meta_grp = f.create_group("meta")
        meta_grp.create_dataset("min_action", data=min_action)
        meta_grp.create_dataset("max_action", data=max_action)
        meta_grp.attrs["info"] = json.dumps(
            {
                "source": "lerobot",
                "repo_id": args.repo_id,
                "root": str(root),
                "codebase_version": info.get("codebase_version"),
                "robot_type": info.get("robot_type"),
                "fps": info.get("fps"),
                "image_size": args.image_size,
                "camera_map": camera_map,
                "episodes": episode_indices,
                "action_mode": args.action_mode,
                "terminal_reward": args.terminal_reward,
                "action_range_eps": args.action_range_eps,
                "action_min_span_frac": args.action_min_span_frac,
                "floored_action_dims": widened,
                "has_next_obs": bool(args.write_next_obs),
            }
        )

    size_mb = output_path.stat().st_size / 1e6
    print(f"Wrote {output_path} ({size_mb:.1f} MB): {len(episode_indices)} demos, {total} transitions")
    return {"output": str(output_path), "num_demos": len(episode_indices), "total": total, "size_mb": size_mb}


def _write_camera(
    *,
    root: Path,
    info: dict,
    feature: str,
    obs_key: str,
    episode_index: int,
    num_samples: int,
    image_size: Optional[int],
    obs_grp: h5py.Group,
    next_obs_grp: Optional[h5py.Group],
    compression: Optional[str],
    record: Optional[dict] = None,
):
    """Stream one camera's frames straight into the HDF5, one frame at a time.

    Never materializes a full episode of pixels: a 720p episode of 900 frames is ~2.5 GB.
    """
    frames = open_camera_stream(root, info, feature, episode_index, image_size, record)
    obs_ds = None
    next_ds = None
    previous: Optional[np.ndarray] = None
    written = 0

    for t, frame in enumerate(
        tqdm(frames, total=num_samples, desc=f"  ep{episode_index} {obs_key}", leave=False, unit="frame")
    ):
        if t >= num_samples:
            # The YAM converter derives `action[t] = joints[t+1]`, so the encoded video can be one
            # frame longer than the parquet. Extra trailing frames have no matching transition.
            break
        if obs_ds is None:
            h, w = frame.shape[:2]
            # Contiguous when uncompressed (what robomimic image datasets use, and the fastest
            # layout for the buffer's one-frame-at-a-time reads). HDF5 requires chunking for any
            # filter, so compressed datasets get one chunk per frame.
            kwargs = {} if compression is None else {"chunks": (1, h, w, 3), "compression": compression}
            obs_ds = obs_grp.create_dataset(obs_key, shape=(num_samples, h, w, 3), dtype=np.uint8, **kwargs)
            if next_obs_grp is not None:
                next_ds = next_obs_grp.create_dataset(
                    obs_key, shape=(num_samples, h, w, 3), dtype=np.uint8, **kwargs
                )
        obs_ds[t] = frame
        if next_ds is not None and t > 0:
            next_ds[t - 1] = frame
        previous = frame
        written = t + 1

    if written != num_samples:
        raise ValueError(
            f"episode {episode_index} camera {feature}: decoded {written} frames but the episode has "
            f"{num_samples} transitions"
        )
    if next_ds is not None and previous is not None:
        # next_obs of the terminal step aliases obs, matching what the buffer does for is_last.
        next_ds[num_samples - 1] = previous


# --------------------------------------------------------------------------------------
# LeRobotDataset-backed path (optional)
# --------------------------------------------------------------------------------------
def convert_via_lerobot_api(args: argparse.Namespace) -> dict:
    """Same output, but sourcing frames through LeRobotDataset's own decoding."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.root, local_files_only=True)
    info = dataset.meta.info
    camera_map = parse_camera_map(args.camera_map, info)
    episode_indices = (
        [int(e) for e in args.episodes.split(",") if e.strip()] if args.episodes else list(range(dataset.num_episodes))
    )

    episodes: Dict[int, Dict[str, list]] = {i: {"state": [], "action": [], "prompt": "", **{k: [] for k in camera_map.values()}} for i in episode_indices}
    for i in tqdm(range(len(dataset)), desc="Reading LeRobotDataset", unit="frame"):
        item = dataset[i]
        episode_index = int(item["episode_index"])
        if episode_index not in episodes:
            continue
        bucket = episodes[episode_index]
        bucket["state"].append(np.asarray(item[STATE_FEATURE], dtype=np.float32))
        bucket["action"].append(np.asarray(item[ACTION_FEATURE], dtype=np.float32))
        bucket["prompt"] = item.get("task", bucket["prompt"])
        for feature, obs_key in camera_map.items():
            # LeRobotDataset returns CHW float tensors in [0, 1].
            frame = np.asarray(item[feature])
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
            bucket[obs_key].append(_resize(frame, args.image_size))

    actions = [np.stack(episodes[e]["action"]) for e in episode_indices]
    if args.action_mode == "delta":
        actions = [a - np.stack(episodes[e]["state"]) for a, e in zip(actions, episode_indices)]
    min_action, max_action, widened = compute_action_bounds(
        actions, args.action_range_eps, args.action_min_span_frac
    )
    if widened:
        print(f"  floored the span of {len(widened)} near-constant action dim(s): {widened}")

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression

    total = 0
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        for demo_idx, episode_index in enumerate(episode_indices):
            bucket = episodes[episode_index]
            action = np.stack(bucket["action"]).astype(np.float32)
            if args.action_mode == "delta":
                action = (action - np.stack(bucket["state"]).astype(np.float32)).astype(np.float32)
            num_samples = len(action)
            demo = data_grp.create_group(f"demo_{demo_idx}")
            demo.create_dataset("actions", data=action)
            rewards = np.zeros(num_samples, dtype=np.float32)
            if args.terminal_reward != 0.0:
                rewards[-1] = float(args.terminal_reward)
            demo.create_dataset("rewards", data=rewards)
            dones = np.zeros(num_samples, dtype=np.int64)
            dones[-1] = 1
            demo.create_dataset("dones", data=dones)
            demo.create_dataset("language_instruction", data=np.bytes_(bucket["prompt"]))
            demo.attrs["num_samples"] = num_samples
            demo.attrs["prompt"] = bucket["prompt"]
            demo.attrs["lerobot_episode_index"] = episode_index

            obs_grp = demo.create_group("obs")
            obs_grp.create_dataset("state", data=np.stack(bucket["state"]).astype(np.float32))
            for obs_key in camera_map.values():
                seq = np.stack(bucket[obs_key])
                kwargs = {} if compression is None else {"chunks": (1,) + seq.shape[1:], "compression": compression}
                obs_grp.create_dataset(obs_key, data=seq, **kwargs)

            if args.write_next_obs:
                next_grp = demo.create_group("next_obs")
                states = np.stack(bucket["state"]).astype(np.float32)
                next_grp.create_dataset("state", data=np.concatenate([states[1:], states[-1:]], axis=0))
                for obs_key in camera_map.values():
                    seq = np.stack(bucket[obs_key])
                    kwargs = {} if compression is None else {"chunks": (1,) + seq.shape[1:], "compression": compression}
                    next_grp.create_dataset(obs_key, data=np.concatenate([seq[1:], seq[-1:]], axis=0), **kwargs)

            total += num_samples
            print(f"  demo_{demo_idx} (episode {episode_index}): {num_samples} steps")

        data_grp.attrs["total"] = total
        data_grp.attrs["env_args"] = ""
        meta_grp = f.create_group("meta")
        meta_grp.create_dataset("min_action", data=min_action)
        meta_grp.create_dataset("max_action", data=max_action)
        meta_grp.attrs["info"] = json.dumps(
            {
                "source": "lerobot (LeRobotDataset api)",
                "repo_id": args.repo_id,
                "image_size": args.image_size,
                "camera_map": camera_map,
                "episodes": episode_indices,
                "action_mode": args.action_mode,
                "floored_action_dims": widened,
                "has_next_obs": bool(args.write_next_obs),
            }
        )

    size_mb = output_path.stat().st_size / 1e6
    print(f"Wrote {output_path} ({size_mb:.1f} MB): {len(episode_indices)} demos, {total} transitions")
    return {"output": str(output_path), "num_demos": len(episode_indices), "total": total, "size_mb": size_mb}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=None, help="LeRobot repo id, resolved under HF_LEROBOT_HOME")
    p.add_argument("--root", default=None, help="Explicit dataset root (overrides --repo-id resolution)")
    p.add_argument("--output", required=True, help="Output .hdf5 path")
    p.add_argument("--image-size", type=int, default=224, help="Square resize for camera frames (0 keeps native)")
    p.add_argument(
        "--camera-map",
        default=None,
        help="Comma-separated src=dst renaming, e.g. scene_camera=image,left_wrist_camera=left_wrist_image",
    )
    p.add_argument("--episodes", default=None, help="Comma-separated episode indices (default: all)")
    p.add_argument(
        "--action-mode",
        default="absolute",
        choices=["absolute", "delta"],
        help="absolute keeps the dataset's joint targets (deployment-ready); delta stores "
        "action - state, which removes the identity-map shortcut for state-conditioned policies",
    )
    p.add_argument("--terminal-reward", type=float, default=1.0, help="Reward on the final step (0 elsewhere)")
    p.add_argument("--write-next-obs", action="store_true", help="Also write a next_obs group (doubles image bytes)")
    p.add_argument("--action-range-eps", type=float, default=1e-3, help="Absolute floor on each action dim's span")
    p.add_argument(
        "--action-min-span-frac",
        type=float,
        default=0.02,
        help="Floor on each action dim's span as a fraction of the largest dim's span (0 disables)",
    )
    p.add_argument(
        "--compression",
        default="none",
        choices=["none", "lzf", "gzip"],
        help="HDF5 image compression. Keep 'none': the buffer reads one frame at a time, and "
        "per-frame gzip decompression costs ~1.1 ms/frame (~1.7 s per 256-sample batch)",
    )
    p.add_argument("--use-lerobot-api", action="store_true", help="Read via LeRobotDataset instead of parquet+mp4")
    return p


def main():
    args = build_parser().parse_args()
    if args.image_size is not None and args.image_size <= 0:
        args.image_size = None
    if args.use_lerobot_api:
        convert_via_lerobot_api(args)
    else:
        convert(args)


if __name__ == "__main__":
    main()
