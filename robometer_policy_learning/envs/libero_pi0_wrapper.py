"""
LIBERO Environment Wrapper for DSRL + Pi0

Wraps LIBERO environments to work with Pi0 and DSRL,
including RFM reward model integration.
"""

import gymnasium as gym
import gymnasium.vector as gym_vector
import numpy as np
from typing import Dict, Any, Optional, List, Union
from sentence_transformers import SentenceTransformer

import sys, os

# Fallback for a LIBERO checkout that was never pip-installed. This must stay an APPEND: LIBERO-plus
# activation shadows the installed LIBERO by inserting its own root at sys.path[0]
# (see robometer_policy_learning/envs/libero_plus.py), and an insert here would fight it.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "LIBERO"))
# NOTE: nothing from `libero` is imported at module level on purpose -- `setup_libero_env` activates
# LIBERO-plus (which shadows the installed LIBERO) lazily, and that only works if importing this
# module has not already bound `libero`.
from robometer_policy_learning.utils.pi0_integration import preprocess_obs_for_pi0
from robometer_policy_learning.utils.robometer_compat import compute_text_embeddings

# LIBERO's no-op action: zero delta pose with the gripper held open. Used to let MuJoCo settle after
# restoring a benchmark init state (the standard LIBERO evaluation protocol).
LIBERO_DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


class LiberoPI0Wrapper(gym.Wrapper):
    """
    Wrapper for LIBERO environments to work with Pi0.

    - Converts LIBERO observations to Pi0 format
    - Handles image preprocessing (resizing, flipping)
    - Manages state vector extraction
    - Optional: Integrates RFM for reward relabeling
    """

    def __init__(
        self,
        env,
        language_instruction: Optional[str] = None,
        init_states: Optional[np.ndarray] = None,
        init_state_mode: Union[int, str, None] = None,
        settle_steps: int = 10,
    ):
        """
        Initialize LIBERO wrapper.

        Args:
            env: Base LIBERO environment
            language_instruction: Task instruction used as the pi0 prompt.
            init_states: Optional ``[N, D]`` array of benchmark init states (MuJoCo flattened sim
                states) from ``Benchmark.get_task_init_states``. When given, every reset restores one
                of them instead of letting the robosuite placement sampler randomize the scene.
                Required for faithful LIBERO-plus evaluation: the Objects Layout perturbations only
                take effect through their init states.
            init_state_mode: which state to restore -- an int index, ``"cycle"`` (episode k uses
                ``k % N``, giving varied but on-distribution starts) or ``"random"``. Ignored when
                ``init_states`` is None.
            settle_steps: no-op steps taken after restoring a state so the controller and gripper
                settle (LIBERO's evaluation protocol uses 10).
        """
        super().__init__(env)

        self._init_states = None if init_states is None else np.asarray(init_states)
        self._init_state_mode = init_state_mode if init_state_mode is not None else 0
        self._settle_steps = max(0, int(settle_steps))
        self._reset_count = 0
        self.last_init_state_index: Optional[int] = None
        if self._init_states is not None and self._init_states.ndim != 2:
            raise ValueError(f"init_states must be [N, D]; got shape {self._init_states.shape}")

        # Define observation space (Pi0 format)
        self.observation_space = gym.spaces.Dict(
            {
                "observation/state": gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32),
                "observation/image": gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
                "observation/wrist_image": gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            }
        )

        # Action space remains the same
        if not hasattr(self.env, "action_space"):
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Task metadata (set during reset)
        self.task_id = None
        self.task_suite = None
        self.language_instruction = language_instruction

        print(f"✓ LIBERO Pi0 Wrapper initialized")

        self.dsrl_key_mapping = {
            "image": ["observation/image"],
            "state": "observation/state",
            "language": "language",
        }

    def _pick_init_state_index(self) -> int:
        """Resolve ``init_state_mode`` to a row of ``self._init_states`` for this episode."""
        n = len(self._init_states)
        mode = self._init_state_mode
        if isinstance(mode, str):
            if mode == "cycle":
                return self._reset_count % n
            if mode == "random":
                return int(np.random.randint(n))
            raise ValueError(f"Unknown init_state_mode {mode!r}; expected an int, 'cycle' or 'random'")
        idx = int(mode)
        if not 0 <= idx < n:
            raise IndexError(f"init_state_mode={idx} out of range for {n} available init states")
        return idx

    def _restore_init_state(self):
        """Restore one benchmark init state and let the sim settle. Returns the raw LIBERO obs."""
        idx = self._pick_init_state_index()
        self.last_init_state_index = idx
        # set_init_state / step are reached through GymToGymnasiumWrapper's attribute forwarding.
        raw_obs = self.env.set_init_state(self._init_states[idx])
        for _ in range(self._settle_steps):
            raw_obs, _, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION)
        # The settle steps must not eat into the episode's time limit.
        if hasattr(self.env, "current_step"):
            self.env.current_step = 0
        return raw_obs

    def reset(self, **kwargs):
        """Reset environment and return Pi0-format observation"""
        # Reset base environment
        raw_obs, _ = self.env.reset()

        if self._init_states is not None:
            raw_obs = self._restore_init_state()
        self._reset_count += 1

        # Some LIBERO env instances do not expose language_instruction; keep the
        # task language supplied by setup_libero_env in that case.
        self.language_instruction = getattr(self.env, "language_instruction", None) or self.language_instruction

        # Convert to Pi0 format
        obs = preprocess_obs_for_pi0(raw_obs)
        obs["prompt"] = self.language_instruction

        return obs, _

    def step(self, action: np.ndarray):
        """
        Execute action and return Pi0-format observation.

        Args:
            action: np.ndarray of shape (7,) - robot action

        Returns:
            obs: Pi0-format observation
            reward: Reward (sparse or RFM-based)
            done: Terminal flag
            truncated: Truncation flag
            info: Info dictionary
        """
        # Execute action
        raw_obs, reward, done, truncated, info = self.env.step(action)

        # Convert observation to Pi0 format
        obs = preprocess_obs_for_pi0(raw_obs)
        obs["prompt"] = self.language_instruction

        info["language_instruction"] = self.language_instruction
        info["sparse_reward"] = reward

        # In LIBERO, done is only True when task succeeds, so success = done
        # But don't overwrite if already present in info
        if "success" not in info:
            info["success"] = done
        if done:
            assert reward == 1.0, "Reward should be 1.0 when task succeeds"

        reward -= 1  # reward is -1, 0

        return obs, np.float64(reward), done, truncated, info

    def close(self):
        """Close environment"""
        self.env.close()


def _vec_language_instruction(env):
    """Read ``language_instruction`` from a vector env, working for Sync AND Async envs.

    ``SyncVectorEnv`` keeps the sub-envs in-process (``env.envs``); ``AsyncVectorEnv`` keeps them in
    subprocesses and instead exposes ``get_attr(name)`` (a per-env tuple). Returns None if neither
    path yields the attribute.
    """
    envs = getattr(env, "envs", None)
    if envs:  # SyncVectorEnv
        return getattr(envs[0], "language_instruction", None)
    get_attr = getattr(env, "get_attr", None)
    if callable(get_attr):  # AsyncVectorEnv
        try:
            vals = get_attr("language_instruction")
            if vals is not None and len(vals):
                return vals[0]
        except Exception:
            return None
    return None


class VectorLiberoPromptWrapper(gym_vector.VectorWrapper):
    """
    Adds 'prompt' (language instruction) to each observation in a vectorized LIBERO env,
    using env.language_instruction from one of the underlying environments.
    'prompt' is included in each observation dict returned by the vector env.
    """

    def __init__(self, env: gym_vector.VectorEnv, sentence_model: SentenceTransformer = None,
                 language_instruction: Optional[str] = None):
        # Always set on the wrapper itself so __getattr__ never forwards it to the base env
        self.sentence_model: Optional[SentenceTransformer] = sentence_model
        self.language_encoding: Optional[np.ndarray] = None

        super().__init__(env)
        # Extract language_instruction from the base env (assumed identical across the vector).
        # ``language_instruction`` overrides when the caller already knows it (avoids querying
        # subprocess sub-envs under AsyncVectorEnv). Falls back to Sync/Async-safe lookup.
        self.language_instruction = language_instruction if language_instruction is not None \
            else _vec_language_instruction(env)

        # Pre-compute language embedding if a model is provided
        if self.sentence_model is not None and self.language_instruction is not None:
            enc = compute_text_embeddings(self.language_instruction, self.sentence_model)
            self.language_encoding = enc.cpu().numpy().astype(np.float32)
        else:
            self.language_encoding = np.zeros((384,), dtype=np.float32)

        # Update observation space to include 'prompt'
        if hasattr(self, "single_observation_space"):
            orig_space = self.single_observation_space
        else:
            orig_space = self.observation_space

        # Defensive: only update if it's a Dict space and doesn't already have 'prompt'
        if isinstance(orig_space, gym.spaces.Dict) and "prompt" not in orig_space.spaces:
            # Guess dtype and shape: a language prompt is a string, so we use gym.spaces.Text if available, else ignore in obs space
            try:
                # gymnasium >=0.29 has spaces.Text
                prompt_space = gym.spaces.Text(min_length=1, max_length=512)
                spaces_dict = dict(orig_space.spaces)
                spaces_dict["prompt"] = prompt_space
                # Only add 'language' to the observation space if we actually have an embedding
                spaces_dict["language"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.language_encoding.shape if self.language_encoding is not None else (384,),
                    dtype=np.float32,
                )
                new_space = gym.spaces.Dict(spaces_dict)
                if hasattr(self, "single_observation_space"):
                    self.single_observation_space = new_space
                else:
                    self.observation_space = new_space
            except Exception:
                # If gym.spaces.Text is not available, just ignore in space (will still insert into dict at runtime)
                pass

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Refresh language_instruction from the sub-envs (Sync or Async). Keep the current value if
        # the lookup yields nothing (e.g. fixed robomimic instruction under AsyncVectorEnv).
        li = _vec_language_instruction(self.env)
        if li is not None:
            self.language_instruction = li
        obs = self._add_prompt(obs)
        try:
            n = self.env.num_envs
        except Exception:
            n = len(obs) if hasattr(obs, "__len__") else 1
        # (Re-)compute language encoding if we have a model; keep the last encoding otherwise
        if self.sentence_model is not None:
            enc = compute_text_embeddings(self.language_instruction, self.sentence_model)
            self.language_encoding = enc.cpu().numpy().astype(np.float32)

        # Add language encoding to observation only if it exists
        if self.language_encoding is not None:
            obs = self._add_language_to_obs(obs, n)
        # Attach to info too, if desired
        if isinstance(info, dict):
            info = {k: dict(v, prompt=self.language_instruction) if isinstance(v, dict) else v for k, v in info.items()}
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = self._add_prompt(obs)
        try:
            n = self.env.num_envs
        except Exception:
            n = len(obs) if hasattr(obs, "__len__") else 1
        # Add language to observation only if we actually have an encoding
        if self.language_encoding is not None:
            obs = self._add_language_to_obs(obs, n)
        # Attach to info for each env
        if isinstance(info, dict):
            info = {k: dict(v, prompt=self.language_instruction) if isinstance(v, dict) else v for k, v in info.items()}
        elif isinstance(info, list):
            info = [dict(i, prompt=self.language_instruction) if isinstance(i, dict) else i for i in info]
        return obs, reward, terminated, truncated, info

    def _add_prompt(self, obs: Dict[str, Any]):
        obs["prompt"] = [self.language_instruction] * self.env.num_envs
        return obs

    def __getattr__(self, name):
        """Forward unknown attributes/methods to wrapped environment."""
        return getattr(self.env, name)

    def get_language_instruction(self) -> str:
        return self.language_instruction

    def _add_language_to_obs(self, obs, n):
        # Adds language instruction under key 'language' for each env
        # language_encoding is already a numpy array from __init__
        if isinstance(obs, dict):
            obs = dict(obs)
            # Create numpy array with shape (n, embedding_dim) instead of list
            obs["language"] = np.tile(self.language_encoding, (n, 1))  # (n, 384)
            return obs
        elif isinstance(obs, (list, tuple)):
            # for list/tuple of dicts
            for i in range(n):
                if isinstance(obs[i], dict):
                    obs[i] = dict(obs[i])
                    obs[i]["language"] = self.language_encoding  # (384,)
            return obs
        else:
            return obs  # in case of unknown obs format
