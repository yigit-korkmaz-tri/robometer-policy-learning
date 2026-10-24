import random
from typing import Any, Dict, List, Optional

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer


class RoundReplayBuffer(ReplayBuffer):
    """
    ReplayBuffer that remembers which collection round each transition was added in, and can sample a
    fixed fraction of every batch from the most recent round.

    Iterative HITL training keeps aggregating rollouts, so with uniform sampling the newest round's
    data shrinks to ``1 / num_rounds`` of each batch. With ``recent_ratio`` set, ``recent_ratio`` of
    each batch comes from the current round (``start_round``) and the rest from all earlier
    rounds; ``recent_ratio=0.5`` weights the newest round equally against everything before it.
    ``None`` (or only one round's data in the buffer) => plain uniform sampling, as ReplayBuffer.

    Works with both RandomSampler (``sample_indices``) and ChunkedSequentialSampler
    (``get_contiguous_chunks``). Chunks never straddle rounds since an episode is added in one round.
    With ``epoch_sampling`` the current and earlier rounds are each cycled in their own epoch passes.
    """

    def __init__(self, *args, recent_ratio: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if recent_ratio is not None and not 0.0 <= recent_ratio <= 1.0:
            raise ValueError(f"recent_ratio must be in [0, 1] or None, got {recent_ratio}")
        self.recent_ratio = recent_ratio
        self.current_round = 0
        self._pos_round: List[Optional[int]] = [None] * self.capacity  # round of each buffer position
        self._episode_round: Dict[Any, int] = {}  # round an episode was (first) added in
        self._position_cache = None  # (key, recent_positions, past_positions)

    def start_round(self, round_idx: int):
        """Mark that transitions added from now on belong to round ``round_idx`` (the "recent" round)."""
        self.current_round = int(round_idx)
        self._position_cache = None

    def _add(self, obs, action, reward, next_obs, done, truncated, episode_id=None, step_in_episode=None, **kwargs):
        pos = self._write_pos
        super()._add(
            obs, action, reward, next_obs, done, truncated,
            episode_id=episode_id, step_in_episode=step_in_episode, **kwargs,
        )
        self._pos_round[pos] = self.current_round
        if episode_id is not None:
            self._episode_round.setdefault(episode_id, self.current_round)
        self._position_cache = None

    def clear(self):
        super().clear()
        self._pos_round = [None] * self.capacity
        self._episode_round.clear()
        self._position_cache = None

    def _epoch_key(self):
        return (*super()._epoch_key(), self.current_round)

    def recent_size(self) -> int:
        """Number of stored transitions from the current round."""
        return len(self._positions()[0])

    def _positions(self):
        """Buffer positions split into (current round, earlier rounds); cached until the next add."""
        key = (self._size, self._write_pos, self.current_round)
        if self._position_cache is None or self._position_cache[0] != key:
            recent, past = [], []
            for i in range(self._size):
                (recent if self._pos_round[i] == self.current_round else past).append(i)
            self._position_cache = (key, recent, past)
        return self._position_cache[1], self._position_cache[2]

    def _num_recent(self, batch_size: int, has_recent: bool, has_past: bool) -> Optional[int]:
        """How many of ``batch_size`` to draw from the current round; None => sample uniformly."""
        if self.recent_ratio is None or not (has_recent and has_past):
            return None
        return round(batch_size * self.recent_ratio)

    def _draw(self, pool: str, population: List[int], k: int) -> List[int]:
        if k <= 0:
            return []
        if self.epoch_sampling:
            return self._epoch_draw(pool, lambda: population, k)
        return random.sample(population, k) if len(population) >= k else random.choices(population, k=k)

    def sample_indices(self, batch_size: int, sampler=None) -> List[int]:
        recent, past = self._positions()
        n_recent = self._num_recent(batch_size, bool(recent), bool(past))
        if n_recent is None:
            return super().sample_indices(batch_size, sampler=sampler)
        return self._draw("recent", recent, n_recent) + self._draw("past", past, batch_size - n_recent)

    def get_contiguous_chunks(self, chunk_size: int, max_chunks: int):
        if self._size == 0 or chunk_size <= 0 or max_chunks <= 0:
            return []
        recent, past = [], []
        for ep_id, steps in self._episode_steps.items():
            if len(steps) >= chunk_size:  # only episodes that can yield a chunk decide the split
                (recent if self._episode_round.get(ep_id) == self.current_round else past).append((ep_id, steps))
        n_recent = self._num_recent(max_chunks, bool(recent), bool(past))
        if n_recent is None:
            return self._chunks_from_episodes(recent + past, chunk_size, max_chunks)
        return self._chunks_from_episodes(recent, chunk_size, n_recent, pool="recent") + self._chunks_from_episodes(
            past, chunk_size, max_chunks - n_recent, pool="past"
        )
