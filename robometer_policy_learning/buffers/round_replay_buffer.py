import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer

# Online HITL intervention labels: 0 = autonomous policy, 1 = human correction (2 = offline demo).
POLICY_LABEL = 0
HUMAN_LABEL = 1


def _label(transition) -> int:
    label = (transition.info or {}).get("intervention", POLICY_LABEL) if transition is not None else POLICY_LABEL
    return round(float(np.asarray(label).reshape(-1)[0]))


def _is_human(transition) -> bool:
    return _label(transition) == HUMAN_LABEL


class RoundReplayBuffer(ReplayBuffer):
    """
    ReplayBuffer that remembers which collection round each transition was added in, and can sample a
    fixed fraction of every batch from the most recent round and/or from human-correction samples.

    Iterative HITL training keeps aggregating rollouts, so with uniform sampling the newest round's
    data shrinks to ``1 / num_rounds`` of each batch. With ``recent_ratio`` set, ``recent_ratio`` of
    each batch comes from the current round (``start_round``) and the rest from all earlier
    rounds; ``recent_ratio=0.5`` weights the newest round equally against everything before it.
    ``None`` (or only one round's data in the buffer) => plain uniform sampling, as ReplayBuffer.

    ``human_ratio`` (FlowDAgger's dual buffer) fixes the fraction of each batch drawn from
    human-correction samples (``info['intervention'] == 1``), the rest from autonomous-policy ones;
    ``recent_ratio`` then applies WITHIN each label. A chunk's label is its first step's, matching
    ChunkedSequentialSampler, which labels a chunk by its conditioning step. ``None`` (or either label
    missing) => label-agnostic sampling, exactly as without it.

    Works with both RandomSampler (``sample_indices``) and ChunkedSequentialSampler
    (``get_contiguous_chunks``). Chunks never straddle rounds since an episode is added in one round.
    With ``epoch_sampling`` each pool (round x label) is cycled in its own epoch passes.

    ``pure_policy_chunks`` restricts which chunk spans are valid: a span starting on a policy (label-0)
    step must be policy throughout (``[R,R,R,R]``), while a span starting on a human/offline step may
    run into whatever follows (``[H,H,R,R]``). Unlike segmenting episodes by label, this keeps human
    takeovers shorter than ``chunk_size`` (as long as H steps remain before the episode ends) while
    never letting a policy-labelled chunk hide an upcoming takeover. Affects chunked sampling only.
    """

    def __init__(
        self,
        *args,
        recent_ratio: Optional[float] = None,
        human_ratio: Optional[float] = None,
        pure_policy_chunks: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if recent_ratio is not None and not 0.0 <= recent_ratio <= 1.0:
            raise ValueError(f"recent_ratio must be in [0, 1] or None, got {recent_ratio}")
        if human_ratio is not None and not 0.0 <= human_ratio <= 1.0:
            raise ValueError(f"human_ratio must be in [0, 1] or None, got {human_ratio}")
        self.recent_ratio = recent_ratio
        self.human_ratio = human_ratio
        self.pure_policy_chunks = bool(pure_policy_chunks)
        self.current_round = 0
        self._pos_round: List[Optional[int]] = [None] * self.capacity  # round of each buffer position
        self._episode_round: Dict[Any, int] = {}  # round an episode was (first) added in
        self._position_cache = None  # (key, recent_positions, past_positions)
        self._label_cache = None  # (key, {label: (recent, past)}) for transitions
        self._span_cache = None  # (key, {label: (recent, past)}) for chunk spans
        self._filtered_span_cache: Dict[Any, List[tuple]] = {}  # (key, chunk_size, pool) -> allowed spans

    def start_round(self, round_idx: int):
        """Mark that transitions added from now on belong to round ``round_idx`` (the "recent" round)."""
        self.current_round = int(round_idx)
        self._invalidate()

    def _invalidate(self):
        self._position_cache = self._label_cache = self._span_cache = None
        self._filtered_span_cache = {}

    def _add(self, obs, action, reward, next_obs, done, truncated, episode_id=None, step_in_episode=None, **kwargs):
        pos = self._write_pos
        super()._add(
            obs, action, reward, next_obs, done, truncated,
            episode_id=episode_id, step_in_episode=step_in_episode, **kwargs,
        )
        self._pos_round[pos] = self.current_round
        if episode_id is not None:
            self._episode_round.setdefault(episode_id, self.current_round)
        self._invalidate()

    def clear(self):
        super().clear()
        self._pos_round = [None] * self.capacity
        self._episode_round.clear()
        self._invalidate()

    def _epoch_key(self):
        return (*super()._epoch_key(), self.current_round)

    def _cache_key(self):
        return (self._size, self._write_pos, self.current_round)

    def recent_size(self) -> int:
        """Number of stored transitions from the current round."""
        return len(self._positions()[0])

    def _positions(self):
        """Buffer positions split into (current round, earlier rounds); cached until the next add."""
        key = self._cache_key()
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

    def _draw(self, pool: str, population: List[Any], k: int) -> List[Any]:
        if k <= 0 or not population:
            return []
        if self.epoch_sampling:
            return self._epoch_draw(pool, lambda: population, k)
        return random.sample(population, k) if len(population) >= k else random.choices(population, k=k)

    # ---- human_ratio: stratify by intervention label, then by round within each label ----
    def _draw_stratified(
        self, groups: Dict[str, Tuple[List[Any], List[Any]]], k: int, pool: str
    ) -> Optional[List[Any]]:
        """Draw ``k`` items from ``{'human': (recent, past), 'policy': (recent, past)}`` pools.

        ``pool`` prefixes the epoch-pass names so transition and chunk draws never share a pass. Returns
        None (=> caller falls back to label-agnostic sampling) when ``human_ratio`` is unset or either label
        has nothing to draw from.
        """
        if self.human_ratio is None or not all(r or p for r, p in groups.values()):
            return None
        k_human = round(k * self.human_ratio)
        out: List[Any] = []
        for label, k_label in (("human", k_human), ("policy", k - k_human)):
            recent, past = groups[label]
            n_recent = self._num_recent(k_label, bool(recent), bool(past))
            if n_recent is None:
                out += self._draw(f"{pool}/{label}/all", recent + past, k_label)
            else:
                out += self._draw(f"{pool}/{label}/recent", recent, n_recent)
                out += self._draw(f"{pool}/{label}/past", past, k_label - n_recent)
        return out

    def _label_positions(self) -> Dict[str, Tuple[List[int], List[int]]]:
        key = self._cache_key()
        if self._label_cache is None or self._label_cache[0] != key:
            groups = {"human": ([], []), "policy": ([], [])}
            for side, positions in enumerate(self._positions()):  # 0 = recent, 1 = past
                for i in positions:
                    groups["human" if _is_human(self.buffer[i]) else "policy"][side].append(i)
            self._label_cache = (key, groups)
        return self._label_cache[1]

    # ---- pure_policy_chunks: filter chunk spans by their labels ----
    def _span_allowed(self, span: tuple) -> bool:
        """A span starting on a non-policy step is always allowed; a policy-start span must be all policy."""
        if _label(self.buffer[span[0]]) != POLICY_LABEL:
            return True
        return all(_label(self.buffer[pos]) == POLICY_LABEL for pos in span[1:])

    def _chunk_spans(self, episodes, chunk_size: int) -> List[tuple]:
        spans = super()._chunk_spans(episodes, chunk_size)
        if not self.pure_policy_chunks:
            return spans
        return [span for span in spans if self._span_allowed(span)]

    def _chunks_from_episodes(self, episodes, chunk_size: int, max_chunks: int, pool: str = "all"):
        # The base random path draws starts and rejects only non-contiguous windows; rejecting
        # label-mixed ones there too would return short batches. Enumerate the allowed spans instead
        # (uniform over spans = the base's episode weighting by number of valid starts). The epoch
        # path already goes through the filtered _chunk_spans.
        if not self.pure_policy_chunks or self.epoch_sampling or chunk_size <= 0 or max_chunks <= 0:
            return super()._chunks_from_episodes(episodes, chunk_size, max_chunks, pool=pool)
        key = (self._cache_key(), chunk_size, pool)
        spans = self._filtered_span_cache.get(key)
        if spans is None:
            spans = self._chunk_spans(list(episodes), chunk_size)
            self._filtered_span_cache[key] = spans
        if not spans:
            return []
        picked = random.sample(spans, max_chunks) if len(spans) >= max_chunks else random.choices(spans, k=max_chunks)
        return [[self.buffer[pos] for pos in span] for span in picked]

    def chunk_span_counts(self, chunk_size: int) -> Dict[str, int]:
        """Number of valid chunk spans by first-step label ('human' / 'policy'), after any filtering."""
        groups = self._label_spans(int(chunk_size))
        return {label: len(recent) + len(past) for label, (recent, past) in groups.items()}

    def _label_spans(self, chunk_size: int) -> Dict[str, Tuple[List[tuple], List[tuple]]]:
        """Every valid chunk span, split by its first step's label and by round; cached until the next add."""
        key = (*self._cache_key(), chunk_size)
        if self._span_cache is None or self._span_cache[0] != key:
            groups = {"human": ([], []), "policy": ([], [])}
            for ep_id, steps in self._episode_steps.items():
                is_recent = self._episode_round.get(ep_id) == self.current_round
                for span in self._chunk_spans([(ep_id, steps)], chunk_size):
                    label = "human" if _is_human(self.buffer[span[0]]) else "policy"
                    groups[label][0 if is_recent else 1].append(span)
            self._span_cache = (key, groups)
        return self._span_cache[1]

    def sample_indices(self, batch_size: int, sampler=None) -> List[int]:
        if self.human_ratio is not None:
            idx = self._draw_stratified(self._label_positions(), batch_size, pool="transitions")
            if idx is not None:
                return idx
        recent, past = self._positions()
        n_recent = self._num_recent(batch_size, bool(recent), bool(past))
        if n_recent is None:
            return super().sample_indices(batch_size, sampler=sampler)
        return self._draw("recent", recent, n_recent) + self._draw("past", past, batch_size - n_recent)

    def get_contiguous_chunks(self, chunk_size: int, max_chunks: int):
        if self._size == 0 or chunk_size <= 0 or max_chunks <= 0:
            return []
        if self.human_ratio is not None:
            spans = self._draw_stratified(self._label_spans(chunk_size), max_chunks, pool="chunks")
            if spans is not None:
                return [[self.buffer[pos] for pos in span] for span in spans]
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
