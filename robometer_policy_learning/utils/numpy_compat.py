"""Load numpy-2 pickles of ``np.random`` generators under numpy 1.x.

Whole-module checkpoints (``torch.save(actor)``) pickle everything the actor references, including the
gymnasium spaces it keeps -- and a seeded space carries a ``np.random.Generator`` in ``_np_random``.
numpy 2 pickles a bit generator by its *class* (``__bit_generator_ctor(PCG64)``) with a
``(state_dict, SeedSequence)`` tuple as its state, and a Generator by the bit-generator *instance*;
numpy 1.x's helpers only accept the *name string* and a plain state dict, and fail with
``ValueError: <class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module``.

That is exactly the cross-environment case here: the ``robometer`` uv group resolves numpy 2, the
default (openpi) group numpy 1.26, so an actor trained in one cannot be ``torch.load``-ed in the other.
:func:`allow_numpy2_rng_pickles` widens numpy 1.x's helpers to accept both encodings. The restored RNG
state is the saved one; nothing else about unpickling changes, and on numpy >= 2 it is a no-op.
"""

import numpy as np

_patched = False
_tuple_state_classes = {}


def _tuple_state_class(cls):
    """A subclass of BitGenerator ``cls`` whose ``__setstate__`` also takes numpy 2's tuple state.

    It keeps ``cls.__name__``, which is what numpy 1.x writes into ``state['bit_generator']`` and hence
    into any re-pickle, so a re-saved checkpoint names the plain class again. The SeedSequence half of
    the tuple is dropped: it only matters for ``spawn()``, not for the restored stream.
    """
    if cls not in _tuple_state_classes:
        def __setstate__(self, state):
            if isinstance(state, tuple):
                state = state[0]
            cls.__setstate__(self, state)

        _tuple_state_classes[cls] = type(cls.__name__, (cls,), {"__setstate__": __setstate__})
    return _tuple_state_classes[cls]


def allow_numpy2_rng_pickles() -> None:
    """Idempotently patch ``numpy.random._pickle`` (numpy < 2 only) to accept numpy-2 RNG pickles."""
    global _patched
    if _patched or int(np.__version__.split(".")[0]) >= 2:
        return
    import numpy.random._pickle as np_pickle

    # getattr/setattr by string: these names start with "__", which would be mangled inside a class.
    orig_bit_generator_ctor = getattr(np_pickle, "__bit_generator_ctor")
    orig_generator_ctor = getattr(np_pickle, "__generator_ctor")
    orig_randomstate_ctor = getattr(np_pickle, "__randomstate_ctor")

    def bit_generator_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):  # numpy 2: the BitGenerator class itself
            return _tuple_state_class(bit_generator_name)()
        return orig_bit_generator_ctor(bit_generator_name)

    # The originals bind the *original* bit_generator_ctor as a default argument, so they are wrapped
    # too; numpy 2 hands them an already-rebuilt BitGenerator instance instead of a name.
    def generator_ctor(bit_generator_name="MT19937", bit_generator_ctor=bit_generator_ctor):
        if isinstance(bit_generator_name, np.random.BitGenerator):
            return np.random.Generator(bit_generator_name)
        return orig_generator_ctor(bit_generator_name, bit_generator_ctor)

    def randomstate_ctor(bit_generator_name="MT19937", bit_generator_ctor=bit_generator_ctor):
        if isinstance(bit_generator_name, np.random.BitGenerator):
            return np.random.RandomState(bit_generator_name)
        return orig_randomstate_ctor(bit_generator_name, bit_generator_ctor)

    # numpy's __reduce__ re-imports these helpers when SAVING, so the wrappers must pickle under numpy's
    # own global names: checkpoints saved from a patched process then reference numpy.random._pickle and
    # load in any process -- unpatched numpy 1 or numpy 2 -- rather than depending on this module.
    for name, fn in (
        ("__bit_generator_ctor", bit_generator_ctor),
        ("__generator_ctor", generator_ctor),
        ("__randomstate_ctor", randomstate_ctor),
    ):
        fn.__module__, fn.__name__, fn.__qualname__ = np_pickle.__name__, name, name
        setattr(np_pickle, name, fn)
    _patched = True
