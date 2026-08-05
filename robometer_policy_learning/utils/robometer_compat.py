"""Lazy accessors for ``robometer`` functions that have no in-repo fallback.

``robometer`` is an optional extra (``uv sync --extra robometer``) and cannot be installed alongside
``openpi`` -- they pin incompatible ``datasets`` / ``torch`` versions, which pyproject.toml declares as
conflicting extras. Modules that only *sometimes* reach robometer (e.g. an env wrapper that computes
language embeddings only when a sentence model is configured) should import from here instead of at
module scope, so merely importing them does not require robometer to be installed.

Each wrapper defers the import to call time and raises a clear, actionable ImportError if the extra is
missing. Unlike :mod:`robometer_policy_learning.utils.logging_compat` -- which provides a real
loguru-based fallback -- these functions genuinely need robometer, so there is nothing to fall back to.
"""

import importlib
from typing import Any

_INSTALL_HINT = (
    "This code path requires the optional `robometer` dependency, which is not installed in this "
    "environment. Install it with `uv sync --extra robometer`. Note that the robometer extra is "
    "mutually exclusive with the openpi extra (conflicting datasets/torch pins), so it cannot be "
    "combined with the pi0/LIBERO environment."
)


def _resolve(module: str, name: str) -> Any:
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f"`{module}.{name}` is unavailable. {_INSTALL_HINT}") from e
    return getattr(mod, name)


def compute_text_embeddings(*args, **kwargs):
    """See ``robometer.utils.embedding_utils.compute_text_embeddings``."""
    return _resolve("robometer.utils.embedding_utils", "compute_text_embeddings")(*args, **kwargs)


def compute_video_embeddings(*args, **kwargs):
    """See ``robometer.utils.embedding_utils.compute_video_embeddings``."""
    return _resolve("robometer.utils.embedding_utils", "compute_video_embeddings")(*args, **kwargs)


def load_model_from_hf(*args, **kwargs):
    """See ``robometer.utils.save.load_model_from_hf``."""
    return _resolve("robometer.utils.save", "load_model_from_hf")(*args, **kwargs)


__all__ = ["compute_text_embeddings", "compute_video_embeddings", "load_model_from_hf"]
