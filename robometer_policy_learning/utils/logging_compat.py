"""Console-logging helpers that work with or without the optional ``robometer`` dependency.

``robometer`` is an optional extra (``uv sync --extra robometer``); the pi0/LIBERO workflows do not
need it, and it cannot even be installed alongside ``openpi`` (conflicting ``datasets`` / ``torch``
pins -- see the conflict declaration in pyproject.toml). ``robometer.utils.logger`` is the only piece
of robometer that most of this package wants, so this module re-exports ``get_logger`` /
``setup_loguru_logging`` from robometer when it is installed and falls back to an equivalent
loguru-only implementation when it is not.

Import these from here rather than from ``robometer.utils.logger`` directly, so a module does not
acquire a hard robometer dependency just to write log lines.
"""

import os
import sys
from typing import Optional

from loguru import logger as _loguru_logger

# Intermediate debug level between TRACE (5) and DEBUG (10); mirrors robometer.utils.logger.
DEBUG2_LEVEL = 8

try:  # pragma: no cover - depends on whether the robometer extra is installed
    from robometer.utils.logger import get_logger, setup_loguru_logging

    HAVE_ROBOMETER_LOGGER = True
except ImportError:
    HAVE_ROBOMETER_LOGGER = False

    def _get_rank() -> int:
        """Best-effort process rank without robometer.utils.distributed."""
        for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
            value = os.environ.get(var)
            if value is not None:
                try:
                    return int(value)
                except ValueError:
                    pass
        return 0

    def _add_custom_log_levels() -> None:
        try:
            _loguru_logger.level("DEBUG2", no=DEBUG2_LEVEL, color="<dim><cyan>")
        except ValueError:
            # Already registered (setup called more than once) -- fine.
            pass

    def setup_loguru_logging(log_level: str = "INFO", output_dir: Optional[str] = None):
        """loguru-only stand-in for ``robometer.utils.logger.setup_loguru_logging``.

        Same rank-prefixed console format and optional ``training.log`` file handler; the rank is
        read from the environment instead of robometer's distributed helpers.
        """
        _add_custom_log_levels()

        _loguru_logger.remove()
        rank = _get_rank()
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            f"[Rank {rank}] "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        _loguru_logger.add(sys.stderr, format=format_string, level=log_level.upper(), colorize=True)

        if output_dir and rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            _loguru_logger.add(
                os.path.join(output_dir, "training.log"),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n",
                level=log_level.upper(),
                rotation="10 MB",
                retention="7 days",
                encoding="utf-8",
            )

    def get_logger():
        """Return the loguru logger instance (matches robometer's ``get_logger``)."""
        return _loguru_logger


__all__ = ["DEBUG2_LEVEL", "HAVE_ROBOMETER_LOGGER", "get_logger", "setup_loguru_logging"]
