from robometer_policy_learning.configs.configs import TrainConfig, DSRLConfig

__all__ = [
    "TrainConfig",
    "DSRLConfig",
]

# ``robometer`` is an optional extra (see pyproject.toml) and is mutually exclusive with ``openpi``,
# so importing this package must not require it -- otherwise every pi0/LIBERO entry point that reads
# a config from here would fail at import time. ``EvalServerConfig`` is re-exported only when the
# robometer extra is installed; the reward-model code paths that actually use it always are.
try:
    from robometer.configs.eval_configs import EvalServerConfig  # noqa: F401

    __all__.append("EvalServerConfig")
except ImportError:
    pass
