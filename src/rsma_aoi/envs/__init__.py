"""Physical-layer models and the AoI Gymnasium environment.

The environment class is imported lazily so that importing a single PHY module
(e.g. in a unit test) does not pull in Gymnasium/PyTorch.
"""

from __future__ import annotations

from typing import Any

__all__ = ["AoIRSMAEnv"]


def __getattr__(name: str) -> Any:
    if name == "AoIRSMAEnv":
        from rsma_aoi.envs.aoi_env import AoIRSMAEnv

        return AoIRSMAEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
