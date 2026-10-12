"""Canonical bounded transforms for the thesis RSMA allocation action.

The direct action uses latent Gaussian variables for the common-power share
``alpha`` and the per-user common-payload fractions ``phi_k``.  A logistic
sigmoid never reaches either physical boundary, so it cannot represent exact
SDMA (``alpha = phi = 0``) without a separate Bernoulli gate.  The clipped
linear sigmoid below is continuous, has useful gradients in its interior, and
has exact atoms at zero and one after transforming an ordinary Gaussian
sample.  PPO probabilities are still evaluated in latent space.
"""

from __future__ import annotations

import numpy as np
import torch


HARD_SIGMOID_LIMIT = 3.0
DIRECT_ACTION_SCHEMA = "direct_alpha_q_phi_hard_sigmoid_v1"


def hard_sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Map real NumPy values to ``[0, 1]`` with exact endpoints."""
    return np.clip(0.5 + np.asarray(x) / (2.0 * HARD_SIGMOID_LIMIT), 0.0, 1.0)


def hard_sigmoid_torch(x: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of :func:`hard_sigmoid_np`."""
    return torch.clamp(0.5 + x / (2.0 * HARD_SIGMOID_LIMIT), 0.0, 1.0)


def hard_sigmoid_inverse(value: float) -> float:
    """Interior inverse used to initialise an allocation head's modal action."""
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("hard-sigmoid inverse requires a value strictly inside (0, 1)")
    return 2.0 * HARD_SIGMOID_LIMIT * (value - 0.5)
