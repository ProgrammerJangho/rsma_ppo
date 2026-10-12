"""The differentiable PHY must agree with the SciPy/NumPy one and stay finite."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsma_aoi.envs import fbl as np_fbl
from rsma_aoi.envs import torch_phy as tphy
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer, split_payload_batch
from rsma_aoi.utils.config import EnvConfig

@pytest.fixture(autouse=True)
def float64_default():
    """Compare against the float64 NumPy PHY without leaking the dtype globally.

    ``torch.set_default_dtype`` is process-wide, so it must be restored or every
    later test module silently builds float64 tensors.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def test_q_function_matches_scipy():
    x = np.linspace(-8.0, 8.0, 65)
    np.testing.assert_allclose(
        tphy.torch_q(torch.as_tensor(x)).numpy(), np_fbl.q_function(x), rtol=1e-10, atol=1e-14
    )


def test_capacity_and_dispersion_match():
    gamma = np.array([1e-9, 0.01, 0.5, 1.0, 10.0, 1e4])
    g = torch.as_tensor(gamma)
    np.testing.assert_allclose(
        tphy.shannon_capacity(g).numpy(), np_fbl.shannon_capacity(gamma), rtol=1e-12
    )
    np.testing.assert_allclose(
        tphy.channel_dispersion(g).numpy(), np_fbl.channel_dispersion(gamma), rtol=1e-12
    )


@pytest.mark.parametrize("blocklength", [16, 128, 512])
def test_bler_matches_scipy(blocklength):
    gamma = np.array([1e-6, 0.05, 0.5, 1.0, 3.0, 10.0, 100.0, 1e5])
    payload = np.full_like(gamma, 128.0)
    got = tphy.fbl_bler(
        torch.as_tensor(gamma), float(blocklength), torch.as_tensor(payload)
    ).numpy()
    expected = np_fbl.fbl_bler(gamma, blocklength, payload)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-14)


def test_zero_payload_gives_zero_bler():
    out = tphy.fbl_bler(torch.tensor([1.0, 2.0]), 128.0, torch.tensor([0.0, 64.0]))
    assert out[0].item() == 0.0
    assert 0.0 < out[1].item() < 1.0


def test_bler_is_finite_at_extremes():
    gamma = torch.tensor([0.0, 1e-30, 1e30])
    payload = torch.tensor([1e6, 1e6, 1.0])
    out = tphy.fbl_bler(gamma, 8.0, payload)
    assert torch.all(torch.isfinite(out))
    assert torch.all((out >= 0.0) & (out <= 1.0))


def test_bler_gradient_is_finite_and_non_zero_near_the_threshold():
    gamma = torch.tensor([3.0], requires_grad=True)
    bler = tphy.fbl_bler(gamma, 128.0, torch.tensor([256.0]))
    bler.backward()
    assert torch.isfinite(gamma.grad).all()
    assert gamma.grad.abs().item() > 1e-6, "no gradient at the decoding threshold"


def test_margin_keeps_a_gradient_where_the_bler_saturates():
    """Far from threshold Q saturates; the margin must still carry signal."""
    gamma = torch.tensor([1e4], requires_grad=True)
    bler = tphy.fbl_bler(gamma, 128.0, torch.tensor([64.0]))
    bler.backward()
    saturated = gamma.grad.abs().item()

    gamma2 = torch.tensor([1e4], requires_grad=True)
    margin = tphy.fbl_margin(gamma2, 128.0, torch.tensor([64.0]))
    margin.backward()
    assert torch.isfinite(gamma2.grad).all()
    assert gamma2.grad.abs().item() > saturated
    assert gamma2.grad.abs().item() > 1e-9


def test_masked_softmax_matches_the_numpy_helper():
    from rsma_aoi.evaluation.oracle import masked_softmax_np

    logits = np.random.default_rng(0).standard_normal((4, 5))
    mask = np.array([[True, True, False, True, False]] * 4)
    got = tphy.masked_softmax(torch.as_tensor(logits), torch.as_tensor(mask)).numpy()
    expected = masked_softmax_np(logits, mask[0])
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_split_payload_matches_numpy():
    rng = np.random.default_rng(1)
    payload = np.array([100.0, 150.0, 200.0])
    active = np.array([True, True, False])
    c = np.zeros((8, 3))
    c[:, active] = rng.dirichlet(np.ones(2), size=8)
    rho = rng.random(8)
    d_c_np, d_p_np = split_payload_batch(payload, active, c, rho)
    d_c_t, d_p_t = tphy.split_payload(
        torch.as_tensor(payload), torch.as_tensor(np.broadcast_to(active, c.shape).copy()),
        torch.as_tensor(c), torch.as_tensor(rho),
    )
    np.testing.assert_allclose(d_c_t.numpy(), d_c_np, rtol=1e-10)
    np.testing.assert_allclose(d_p_t.numpy(), d_p_np, rtol=1e-10)


@pytest.mark.parametrize("use_common", [True, False])
def test_success_probability_matches_the_numpy_phy(use_common):
    cfg = EnvConfig(num_users=3, num_antennas=2, snr_db=None, p_max=10.0, noise_power=1.0,
                    blocklength=128, payload_bits=192.0, private_beam="rzf",
                    common_beam="principal",
                    multiple_access="rsma" if use_common else "sdma")
    phy = RSMAPhysicalLayer(cfg)
    rng = np.random.default_rng(2)
    h = (rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))) / np.sqrt(2)
    h_hat = (rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))) / np.sqrt(2)
    active = np.array([True, True, False])

    n = 12
    pf = np.zeros((n, 4))
    rf = np.zeros((n, 3))
    for i in range(n):
        w = rng.dirichlet(np.ones(int(active.sum()) + 1))
        pf[i, 0] = w[0]
        pf[i, 1:][active] = w[1:]
        rf[i, active] = rng.dirichlet(np.ones(int(active.sum())))
    rho = rng.random(n)

    reference = phy.evaluate_allocations(h, active, pf, rf, rho, h_hat=h_hat)
    u_priv, u_common = phy.rsma_beams(h_hat, active)
    gain_priv, gain_common = tphy.channel_gains(
        torch.as_tensor(h)[None].expand(n, -1, -1),
        torch.as_tensor(u_priv)[None].expand(n, -1, -1),
        torch.as_tensor(u_common)[None].expand(n, -1),
    )
    out = tphy.rsma_success_probability(
        gain_priv, gain_common,
        torch.as_tensor(np.broadcast_to(active, (n, 3)).copy()),
        torch.as_tensor(pf), torch.as_tensor(rf), torch.as_tensor(rho),
        torch.as_tensor(np.asarray(cfg.payload_vector)), float(cfg.blocklength),
        cfg.noise_power, cfg.total_power, use_common=use_common,
    )
    np.testing.assert_allclose(
        out["success_prob"].numpy(), reference["success_prob"], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        out["private_sinr"].numpy(), reference["private_sinr"], rtol=1e-9
    )
    np.testing.assert_allclose(
        out["common_payload"].numpy(), reference["common_payload"], rtol=1e-9
    )


def test_success_probability_gradient_flows_to_the_latents():
    """End-to-end: latents -> simplex -> SINR -> BLER -> objective."""
    k, n = 3, 4
    rng = np.random.default_rng(3)
    gain_priv = torch.as_tensor(np.abs(rng.standard_normal((n, k, k))) + 0.1)
    gain_common = torch.as_tensor(np.abs(rng.standard_normal((n, k))) + 0.1)
    active = torch.ones(n, k, dtype=torch.bool)
    latent = torch.zeros(n, 2 * k + 2, requires_grad=True)

    power = tphy.masked_softmax(latent[:, : k + 1], torch.ones(n, k + 1, dtype=torch.bool))
    rate = tphy.masked_softmax(latent[:, k + 1 : 2 * k + 1], active)
    rho = torch.sigmoid(latent[:, -1])
    out = tphy.rsma_success_probability(
        gain_priv, gain_common, active, power, rate, rho,
        torch.full((k,), 128.0), 128.0, 1.0, 10.0,
    )
    urgency = torch.ones(k)
    objective = (out["success_prob"] @ urgency).mean()
    objective.backward()
    assert torch.isfinite(latent.grad).all()
    assert latent.grad.abs().sum().item() > 0.0
