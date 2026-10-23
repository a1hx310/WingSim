import numpy as np
import pytest

from wingsim import airfoil
from wingsim.lbm3d import CX, CY, CZ, Tunnel3D, equilibrium


def test_equilibrium_moments():
    feq = equilibrium(1.2, 0.05, -0.03, 0.02)
    assert feq.sum() == pytest.approx(1.2)
    assert (CX * feq).sum() == pytest.approx(1.2 * 0.05)
    assert (CY * feq).sum() == pytest.approx(1.2 * -0.03)
    assert (CZ * feq).sum() == pytest.approx(1.2 * 0.02)


def test_empty_tunnel_stays_uniform():
    t = Tunnel3D(40, 20, 20)
    t.step(100)
    assert np.allclose(t.ux[1:-1, 1:-1, 1:-1], t.u, atol=1e-6)
    assert t.force == (0.0, 0.0, 0.0)


NX, NY, NZ, CHORD = 72, 36, 40, 16


def _run(code, aoa, span=None, steps=1500):
    """CD, CL and side force coefficient of a wing centred in the tunnel. Without a
    span the wing runs wall to wall, which makes the flow two-dimensional."""
    t = Tunnel3D(NX, NY, NZ)
    t.set_reynolds(100.0, CHORD)
    section = airfoil.rasterize(code, CHORD, aoa, (20.0, (NY - 1) / 2), (NY, NX))
    mid = (NZ - 1) / 2
    z0, z1 = (mid - span / 2, mid + span / 2) if span else (0, NZ)
    t.set_solid(airfoil.extrude(section, NZ, z0, z1))
    t.step(steps)
    cd, cl = t.coefficients()
    return cd, cl, t.force[2] / (0.5 * t.u**2 * CHORD * t.span)


def test_symmetric_wing_at_zero_incidence_has_no_lift():
    cd, cl, cs = _run("0012", 0.0, span=24)
    assert abs(cl) < 1e-3
    assert abs(cs) < 1e-6
    assert cd > 0


def test_lift_grows_with_angle_of_attack():
    _, cl_low, _ = _run("0012", 2.0, span=24)
    _, cl_high, _ = _run("0012", 8.0, span=24)
    assert 0 < cl_low < cl_high


def test_wing_centred_in_span_has_no_side_force():
    _, _, cs = _run("2412", 8.0, span=24)
    assert abs(cs) < 1e-6


def test_finite_wing_loses_lift_to_its_tips():
    _, cl_finite, _ = _run("0012", 8.0, span=24)
    _, cl_endless, _ = _run("0012", 8.0)
    assert cl_finite < 0.9 * cl_endless
