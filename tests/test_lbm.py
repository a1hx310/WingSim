import numpy as np
import pytest

from wingsim import airfoil
from wingsim.lbm import Tunnel, equilibrium


def test_equilibrium_moments():
    feq = equilibrium(1.2, 0.05, -0.03)
    from wingsim.lbm import CX, CY
    assert feq.sum() == pytest.approx(1.2)
    assert (CX * feq).sum() == pytest.approx(1.2 * 0.05)
    assert (CY * feq).sum() == pytest.approx(1.2 * -0.03)


def test_empty_tunnel_stays_uniform():
    t = Tunnel(80, 40)
    t.step(200)
    assert np.allclose(t.ux[1:-1, 1:-1], t.u, atol=1e-6)
    assert t.force == (0.0, 0.0)


def _run(code, aoa, steps=6000, re=200.0):
    t = Tunnel(240, 120)
    chord = 40
    t.set_reynolds(re, chord)
    t.set_solid(airfoil.rasterize(code, chord, aoa, (80.0, 59.5), (t.ny, t.nx)))
    t.step(steps)
    return t.coefficients()


def test_symmetric_airfoil_at_zero_incidence_has_no_lift():
    cd, cl = _run("0012", 0.0)
    assert abs(cl) < 1e-3
    assert cd > 0


def test_lift_grows_with_angle_of_attack():
    _, cl_low = _run("0012", 2.0)
    _, cl_high = _run("0012", 8.0)
    assert 0 < cl_low < cl_high


def test_start_then_finish_advances_the_flow():
    t = Tunnel(40, 20)
    t.start(7)
    assert t.finish() == 7
    assert t.steps == 7
    assert t.finish() == 0
