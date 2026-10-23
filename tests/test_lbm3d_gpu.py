"""Checks the 3D GPU kernels against the CPU solver.

Runs on a real NVIDIA GPU when one is present. Without one, run it under
Numba's CUDA simulator: NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_lbm3d_gpu.py
"""

import numpy as np
import pytest

from wingsim import airfoil
from wingsim.lbm3d import Tunnel3D

lbm_gpu = pytest.importorskip("wingsim.lbm_gpu")
if lbm_gpu.gpu_name() is None:
    pytest.skip("no CUDA device or simulator", allow_module_level=True)
lbm3d_gpu = pytest.importorskip("wingsim.lbm3d_gpu")


def test_gpu_matches_cpu():
    nx, ny, nz, chord = 32, 16, 16, 8
    section = airfoil.rasterize("2412", chord, 8.0, (10.0, 7.5), (ny, nx))
    mask = airfoil.extrude(section, nz, 4, 12)
    rng = np.random.default_rng(0)
    pos = np.c_[rng.uniform(2, nx - 4, 200), rng.uniform(2, ny - 4, 200),
                rng.uniform(2, nz - 4, 200)]
    results = []
    for cls in (Tunnel3D, lbm3d_gpu.GpuTunnel3D):
        t = cls(nx, ny, nz)
        t.set_reynolds(100.0, chord)
        t.set_solid(mask)
        t.step(5)
        t.set_particles(pos, pos)
        slices = [t.slice(kind, axis, k) for kind in ("speed", "vorticity", "pressure")
                  for axis, k in (("x", 16), ("z", 8))]
        results.append((t.fields(), t.force, slices, t.q_criterion(2), t.advect(5.0)))
    (f_c, force_c, s_c, q_c, p_c), (f_g, force_g, s_g, q_g, p_g) = results
    for a, b in zip(f_c, f_g):
        assert np.allclose(b, a, atol=1e-5)
    assert np.allclose(force_g, force_c, rtol=1e-3, atol=1e-6)
    for a, b in zip(s_c, s_g):
        # Pressure coefficients scale density errors up by about 100.
        assert b.shape == a.shape and np.allclose(b, a, atol=1e-3)
    assert np.allclose(q_g, q_c, atol=1e-7)
    assert np.allclose(p_g, p_c, atol=1e-4)
