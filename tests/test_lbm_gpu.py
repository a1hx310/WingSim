"""Checks the GPU kernels against the CPU solver.

Runs on a real NVIDIA GPU when one is present. Without one, run it under
Numba's CUDA simulator: NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_lbm_gpu.py
"""

import numpy as np
import pytest

from wingsim import airfoil
from wingsim.lbm import Tunnel

lbm_gpu = pytest.importorskip("wingsim.lbm_gpu")
if lbm_gpu.gpu_name() is None:
    pytest.skip("no CUDA device or simulator", allow_module_level=True)


def test_gpu_matches_cpu():
    nx, ny, chord = 48, 24, 16
    mask = airfoil.rasterize("2412", chord, 8.0, (16.0, 11.5), (ny, nx))
    rng = np.random.default_rng(0)
    pos = np.c_[rng.uniform(2, nx - 4, 200), rng.uniform(2, ny - 4, 200)]
    results = []
    for cls in (Tunnel, lbm_gpu.GpuTunnel):
        t = cls(nx, ny)
        t.set_reynolds(100.0, chord)
        t.set_solid(mask)
        t.step(5)
        t.set_particles(pos, pos)
        results.append((t.ux.copy(), t.uy.copy(), t.force, t.advect(5.0)))
    (ux_c, uy_c, f_c, p_c), (ux_g, uy_g, f_g, p_g) = results
    assert np.allclose(ux_g, ux_c, atol=1e-5)
    assert np.allclose(uy_g, uy_c, atol=1e-5)
    assert np.allclose(f_g, f_c, rtol=1e-3, atol=1e-6)
    assert np.allclose(p_g, p_c, atol=1e-4)
