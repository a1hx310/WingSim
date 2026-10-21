"""NVIDIA GPU backend for the Lattice Boltzmann tunnel, using Numba CUDA.

The kernels mirror the CPU ones in lbm.py, one GPU thread per grid cell. The
GPU works in single precision, since consumer cards run double precision far
slower; at these Mach numbers single precision is plenty. Populations stay on
the card between steps, and only the density and velocity fields come back to
the host for drawing.
"""

import math

import numpy as np
from numba import cuda, float32

from .lbm import CX, CY, OPP, W, Tunnel

W32 = W.astype(np.float32)
# Float copies of the lattice velocities keep the arithmetic in single precision.
CXF = CX.astype(np.float32)
CYF = CY.astype(np.float32)
BLOCK = (32, 8)


def gpu_name():
    """Name of the CUDA device, or None when no usable NVIDIA GPU is present."""
    try:
        if not cuda.is_available():
            return None
        name = getattr(cuda.get_current_device(), "name", "CUDA simulator")
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return None


@cuda.jit(fastmath=True)
def _stream_collide(f, fn, solid, tau0, cs2, rho_out, ux_out, uy_out, force):
    x, y = cuda.grid(2)
    ny = f.shape[1]
    nx = f.shape[2]
    if x < 1 or y < 1 or x >= nx - 1 or y >= ny - 1:
        return
    if solid[y, x]:
        for i in range(9):
            fn[i, y, x] = W32[i]
        rho_out[y, x] = 1.0
        ux_out[y, x] = 0.0
        uy_out[y, x] = 0.0
        return
    fl = cuda.local.array(9, float32)
    fx_acc = float32(0.0)
    fy_acc = float32(0.0)
    # Pull streaming, with half-way bounce-back off solid neighbours.
    for i in range(9):
        sx = x - CX[i]
        sy = y - CY[i]
        if solid[sy, sx]:
            o = OPP[i]
            v = f[o, y, x]
            fl[i] = v
            # The population left with velocity c_o and came back with -c_o.
            fx_acc += float32(2.0) * CXF[o] * v
            fy_acc += float32(2.0) * CYF[o] * v
        else:
            fl[i] = f[i, sy, sx]
    rho = float32(0.0)
    mx = float32(0.0)
    my = float32(0.0)
    for i in range(9):
        rho += fl[i]
        mx += CXF[i] * fl[i]
        my += CYF[i] * fl[i]
    ux = mx / rho
    uy = my / rho
    usq = float32(1.5) * (ux * ux + uy * uy)
    # Non-equilibrium momentum flux drives the Smagorinsky eddy viscosity.
    pxx = float32(0.0)
    pyy = float32(0.0)
    pxy = float32(0.0)
    for i in range(9):
        cu = float32(3.0) * (CXF[i] * ux + CYF[i] * uy)
        feq = W32[i] * rho * (float32(1.0) + cu + float32(0.5) * cu * cu - usq)
        d = fl[i] - feq
        pxx += CXF[i] * CXF[i] * d
        pyy += CYF[i] * CYF[i] * d
        pxy += CXF[i] * CYF[i] * d
    q = math.sqrt(float32(2.0) * (pxx * pxx + pyy * pyy + float32(2.0) * pxy * pxy))
    tau = 0.5 * (tau0 + math.sqrt(tau0 * tau0 + 18.0 * cs2 * q / rho))
    omega = float32(1.0 / tau)
    for i in range(9):
        cu = float32(3.0) * (CXF[i] * ux + CYF[i] * uy)
        feq = W32[i] * rho * (float32(1.0) + cu + float32(0.5) * cu * cu - usq)
        fn[i, y, x] = fl[i] + omega * (feq - fl[i])
    rho_out[y, x] = rho
    ux_out[y, x] = ux
    uy_out[y, x] = uy
    if fx_acc != 0.0 or fy_acc != 0.0:
        cuda.atomic.add(force, 0, fx_acc)
        cuda.atomic.add(force, 1, fy_acc)


@cuda.jit
def _boundaries(fn, inflow):
    x, y = cuda.grid(2)
    ny = fn.shape[1]
    nx = fn.shape[2]
    if x >= nx or y >= ny:
        return
    if x == 0 or y == 0 or y == ny - 1:
        for i in range(9):
            fn[i, y, x] = inflow[i]
    elif x == nx - 1:
        for i in range(9):
            fn[i, y, x] = fn[i, y, x - 1]


class GpuTunnel(Tunnel):
    """Same tunnel and API as Tunnel, with the time stepping on an NVIDIA GPU."""

    device = "gpu"
    device_label = "GPU"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_label = f"GPU ({gpu_name()})"
        self._grid = ((self.nx + BLOCK[0] - 1) // BLOCK[0],
                      (self.ny + BLOCK[1] - 1) // BLOCK[1])
        self._d_f = cuda.to_device(self.f.astype(np.float32))
        self._d_fn = cuda.to_device(self.f.astype(np.float32))
        self._d_solid = cuda.to_device(self.solid)
        self._d_rho = cuda.to_device(self.rho.astype(np.float32))
        self._d_ux = cuda.to_device(self.ux.astype(np.float32))
        self._d_uy = cuda.to_device(self.uy.astype(np.float32))
        self._d_inflow = cuda.to_device(self._inflow[:, 0].astype(np.float32))
        self._d_force = cuda.to_device(np.zeros(2, dtype=np.float64))

    def set_solid(self, mask):
        self.f = self._d_f.copy_to_host().astype(np.float64)
        super().set_solid(mask)
        self._d_f = cuda.to_device(self.f.astype(np.float32))
        self._d_solid = cuda.to_device(self.solid)

    def step(self, n=1):
        self._d_force.copy_to_device(np.zeros(2, dtype=np.float64))
        tau0, cs2 = np.float32(self.tau), np.float32(self.cs2)
        for _ in range(n):
            _stream_collide[self._grid, BLOCK](self._d_f, self._d_fn, self._d_solid, tau0, cs2,
                                               self._d_rho, self._d_ux, self._d_uy, self._d_force)
            _boundaries[self._grid, BLOCK](self._d_fn, self._d_inflow)
            self._d_f, self._d_fn = self._d_fn, self._d_f
        self.rho = self._d_rho.copy_to_host()
        self.ux = self._d_ux.copy_to_host()
        self.uy = self._d_uy.copy_to_host()
        fx, fy = self._d_force.copy_to_host()
        self.steps += n
        self.force = (fx / n, fy / n)
        return self.force
