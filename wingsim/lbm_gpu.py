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

from .lbm import CX, CY, OPP, W, Tunnel, _jitter

W32 = W.astype(np.float32)
# Float copies of the lattice velocities keep the arithmetic in single precision.
CXF = CX.astype(np.float32)
CYF = CY.astype(np.float32)
BLOCK = (32, 8)
BLOCK_1D = 256
# The CPU smoke jitter compiled for the GPU, so both release particles alike.
_jitter = cuda.jit(device=True)(_jitter.py_func)


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
def _stream_collide(f, fn, solid, tau0, cs2, inflow, rho_out, ux_out, uy_out, force):
    x, y = cuda.grid(2)
    ny = f.shape[1]
    nx = f.shape[2]
    if x >= nx or y >= ny:
        return
    # The tunnel boundaries, which the CPU solver applies after each step, are set here
    # rather than in a second kernel: at this grid size a launch costs about as much as
    # the step itself.
    if x == 0 or y == 0 or y == ny - 1:
        for i in range(9):
            fn[i, y, x] = inflow[i]
        return
    # The open outlet copies the cell just upstream, so its thread computes that cell.
    xo = x
    outlet = x == nx - 1
    if outlet:
        x = nx - 2
    if solid[y, x]:
        for i in range(9):
            fn[i, y, xo] = W32[i]
        if not outlet:
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
        fn[i, y, xo] = fl[i] + omega * (feq - fl[i])
    if outlet:
        return
    rho_out[y, x] = rho
    ux_out[y, x] = ux
    uy_out[y, x] = uy
    if fx_acc != 0.0 or fy_acc != 0.0:
        cuda.atomic.add(force, 0, fx_acc)
        cuda.atomic.add(force, 1, fy_acc)


@cuda.jit
def _advect(pos, release, ux, uy, solid, dt, spread, seed):
    k = cuda.grid(1)
    if k >= pos.shape[0]:
        return
    ny = ux.shape[0]
    nx = ux.shape[1]
    # Keep each particle move under about half a cell so streaks stay smooth.
    substeps = max(1, int(dt * 0.2) + 1)
    h = dt / substeps
    x = pos[k, 0]
    y = pos[k, 1]
    for _ in range(substeps):
        if not (0 <= x < nx - 1 and 0 <= y < ny - 1):
            break
        i, j = int(x), int(y)
        fx, fy = x - i, y - j
        u = ((1 - fx) * (1 - fy) * ux[j, i] + fx * (1 - fy) * ux[j, i + 1]
             + (1 - fx) * fy * ux[j + 1, i] + fx * fy * ux[j + 1, i + 1])
        v = ((1 - fx) * (1 - fy) * uy[j, i] + fx * (1 - fy) * uy[j, i + 1]
             + (1 - fx) * fy * uy[j + 1, i] + fx * fy * uy[j + 1, i + 1])
        x += u * h
        y += v * h
    if x < 1 or x >= nx - 2 or y < 1 or y >= ny - 2 or solid[int(y), int(x)]:
        # Re-release from the smoke rake at the inlet, spread over the distance the
        # free stream covers in dt so the streaks stay continuous.
        x = 1.0 + spread * _jitter(k, seed)
        y = release[k, 1]
    pos[k, 0] = x
    pos[k, 1] = y


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
        self._d_zero = cuda.to_device(np.zeros(2, dtype=np.float64))
        # Steps run on their own stream, and the results land in pinned host memory,
        # so start() can return while the card works.
        self._stream = cuda.stream()
        self._events = (cuda.event(), cuda.event())
        self._h_force = cuda.pinned_array(2, dtype=np.float64)
        self._h_fields = [cuda.pinned_array((self.ny, self.nx), dtype=np.float32) for _ in range(3)]

    def set_solid(self, mask):
        self.f = self._d_f.copy_to_host().astype(np.float64)
        super().set_solid(mask)
        self._d_f = cuda.to_device(self.f.astype(np.float32))
        self._d_solid = cuda.to_device(self.solid)

    def step(self, n=1):
        self.start(n)
        self.finish()
        return self.force

    def start(self, n=1):
        s = self._stream
        self._d_force.copy_to_device(self._d_zero, stream=s)
        tau0, cs2 = np.float32(self.tau), np.float32(self.cs2)
        self._events[0].record(s)
        # Launching is the slow part at this grid size, so do as little else per step.
        kernel = _stream_collide[self._grid, BLOCK, s]
        args = [self._d_f, self._d_fn, self._d_solid, tau0, cs2, self._d_inflow,
                self._d_rho, self._d_ux, self._d_uy, self._d_force]
        for _ in range(n):
            kernel(*args)
            args[0], args[1] = args[1], args[0]
        self._d_f, self._d_fn = args[0], args[1]
        self._events[1].record(s)
        self._d_force.copy_to_host(self._h_force, stream=s)
        for d, h in zip((self._d_rho, self._d_ux, self._d_uy), self._h_fields):
            d.copy_to_host(h, stream=s)
        self._pending = n

    def finish(self):
        n, self._pending = self._pending, 0
        if n:
            self._stream.synchronize()
            self.batch_ms = self._events[0].elapsed_time(self._events[1])
            # Copies, so the next batch can refill the pinned buffers while these are drawn.
            self.rho, self.ux, self.uy = (h.copy() for h in self._h_fields)
            self.steps += n
            self.force = tuple(self._h_force / n)
        return n

    def set_particles(self, pos, release):
        self._d_pos = cuda.to_device(np.asarray(pos, dtype=np.float32).reshape(-1, 2))
        self._d_release = cuda.to_device(np.asarray(release, dtype=np.float32).reshape(-1, 2))

    def advect(self, dt):
        n = self._d_pos.shape[0]
        if n:
            _advect[(n + BLOCK_1D - 1) // BLOCK_1D, BLOCK_1D](
                self._d_pos, self._d_release, self._d_ux, self._d_uy, self._d_solid,
                np.float32(dt), np.float32(self._spread(dt)), self.steps)
        return self._d_pos.copy_to_host()
