"""NVIDIA GPU backend for the 3D tunnel, using Numba CUDA.

The kernels mirror the CPU ones in lbm3d.py, one GPU thread per grid cell, in single
precision. Everything stays on the card: the populations, the flow fields and the
smoke particles. Only what gets drawn comes back to the host each frame, which is a
few planes, the particle positions and now and then a coarse Q-criterion field.
"""

import math

import numpy as np
from numba import cuda, float32

from .lbm import _jitter
from .lbm3d import CX, CY, CZ, OPP, SLICE_KINDS, W, Tunnel3D
from .lbm_gpu import BLOCK as BLOCK_2D
from .lbm_gpu import gpu_name

W32 = W.astype(np.float32)
# Float copies of the lattice velocities keep the arithmetic in single precision.
CXF = CX.astype(np.float32)
CYF = CY.astype(np.float32)
CZF = CZ.astype(np.float32)
BLOCK = (32, 4, 2)
BLOCK_1D = 256
# The CPU smoke jitter compiled for the GPU, so both release particles alike.
_jitter = cuda.jit(device=True)(_jitter.py_func)


def _blocks(sizes, block):
    """Thread blocks needed to cover sizes (x first, like the block shape)."""
    return tuple((n + b - 1) // b for n, b in zip(sizes, block))


@cuda.jit(fastmath=True, cache=True)
def _stream_collide(f, fn, solid, tau0, cs2, store, rho_out, ux_out, uy_out, uz_out, force):
    x, y, z = cuda.grid(3)
    nz = f.shape[1]
    ny = f.shape[2]
    nx = f.shape[3]
    if x < 1 or y < 1 or z < 1 or x >= nx - 1 or y >= ny - 1 or z >= nz - 1:
        return
    if solid[z, y, x]:
        for i in range(19):
            fn[i, z, y, x] = W32[i]
        if store:
            rho_out[z, y, x] = 1.0
            ux_out[z, y, x] = 0.0
            uy_out[z, y, x] = 0.0
            uz_out[z, y, x] = 0.0
        return
    fl = cuda.local.array(19, float32)
    fx_acc = float32(0.0)
    fy_acc = float32(0.0)
    fz_acc = float32(0.0)
    # Pull streaming, with half-way bounce-back off solid neighbours.
    for i in range(19):
        sx = x - CX[i]
        sy = y - CY[i]
        sz = z - CZ[i]
        if solid[sz, sy, sx]:
            o = OPP[i]
            v = f[o, z, y, x]
            fl[i] = v
            # The population left with velocity c_o and came back with -c_o.
            fx_acc += float32(2.0) * CXF[o] * v
            fy_acc += float32(2.0) * CYF[o] * v
            fz_acc += float32(2.0) * CZF[o] * v
        else:
            fl[i] = f[i, sz, sy, sx]
    rho = float32(0.0)
    mx = float32(0.0)
    my = float32(0.0)
    mz = float32(0.0)
    for i in range(19):
        rho += fl[i]
        mx += CXF[i] * fl[i]
        my += CYF[i] * fl[i]
        mz += CZF[i] * fl[i]
    ux = mx / rho
    uy = my / rho
    uz = mz / rho
    usq = float32(1.5) * (ux * ux + uy * uy + uz * uz)
    # Non-equilibrium momentum flux drives the Smagorinsky eddy viscosity.
    pxx = float32(0.0)
    pyy = float32(0.0)
    pzz = float32(0.0)
    pxy = float32(0.0)
    pxz = float32(0.0)
    pyz = float32(0.0)
    for i in range(19):
        cu = float32(3.0) * (CXF[i] * ux + CYF[i] * uy + CZF[i] * uz)
        feq = W32[i] * rho * (float32(1.0) + cu + float32(0.5) * cu * cu - usq)
        d = fl[i] - feq
        pxx += CXF[i] * CXF[i] * d
        pyy += CYF[i] * CYF[i] * d
        pzz += CZF[i] * CZF[i] * d
        pxy += CXF[i] * CYF[i] * d
        pxz += CXF[i] * CZF[i] * d
        pyz += CYF[i] * CZF[i] * d
    q = math.sqrt(float32(2.0) * (pxx * pxx + pyy * pyy + pzz * pzz)
                  + float32(4.0) * (pxy * pxy + pxz * pxz + pyz * pyz))
    tau = 0.5 * (tau0 + math.sqrt(tau0 * tau0 + 18.0 * cs2 * q / rho))
    omega = float32(1.0 / tau)
    for i in range(19):
        cu = float32(3.0) * (CXF[i] * ux + CYF[i] * uy + CZF[i] * uz)
        feq = W32[i] * rho * (float32(1.0) + cu + float32(0.5) * cu * cu - usq)
        fn[i, z, y, x] = fl[i] + omega * (feq - fl[i])
    if store:
        rho_out[z, y, x] = rho
        ux_out[z, y, x] = ux
        uy_out[z, y, x] = uy
        uz_out[z, y, x] = uz
    if fx_acc != 0.0 or fy_acc != 0.0 or fz_acc != 0.0:
        cuda.atomic.add(force, 0, fx_acc)
        cuda.atomic.add(force, 1, fy_acc)
        cuda.atomic.add(force, 2, fz_acc)


@cuda.jit(cache=True)
def _boundaries(fn, inflow):
    x, y, z = cuda.grid(3)
    nz = fn.shape[1]
    ny = fn.shape[2]
    nx = fn.shape[3]
    if x >= nx or y >= ny or z >= nz:
        return
    if x == 0 or y == 0 or y == ny - 1 or z == 0 or z == nz - 1:
        for i in range(19):
            fn[i, z, y, x] = inflow[i]
    elif x == nx - 1:
        for i in range(19):
            fn[i, z, y, x] = fn[i, z, y, x - 1]


@cuda.jit(cache=True)
def _fill(f, values):
    x, y, z = cuda.grid(3)
    if x < f.shape[3] and y < f.shape[2] and z < f.shape[1]:
        for i in range(19):
            f[i, z, y, x] = values[i]


@cuda.jit(cache=True)
def _free_cells(f, old, new):
    """Cells that turn from solid to fluid start at rest."""
    x, y, z = cuda.grid(3)
    if x < f.shape[3] and y < f.shape[2] and z < f.shape[1]:
        if old[z, y, x] and not new[z, y, x]:
            for i in range(19):
                f[i, z, y, x] = W32[i]


@cuda.jit(device=True)
def _diff(a, z, y, x, axis):
    """Central difference of a along an array axis, one-sided at the edges like np.gradient."""
    if axis == 0:
        lo = max(z - 1, 0)
        hi = min(z + 1, a.shape[0] - 1)
        return (a[hi, y, x] - a[lo, y, x]) / (hi - lo)
    if axis == 1:
        lo = max(y - 1, 0)
        hi = min(y + 1, a.shape[1] - 1)
        return (a[z, hi, x] - a[z, lo, x]) / (hi - lo)
    lo = max(x - 1, 0)
    hi = min(x + 1, a.shape[2] - 1)
    return (a[z, y, hi] - a[z, y, lo]) / (hi - lo)


@cuda.jit(cache=True)
def _slice(rho, ux, uy, uz, crossflow, index, kind, cp_scale, out):
    a, b = cuda.grid(2)
    if a >= out.shape[1] or b >= out.shape[0]:
        return
    if crossflow:  # out[z, y] on the plane x = index
        z, y, x = b, a, index
    else:  # out[y, x] on the plane z = index
        z, y, x = index, b, a
    if kind == 0:
        u = ux[z, y, x]
        v = uy[z, y, x]
        w = uz[z, y, x]
        out[b, a] = math.sqrt(u * u + v * v + w * w)
    elif kind == 2:
        out[b, a] = (rho[z, y, x] - 1.0) * cp_scale
    elif crossflow:
        out[b, a] = _diff(uz, z, y, x, 1) - _diff(uy, z, y, x, 0)
    else:
        out[b, a] = _diff(uy, z, y, x, 2) - _diff(ux, z, y, x, 1)


@cuda.jit(cache=True)
def _q_criterion(ux, uy, uz, solid, stride, out):
    i, j, k = cuda.grid(3)
    if i >= out.shape[2] or j >= out.shape[1] or k >= out.shape[0]:
        return
    x, y, z = i * stride, j * stride, k * stride
    if solid[z, y, x]:
        out[k, j, i] = 0.0
        return
    g = cuda.local.array((3, 3), float32)
    for d in range(3):
        axis = 2 - d  # array axis of x, y, z
        g[0, d] = _diff(ux, z, y, x, axis)
        g[1, d] = _diff(uy, z, y, x, axis)
        g[2, d] = _diff(uz, z, y, x, axis)
    q = float32(0.0)
    for a in range(3):
        for b in range(3):
            q -= float32(0.5) * g[a, b] * g[b, a]
    out[k, j, i] = q


@cuda.jit(cache=True)
def _advect(pos, release, ux, uy, uz, solid, dt, spread, seed):
    k = cuda.grid(1)
    if k >= pos.shape[0]:
        return
    nz = ux.shape[0]
    ny = ux.shape[1]
    nx = ux.shape[2]
    # Keep each particle move under about half a cell so streaks stay smooth.
    substeps = max(1, int(dt * 0.2) + 1)
    h = dt / substeps
    x = pos[k, 0]
    y = pos[k, 1]
    z = pos[k, 2]
    for _ in range(substeps):
        if not (0 <= x < nx - 1 and 0 <= y < ny - 1 and 0 <= z < nz - 1):
            break
        i, j, l = int(x), int(y), int(z)
        fx, fy, fz = x - i, y - j, z - l
        u = float32(0.0)
        v = float32(0.0)
        w = float32(0.0)
        # Trilinear interpolation from the eight surrounding cells.
        for c in range(8):
            di, dj, dl = c & 1, (c >> 1) & 1, (c >> 2) & 1
            wt = ((fx if di else 1 - fx) * (fy if dj else 1 - fy)
                  * (fz if dl else 1 - fz))
            u += wt * ux[l + dl, j + dj, i + di]
            v += wt * uy[l + dl, j + dj, i + di]
            w += wt * uz[l + dl, j + dj, i + di]
        x += u * h
        y += v * h
        z += w * h
    if (x < 1 or x >= nx - 2 or y < 1 or y >= ny - 2 or z < 1 or z >= nz - 2
            or solid[int(z), int(y), int(x)]):
        # Re-release from the smoke rake at the inlet, spread over the distance the
        # free stream covers in dt so the streaks stay continuous.
        x = 1.0 + spread * _jitter(k, seed)
        y = release[k, 1]
        z = release[k, 2]
    pos[k, 0] = x
    pos[k, 1] = y
    pos[k, 2] = z


class GpuTunnel3D(Tunnel3D):
    """Same tunnel and API as Tunnel3D, with everything on an NVIDIA GPU."""

    device = "gpu"
    device_label = "GPU"

    def _allocate(self):
        self.device_label = f"GPU ({gpu_name()})"
        shape = (self.nz, self.ny, self.nx)
        self._grid = _blocks(shape[::-1], BLOCK)
        self._d_f = cuda.device_array((19,) + shape, dtype=np.float32)
        self._d_fn = cuda.device_array((19,) + shape, dtype=np.float32)
        self._d_inflow = cuda.to_device(self._inflow.astype(np.float32))
        _fill[self._grid, BLOCK](self._d_f, self._d_inflow)
        _fill[self._grid, BLOCK](self._d_fn, self._d_inflow)
        self._d_solid = cuda.to_device(self.solid)
        self._d_rho = cuda.to_device(np.ones(shape, dtype=np.float32))
        self._d_ux = cuda.to_device(np.full(shape, self.u, dtype=np.float32))
        self._d_uy = cuda.to_device(np.zeros(shape, dtype=np.float32))
        self._d_uz = cuda.to_device(np.zeros(shape, dtype=np.float32))
        self._d_force = cuda.to_device(np.zeros(3, dtype=np.float64))
        self._d_zero = cuda.to_device(np.zeros(3, dtype=np.float64))
        # Steps run on their own stream so start() can return while the card works.
        self._stream = cuda.stream()
        self._events = (cuda.event(), cuda.event())
        self._h_force = cuda.pinned_array(3, dtype=np.float64)

    def set_solid(self, mask):
        mask = self._interior(mask)
        d_mask = cuda.to_device(mask)
        _free_cells[self._grid, BLOCK](self._d_f, self._d_solid, d_mask)
        self._d_solid = d_mask
        self.solid = mask
        self.span = int(mask.any(axis=(1, 2)).sum())

    def step(self, n=1):
        self.start(n)
        self.finish()
        return self.force

    def start(self, n=1):
        s = self._stream
        self._d_force.copy_to_device(self._d_zero, stream=s)
        tau0, cs2 = np.float32(self.tau), np.float32(self.cs2)
        self._events[0].record(s)
        for k in range(n):
            _stream_collide[self._grid, BLOCK, s](self._d_f, self._d_fn, self._d_solid, tau0, cs2,
                                                  k == n - 1, self._d_rho, self._d_ux,
                                                  self._d_uy, self._d_uz, self._d_force)
            _boundaries[self._grid, BLOCK, s](self._d_fn, self._d_inflow)
            self._d_f, self._d_fn = self._d_fn, self._d_f
        self._events[1].record(s)
        self._d_force.copy_to_host(self._h_force, stream=s)
        self._pending = n

    def finish(self):
        n, self._pending = self._pending, 0
        if n:
            self._stream.synchronize()
            self.batch_ms = self._events[0].elapsed_time(self._events[1])
            self.steps += n
            self.force = tuple(self._h_force / n)
        return n

    def fields(self):
        return tuple(a.copy_to_host() for a in (self._d_rho, self._d_ux, self._d_uy, self._d_uz))

    def slice(self, kind, axis, index):
        crossflow = axis == "x"
        shape = (self.nz, self.ny) if crossflow else (self.ny, self.nx)
        out = cuda.device_array(shape, dtype=np.float32)
        grid = _blocks(shape[::-1], BLOCK_2D)
        cp_scale = np.float32(1.0 / 3.0 / (0.5 * self.u**2))
        _slice[grid, BLOCK_2D](self._d_rho, self._d_ux, self._d_uy, self._d_uz, crossflow,
                               int(index), SLICE_KINDS.index(kind), cp_scale, out)
        return out.copy_to_host()

    def q_criterion(self, stride=2):
        shape = tuple((n + stride - 1) // stride for n in self.solid.shape)
        out = cuda.device_array(shape, dtype=np.float32)
        grid = _blocks(shape[::-1], BLOCK)
        _q_criterion[grid, BLOCK](self._d_ux, self._d_uy, self._d_uz, self._d_solid,
                                  stride, out)
        return out.copy_to_host()

    def set_particles(self, pos, release):
        self._d_pos = cuda.to_device(np.asarray(pos, dtype=np.float32).reshape(-1, 3))
        self._d_release = cuda.to_device(np.asarray(release, dtype=np.float32).reshape(-1, 3))

    def advect(self, dt):
        n = self._d_pos.shape[0]
        if n:
            _advect[_blocks((n,), (BLOCK_1D,)), BLOCK_1D](
                self._d_pos, self._d_release, self._d_ux, self._d_uy, self._d_uz,
                self._d_solid, np.float32(dt), np.float32(self._spread(dt)), self.steps)
        return self._d_pos.copy_to_host()
