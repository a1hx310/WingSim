"""D3Q19 Lattice Boltzmann solver for flow past a 3D body, such as a finite wing.

The physics is the same as the 2D solver in lbm.py, on a 3D lattice: BGK collision
with a Smagorinsky subgrid model, half-way bounce-back walls, and the body force from
momentum exchange across every fluid-solid link. Axes are x downstream, y up and z
along the span. Everything is in lattice units, with free-stream density 1.
"""

import numpy as np
from numba import njit, prange

from .lbm import Tunnel, _jitter

# Rest, the six faces, then the twelve edges, with each velocity next to its opposite.
CX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=np.int64)
CY = np.array([0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.int64)
CZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1], dtype=np.int64)
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int64)
SLICE_KINDS = ("speed", "vorticity", "pressure")


def equilibrium(rho, ux, uy, uz):
    """Equilibrium populations for array (or scalar) density and velocity."""
    rho, ux, uy, uz = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (rho, ux, uy, uz)))
    usq = 1.5 * (ux**2 + uy**2 + uz**2)
    feq = np.empty((19,) + rho.shape)
    for i in range(19):
        cu = 3.0 * (CX[i] * ux + CY[i] * uy + CZ[i] * uz)
        feq[i] = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
    return feq


@njit(parallel=True, fastmath=True, cache=True)
def _stream_collide(f, fn, solid, tau0, cs2, store, rho_out, ux_out, uy_out, uz_out, f_planes):
    # Density and velocity are only read between frames, so they are stored only when
    # store is set, which saves memory traffic on the other steps.
    nq, nz, ny, nx = f.shape
    for z in prange(1, nz - 1):
        fl = np.empty(19)
        fx_acc = 0.0
        fy_acc = 0.0
        fz_acc = 0.0
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                if solid[z, y, x]:
                    for i in range(19):
                        fn[i, z, y, x] = W[i]
                    if store:
                        rho_out[z, y, x] = 1.0
                        ux_out[z, y, x] = 0.0
                        uy_out[z, y, x] = 0.0
                        uz_out[z, y, x] = 0.0
                    continue
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
                        fx_acc += 2.0 * CX[o] * v
                        fy_acc += 2.0 * CY[o] * v
                        fz_acc += 2.0 * CZ[o] * v
                    else:
                        fl[i] = f[i, sz, sy, sx]
                rho = 0.0
                mx = 0.0
                my = 0.0
                mz = 0.0
                for i in range(19):
                    rho += fl[i]
                    mx += CX[i] * fl[i]
                    my += CY[i] * fl[i]
                    mz += CZ[i] * fl[i]
                ux = mx / rho
                uy = my / rho
                uz = mz / rho
                usq = 1.5 * (ux * ux + uy * uy + uz * uz)
                # Non-equilibrium momentum flux drives the Smagorinsky eddy viscosity.
                pxx = 0.0
                pyy = 0.0
                pzz = 0.0
                pxy = 0.0
                pxz = 0.0
                pyz = 0.0
                for i in range(19):
                    cu = 3.0 * (CX[i] * ux + CY[i] * uy + CZ[i] * uz)
                    feq = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
                    d = fl[i] - feq
                    pxx += CX[i] * CX[i] * d
                    pyy += CY[i] * CY[i] * d
                    pzz += CZ[i] * CZ[i] * d
                    pxy += CX[i] * CY[i] * d
                    pxz += CX[i] * CZ[i] * d
                    pyz += CY[i] * CZ[i] * d
                q = np.sqrt(2.0 * (pxx * pxx + pyy * pyy + pzz * pzz)
                            + 4.0 * (pxy * pxy + pxz * pxz + pyz * pyz))
                tau = 0.5 * (tau0 + np.sqrt(tau0 * tau0 + 18.0 * cs2 * q / rho))
                omega = 1.0 / tau
                for i in range(19):
                    cu = 3.0 * (CX[i] * ux + CY[i] * uy + CZ[i] * uz)
                    feq = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
                    fn[i, z, y, x] = fl[i] + omega * (feq - fl[i])
                if store:
                    rho_out[z, y, x] = rho
                    ux_out[z, y, x] = ux
                    uy_out[z, y, x] = uy
                    uz_out[z, y, x] = uz
        f_planes[0, z] = fx_acc
        f_planes[1, z] = fy_acc
        f_planes[2, z] = fz_acc


@njit(cache=True)
def _advect(pos, release, ux, uy, uz, solid, dt, spread, seed):
    nz, ny, nx = ux.shape
    # Keep each particle move under about half a cell so streaks stay smooth.
    substeps = max(1, int(dt * 0.2) + 1)
    h = dt / substeps
    for k in range(pos.shape[0]):
        x, y, z = pos[k, 0], pos[k, 1], pos[k, 2]
        for _ in range(substeps):
            if not (0 <= x < nx - 1 and 0 <= y < ny - 1 and 0 <= z < nz - 1):
                break
            i, j, l = int(x), int(y), int(z)
            fx, fy, fz = x - i, y - j, z - l
            u = 0.0
            v = 0.0
            w = 0.0
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
        pos[k, 0], pos[k, 1], pos[k, 2] = x, y, z


class Tunnel3D:
    """A 3D wind tunnel: uniform inflow on the left, open outlet on the right,
    free-stream far field on the four side faces, and a solid body in between."""

    device = "cpu"
    device_label = "CPU"

    def __init__(self, nx=224, ny=112, nz=176, u_lattice=0.08, smagorinsky=0.14):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.u = u_lattice  # inflow speed in lattice units; keep well below 0.3 (Mach limit)
        self.cs2 = smagorinsky**2
        self.solid = np.zeros((nz, ny, nx), dtype=np.bool_)
        self._inflow = equilibrium(1.0, self.u, 0.0, 0.0)
        self.force = (0.0, 0.0, 0.0)
        self.steps = 0
        self.batch_ms = 0.0  # how long the last start() batch took to compute
        self._pending = 0
        self.length = 1.0
        self.span = 0  # span layers the body fills, for the planform area
        self._allocate()
        self.set_particles(np.zeros((0, 3)), np.zeros((0, 3)))
        self.set_reynolds(100.0)

    def _allocate(self):
        shape = (self.nz, self.ny, self.nx)
        self.f = np.empty((19,) + shape)
        self.f[:] = self._inflow[:, None, None, None]
        self.fn = self.f.copy()
        self.rho = np.ones(shape)
        self.ux = np.full(shape, self.u)
        self.uy = np.zeros(shape)
        self.uz = np.zeros(shape)
        self._f_planes = np.zeros((3, self.nz))

    set_reynolds = Tunnel.set_reynolds
    start = Tunnel.start
    finish = Tunnel.finish
    set_particles = Tunnel.set_particles
    _spread = Tunnel._spread

    @staticmethod
    def _interior(mask):
        """Copy of a body mask with the boundary shell cleared."""
        mask = np.asarray(mask, dtype=np.bool_).copy()
        mask[[0, -1], :, :] = False
        mask[:, [0, -1], :] = False
        mask[:, :, [0, -1]] = False
        return mask

    def set_solid(self, mask):
        """Replace the body. Cells that turn from solid to fluid start at rest."""
        mask = self._interior(mask)
        freed = self.solid & ~mask
        if freed.any():
            self.f[:, freed] = equilibrium(1.0, 0.0, 0.0, 0.0)[:, None]
        self.solid = mask
        self.span = int(mask.any(axis=(1, 2)).sum())

    def step(self, n=1):
        """Advance n time steps. self.force holds the mean body force over those steps."""
        total = np.zeros(3)
        inflow = self._inflow[:, None, None]
        for k in range(n):
            _stream_collide(self.f, self.fn, self.solid, self.tau, self.cs2, k == n - 1,
                            self.rho, self.ux, self.uy, self.uz, self._f_planes)
            fn = self.fn
            fn[:, :, :, 0] = inflow
            fn[:, :, 0, :] = inflow
            fn[:, :, -1, :] = inflow
            fn[:, 0, :, :] = inflow
            fn[:, -1, :, :] = inflow
            fn[:, 1:-1, 1:-1, -1] = fn[:, 1:-1, 1:-1, -2]
            self.f, self.fn = fn, self.f
            total += self._f_planes.sum(axis=1)
        self.steps += n
        self.force = tuple(total / n)
        return self.force

    def coefficients(self):
        """Drag and lift coefficients for the current force, on the planform area: the
        reference length from set_reynolds (the chord) times the span."""
        q = 0.5 * self.u**2 * self.length * max(self.span, 1)
        return self.force[0] / q, self.force[1] / q

    def fields(self):
        """Density and the three velocity components, as (nz, ny, nx) host arrays."""
        return self.rho, self.ux, self.uy, self.uz

    def slice(self, kind, axis, index):
        """A plane of the flow: axis "z" gives a span station as (ny, nx), axis "x" a
        crossflow plane as (nz, ny). kind is "speed", "pressure" (as a pressure
        coefficient) or "vorticity" (the component normal to the plane)."""
        pick = (lambda a: a[index]) if axis == "z" else (lambda a: a[:, :, index])
        rho, ux, uy, uz = (pick(a) for a in self.fields())
        if kind == "speed":
            return np.sqrt(ux**2 + uy**2 + uz**2)
        if kind == "pressure":
            # Lattice pressure is rho * cs^2 with cs^2 = 1/3.
            return (rho - 1.0) / 3.0 / (0.5 * self.u**2)
        if axis == "z":
            return np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
        return np.gradient(uz, axis=1) - np.gradient(uy, axis=0)

    def q_criterion(self, stride=2):
        """Q-criterion (positive inside vortex cores) on every stride-th cell, zero in
        the body. Q = (|rotation|^2 - |strain|^2) / 2 = -tr(grad u . grad u) / 2."""
        _, *vel = self.fields()
        # grad[i][j] is d u_i / d x_j, with the axes ordered (z, y, x).
        grad = [np.gradient(c) for c in vel]
        axis = [2, 1, 0]  # array axis of x, y, z
        q = np.zeros(self.solid.shape)
        for i in range(3):
            for j in range(3):
                q -= 0.5 * grad[i][axis[j]] * grad[j][axis[i]]
        q[self.solid] = 0.0
        return q[::stride, ::stride, ::stride]

    def advect(self, dt):
        """Move the smoke particles with the flow for dt steps; returns their positions."""
        _advect(self._pos, self._release, self.ux, self.uy, self.uz, self.solid,
                float(dt), self._spread(dt), self.steps)
        return self._pos
