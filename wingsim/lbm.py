"""D2Q9 Lattice Boltzmann solver for flow past a 2D body.

The fluid is a weakly compressible lattice gas that recovers the incompressible
Navier-Stokes equations at low Mach number. Collision is BGK with a Smagorinsky
subgrid model so the solver stays stable at higher Reynolds numbers. Walls use
half-way bounce-back, and the force on the body comes from momentum exchange
across every fluid-solid link.

Everything here is in lattice units: one cell is one unit of length, one step is
one unit of time, and the free-stream density is 1.
"""

import numpy as np
from numba import njit, prange

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


def equilibrium(rho, ux, uy):
    """Equilibrium populations for array (or scalar) density and velocity."""
    rho, ux, uy = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (rho, ux, uy)))
    usq = 1.5 * (ux**2 + uy**2)
    feq = np.empty((9,) + rho.shape)
    for i in range(9):
        cu = 3.0 * (CX[i] * ux + CY[i] * uy)
        feq[i] = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
    return feq


@njit(parallel=True, fastmath=True, cache=True)
def _stream_collide(f, fn, solid, tau0, cs2, rho_out, ux_out, uy_out, fx_rows, fy_rows):
    nq, ny, nx = f.shape
    for y in prange(1, ny - 1):
        fl = np.empty(9)
        fx_acc = 0.0
        fy_acc = 0.0
        for x in range(1, nx - 1):
            if solid[y, x]:
                for i in range(9):
                    fn[i, y, x] = W[i]
                rho_out[y, x] = 1.0
                ux_out[y, x] = 0.0
                uy_out[y, x] = 0.0
                continue
            # Pull streaming, with half-way bounce-back off solid neighbours.
            for i in range(9):
                sx = x - CX[i]
                sy = y - CY[i]
                if solid[sy, sx]:
                    o = OPP[i]
                    v = f[o, y, x]
                    fl[i] = v
                    # The population left with velocity c_o and came back with -c_o.
                    fx_acc += 2.0 * CX[o] * v
                    fy_acc += 2.0 * CY[o] * v
                else:
                    fl[i] = f[i, sy, sx]
            rho = 0.0
            mx = 0.0
            my = 0.0
            for i in range(9):
                rho += fl[i]
                mx += CX[i] * fl[i]
                my += CY[i] * fl[i]
            ux = mx / rho
            uy = my / rho
            usq = 1.5 * (ux * ux + uy * uy)
            # Non-equilibrium momentum flux drives the Smagorinsky eddy viscosity.
            pxx = 0.0
            pyy = 0.0
            pxy = 0.0
            for i in range(9):
                cu = 3.0 * (CX[i] * ux + CY[i] * uy)
                feq = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
                d = fl[i] - feq
                pxx += CX[i] * CX[i] * d
                pyy += CY[i] * CY[i] * d
                pxy += CX[i] * CY[i] * d
            q = np.sqrt(2.0 * (pxx * pxx + pyy * pyy + 2.0 * pxy * pxy))
            tau = 0.5 * (tau0 + np.sqrt(tau0 * tau0 + 18.0 * cs2 * q / rho))
            omega = 1.0 / tau
            for i in range(9):
                cu = 3.0 * (CX[i] * ux + CY[i] * uy)
                feq = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
                fn[i, y, x] = fl[i] + omega * (feq - fl[i])
            rho_out[y, x] = rho
            ux_out[y, x] = ux
            uy_out[y, x] = uy
        fx_rows[y] = fx_acc
        fy_rows[y] = fy_acc


class Tunnel:
    """A 2D wind tunnel: uniform inflow on the left, open outlet on the right,
    free-stream far field on the top and bottom, and a solid body in between."""

    device = "cpu"
    device_label = "CPU"

    def __init__(self, nx=520, ny=220, u_lattice=0.08, smagorinsky=0.14):
        self.nx, self.ny = nx, ny
        self.u = u_lattice  # inflow speed in lattice units; keep well below 0.3 (Mach limit)
        self.cs2 = smagorinsky**2
        self.solid = np.zeros((ny, nx), dtype=np.bool_)
        self.f = equilibrium(np.ones((ny, nx)), self.u, 0.0)
        self.fn = self.f.copy()
        self.rho = np.ones((ny, nx))
        self.ux = np.full((ny, nx), self.u)
        self.uy = np.zeros((ny, nx))
        self._fx = np.zeros(ny)
        self._fy = np.zeros(ny)
        self._inflow = equilibrium(1.0, self.u, 0.0)[:, None]
        self.force = (0.0, 0.0)
        self.steps = 0
        self.length = 1.0
        self.set_reynolds(100.0)

    def set_reynolds(self, re, length=None):
        """Set viscosity so the flow has Reynolds number re based on the given length (cells)."""
        if length is not None:
            self.length = float(length)
        self.re = float(re)
        nu = self.u * self.length / self.re
        self.tau = 3.0 * nu + 0.5

    def set_solid(self, mask):
        """Replace the body. Cells that turn from solid to fluid start at rest."""
        mask = np.asarray(mask, dtype=np.bool_).copy()
        mask[0, :] = mask[-1, :] = False
        mask[:, 0] = mask[:, -1] = False
        freed = self.solid & ~mask
        if freed.any():
            rest = equilibrium(1.0, 0.0, 0.0)
            self.f[:, freed] = rest[:, None]
        self.solid = mask

    def step(self, n=1):
        """Advance n time steps. self.force holds the mean body force over those steps."""
        fx = fy = 0.0
        for _ in range(n):
            _stream_collide(self.f, self.fn, self.solid, self.tau, self.cs2,
                            self.rho, self.ux, self.uy, self._fx, self._fy)
            fn = self.fn
            fn[:, :, 0] = self._inflow
            fn[:, 0, :] = self._inflow
            fn[:, -1, :] = self._inflow
            fn[:, 1:-1, -1] = fn[:, 1:-1, -2]
            self.f, self.fn = fn, self.f
            fx += self._fx.sum()
            fy += self._fy.sum()
        self.steps += n
        self.force = (fx / n, fy / n)
        return self.force

    def coefficients(self):
        """Drag and lift coefficients for the current force, per unit span."""
        q = 0.5 * self.u**2 * self.length
        return self.force[0] / q, self.force[1] / q

    def vorticity(self):
        return np.gradient(self.uy, axis=1) - np.gradient(self.ux, axis=0)

    def pressure_coefficient(self):
        # Lattice pressure is rho * cs^2 with cs^2 = 1/3.
        return (self.rho - 1.0) / 3.0 / (0.5 * self.u**2)
