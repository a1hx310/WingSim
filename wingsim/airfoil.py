"""NACA 4-digit airfoil geometry and rasterization onto the simulation grid."""

import numpy as np


def parse_naca4(code):
    """Return (max camber, camber position, thickness) as chord fractions."""
    code = code.strip().upper().removeprefix("NACA").strip()
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"expected a 4-digit NACA code, got {code!r}")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    if t <= 0:
        raise ValueError("thickness must be greater than zero")
    if m > 0 and p == 0:
        raise ValueError("a cambered airfoil needs a camber position (second digit)")
    return m, p, t


def naca4_surfaces(code, n=200):
    """Upper and lower surface points (x, y) in chord units, leading edge at the origin."""
    m, p, t = parse_naca4(code)
    beta = np.linspace(0.0, np.pi, n)
    xc = 0.5 * (1.0 - np.cos(beta))  # cosine spacing clusters points at the edges
    # Closed trailing edge variant (last coefficient -0.1036).
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2
                  + 0.2843 * xc**3 - 0.1036 * xc**4)
    if m == 0:
        yc = np.zeros_like(xc)
        dyc = np.zeros_like(xc)
    else:
        front = xc < p
        yc = np.where(front, m / p**2 * (2 * p * xc - xc**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xc - xc**2))
        dyc = np.where(front, 2 * m / p**2 * (p - xc),
                       2 * m / (1 - p)**2 * (p - xc))
    theta = np.arctan(dyc)
    upper = np.stack([xc - yt * np.sin(theta), yc + yt * np.cos(theta)], axis=1)
    lower = np.stack([xc + yt * np.sin(theta), yc - yt * np.cos(theta)], axis=1)
    return upper, lower


def outline(code, chord, aoa_deg, pivot, n=200):
    """Closed outline in grid coordinates, rotated nose-up by aoa_deg about the quarter chord."""
    upper, lower = naca4_surfaces(code, n)
    pts = np.concatenate([upper, lower[::-1][1:]]) - [0.25, 0.0]
    a = np.radians(aoa_deg)
    # Nose-up for flow in +x means rotating clockwise in a y-up frame.
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return pts @ rot.T * chord + pivot


def rasterize(code, chord, aoa_deg, pivot, shape):
    """Boolean mask (ny, nx) of grid cells whose centres lie inside the airfoil."""
    ny, nx = shape
    upper, lower = naca4_surfaces(code, 400)
    ys, xs = np.mgrid[0:ny, 0:nx].astype(np.float64)
    a = np.radians(aoa_deg)
    dx, dy = xs - pivot[0], ys - pivot[1]
    # Inverse rotation takes grid points back into the airfoil's own frame.
    xa = (np.cos(a) * dx - np.sin(a) * dy) / chord + 0.25
    ya = (np.sin(a) * dx + np.cos(a) * dy) / chord
    inside = (xa >= upper[:, 0].min()) & (xa <= 1.0)
    yu = np.interp(xa, *_monotonic(upper))
    yl = np.interp(xa, *_monotonic(lower))
    return inside & (ya <= yu) & (ya >= yl)


def _monotonic(surface):
    order = np.argsort(surface[:, 0], kind="stable")
    return surface[order, 0], surface[order, 1]
