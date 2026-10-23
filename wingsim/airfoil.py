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


def naca4_vertices(code, n=11):
    """Editable vertices for a section: a closed loop in chord units that starts at the
    trailing edge, runs forward over the upper surface and back along the lower one."""
    upper, lower = naca4_surfaces(code, n)
    return np.concatenate([upper[::-1], lower[1:-1]])


def smooth_outline(vertices, samples=12):
    """Closed curve through the vertices, sharp at vertex 0 (the trailing edge) and
    smooth elsewhere. It is a centripetal Catmull-Rom spline, which never overshoots
    into loops however unevenly the vertices are spaced."""
    p = np.asarray(vertices, dtype=np.float64)
    q = np.concatenate([p, p[:1]])
    # Reflected end points leave the two curve ends free, so they meet at a corner.
    q = np.concatenate([2 * q[:1] - q[1:2], q, 2 * q[-1:] - q[-2:-1]])
    d = np.linalg.norm(np.diff(q, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(np.maximum(d, 1e-9) ** 0.5)])
    s = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    out = []
    for k in range(1, len(q) - 2):
        t0, t1, t2, t3 = t[k - 1:k + 3]
        p0, p1, p2, p3 = q[k - 1:k + 3]
        u = t1 + s * (t2 - t1)
        # Barry-Goldman pyramid for the segment from p1 to p2.
        a1 = ((t1 - u) * p0 + (u - t0) * p1) / (t1 - t0)
        a2 = ((t2 - u) * p1 + (u - t1) * p2) / (t2 - t1)
        a3 = ((t3 - u) * p2 + (u - t2) * p3) / (t3 - t2)
        b1 = ((t2 - u) * a1 + (u - t0) * a2) / (t2 - t0)
        b2 = ((t3 - u) * a2 + (u - t1) * a3) / (t3 - t1)
        out.append(((t2 - u) * b1 + (u - t1) * b2) / (t2 - t1))
    return np.concatenate(out)


def _rotation(aoa_deg):
    a = np.radians(aoa_deg)
    # Nose-up for flow in +x means rotating clockwise in a y-up frame.
    return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def place(pts, chord, aoa_deg, pivot):
    """Chord-unit points (leading edge at the origin) to grid coordinates, scaled by
    chord and rotated nose-up by aoa_deg about the quarter chord at pivot."""
    return (np.asarray(pts) - [0.25, 0.0]) @ _rotation(aoa_deg).T * chord + pivot


def unplace(pts, chord, aoa_deg, pivot):
    """Inverse of place: grid coordinates back to chord units."""
    return (np.asarray(pts) - pivot) / chord @ _rotation(aoa_deg) + [0.25, 0.0]


def outline(code, chord, aoa_deg, pivot, n=200):
    """Closed outline in grid coordinates, rotated nose-up by aoa_deg about the quarter chord."""
    upper, lower = naca4_surfaces(code, n)
    return place(np.concatenate([upper, lower[::-1][1:]]), chord, aoa_deg, pivot)


def rasterize(code, chord, aoa_deg, pivot, shape):
    """Boolean mask (ny, nx) of grid cells whose centres lie inside the airfoil."""
    return rasterize_polygon(outline(code, chord, aoa_deg, pivot, 400), shape)


def rasterize_polygon(poly, shape):
    """Boolean mask (ny, nx) of grid cells whose centres lie inside a closed polygon
    given in grid coordinates (even-odd rule, so a self-crossing outline still works)."""
    ny, nx = shape
    mask = np.zeros(shape, dtype=np.bool_)
    ax, ay = poly[:, 0], poly[:, 1]
    bx, by = np.roll(ax, -1), np.roll(ay, -1)
    y0 = max(int(np.ceil(ay.min())), 0)
    y1 = min(int(np.floor(ay.max())) + 1, ny)
    for j in range(y0, y1):
        # Scanline fill between successive pairs of edge crossings on this row.
        c = (ay > j) != (by > j)
        xc = np.sort(ax[c] + (j - ay[c]) * (bx[c] - ax[c]) / (by[c] - ay[c]))
        for left, right in zip(xc[::2], xc[1::2]):
            mask[j, max(int(np.ceil(left)), 0):max(min(int(np.ceil(right)), nx), 0)] = True
    return mask


def extrude(mask, nz, z0, z1):
    """Boolean mask (nz, ny, nx) of a straight wing: the section mask (ny, nx) in every
    span layer whose cell centre lies between z0 and z1."""
    out = np.zeros((nz,) + mask.shape, dtype=np.bool_)
    out[max(int(np.ceil(z0)), 0):max(min(int(np.ceil(z1)), nz), 0)] = mask
    return out


def signed_area(poly):
    """Area of a closed polygon (n, 2), positive when it runs counter-clockwise."""
    x, y = np.asarray(poly, dtype=np.float64).T
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _cross(o, a, b):
    """z of (a - o) x (b - o); positive when b is left of the line from o to a.
    b may be an array of points."""
    return (a[0] - o[0]) * (b[..., 1] - o[1]) - (a[1] - o[1]) * (b[..., 0] - o[0])


def _triangulate(poly):
    """Triangles (index triples) covering a simple polygon given counter-clockwise,
    by ear clipping. A self-crossing outline falls back to a fan for what is left."""
    idx = list(range(len(poly)))
    tris = []
    k = 0
    misses = 0
    while len(idx) > 3 and misses < len(idx):
        a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
        pa, pb, pc = poly[a], poly[b], poly[c]
        ear = _cross(pa, pb, pc) > 0
        if ear:
            # No other vertex may lie inside the candidate triangle.
            rest = poly[[i for i in idx if i not in (a, b, c)]]
            ear = not ((_cross(pa, pb, rest) >= 0) & (_cross(pb, pc, rest) >= 0)
                       & (_cross(pc, pa, rest) >= 0)).any()
        if ear:
            tris.append((a, b, c))
            idx.pop(k)
            k %= len(idx)
            misses = 0
        else:
            k = (k + 1) % len(idx)
            misses += 1
    tris += [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
    return np.array(tris, dtype=np.int64)


def wing_mesh(outline, z0, z1):
    """Closed triangle mesh (vertices (2n, 3), faces) of a straight wing with the given
    section outline (n, 2) in grid coordinates, spanning z0 to z1. Faces wind
    counter-clockwise seen from outside, so their normals point out of the wing."""
    pts = np.asarray(outline, dtype=np.float64)
    if signed_area(pts) < 0:
        pts = pts[::-1]
    n = len(pts)
    verts = np.concatenate([np.c_[pts, np.full(n, z0)], np.c_[pts, np.full(n, z1)]])
    k = np.arange(n)
    k1 = (k + 1) % n
    side = np.concatenate([np.c_[k, k1, n + k1], np.c_[k, n + k1, n + k]])
    cap = _triangulate(pts)
    faces = np.concatenate([side, cap[:, ::-1], cap + n])
    return verts, faces
