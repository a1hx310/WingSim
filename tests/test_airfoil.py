import numpy as np
import pytest

from wingsim import airfoil


def test_parse_naca4():
    assert airfoil.parse_naca4("2412") == pytest.approx((0.02, 0.4, 0.12))
    assert airfoil.parse_naca4("NACA 0012") == pytest.approx((0.0, 0.0, 0.12))
    for bad in ["12", "abcd", "0000", "2012"]:
        with pytest.raises(ValueError):
            airfoil.parse_naca4(bad)


def test_mask_area_matches_section_area():
    # A NACA 00xx section has area of about 0.685 * t * c^2.
    chord = 100
    mask = airfoil.rasterize("0012", chord, 0.0, (100.0, 60.0), (120, 240))
    assert mask.sum() == pytest.approx(0.685 * 0.12 * chord**2, rel=0.05)


def test_symmetric_section_is_mirror_symmetric():
    mask = airfoil.rasterize("0012", 80, 0.0, (60.0, 40.0), (81, 160))
    assert np.array_equal(mask, mask[::-1])


def test_positive_angle_raises_the_nose():
    pivot = (100.0, 60.0)
    pts = airfoil.outline("0012", 100, 10.0, pivot)
    nose = pts[np.argmin(pts[:, 0])]
    tail = pts[np.argmax(pts[:, 0])]
    assert nose[1] > pivot[1] > tail[1]


def test_extrude_fills_only_the_span():
    section = airfoil.rasterize("0012", 20, 4.0, (15.0, 10.0), (21, 40))
    wing = airfoil.extrude(section, 30, 9.5, 19.5)
    z = np.arange(30)
    inside = (z >= 10) & (z < 20)
    assert all(np.array_equal(layer, section) for layer in wing[inside])
    assert not wing[~inside].any()


def test_wing_mesh_is_closed_and_faces_outward():
    curve = airfoil.smooth_outline(airfoil.naca4_vertices("2412"))
    pts = airfoil.place(curve, 40, 8.0, (50.0, 30.0))
    verts, faces = airfoil.wing_mesh(pts, 10.0, 60.0)
    assert faces.min() >= 0 and faces.max() < len(verts)
    # Every edge of a closed surface borders exactly two triangles.
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    # Outward normals give a positive volume equal to section area times span.
    a, b, c = (verts[faces[:, k]] for k in range(3))
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    assert volume == pytest.approx(abs(airfoil.signed_area(pts)) * 50.0, rel=1e-6)
