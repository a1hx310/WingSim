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
