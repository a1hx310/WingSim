"""Orbitable 3D view of the tunnel: the wing, flow slices, smoke and vortex surfaces.

Drag with the left button to orbit, with the middle button to pan, and scroll to zoom.
Every item lives in simulation coordinates (x downstream, y up, z along the span)
under one root item that turns them into the view's z-up frame.
"""

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtGui
from skimage.measure import marching_cubes

BODY_RGBA = (60, 60, 70, 255)  # the wing's colour in the 2D view, used where a slice cuts it

# Azimuth and elevation in degrees. Azimuth 0 looks upstream from behind the wing, and
# -90 looks along the span with the flow going left to right, like the 2D view.
CAMERAS = {
    "Front": (180.0, 8.0),
    "Side": (-90.0, 0.0),
    "Top": (-90.0, 90.0),
    "Tip": (-35.0, 12.0),
    "Reset": (-130.0, 25.0),
}


def colorize(field, cmap, levels, fade=False):
    """RGBA bytes for a 2D field (rows, cols) laid out as the image item expects
    (cols, rows, 4). NaN cells, which are inside the body, get the body colour. With
    fade, values near zero turn see-through, so a vorticity plane only hides the
    parts of the scene behind actual vortices."""
    lo, hi = levels
    rgba = pg.colormap.get(cmap).map(np.clip((field - lo) / (hi - lo), 0.0, 1.0), mode="byte")
    rgba = rgba.reshape(field.shape + (4,))
    if fade:
        rgba[..., 3] = 255 * np.clip(np.abs(np.nan_to_num(field)) / (0.3 * hi), 0.0, 1.0)
    rgba[np.isnan(field)] = BODY_RGBA
    return np.ascontiguousarray(rgba.transpose(1, 0, 2))


class FlowView3D(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor(24, 24, 28)
        self.grid_shape = (1, 1, 1)
        self.target = (0.0, 0.0, 0.0)

        # Simulation (x, y, z) to view (x, -z, y): a rotation, so nothing is mirrored.
        self.root = gl.GLGraphicsItem.GLGraphicsItem()
        self.root.setTransform(QtGui.QMatrix4x4(1, 0, 0, 0,
                                                0, 0, -1, 0,
                                                0, 1, 0, 0,
                                                0, 0, 0, 1))
        self.addItem(self.root)
        self.box = self._add(gl.GLBoxItem(color=(255, 255, 255, 50)))
        self.wing = self._add(gl.GLMeshItem(color=(0.78, 0.78, 0.82, 1.0), shader="shaded",
                                            smooth=False))
        self.span_slice = self._add(gl.GLImageItem(np.zeros((1, 1, 4), np.ubyte)))
        self.cross_slice = self._add(gl.GLImageItem(np.zeros((1, 1, 4), np.ubyte)))
        self.smoke = self._add(gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=2.0, pxMode=True,
                                                    color=(1.0, 1.0, 1.0, 0.35)))
        self.smoke.setGLOptions("translucent")
        # Flat shading: pyqtgraph averages smooth vertex normals in a Python loop, which
        # costs about 100 ms per update on a developed wake.
        self.vortex = self._add(gl.GLMeshItem(color=(0.3, 0.85, 1.0, 0.45), shader="shaded",
                                              smooth=False, glOptions="translucent"))
        self.vortex.setVisible(False)
        # See-through items draw after the solid wing, each over what is already drawn,
        # and the slices last: an item drawn after a slice and behind it would vanish.
        self.smoke.setDepthValue(1)
        self.vortex.setDepthValue(2)
        self.span_slice.setDepthValue(3)
        self.cross_slice.setDepthValue(3)

    def _add(self, item):
        item.setParentItem(self.root)
        self.addItem(item)
        return item

    def set_grid(self, nx, ny, nz, target):
        """Size the tunnel outline and aim the camera at target, the middle of the wing."""
        self.grid_shape = (nx, ny, nz)
        self.target = target
        self.box.setSize(nx - 1, ny - 1, nz - 1)
        self.set_camera("Reset")

    def set_camera(self, name):
        azimuth, elevation = CAMERAS[name]
        x, y, z = self.target
        nx, ny, nz = self.grid_shape
        self.setCameraPosition(pos=QtGui.QVector3D(x, -z, y), distance=1.1 * max(nx, nz),
                               elevation=elevation, azimuth=azimuth)

    def set_wing(self, verts, faces):
        self.wing.setMeshData(vertexes=verts, faces=faces)

    def set_slice(self, item, rgba, axis, index):
        """Show RGBA bytes from colorize on a span plane (axis "z") or a crossflow plane
        (axis "x") at the given cell index."""
        item.setData(rgba)
        if axis == "z":
            # Image (x, y) onto the plane z = index, pixels centred on cell centres.
            tr = QtGui.QMatrix4x4()
            tr.translate(-0.5, -0.5, index)
        else:
            # Image (y, z) onto the plane x = index.
            tr = QtGui.QMatrix4x4(0, 0, 1, index,
                                  1, 0, 0, -0.5,
                                  0, 1, 0, -0.5,
                                  0, 0, 0, 1)
        item.setTransform(tr)

    def set_smoke(self, pos):
        self.smoke.setData(pos=pos)

    def set_vortex(self, q, level, stride):
        """Surfaces where q, sampled on every stride-th cell as (z, y, x), crosses level."""
        if q.max() <= level:
            self.clear_vortex()
            return
        verts, faces, _, _ = marching_cubes(q, level)
        self.vortex.setMeshData(vertexes=verts[:, ::-1] * stride, faces=faces)
        self.vortex.setVisible(True)

    def clear_vortex(self):
        self.vortex.setVisible(False)
