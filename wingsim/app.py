"""Desktop wind tunnel: live flow view, wing controls, and lift/drag readouts."""

import sys
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import airfoil
from .gpu_panel import GpuPanel
from .lbm import Tunnel
from .lbm3d import Tunnel3D
from .view3d import CAMERAS, FlowView3D, colorize


AIR_DENSITY = 1.225  # kg/m^3, sea level
AIR_NU = 1.5e-5  # m^2/s, kinematic viscosity at about 20 C
# Above this the grid cannot resolve the flow round an 80-cell chord even with the
# subgrid model, so the simulation is held here and the UI says so. The limit scales
# with the wing's size in the tunnel.
MAX_SIM_RE = 20000.0

NX, NY = 560, 280
NZ = 440  # depth of the 3D tunnel along the span, in the same units as NX and NY
CHORD_CELLS = 80
PIVOT = (180.0, (NY - 1) / 2)  # centred so a symmetric wing sees a symmetric tunnel
# The 3D tunnel is the 2D one at a coarser resolution, plus depth: about 4.4 million
# cells on the GPU, and a quarter of the resolution again on the CPU.
SCALE_3D = {"gpu": 0.4, "cpu": 0.2}
SMOKE_2D = (36, 500)  # rakes across the inlet, particles per rake
SMOKE_3D = (10, 28, 140)  # rake heights, rake span stations, particles per rake point
VORTEX_Q = 2.0  # Q-criterion level of the vortex surfaces, in (free stream / chord)^2


def make_tunnel(prefer_gpu=True, dims=2):
    """Build a 2D or 3D tunnel on the NVIDIA GPU when one is available, otherwise on the CPU."""
    if prefer_gpu:
        try:
            from .lbm_gpu import GpuTunnel, gpu_name
            if gpu_name() is not None:
                if dims == 2:
                    GpuTunnel(16, 8).step(1)  # compiles the kernels, so failures surface here
                    return GpuTunnel(NX, NY)
                from .lbm3d_gpu import GpuTunnel3D
                GpuTunnel3D(16, 8, 8).step(1)
                return GpuTunnel3D(*grid_3d("gpu"))
        except Exception as exc:  # missing numba-cuda, driver problems, and so on
            print(f"GPU unavailable, using CPU: {exc}", file=sys.stderr)
    return Tunnel(NX, NY) if dims == 2 else Tunnel3D(*grid_3d("cpu"))


def grid_3d(device):
    """Size (nx, ny, nz) of the 3D tunnel on the given device, "gpu" or "cpu"."""
    s = SCALE_3D[device]
    return round(NX * s), round(NY * s), round(NZ * s)


VIEWS = {
    "Speed": ("speed", "inferno"),
    "Vorticity": ("vorticity", "CET-D1"),
    "Pressure": ("pressure", "CET-D1"),
}


class PointCloud(pg.GraphicsObject):
    """Many small dots of one colour, such as smoke particles. Drawing them as plain
    points is about twenty times faster than a ScatterPlotItem of the same size."""

    def __init__(self, nx, ny):
        super().__init__()
        self.rect = QtCore.QRectF(0, 0, nx, ny)
        self.points = QtGui.QPolygonF()
        self.pen = None

    def set_points(self, x, y):
        self.points = pg.functions.arrayToQPolygonF(x, y)
        self.update()

    def set_shade(self, shade, alpha=110):
        self.pen = pg.mkPen(shade, shade, shade, alpha, width=1.5, cosmetic=True)

    def boundingRect(self):
        return self.rect

    def paint(self, p, *args):
        p.setPen(self.pen)
        p.drawPoints(self.points)


class ShapeEditor(pg.GraphicsObject):
    """Vertex handles drawn over the wing. Drag a vertex to move it, double-click the
    outline to add one, right-click a vertex to remove it, and scroll over the wing to
    resize it. Positions are in grid coordinates; the window owns the actual shape."""

    moved = QtCore.Signal(int, float, float)  # vertex index, new position
    added = QtCore.Signal(int, float, float)  # index the new vertex takes, its position
    removed = QtCore.Signal(int)
    scrolled = QtCore.Signal(float)  # wheel notches, positive away from the user
    RADIUS = 4  # handle size in screen pixels
    PICK = 9  # how close the pointer must be, in screen pixels

    def __init__(self, nx, ny):
        super().__init__()
        self.rect = QtCore.QRectF(0, 0, nx, ny)
        self.pts = np.zeros((0, 2))
        self.hover = None
        self.drag = None
        self.setZValue(10)

    def set_points(self, pts):
        self.pts = np.asarray(pts, dtype=np.float64)
        self.update()

    def boundingRect(self):
        return self.rect

    def paint(self, p, *args):
        # Draw in screen pixels so the handles keep their size as the window resizes.
        tr = p.transform()
        p.resetTransform()
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(pg.mkPen(20, 20, 20, width=1))
        for k, (x, y) in enumerate(self.pts):
            active = k in (self.hover, self.drag)
            # The trailing-edge corner is coloured differently since it cannot be removed.
            p.setBrush(pg.mkBrush("#ffeb3b" if active else "#4fc3f7" if k == 0 else "w"))
            r = self.RADIUS + 2 if active else self.RADIUS
            p.drawEllipse(tr.map(QtCore.QPointF(x, y)), r, r)

    def _pick_range(self):
        return self.PICK * self.pixelSize()[0]

    def nearest_vertex(self, pos):
        if not len(self.pts):
            return None
        d = np.hypot(self.pts[:, 0] - pos.x(), self.pts[:, 1] - pos.y())
        k = int(np.argmin(d))
        return k if d[k] <= self._pick_range() else None

    def nearest_edge(self, pos):
        """Index of the outline edge (from vertex k to k + 1) near pos, or None."""
        a = self.pts
        ab = np.roll(a, -1, axis=0) - a
        p = np.array([pos.x(), pos.y()])
        s = np.clip(((p - a) * ab).sum(1) / np.maximum((ab**2).sum(1), 1e-12), 0.0, 1.0)
        d = np.hypot(*(a + s[:, None] * ab - p).T)
        k = int(np.argmin(d))
        return k if d[k] <= 2 * self._pick_range() else None

    def hoverEvent(self, ev):
        k = None if ev.isExit() else self.nearest_vertex(ev.pos())
        if k is not None:
            ev.acceptDrags(QtCore.Qt.LeftButton)
            ev.acceptClicks(QtCore.Qt.RightButton)
            self.setCursor(QtCore.Qt.OpenHandCursor)
        elif self.drag is None:
            self.unsetCursor()
        if k != self.hover:
            self.hover = k
            self.update()

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        if ev.isStart():
            self.drag = self.nearest_vertex(ev.buttonDownPos())
            if self.drag is not None:
                self.setCursor(QtCore.Qt.ClosedHandCursor)
        if self.drag is None:
            ev.ignore()
            return
        ev.accept()
        self.moved.emit(self.drag, ev.pos().x(), ev.pos().y())
        if ev.isFinish():
            self.drag = None
            self.setCursor(QtCore.Qt.OpenHandCursor)
            self.update()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            k = self.nearest_vertex(ev.pos())
            if k is not None:
                ev.accept()
                self.removed.emit(k)
        elif ev.button() == QtCore.Qt.LeftButton and ev.double():
            k = self.nearest_edge(ev.pos())
            if k is not None:
                ev.accept()
                self.added.emit(k + 1, ev.pos().x(), ev.pos().y())

    def wheelEvent(self, ev):
        poly = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in self.pts])
        if poly.containsPoint(ev.pos(), QtCore.Qt.OddEvenFill):
            ev.accept()
            self.scrolled.emit(ev.delta() / 120)
        else:
            ev.ignore()


class Controls(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        form = QtWidgets.QFormLayout()
        outer.addLayout(form)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["2D section", "3D wing"])
        self.mode.setToolTip("2D simulates a slice of an endless wing. 3D simulates a wing "
                             "with tips, so tip vortices and induced drag appear.")
        form.addRow("Simulation", self.mode)

        self.naca = QtWidgets.QLineEdit("2412")
        self.naca.setMaxLength(4)
        self.naca.setToolTip("NACA 4-digit code, for example 0012, 2412, 4415. "
                             "A new code replaces any edits to the shape.")
        self.naca.editingFinished.connect(self.changed)
        form.addRow("NACA airfoil", self.naca)

        self.size = self._slider_spin(30, 200, CHORD_CELLS, 1, " cells", decimals=0)
        self.size[2].setToolTip("Chord of the original section in grid cells. "
                                "You can also scroll over the wing.")
        form.addRow("Size in tunnel", self.size[2])

        shape = QtWidgets.QHBoxLayout()
        self.handles = QtWidgets.QCheckBox("Edit vertices")
        self.handles.setChecked(True)
        self.handles.setToolTip("Drag a vertex to reshape the wing. Double-click the outline "
                                "to add a vertex, right-click a vertex to remove it.")
        self.reset_shape = QtWidgets.QPushButton("Reset shape")
        shape.addWidget(self.handles)
        shape.addWidget(self.reset_shape)
        form.addRow(shape)

        self.aoa = self._slider_spin(-15, 25, 5.0, 0.5, " °")
        form.addRow("Angle of attack", self.aoa[2])

        self.speed = self._slider_spin(0.1, 30, 1.0, 0.1, " m/s")
        form.addRow("Wind speed", self.speed[2])

        self.chord = QtWidgets.QDoubleSpinBox()
        self.chord.setRange(0.01, 5.0)
        self.chord.setDecimals(3)
        self.chord.setValue(0.15)
        self.chord.setSuffix(" m")
        self.chord.setToolTip("Real-world chord, used for the Reynolds number and the forces")
        self.chord.valueChanged.connect(self.changed)
        form.addRow("Chord length", self.chord)

        self.view = QtWidgets.QComboBox()
        self.view.addItems(list(VIEWS))
        form.addRow("Show", self.view)

        self.smoke = QtWidgets.QCheckBox("Smoke particles")
        self.smoke.setChecked(True)
        form.addRow(self.smoke)

        self.steps = QtWidgets.QSpinBox()
        self.steps.setRange(1, 1000)
        self.steps.setValue(20)
        self.steps.setEnabled(False)
        self.auto_steps = QtWidgets.QCheckBox("Auto")
        self.auto_steps.setChecked(True)
        self.auto_steps.setToolTip("Keep the solver busy while the window draws each frame, "
                                   "for the fastest flow at a smooth frame rate")
        self.auto_steps.toggled.connect(lambda on: self.steps.setEnabled(not on))
        steps = QtWidgets.QHBoxLayout()
        steps.addWidget(self.steps, 1)
        steps.addWidget(self.auto_steps)
        form.addRow("Steps per frame", steps)

        buttons = QtWidgets.QHBoxLayout()
        self.pause = QtWidgets.QPushButton("Pause")
        self.pause.setCheckable(True)
        self.reset = QtWidgets.QPushButton("Reset flow")
        buttons.addWidget(self.pause)
        buttons.addWidget(self.reset)
        form.addRow(buttons)

        self.box_3d = QtWidgets.QGroupBox("3D wing")
        form3 = QtWidgets.QFormLayout(self.box_3d)
        self.aspect = self._slider_spin(1, 8, 3.0, 0.1, "")
        self.aspect[2].setToolTip("Span divided by chord. The tunnel width limits it, "
                                  "leaving a chord of clear air past each tip.")
        form3.addRow("Aspect ratio", self.aspect[2])
        self.span_slice, self.span_pos = self._slice_row(
            form3, "Span slice", "Flow on a plane along the wing, at the chosen span station")
        self.cross_slice, self.cross_pos = self._slice_row(
            form3, "Crossflow slice", "Flow on a plane across the tunnel, at the chosen "
            "distance downstream. Vorticity here shows the tip vortices.")
        self.vortex = QtWidgets.QCheckBox("Vortex surfaces")
        self.vortex.setToolTip("Surfaces round the vortex cores (Q-criterion), refreshed "
                               "twice a second")
        form3.addRow(self.vortex)
        cameras = QtWidgets.QHBoxLayout()
        self.cameras = {}
        for name in CAMERAS:
            b = QtWidgets.QPushButton(name)
            b.setMinimumWidth(10)
            cameras.addWidget(b)
            self.cameras[name] = b
        form3.addRow(cameras)
        form3.addRow(QtWidgets.QLabel("Drag to orbit, middle-drag to pan, scroll to zoom."))
        outer.addWidget(self.box_3d)
        self.box_3d.hide()

    def _slice_row(self, form, label, tip):
        box = QtWidgets.QCheckBox(label)
        box.setChecked(True)
        box.setToolTip(tip)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        form.addRow(box, slider)
        return box, slider

    def _slider_spin(self, lo, hi, value, step, suffix, decimals=1):
        scale = 1.0 / step
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(lo * scale), int(hi * scale))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        slider.valueChanged.connect(lambda v: spin.setValue(v / scale))
        spin.valueChanged.connect(lambda v: slider.setValue(round(v * scale)))
        spin.valueChanged.connect(self.changed)
        spin.setValue(value)
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        return slider, spin, box


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, prefer_gpu=True):
        super().__init__()
        self.setWindowTitle("WingSim wind tunnel")
        self.prefer_gpu = prefer_gpu
        self.dims = 2
        self.tunnel = make_tunnel(prefer_gpu)
        self.history = deque(maxlen=600)
        self.code = None
        self.vertices = None  # editable outline in chord units, see airfoil.naca4_vertices
        self.aoa = None
        self.size = None
        self.ref_length = float(CHORD_CELLS)  # actual chord in cells, after any edits
        self.aspect = None
        self.steps_per_frame = {2: 20, 3: 5}
        self._vortex_time = 0.0

        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
        self.flow = pg.GraphicsLayoutWidget()
        vb = self.flow.addViewBox(lockAspect=True, invertY=False, enableMouse=False)
        vb.setRange(xRange=(0, NX), yRange=(0, NY), padding=0)
        self.image = pg.ImageItem()
        vb.addItem(self.image)
        self.particles = PointCloud(NX, NY)
        vb.addItem(self.particles)
        self.body = QtWidgets.QGraphicsPolygonItem()
        self.body.setPen(pg.mkPen("w", width=1.5, cosmetic=True))
        self.body.setBrush(pg.mkBrush(60, 60, 70))
        vb.addItem(self.body)
        self.editor = ShapeEditor(NX, NY)
        self.editor.moved.connect(self.move_vertex)
        self.editor.added.connect(self.add_vertex)
        self.editor.removed.connect(self.remove_vertex)
        self.editor.scrolled.connect(self.scale_wing)
        vb.addItem(self.editor)

        self.controls = Controls()
        self.controls.changed.connect(self.apply_settings)
        self.controls.reset.clicked.connect(self.reset_flow)
        self.controls.reset_shape.clicked.connect(self.reset_shape)
        self.controls.handles.toggled.connect(self.editor.setVisible)
        self.controls.mode.currentIndexChanged.connect(lambda i: self.set_mode(3 if i else 2))
        self.view3d = FlowView3D()
        for name, button in self.controls.cameras.items():
            button.clicked.connect(lambda _=False, n=name: self.view3d.set_camera(n))
        self.controls.vortex.toggled.connect(self.toggle_vortex)

        self.readout = QtWidgets.QLabel()
        self.readout.setTextFormat(QtCore.Qt.RichText)
        self.readout.setWordWrap(True)
        self.readout.setMinimumWidth(280)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "time steps")
        self.plot.addLegend(offset=(10, 5))
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.cl_curve = self.plot.plot(pen=pg.mkPen("#4fc3f7", width=2), name="CL (lift)")
        self.cd_curve = self.plot.plot(pen=pg.mkPen("#ffb74d", width=2), name="CD (drag)")
        self.plot.setMinimumHeight(180)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.controls)
        side.addWidget(self.readout)
        side.addStretch(1)
        side.addWidget(GpuPanel())
        side_box = QtWidgets.QWidget()
        side_box.setLayout(side)
        side_box.setFixedWidth(330)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.flow)
        self.stack.addWidget(self.view3d)
        left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left.addWidget(self.stack)
        left.addWidget(self.plot)
        left.setSizes([520, 200])

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(left, 1)
        layout.addWidget(side_box)
        self.setCentralWidget(central)
        self.resize(1400, 820)

        self.apply_settings()
        self.reset_smoke()
        self._last = time.perf_counter()
        self._rate = 0.0
        self._per_step_s = 0.0  # time the last frame spent on work that grows with its steps
        self._plot_time = 0.0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(0)

    # Settings -------------------------------------------------------------

    def physical_re(self):
        return self.controls.speed[1].value() * self.controls.chord.value() / AIR_NU

    def max_sim_re(self):
        # Holding the lattice viscosity at its limit makes the limit grow with size.
        return MAX_SIM_RE * self.ref_length / CHORD_CELLS

    def update_reynolds(self):
        self.tunnel.set_reynolds(min(self.physical_re(), self.max_sim_re()), self.ref_length)

    def apply_settings(self):
        c = self.controls
        code = c.naca.text().strip()
        try:
            airfoil.parse_naca4(code)
            c.naca.setStyleSheet("")
        except ValueError:
            c.naca.setStyleSheet("background: #5a1f1f;")
            code = self.code or "0012"
        if code != self.code:
            self.code = code
            self.vertices = airfoil.naca4_vertices(code)
            self.update_body()
        elif (c.aoa[1].value(), c.size[1].value(), c.aspect[1].value()) != (
                self.aoa, self.size, self.aspect):
            self.update_body()
        self.update_reynolds()

    def scale(self):
        """Cells per 2D-grid cell: 1 in 2D, less on the coarser 3D grid."""
        return self.tunnel.nx / NX

    def pivot(self):
        """The wing's pivot point on the current grid."""
        return PIVOT[0] * self.scale(), (self.tunnel.ny - 1) / 2

    def update_body(self):
        """Put the current shape, size and angle into the tunnel and redraw the wing."""
        c = self.controls
        t = self.tunnel
        self.aoa, self.size, self.aspect = c.aoa[1].value(), c.size[1].value(), c.aspect[1].value()
        curve = airfoil.smooth_outline(self.vertices)
        chord = self.size * self.scale()
        self.ref_length = chord * np.ptp(curve[:, 0])
        pts = airfoil.place(curve, chord, self.aoa, self.pivot())
        mask = airfoil.rasterize_polygon(pts, (t.ny, t.nx))
        if self.dims == 2:
            t.set_solid(mask)
            self.body.setPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in pts]))
            self.editor.set_points(airfoil.place(self.vertices, self.size, self.aoa, PIVOT))
        else:
            z0, z1 = self.span_stations()
            t.set_solid(airfoil.extrude(mask, t.nz, z0, z1))
            self.view3d.set_wing(*airfoil.wing_mesh(pts, z0, z1))
            self.reset_smoke()
        self.history.clear()
        self.update_reynolds()

    def span_limit(self):
        """Largest span in cells that leaves a chord of clear air past each tip."""
        return max(self.tunnel.nz - 2 - 2 * self.ref_length, 2.0)

    def span_stations(self):
        """Span positions (z0, z1) of the wing tips, centred in the tunnel."""
        span = min(self.aspect * self.ref_length, self.span_limit())
        mid = (self.tunnel.nz - 1) / 2
        return mid - span / 2, mid + span / 2

    def reset_smoke(self):
        """Release smoke afresh: from rakes across the inlet in 2D, and in 3D from a grid
        of rake points round the wing, tips included."""
        t = self.tunnel
        if self.dims == 2:
            n_rakes, per_rake = SMOKE_2D
            rake = np.repeat(np.linspace(8, t.ny - 8, n_rakes), per_rake)[:, None]
        else:
            n_y, n_z, per_point = SMOKE_3D
            z0, z1 = self.span_stations()
            mid_y, reach = self.pivot()[1], 0.6 * self.ref_length
            ys = np.linspace(max(mid_y - reach, 2), min(mid_y + reach, t.ny - 3), n_y)
            zs = np.linspace(max(z0 - reach, 2), min(z1 + reach, t.nz - 3), n_z)
            rake = np.stack(np.meshgrid(ys, zs, indexing="ij"), axis=-1).reshape(-1, 2)
            rake = np.repeat(rake, per_point, axis=0)
        release = np.c_[np.ones(len(rake)), rake]
        pos = release.copy()
        pos[:, 0] = np.random.uniform(1, t.nx - 3, len(pos))
        t.set_particles(pos, release)
        self.show_smoke(pos)

    def show_smoke(self, pos):
        if self.dims == 2:
            self.particles.set_points(pos[:, 0], pos[:, 1])
        else:
            self.view3d.set_smoke(pos)

    def reset_shape(self):
        self.vertices = airfoil.naca4_vertices(self.code)
        self.update_body()

    def to_chord_units(self, x, y):
        # Keep vertices off the tunnel walls.
        x, y = np.clip(x, 2, NX - 3), np.clip(y, 2, NY - 3)
        return airfoil.unplace([x, y], self.size, self.aoa, PIVOT)

    def move_vertex(self, k, x, y):
        self.vertices[k] = self.to_chord_units(x, y)
        self.update_body()

    def add_vertex(self, k, x, y):
        self.vertices = np.insert(self.vertices, k, self.to_chord_units(x, y), axis=0)
        self.update_body()

    def remove_vertex(self, k):
        # Vertex 0 is the trailing-edge corner, and a wing needs a few vertices.
        if k == 0 or len(self.vertices) <= 4:
            return
        self.vertices = np.delete(self.vertices, k, axis=0)
        self.update_body()

    def scale_wing(self, notches):
        spin = self.controls.size[1]
        spin.setValue(spin.value() * 1.1**notches)

    def reset_flow(self):
        self.tunnel = None  # frees the old tunnel's GPU memory before building the new one
        self.tunnel = make_tunnel(self.prefer_gpu, self.dims)
        self.update_body()
        if self.dims == 2:
            self.reset_smoke()  # 3D smoke follows the wing, so update_body releases it

    def set_mode(self, dims):
        """Switch between the 2D and 3D tunnels. Only one exists at a time."""
        if dims == self.dims:
            return
        c = self.controls
        self.steps_per_frame[self.dims] = c.steps.value()
        self.dims = dims
        self.reset_flow()
        three = dims == 3
        self.stack.setCurrentIndex(int(three))
        c.box_3d.setVisible(three)
        c.handles.setEnabled(not three)
        c.steps.setValue(self.steps_per_frame[dims])
        if three:
            t = self.tunnel
            px, py = self.pivot()
            c.span_pos.setRange(1, t.nz - 2)
            c.span_pos.setValue(t.nz // 2)
            c.cross_pos.setRange(1, t.nx - 2)
            c.cross_pos.setValue(min(int(px + 1.75 * self.ref_length), t.nx - 2))
            self.view3d.set_grid(t.nx, t.ny, t.nz, (px, py, (t.nz - 1) / 2))
            self.toggle_vortex(c.vortex.isChecked())

    def toggle_vortex(self, on):
        """Clear the old vortex surfaces; when on, the next frame draws fresh ones."""
        self.view3d.clear_vortex()
        self._vortex_time = 0.0

    # Main loop ------------------------------------------------------------

    def tick(self):
        """One frame. A GPU computes each batch of steps while the window draws the
        previous one, so the card and the drawing don't take turns."""
        c = self.controls
        t = self.tunnel
        paused = c.pause.isChecked()
        self.timer.setInterval(50 if paused else 0)  # no need to spin while the flow is still
        began = time.perf_counter()
        n = t.finish()  # the steps started last frame
        now = time.perf_counter()
        if n:
            if not np.isfinite(t.force[0]):
                self.reset_flow()
                return
            cd, cl = t.coefficients()
            self.history.append((t.steps, cl, cd))
            frame = now - self._last
            self._rate = 0.9 * self._rate + 0.1 * n / max(frame, 1e-6)

        # Everything that reads from the GPU comes before the next batch is queued.
        smoke_s = time.perf_counter()
        if n and c.smoke.isChecked():
            self.show_smoke(t.advect(n))
        smoke_s = time.perf_counter() - smoke_s
        if self.dims == 3:
            self.draw_3d()
        launch_s = time.perf_counter()
        steps = 0 if paused else c.steps.value()
        if steps:
            t.start(steps)
        launch_s = time.perf_counter() - launch_s
        if self.dims == 2:
            self.draw()
        self.draw_plot()

        if n and steps and c.auto_steps.isChecked():
            self.tune_steps(n, frame, now - began, smoke_s / n + launch_s / steps)
        self._last = now
        self._per_step_s = smoke_s + launch_s

    def tune_steps(self, n, frame, waited, host_per_step):
        """Choose the next batch size. Drawing takes a fixed time per frame, while this
        thread's queueing of steps and moving of smoke, and the solver itself, take time
        per step. Aim for a 60 Hz frame, or a quarter longer than the drawing alone if
        that is slower, and fit in as many steps as both this thread and the solver
        allow. The CPU solver works inside start(), which counts as this thread's time."""
        drawing = max(frame - waited - self._per_step_s, 1e-3)
        target = max(1 / 60, 1.25 * drawing)
        solver_per_step = self.tunnel.batch_ms / 1000 / n
        ideal = min((target - drawing) / max(host_per_step, 1e-7),
                    target / max(solver_per_step, 1e-7))
        steps = self.controls.steps
        steps.setValue(round(np.clip(0.7 * n + 0.3 * ideal, 1, steps.maximum())))

    def display(self, kind, raw):
        """A raw lattice field in the units the view shows, with its colour range."""
        if kind == "speed":
            return raw / self.tunnel.u, (0.0, 1.8)
        if kind == "vorticity":
            return raw * self.ref_length / self.tunnel.u, (-15.0, 15.0)
        return raw, (-2.0, 1.0)

    def draw(self):
        t = self.tunnel
        kind, cmap = VIEWS[self.controls.view.currentText()]
        if kind == "speed":
            raw = np.hypot(t.ux, t.uy)
        elif kind == "vorticity":
            raw = t.vorticity()
        else:
            raw = t.pressure_coefficient()
        field, levels = self.display(kind, raw)
        field = np.where(t.solid, np.nan, field)
        self.image.setImage(field, levels=levels, autoLevels=False)
        self.image.setColorMap(pg.colormap.get(cmap))
        self.particles.setVisible(self.controls.smoke.isChecked())
        self.particles.set_shade(20 if kind == "vorticity" else 255)  # dark on the light map

    def draw_3d(self):
        t = self.tunnel
        c = self.controls
        v = self.view3d
        kind, cmap = VIEWS[c.view.currentText()]
        for item, box, slider, axis in ((v.span_slice, c.span_slice, c.span_pos, "z"),
                                        (v.cross_slice, c.cross_slice, c.cross_pos, "x")):
            item.setVisible(box.isChecked())
            if box.isChecked():
                k = slider.value()
                field, levels = self.display(kind, t.slice(kind, axis, k))
                solid = t.solid[k] if axis == "z" else t.solid[:, :, k]
                field = np.where(solid, np.nan, field)
                v.set_slice(item, colorize(field, cmap, levels, fade=kind == "vorticity"), axis, k)
        v.smoke.setVisible(c.smoke.isChecked())
        now = time.perf_counter()
        if c.vortex.isChecked() and now - self._vortex_time > 0.5:
            self._vortex_time = now
            self.update_vortex()

    def update_vortex(self):
        """Surfaces where the Q-criterion, in chord and free-stream units, crosses a level."""
        stride = 2
        q = self.tunnel.q_criterion(stride) * (self.ref_length / self.tunnel.u) ** 2
        self.view3d.set_vortex(q, VORTEX_Q, stride)

    def draw_plot(self):
        """Refresh the CL/CD plot and the readout, ten times a second. They change slowly,
        and repainting them every frame costs a third of a frame on a large screen."""
        now = time.perf_counter()
        if now - self._plot_time < 0.1:
            return
        self._plot_time = now
        if self.history:
            h = np.array(self.history)
            self.cl_curve.setData(h[:, 0], h[:, 1])
            self.cd_curve.setData(h[:, 0], h[:, 2])
        self.readout.setText(self.readout_html())

    def readout_html(self):
        c = self.controls
        h = np.array(self.history) if self.history else np.zeros((1, 3))
        # Average over the most recent stretch so unsteady (shedding) flows read steadily.
        recent = h[-min(len(h), 150):]
        cl, cd = recent[:, 1].mean(), recent[:, 2].mean()
        u = c.speed[1].value()
        chord = c.chord.value()
        q = 0.5 * AIR_DENSITY * u**2 * chord
        re_phys = self.physical_re()
        re_sim = self.tunnel.re
        max_re = self.max_sim_re()
        capped = re_phys > max_re
        ld = cl / cd if abs(cd) > 1e-9 else float("nan")
        warn = (f"<br><span style='color:#ffb74d'>Real Re is {re_phys:,.0f}; at this size the "
                f"grid resolves up to {max_re:,.0f}, so the flow runs at that.</span>"
                if capped else "")
        if self.dims == 2:
            forces = (f"<b>Lift</b> {cl * q:.3f} N per metre of span<br>"
                      f"<b>Drag</b> {cd * q:.3f} N per metre of span<br><br>")
        else:
            aspect = self.tunnel.span / self.ref_length
            limited = (" <span style='color:#ffb74d'>(limited by the tunnel width)</span>"
                       if aspect < self.aspect - 0.05 else "")
            q *= aspect * chord  # whole wing: span = aspect ratio x chord
            forces = (f"<b>Lift</b> {cl * q:.3f} N for the whole wing<br>"
                      f"<b>Drag</b> {cd * q:.3f} N for the whole wing<br>"
                      f"<b>Aspect ratio</b> {aspect:.2f}{limited}<br><br>")
        return (
            f"<b>Lift coefficient CL</b> {cl:+.3f}<br>"
            f"<b>Drag coefficient CD</b> {cd:.3f}<br>"
            f"<b>L/D</b> {ld:.2f}<br><br>"
            + forces +
            f"<b>Reynolds number</b> {re_sim:,.0f}{warn}<br>"
            f"<b>Chord in tunnel</b> {self.ref_length:.0f} cells<br>"
            f"<b>Solver</b> {self.tunnel.device_label}, {self._rate:,.0f} steps/s, step {self.tunnel.steps:,}"
        )


def main():
    prefer_gpu = "--cpu" not in sys.argv
    # Don't make each frame wait for the display: the frame loop paces itself, and on
    # Wayland a waiting buffer swap can stall the window for 100 ms or more.
    fmt = QtGui.QSurfaceFormat.defaultFormat()
    fmt.setSwapInterval(0)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(prefer_gpu)
    win.show()
    return app.exec()
