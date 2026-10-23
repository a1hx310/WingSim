"""Desktop wind tunnel: live flow view, wing controls, and lift/drag readouts."""

import sys
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from numba import njit
from PySide6 import QtCore, QtGui, QtWidgets

from . import airfoil
from .lbm import Tunnel


AIR_DENSITY = 1.225  # kg/m^3, sea level
AIR_NU = 1.5e-5  # m^2/s, kinematic viscosity at about 20 C
# Above this the grid cannot resolve the flow round an 80-cell chord even with the
# subgrid model, so the simulation is held here and the UI says so. The limit scales
# with the wing's size in the tunnel.
MAX_SIM_RE = 20000.0

NX, NY = 560, 280
CHORD_CELLS = 80
PIVOT = (180.0, (NY - 1) / 2)  # centred so a symmetric wing sees a symmetric tunnel


def make_tunnel(prefer_gpu=True):
    """Build the tunnel on the NVIDIA GPU when one is available, otherwise on the CPU."""
    if prefer_gpu:
        try:
            from .lbm_gpu import GpuTunnel, gpu_name
            if gpu_name() is not None:
                GpuTunnel(16, 8).step(1)  # compiles the kernels, so failures surface here
                return GpuTunnel(NX, NY)
        except Exception as exc:  # missing numba-cuda, driver problems, and so on
            print(f"GPU unavailable, using CPU: {exc}", file=sys.stderr)
    return Tunnel(NX, NY)


VIEWS = {
    "Speed": ("speed", "inferno"),
    "Vorticity": ("vorticity", "CET-D1"),
    "Pressure": ("pressure", "CET-D1"),
}


@njit(cache=True)
def _advect(px, py, ux, uy, solid, dt, rake_y, rng_seed):
    np.random.seed(rng_seed)
    ny, nx = ux.shape
    # Keep each particle move under about half a cell so streaks stay smooth.
    substeps = max(1, int(dt * 0.2) + 1)
    h = dt / substeps
    for k in range(px.size):
        x, y = px[k], py[k]
        for _ in range(substeps):
            i, j = int(x), int(y)
            if not (0 <= i < nx - 1 and 0 <= j < ny - 1):
                break
            fx, fy = x - i, y - j
            u = ((1 - fx) * (1 - fy) * ux[j, i] + fx * (1 - fy) * ux[j, i + 1]
                 + (1 - fx) * fy * ux[j + 1, i] + fx * fy * ux[j + 1, i + 1])
            v = ((1 - fx) * (1 - fy) * uy[j, i] + fx * (1 - fy) * uy[j, i + 1]
                 + (1 - fx) * fy * uy[j + 1, i] + fx * fy * uy[j + 1, i + 1])
            x += u * h
            y += v * h
        ii, jj = int(x), int(y)
        if x < 1 or x >= nx - 2 or y < 1 or y >= ny - 2 or solid[jj, ii]:
            # Re-release from the smoke rake at the inlet.
            x = 1.0 + np.random.random() * 2.0
            y = rake_y[k]
        px[k], py[k] = x, y


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
        form = QtWidgets.QFormLayout(self)

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
        self.steps.setRange(1, 200)
        self.steps.setValue(20)
        form.addRow("Steps per frame", self.steps)

        buttons = QtWidgets.QHBoxLayout()
        self.pause = QtWidgets.QPushButton("Pause")
        self.pause.setCheckable(True)
        self.reset = QtWidgets.QPushButton("Reset flow")
        buttons.addWidget(self.pause)
        buttons.addWidget(self.reset)
        form.addRow(buttons)

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
        self.tunnel = make_tunnel(prefer_gpu)
        self.history = deque(maxlen=600)
        self.code = None
        self.vertices = None  # editable outline in chord units, see airfoil.naca4_vertices
        self.aoa = None
        self.size = None
        self.ref_length = float(CHORD_CELLS)  # actual chord in cells, after any edits

        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
        self.flow = pg.GraphicsLayoutWidget()
        vb = self.flow.addViewBox(lockAspect=True, invertY=False, enableMouse=False)
        vb.setRange(xRange=(0, NX), yRange=(0, NY), padding=0)
        self.image = pg.ImageItem()
        vb.addItem(self.image)
        self.particles = pg.ScatterPlotItem(size=1.5, pen=None, brush=pg.mkBrush(255, 255, 255, 110))
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
        n_rakes, per_rake = 36, 500
        self.rake_y = np.repeat(np.linspace(8, NY - 8, n_rakes), per_rake)
        self.px = np.random.uniform(1, NX - 3, self.rake_y.size)
        self.py = self.rake_y.copy()

        self.controls = Controls()
        self.controls.changed.connect(self.apply_settings)
        self.controls.reset.clicked.connect(self.reset_flow)
        self.controls.reset_shape.clicked.connect(self.reset_shape)
        self.controls.handles.toggled.connect(self.editor.setVisible)

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
        side_box = QtWidgets.QWidget()
        side_box.setLayout(side)
        side_box.setFixedWidth(330)

        left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left.addWidget(self.flow)
        left.addWidget(self.plot)
        left.setSizes([520, 200])

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(left, 1)
        layout.addWidget(side_box)
        self.setCentralWidget(central)
        self.resize(1400, 820)

        self.apply_settings()
        self._last = time.perf_counter()
        self._rate = 0.0
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
        elif (c.aoa[1].value(), c.size[1].value()) != (self.aoa, self.size):
            self.update_body()
        self.update_reynolds()

    def update_body(self):
        """Put the current shape, size and angle into the tunnel and redraw the wing."""
        c = self.controls
        self.aoa, self.size = c.aoa[1].value(), c.size[1].value()
        curve = airfoil.smooth_outline(self.vertices)
        self.ref_length = self.size * np.ptp(curve[:, 0])
        pts = airfoil.place(curve, self.size, self.aoa, PIVOT)
        self.tunnel.set_solid(airfoil.rasterize_polygon(pts, (NY, NX)))
        self.body.setPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in pts]))
        self.editor.set_points(airfoil.place(self.vertices, self.size, self.aoa, PIVOT))
        self.history.clear()
        self.update_reynolds()

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
        solid = self.tunnel.solid
        self.tunnel = make_tunnel(self.tunnel.device == "gpu")
        self.tunnel.set_solid(solid)
        self.update_reynolds()
        self.history.clear()

    # Main loop ------------------------------------------------------------

    def tick(self):
        c = self.controls
        if not c.pause.isChecked():
            n = c.steps.value()
            self.tunnel.step(n)
            if not np.isfinite(self.tunnel.force[0]):
                self.reset_flow()
                return
            cd, cl = self.tunnel.coefficients()
            self.history.append((self.tunnel.steps, cl, cd))
            if c.smoke.isChecked():
                _advect(self.px, self.py, self.tunnel.ux, self.tunnel.uy, self.tunnel.solid,
                        float(n), self.rake_y, self.tunnel.steps)
            now = time.perf_counter()
            self._rate = 0.9 * self._rate + 0.1 * n / max(now - self._last, 1e-6)
            self._last = now
        self.draw()

    def draw(self):
        t = self.tunnel
        kind, cmap = VIEWS[self.controls.view.currentText()]
        if kind == "speed":
            field = np.hypot(t.ux, t.uy) / t.u
            levels = (0.0, 1.8)
        elif kind == "vorticity":
            field = t.vorticity() * self.ref_length / t.u
            levels = (-15.0, 15.0)
        else:
            field = t.pressure_coefficient()
            levels = (-2.0, 1.0)
        field = np.where(t.solid, np.nan, field)
        self.image.setImage(field, levels=levels, autoLevels=False)
        self.image.setColorMap(pg.colormap.get(cmap))
        if self.controls.smoke.isChecked():
            shade = 20 if kind == "vorticity" else 255  # dark smoke on the light vorticity map
            self.particles.setBrush(pg.mkBrush(shade, shade, shade, 110))
            self.particles.setData(self.px, self.py)
        else:
            self.particles.clear()

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
        return (
            f"<b>Lift coefficient CL</b> {cl:+.3f}<br>"
            f"<b>Drag coefficient CD</b> {cd:.3f}<br>"
            f"<b>L/D</b> {ld:.2f}<br><br>"
            f"<b>Lift</b> {cl * q:.3f} N per metre of span<br>"
            f"<b>Drag</b> {cd * q:.3f} N per metre of span<br><br>"
            f"<b>Reynolds number</b> {re_sim:,.0f}{warn}<br>"
            f"<b>Chord in tunnel</b> {self.ref_length:.0f} cells<br>"
            f"<b>Solver</b> {self.tunnel.device_label}, {self._rate:,.0f} steps/s, step {self.tunnel.steps:,}"
        )


def main():
    prefer_gpu = "--cpu" not in sys.argv
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(prefer_gpu)
    win.show()
    return app.exec()
