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
# Above this the grid cannot resolve the flow even with the subgrid model, so the
# simulation is held here and the UI says so.
MAX_SIM_RE = 20000.0

NX, NY = 560, 280
CHORD_CELLS = 80
PIVOT = (180.0, (NY - 1) / 2)  # centred so a symmetric wing sees a symmetric tunnel

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


class Controls(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        form = QtWidgets.QFormLayout(self)

        self.naca = QtWidgets.QLineEdit("2412")
        self.naca.setMaxLength(4)
        self.naca.setToolTip("NACA 4-digit code, for example 0012, 2412, 4415")
        self.naca.editingFinished.connect(self.changed)
        form.addRow("NACA airfoil", self.naca)

        self.aoa = self._slider_spin(-15, 25, 5.0, 0.5, " °")
        form.addRow("Angle of attack", self.aoa[2])

        self.speed = self._slider_spin(0.1, 30, 1.0, 0.1, " m/s")
        form.addRow("Wind speed", self.speed[2])

        self.chord = QtWidgets.QDoubleSpinBox()
        self.chord.setRange(0.01, 5.0)
        self.chord.setDecimals(3)
        self.chord.setValue(0.15)
        self.chord.setSuffix(" m")
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

    def _slider_spin(self, lo, hi, value, step, suffix):
        scale = 1.0 / step
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(lo * scale), int(hi * scale))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(1)
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WingSim wind tunnel")
        self.tunnel = Tunnel(NX, NY)
        self.history = deque(maxlen=600)
        self.code = None
        self.aoa = None

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
        n_rakes, per_rake = 36, 500
        self.rake_y = np.repeat(np.linspace(8, NY - 8, n_rakes), per_rake)
        self.px = np.random.uniform(1, NX - 3, self.rake_y.size)
        self.py = self.rake_y.copy()

        self.controls = Controls()
        self.controls.changed.connect(self.apply_settings)
        self.controls.reset.clicked.connect(self.reset_flow)

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

    def apply_settings(self):
        c = self.controls
        code = c.naca.text().strip()
        try:
            airfoil.parse_naca4(code)
            c.naca.setStyleSheet("")
        except ValueError:
            c.naca.setStyleSheet("background: #5a1f1f;")
            code = self.code or "0012"
        aoa = c.aoa[1].value()
        if code != self.code or aoa != self.aoa:
            self.code, self.aoa = code, aoa
            self.tunnel.set_solid(airfoil.rasterize(code, CHORD_CELLS, aoa, PIVOT, (NY, NX)))
            pts = airfoil.outline(code, CHORD_CELLS, aoa, PIVOT)
            self.body.setPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in pts]))
            self.history.clear()
        self.tunnel.set_reynolds(min(self.physical_re(), MAX_SIM_RE), CHORD_CELLS)

    def reset_flow(self):
        solid = self.tunnel.solid
        self.tunnel = Tunnel(NX, NY)
        self.tunnel.set_solid(solid)
        self.tunnel.set_reynolds(min(self.physical_re(), MAX_SIM_RE), CHORD_CELLS)
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
            field = t.vorticity() * CHORD_CELLS / t.u
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
        capped = re_phys > MAX_SIM_RE
        ld = cl / cd if abs(cd) > 1e-9 else float("nan")
        warn = (f"<br><span style='color:#ffb74d'>Real Re is {re_phys:,.0f}; the grid "
                f"resolves up to {MAX_SIM_RE:,.0f}, so the flow runs at that.</span>"
                if capped else "")
        return (
            f"<b>Lift coefficient CL</b> {cl:+.3f}<br>"
            f"<b>Drag coefficient CD</b> {cd:.3f}<br>"
            f"<b>L/D</b> {ld:.2f}<br><br>"
            f"<b>Lift</b> {cl * q:.3f} N per metre of span<br>"
            f"<b>Drag</b> {cd * q:.3f} N per metre of span<br><br>"
            f"<b>Reynolds number</b> {re_sim:,.0f}{warn}<br>"
            f"<b>Solver</b> {self._rate:,.0f} steps/s, step {self.tunnel.steps:,}"
        )


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec()
