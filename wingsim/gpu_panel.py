"""Live readout of the NVIDIA GPU's memory and load, from NVIDIA's management library
(NVML, through the nvidia-ml-py package). Hidden when there is no NVIDIA GPU."""

import os

from PySide6 import QtCore, QtWidgets

GIB = 2**30


class GpuPanel(QtWidgets.QGroupBox):
    def __init__(self, interval_ms=500):
        super().__init__("GPU")
        form = QtWidgets.QFormLayout(self)
        self.memory = QtWidgets.QProgressBar()
        self.memory.setToolTip("Memory in use on the whole card, by every program")
        self.load = QtWidgets.QProgressBar()
        self.load.setToolTip("Share of the last moment the GPU spent running work")
        self.own = QtWidgets.QLabel()
        form.addRow("Memory", self.memory)
        form.addRow("Load", self.load)
        form.addRow(self.own)
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.setToolTip(pynvml.nvmlDeviceGetName(self.handle))
        except Exception:  # no NVIDIA driver, or nvidia-ml-py is not installed
            self.hide()
            return
        self.nvml = pynvml
        self.refresh()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(interval_ms)

    def refresh(self):
        nvml, h = self.nvml, self.handle
        mem = nvml.nvmlDeviceGetMemoryInfo(h)
        share = 100 * mem.used / mem.total
        self.memory.setValue(round(share))
        self.memory.setFormat(f"{mem.used / GIB:.1f} of {mem.total / GIB:.1f} GB ({share:.0f}%)")
        load = nvml.nvmlDeviceGetUtilizationRates(h).gpu
        self.load.setValue(load)
        self.load.setFormat(f"{load}%")
        own = sum(p.usedGpuMemory or 0 for p in nvml.nvmlDeviceGetComputeRunningProcesses(h)
                  if p.pid == os.getpid())
        self.own.setText(f"WingSim is using {own / GIB:.2f} GB of it")
