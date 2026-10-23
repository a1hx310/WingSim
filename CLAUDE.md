# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
source .venv/bin/activate
pip install -r requirements.txt        # add requirements-gpu.txt for NVIDIA GPU support
python -m wingsim                      # run the app (--cpu forces the CPU solver)
python -m pytest                       # all tests
python -m pytest tests/test_lbm.py::test_name   # single test
NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/test_lbm_gpu.py tests/test_lbm3d_gpu.py   # GPU kernels vs CPU, no GPU needed
```

No linter or build step is configured. The first launch is slow while Numba compiles (`cache=True` on the CPU kernel).

## Architecture

A 2D wind tunnel: a PySide6/pyqtgraph GUI (`wingsim/app.py`) drives a D2Q9 Lattice Boltzmann solver and shows flow, smoke particles and live CL/CD.

- `wingsim/lbm.py` is the CPU solver and defines the `Tunnel` class and the lattice constants (`CX`, `CY`, `W`, `OPP`). Everything is in **lattice units** (cell = length, step = time, free-stream density 1). `Tunnel.step` runs a fused pull-stream + BGK/Smagorinsky collide kernel, with half-way bounce-back off solid cells. Force comes from momentum exchange across fluid-solid links, accumulated per row inside the kernel. Boundaries (inflow left/top/bottom, copy outlet right) are applied in Python after each kernel call.
- `wingsim/lbm_gpu.py` has `GpuTunnel`, a Numba CUDA mirror of the CPU kernel (one thread per cell, float32). It subclasses/reuses `Tunnel` and keeps populations on the card, copying only rho/u back for drawing. **Any change to the CPU kernel's physics must be mirrored in the GPU kernel**; `tests/test_lbm_gpu.py` checks they agree.
- Stepping is split into `start(n)` and `finish()`. GPU tunnels queue the batch on their own CUDA stream and return at once, so `MainWindow.tick` overlaps the next batch with drawing: it calls `finish()`, does everything that reads from the GPU (smoke `advect`, 3D slices and Q-criterion), then `start()`s the next batch, then draws. Anything new that reads GPU data per frame must go before `start()`, or it queues behind the batch. With "Auto" steps per frame, `tune_steps` sizes each batch from measured costs to aim for 60 Hz. Smoke particles live in the solvers (`set_particles`/`advect`) in both 2D and 3D. The 2D GPU kernel applies the tunnel boundaries itself (one launch per step, since launches dominate at that grid size); the CPU solver still applies them in Python, and the GPU test checks they agree.
- `main()` sets the OpenGL swap interval to 0: on Wayland, vsync'd swaps stall the window for 100+ ms. The plot and readout refresh at 10 Hz, since repainting them every frame costs a third of a frame on a high-DPI screen. `wingsim/gpu_panel.py` shows card memory and load via NVML (`nvidia-ml-py`, in requirements-gpu.txt) and hides itself without an NVIDIA GPU.
- `wingsim/app.py` `make_tunnel(prefer_gpu)` picks `GpuTunnel` when `gpu_name()` finds a device, otherwise falls back to the CPU `Tunnel` (printing the reason). Solvers expose `device`/`device_label`, which the UI shows. The GUI also clamps Reynolds number to what the grid can resolve (about Re 20,000) and says so.
- `wingsim/lbm3d.py` (`Tunnel3D`, D3Q19) and `wingsim/lbm3d_gpu.py` (`GpuTunnel3D`) are the 3D counterparts, for a finite wing: x downstream, y up, z along the span. The same mirroring rule applies between them, checked by `tests/test_lbm3d_gpu.py`. The GPU class keeps everything on the card (populations, fields, smoke particles), so the view asks the solver for what it draws via `slice()`, `advect()` and `q_criterion()` instead of reading whole fields.
- The app has a 2D/3D mode switch; only one tunnel exists at a time. The 3D grid is the 2D one scaled by `SCALE_3D` (0.4 on GPU, 0.2 on CPU) plus span depth `NZ`, so the "Size in tunnel" control means the same wing in both. `wingsim/view3d.py` (`FlowView3D`, pyqtgraph.opengl) draws the wing mesh, flow slices, smoke and Q-criterion vortex surfaces; its items sit under one root item that maps simulation axes to the view's z-up axes.
- `wingsim/airfoil.py` generates NACA 4-digit sections and rasterizes them to the boolean solid mask passed to `Tunnel.set_solid`; `extrude` and `wing_mesh` turn a section into the 3D mask and the drawn wing.

Physical units (newtons, Reynolds number) are derived in the app from lattice quantities via `set_reynolds(re, length)` and `coefficients()`; the solver itself only knows lattice units and keeps inflow speed `u` well below the Mach limit (~0.3).
