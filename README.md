# WingSim

A virtual wind tunnel for testing wings. This first milestone is a 2D tunnel
for NACA airfoils: pick a section, change the angle of attack and wind speed
while it runs, and watch the flow and the lift and drag respond live.

![WingSim running a NACA 2412 at 6 degrees](docs/screenshot.png)

## Run it

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m wingsim
```

The first launch takes a few seconds while Numba compiles the solver. Later
launches use the cached build.

## What you can change

| Control | What it does |
|---|---|
| NACA airfoil | Any 4-digit section, such as 0012, 2412 or 4415 |
| Angle of attack | Rotates the wing about its quarter chord, live |
| Wind speed, chord | Set the Reynolds number and the lift and drag in newtons |
| Show | Speed, vorticity or pressure coefficient |
| Smoke particles | Tracers released from rakes at the inlet |

The panel reports CL, CD, L/D, lift and drag per metre of span, and the
Reynolds number. The plot under the flow shows CL and CD over time, so you can
see vortex shedding when the wing starts to stall.

## How the physics works

The fluid solver is a D2Q9 **Lattice Boltzmann** method (`wingsim/lbm.py`).
Instead of solving the Navier-Stokes equations directly, it tracks particle
populations moving between grid cells and colliding. At low Mach number this
recovers incompressible Navier-Stokes flow. It suits this project because each
cell updates on its own, so it parallelises across CPU cores now and maps
directly onto a GPU later.

* Collision: BGK with a Smagorinsky subgrid model, so higher Reynolds numbers stay stable.
* Walls: half-way bounce-back on the cells inside the airfoil.
* Forces: momentum exchange across every fluid-to-wall link, giving lift and drag directly.
* Boundaries: uniform inflow on the left, free-stream far field top and bottom, open outlet on the right.
* Speed: Numba compiles the kernel and runs it on all CPU cores.

### Accuracy and limits

* For a NACA 0012 at Re 1000 and 0 degrees the solver gives CD = 0.131.
  [Published LBM and XFOIL results](https://arxiv.org/abs/2006.10487) give 0.119.
  The difference comes from the stair-stepped wall and the finite tunnel size.
* The grid is 560 x 280 cells with an 80-cell chord. That resolves the flow up to
  about Re 20,000. Real wings at everyday speeds run at Re in the hundreds of
  thousands. When your settings exceed the limit the app runs at Re 20,000 and
  says so. Expect lower lift slopes and higher drag than textbook high-Re data.
* The flow is 2D, so it models an infinitely long wing with no tip vortices.

## Tests

```bash
pip install pytest
python -m pytest
```

## Roadmap

1. Airfoil wind tunnel (this milestone)
2. GPU solver and finer grids for higher Reynolds numbers
3. 3D wings with span, taper and tip vortices
4. Bird wing shapes and flapping
5. Human-scale wings, with every parameter adjustable live
