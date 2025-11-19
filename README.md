# Warp PBF

Naive Warp implementation of a Position-Based Fluids (PBF) solver. The simulation packs Warp kernels for density estimation, constraint solving, and position/velocity updates, then uses `warp.render` to display the particles in real time.

## Requirements

- Python 3.10+ (Warp requires a modern Python build)
- [NVIDIA Warp](https://github.com/NVIDIA/warp) with GPU drivers/toolkit that Warp supports
- NumPy and Pyglet (listed in `requirements.txt`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Warp selects a CUDA device when available; otherwise it will fall back to CPU. Make sure your GPU drivers meet the version expectations from the Warp release you install.

## Running the simulation

```bash
python pbf.py
```

Useful command-line switches:

- `--device=<name>` – Override the device Warp chooses (e.g. `cuda:0`, `cpu`).
- `--num_frames=<n>` – Number of frames to simulate before exiting (default 4800).
- `--verbose` – Prints `ScopedTimer` timings for each major step and render pass.

Once the window opens you should immediately see the particle block fall, interact, and bounce inside the domain box. The solver is configured for 60 FPS output with four sub-steps per render frame, but you can experiment with the parameters inside `Example.__init__` to explore different fluid densities, smoothing lengths, or domains.
