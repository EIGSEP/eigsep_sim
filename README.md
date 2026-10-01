# eigsep-sim

Simulation and recovery tools for EIGSEP radiometry: beam, sky and terrain
models, observers (Earth surface, lunar surface, lunar orbit), a JAX forward
model, and linear and nonlinear recovery.

## Linear recovery

- `recovery.py`: design-matrix construction and normal-equation solves
  shared with the lunar-orbit (bloom21cm) studies.
- `design_matrix.py`: the fixed-site version, for a ground-based antenna.
  It gives sky pixels, ground temperature regions below a terrain horizon,
  and additive offsets per frequency, for an assumed beam (`HealpixBeam`
  reads the HFSS and empirical-beam npz files). It also has a Gaussian-prior
  solve and a Fisher summary that flags unconstrained columns. The Marjum
  studies built on it are in
  `data-analysis/scripts/marjum-2026-07/ground_sky/`.

## Recent changes

- 2026-10-01: added `design_matrix.py`, for solving ground temperature and
  sky from an assumed beam at a fixed site.
