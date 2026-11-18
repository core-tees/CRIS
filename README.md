# CRIS (Coupled Recurrence Inverse System)

CRIS approximates the solution of a recurring elemental nonlinear equation, then couples these local predictions into a global solve. The examples compare the original discretized equations with the approximate CRIS reformulation using the same physical model and grid.

Four workflows cover Bratu branch selection and convergence, inverse construction, transport comparisons, and accuracy under refinement. Supplied models let you run the simulations without retraining.

## Setup

Install Python 3.11 or later, Git and Git LFS. Clone the repository and create a Python environment:

```bash
git lfs install
git clone https://github.com/core-tees/CRIS.git
cd CRIS
git lfs pull
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

Git LFS downloads the fitted models, training data and benchmark grids. Run the commands below from the repository root. New calculations write to `runs/`; choose a new output directory for each transport comparison.

## 1. Bratu: branch selection and convergence

Build the PETSc executables following [the build instructions](docs/BUILD.md), then run:

```bash
python experiments/bratu/two_cell.py
```

This compares classical Newton, cellwise ASPIN and neural CRIS from 8,281 initial states using full Newton steps. The same two-input fit is used in both the two-cell and three-dimensional examples. The command saves the solver outcomes, Newton counts and `runs/bratu/convergence.png`.

For a short installation check, run `python experiments/bratu/two_cell.py --limit 100 --out runs/bratu_smoke`.

For the three-dimensional example or the 16 reaction--forcing combinations:

```bash
python tools/run_bratu.py --case spatial --out runs/bratu_spatial
python tools/run_bratu.py --case sweep --jobs 4 --out runs/bratu_sweep
```

Use `--method original`, `neural` or `aspin` for one formulation, `--limit` for a short check and `--newton backtracking` to select a different outer Newton policy. The default `ordinary` setting reproduces the full-step comparisons. Both cases use the same saved fit and desired-branch reference solutions. `--dry-run` prints the commands.

To independently check the packaged map and its derivative, run `python experiments/bratu/verify.py` after building PETSc.

## 2. Construct an inverse

The supplied neural and Chebyshev models approximate the same three-input transport inverse. Evaluate them independently:

```bash
python experiments/transport/local_accuracy.py
```

[Inverse construction](docs/FITTING.md) covers target generation, fitting and exporting a new model for simulation. Both training data and the fitted models are included.

## 3. Compare transport formulations

The simulator uses PETSc for the nonlinear and linear solves. Build the CPU or CUDA version using [the build instructions](docs/BUILD.md). Start with one scheduled interval of the column:

```bash
python tools/run_transport.py --case column --steps 1 --executable build-linux/petsc_transport --out runs/column
```

The command runs the original formulation, neural CRIS and Chebyshev CRIS, then writes work counts, field differences and a comparison plot. Use `--method original`, `neural` or `chebyshev` to run one formulation. `--newton ordinary` selects ordinary Newton; the default is backtracking. `--dry-run` prints the commands.

The same inverses support the fractured grid (`edfm`, Case A), million-cell Cartesian grid (`spe10`, Case B), faulted corner-point grid (`norne`, Case C), and Cartesian slices (`layer50`, `layer75`, `layer84`). Select the constitutive exponent with `--exponent 2`, `0.8` or `0.2`. Supported case, exponent and solver configurations are listed in [cases.json](experiments/transport/cases.json); the launcher checks your selection before running.

```bash
python tools/run_transport.py --case edfm --exponent 0.8 --out runs/edfm
python tools/run_transport.py --case norne --exponent 0.2 --newton ordinary --out runs/norne
# One independent step with no timestep subdivision:
python tools/run_transport.py --case edfm --dt 1000 --out runs/step1000
```

Omit `--steps` for complete trajectories. For CUDA-capable GPUs, build the GPU executable and pass `--executable build-gpu/petsc_transport_gpu`. The paper's timing measurements used the GPU implementation. Field differences between adaptive trajectories can include temporal-discretization differences.

## 4. Accuracy and refinement

The accuracy studies use the same CPU PETSc transport solver. Build it following [the build instructions](docs/BUILD.md), then run:

```bash
python experiments/transport/accuracy.py checkpoints
python experiments/transport/accuracy.py refinement --limit 3
```

The checkpoint study shows how improving the local inverse changes field accuracy at fixed resolution. The refinement study holds the neural inverse fixed to distinguish approximation error from discretization error as the mesh and timestep shrink. Results and plots are written to `runs/checkpoints/` and `runs/refinement/`. Omit `--limit` for all 19 refinement pairs. The optional `norne-reference` study constructs time-refined Case C references on the same spatial mesh.

## Simulator organization

| Component | Source |
|---|---|
| Grid storage, connectivity and prescribed flow | `src/CORE_SCL/MeshRepresentations/TransportGraph.hpp` |
| Constitutive flux | `src/TwoPhaseTransport/FluxFunctors.hpp` |
| Original and CRIS discrete equations | `src/CORE_SCL/DiscreteProblem/TransportProblem.hpp` |
| Inverse values and derivatives | `src/CRIS/LocalInverse.hpp` |
| PETSc residual/Jacobian interfaces | `src/CORE_SCL/Solvers/Nonlinear/PetscTransportSolver.hpp` |
| Time advancement | `src/TwoPhaseTransport/Simulation.hpp`, `CpuSimulation.hpp` |
| GPU evaluation | `src/TwoPhaseTransport/CUDA/` |

`experiments/` holds the example calculations and settings; `training/` constructs the inverses; `models/` contains deployable fits; `data/` contains inputs. `results/` retains reference measurements and final fields for comparison.

The supplied graphs contain cell storage, directed flow connections, source terms and timestep schedules. The pressure solve and reservoir mesh preprocessing precede these transport calculations. The graph format is documented in [BUILD.md](docs/BUILD.md).
