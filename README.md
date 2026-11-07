# CRIS (Coupled Recurrence Inverse System)

CRIS approximates the solution of a recurring elemental nonlinear equation, then couples these local predictions into a global solve. The examples compare the original discretized equations with the approximate CRIS reformulation using the same physical model and grid.

Four workflows cover Bratu branch selection and convergence, inverse construction, transport comparisons, and accuracy under refinement. Start with the Python-only Bratu example, then build the transport solver. Supplied models let you run the simulations without retraining.

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

```bash
python experiments/bratu/two_cell.py
```

This solves the two-cell problem from a grid of initial states and compares the original formulation with learned CRIS. The analytic inverse checks the selected branch. The command saves convergence outcomes, Newton counts and `runs/bratu/convergence.png`.

For the spatial error example:

```bash
python experiments/bratu/field_accuracy.py
```

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

Omit `--steps` for complete trajectories. For GPU runs, pass `--executable build-gpu/petsc_transport_gpu`. The paper's timing measurements used the GPU implementation. Field differences between adaptive trajectories can include temporal-discretization differences.

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
