# CRIS: code accompanying the Nature manuscript

CRIS learns an inverse of a recurring elemental equation and evaluates it inside a coupled nonlinear solver. This repository contains the Bratu and nonlinear transport examples, fitted inverses, benchmark inputs, and results used in the manuscript.

**Start with the saved results or the small Bratu example.** Neither requires a GPU or retraining. The complete transport performance experiments use PETSc and CUDA; fitting is a separate, optional workflow.

## Simulator structure

Start with the grid, then the discrete problem, then the solver:

| Component | Source | Responsibility |
|---|---|---|
| Grid | [TransportGraph](src/CORE_SCL/MeshRepresentations/TransportGraph.hpp) | Cell storage, connectivity, prescribed flow and schedule |
| Physics | [FluxFunctors](src/TwoPhaseTransport/FluxFunctors.hpp) | Constitutive flux and its derivative |
| Discrete problem | [TransportProblem](src/CORE_SCL/DiscreteProblem/TransportProblem.hpp) | SRDM and CRIS cell equations and Jacobian entries |
| CRIS inverse | [LocalInverse](src/CRIS/LocalInverse.hpp) | Neural or Chebyshev inverse values and derivatives |
| Nonlinear solver | [PetscTransportSolver](src/CORE_SCL/Solvers/Nonlinear/PetscTransportSolver.hpp) | PETSc residual/Jacobian callbacks and solver setup |
| Simulation | [Simulation](src/TwoPhaseTransport/Simulation.hpp) | Initial state, timestep advancement and output; [CPU version](src/TwoPhaseTransport/CpuSimulation.hpp) |

The entry points are `src/TwoPhaseTransport/main.cu` and `main.cpp`. CUDA evaluation lives in `src/TwoPhaseTransport/CUDA/`.

A case combines a graph and initial state from `data/transport/`, a constitutive law, a choice of SRDM or CRIS, and solver controls. CRIS additionally selects an inverse from `models/transport/`. The named cases in `experiments/transport/configurations.json` assemble these inputs; `tools/run_transport.py` runs them.

The supplied reservoir graphs already contain the discretized storage and flow coefficients. Mesh construction and the pressure solve precede this package's transport calculations. [Data conventions](docs/PAPER_MAP.md#data-conventions) describe the stored quantities.

## Quick start

Run commands from this directory with Python 3.11 or later:

The binary datasets and checkpoints are tracked with Git LFS. After cloning, run `git lfs pull` before the commands below.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python tools/verify_package.py
python tools/reproduce.py --list
python tools/reproduce.py saved-plots
python tools/reproduce.py bratu-two-cell
```

New output goes in `runs/`. Saved paper evidence stays in `results/`. The `saved-plots` command draws the main, Extended Data and supplementary numerical plots with Python/Matplotlib; it needs neither LaTeX nor plotting software outside Python. Fig. 1 is supplied as workflow artwork, and the SPE-10 ParaView view is an image artifact. See [the plot guide](docs/PLOTS.md) and [paper-to-code map](docs/PAPER_MAP.md).

## Repository layout

| Directory | Contents |
|---|---|
| `src/` | Transport residuals, Jacobians and inverse evaluation; PETSc/CUDA drivers; CPU accuracy solvers; local-root generators |
| `experiments/` | Named paper configurations and small experiment entry points |
| `models/` | Frozen Bratu checkpoints, the 976-parameter transport network, and adaptive Chebyshev trees |
| `data/` | Fixed flow graphs, initial states, fitting inputs and accuracy references |
| `training/` | Local target generation and neural, Chebyshev, spline and RBF fitting |
| `results/` | Retained numerical measurements and comparison fields |
| `Figures/` | Plotting scripts and the manuscript's final artwork |
| `tools/` | Reproduction commands, model export and package verification |

## Build and run a CPU example

The default CMake build needs a C++17 compiler and OpenMP, but no PETSc, CUDA or neural-network runtime.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
python experiments/transport/accuracy.py checkpoints
python experiments/transport/accuracy.py refinement --limit 1
```

Omit `--limit 1` to reproduce all 19 refinement calculations. These are accuracy studies, not runtime benchmarks. The separate `norne-reference` study constructs the two time-refined original-balance references and is substantially longer.

## Transport simulations

The frozen graphs contain storage, prescribed flow, connectivity and the nominal time schedule. They avoid a dependency on the original reservoir preprocessing software. The same solver evaluates SRDM, neural CRIS and Chebyshev CRIS through their residual and Jacobian interfaces.

```bash
# On a Linux system with a CUDA-enabled, real-double PETSc installation:
export PKG_CONFIG_PATH="$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH"
cmake -S . -B build-gpu -DCRIS_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu -j 4

python tools/run_transport.py --list
python tools/run_transport.py edfm_quadratic_nn --executable build-gpu/petsc_transport_gpu
python tools/run_transport.py edfm_quadratic_nn --executable build-gpu/petsc_transport_gpu --execute
```

The first command prints the chosen configuration; `--execute` runs it. Each configuration fixes its graph, model, Newton policy and timestep controls. Select a new `--out` prefix to repeat an experiment without overwriting a result. [Build details](docs/BUILD.md) describe the GPU environment and optional CPU PETSc target.

## Fitting the inverses

The supplied models reproduce the simulation results without fitting. For reconstruction, [the fitting guide](docs/FITTING.md) gives commands for the saved input sets, PETSc target generation, and all four approximators. Training data consist of independent elemental solves; complete PDE trajectories are not needed.

The three transport `.model` exports fix the exponent to the three laws used in the experiments. They come from **one** three-input network, whose original checkpoint is `models/transport/network.npz`. The exporter verifies their values and derivatives against that checkpoint.

## Reading the results

`results/transport/timing.json` holds the headline measurements; `overview.json` covers the wider geometry/law suite; `edfm_single_step.json` holds the independent step sweep. Accuracy and construction records are separate so they can be inspected without running performance experiments. Failed trajectories remain identified as incomplete.

`SHA256SUMS.json` identifies the packaged files. `python tools/verify_package.py` checks it, the graph layouts, and the configuration dependencies. Machine-specific source locations have been replaced by archive-relative provenance labels; executable commands use paths within this repository.
