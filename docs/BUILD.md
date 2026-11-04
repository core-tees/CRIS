# Building the solvers

## CPU accuracy calculations

The default build produces `column_neural`, `column_chebyshev`, `norne_reference` and `check_export`. These use C++17 and OpenMP. A Visual Studio multi-configuration build places executables in `build/Release`; the accuracy wrapper detects this automatically.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## PETSc and CUDA

Build PETSc with real, double-precision scalars, MPI and CUDA support. The CUDA backend must provide sequential CUDA vectors and CUDA AIJ matrices. The paper's GPU is an NVIDIA A100 (compute capability 8.0); set `CMAKE_CUDA_ARCHITECTURES` explicitly for another supported device.

The native transport runs used PETSc 3.24.0, GCC 12.3.0 and CUDA 12.6.0. The Python fitting environment used PyTorch 2.1.2 with CUDA 12.1.1. These are the recorded reference versions; the default CPU targets also build with MSVC on Windows.

The package has no FASTER-specific build or runtime dependency. Use a suitable local machine or cluster. The GPU fitters require CUDA and double-precision operations; they do not require an A100. Available memory and FP64 throughput affect which fitting workloads are practical.

```bash
export PKG_CONFIG_PATH="$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH"
cmake -S . -B build-gpu -DCRIS_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-gpu -j 4
```

Targets: `petsc_transport_gpu`, `labels_gpu`, and `parity_coordinates`. `-DCRIS_PETSC=ON` additionally builds `petsc_transport` and `labels_cpu`. The CPU PETSc driver is useful for functional experiments; the published transport timings use the GPU driver and the options stored in `experiments/transport/configurations.json`.

Use one MPI rank for the supplied graph experiments. Run GPU commands inside a resource allocation on a cluster; the wrapper does not submit jobs or choose an account. The largest graph is full SPE-10. The fitting comparison requires more memory than the small accuracy examples; the original spline/RBF workflow used 64 GB of host memory and a 40 GB GPU.

## Bratu fitting

The C++ Bratu trainer additionally needs LibTorch and Intel MKL. Point `CMAKE_PREFIX_PATH` at their CMake package directories:

```bash
cmake -S . -B build-train -DCRIS_BRATU_TRAINING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-train --target bratu_lm_trainer --config Release
```

Training is optional. The Python Bratu experiments use the bundled checkpoints with NumPy/SciPy and do not depend on LibTorch.

## Tested platforms

| Environment | Checks performed |
|---|---|
| Windows, MSVC 19.42 | Four default C++ targets; three model-export tests; Bratu and transport accuracy examples; saved-result plots |
| Ubuntu 26.04 under WSL2, GCC 15.2, PETSc 3.24.4 | All six CPU targets; three model-export tests; neural/Chebyshev accuracy calculations; Bratu and transport target generation; one-step PETSc solves for SRDM and both CRIS approximators |

The Linux and Windows accuracy fields agree to roundoff. The Ubuntu environment has no CUDA compiler, so these local checks cover CPU execution and PETSc integration. The original GPU experiments used the reference environment listed above.

## Plotting

`python tools/reproduce.py saved-plots` redraws the numerical figures with Matplotlib from the bundled data. Mathematical labels use Matplotlib's built-in mathtext; no LaTeX installation, PDF-composition library or ParaView installation is required. [The plot guide](PLOTS.md) maps each figure to its script and inputs.

## Generated output

`build*/`, `runs/` and local virtual environments are disposable outputs. They are excluded from version control. The package verifier records local verification separately under `verification/`.
