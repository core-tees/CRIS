# Building

The default C++17/OpenMP build provides the column accuracy solvers, Case C reference solver and inverse-export checks:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

On Windows, CMake can use Visual Studio; the example scripts find executables under `build/Release`. On Linux, install a C++ compiler, CMake and OpenMP.

## PETSc transport

Use a real, double-precision PETSc installation and one MPI rank. CPU build:

```bash
export PKG_CONFIG_PATH="$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH"
cmake -S . -B build-linux -DCRIS_PETSC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux -j 4
```

This builds `petsc_transport` and `labels_cpu`. PETSc installed by a package manager may already be discoverable without setting `PKG_CONFIG_PATH`.

For CUDA, PETSc must support CUDA vectors and matrices:

```bash
cmake -S . -B build-gpu -DCRIS_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-gpu -j 4
```

Set the CUDA architecture for your GPU. The paper used an A100, PETSc 3.24.0, GCC 12.3 and CUDA 12.6. The code has no FASTER-specific dependency. GPU fitting uses PyTorch independently of PETSc.

## Bratu training

The optional native Bratu trainer requires LibTorch and Intel MKL. Set `CMAKE_PREFIX_PATH` to their CMake package directories and configure with `-DCRIS_BRATU_TRAINING=ON`. The supplied Python Bratu examples use saved weights and do not require these libraries.

## Stored inputs

`data/transport/*.bin` starts with five little-endian 64-bit integers: format magic, cell count, connection count, scheduled-step count and law marker. The payload contains cell storage, outgoing flow, injection, compressed upstream adjacency, edge flow rates and timesteps. `TransportGraph.hpp` specifies the ordering. `.initial` and `.field` contain one little-endian double per active cell.

Bratu datasets start with `BRATUD01` and a 64-bit count, followed by rows of `a`, beta, root and local slope as little-endian doubles. Transport fitting `.npz` files store raw inputs, normalized inputs (`x`), root targets (`y`) and sampling membership. Model exports are read by `LocalInverse.hpp`; Chebyshev trees contain 74 doubles per node.

Build directories and generated runs are ignored by Git. Use a new output directory for each comparison.
