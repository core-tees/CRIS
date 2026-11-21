# Building

The default C++17/OpenMP build provides the inverse-export checks:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

On Windows, CMake can use Visual Studio; `--config Release` and `-C Release` select the Release configuration for the build and tests. On Linux, install a C++ compiler, CMake and OpenMP.

## PETSc transport

Use a real, double-precision PETSc installation and one MPI rank. CPU build:

```bash
export PKG_CONFIG_PATH="$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH"
cmake -S . -B build-linux -DCRIS_PETSC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux -j 4
```

This builds `petsc_transport`, `petsc_bratu` and `labels_cpu`. Transport comparisons, column accuracy studies and Case C references use `petsc_transport`; the Bratu comparisons use `petsc_bratu`. PETSc installed by a package manager may already be discoverable without setting `PKG_CONFIG_PATH`.

The transport and Bratu launchers default to executables directly in `build-linux/`. For a Visual Studio PETSc build in that directory, build with `cmake --build build-linux --config Release`, then pass `--executable build-linux/Release/petsc_transport.exe` to `tools/run_transport.py` or `--executable build-linux/Release/petsc_bratu.exe` to `tools/run_bratu.py`, `experiments/bratu/two_cell.py` and `experiments/bratu/verify.py`. The accuracy driver, `experiments/transport/accuracy.py`, accepts `--build build-linux` and searches both that directory and its `Release` subdirectory. Adjust these paths if you choose another build directory.

For GPU execution, use a CUDA-capable NVIDIA GPU and a compatible CUDA-enabled PETSc installation. Set `CUDA_ARCH` to the compute capability of the target GPU (for example, `86`):

```bash
export CUDA_ARCH=86  # Replace with the target GPU's compute capability.
cmake -S . -B build-gpu -DCRIS_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build build-gpu -j 4
```

The GPU solver uses double precision; GPU fitting uses CUDA-enabled PyTorch independently of PETSc.

## Bratu training

The optional native Bratu trainer requires LibTorch and Intel MKL. Set `CMAKE_PREFIX_PATH` to their CMake package directories and configure with `-DCRIS_BRATU_TRAINING=ON`. The Bratu simulations use saved weights and do not require these libraries.

## Stored inputs

`data/transport/*.bin` starts with five little-endian 64-bit integers: format magic, cell count, connection count, scheduled-step count and law marker. The payload contains cell storage, outgoing flow, injection, compressed upstream adjacency, edge flow rates and timesteps. `TransportGraph.hpp` specifies the ordering. `.initial` and `.field` contain one little-endian double per active cell.

The older Bratu datasets start with `BRATUD01` and a 64-bit count, followed by rows of `a`, beta, root and local slope as little-endian doubles. The regularized Bratu training data contain raw rows of four little-endian doubles: input, reaction coefficient, target and weight. Transport fitting `.npz` files store raw inputs, normalized inputs (`x`), root targets (`y`) and sampling membership. Model exports are read by `LocalInverse.hpp`; Chebyshev trees contain 74 doubles per node.

Build directories and generated runs are ignored by Git. Use a new output directory for each comparison.
