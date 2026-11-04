# Constructing the elemental inverses

## Transport

Install the plotting dependencies and a PyTorch build suitable for the intended device:

```bash
python -m pip install -r requirements-training.txt
```

`data/fitting/` contains the exact saved input sets and the corresponding PETSc-generated targets. The fixed splits are: 1,048,576 training inputs, 65,536 validation inputs, 65,536 withheld test inputs, and 8,192 inputs in each endpoint/quadratic/singular set. Chebyshev and the spline/RBF comparison use a 4,194,304-input fitting pool.

`python tools/reproduce.py transport-local --alternatives` recomputes validation, endpoint and fixed-exponent errors from all the saved approximators. It runs on the CPU without retraining and leaves the withheld test set unused.

To regenerate targets at the saved inputs with the native solver:

```bash
python training/transport/prepare_data.py --solver build-gpu/labels_gpu --from-saved data/fitting --out runs/regenerated_targets --chebyshev
```

The scalar local solver uses Newton with backtracking, continuation for unresolved inputs, and a safeguarded fallback. It records the resulting roots and residuals. `--smoke` requests only 256 inputs per split. Omitting `--from-saved` reconstructs the seeded sampling design; the saved arrays are the authoritative paper inputs.

Fit the network:

```bash
python training/transport/train_impl.py --data data/fitting --out runs/transport_nn --width 13 --method br --seconds 1800 --active 65536 --growth
```

The network has three inputs, six width-13 hidden layers and 976 parameters. The frozen checkpoint and its training records are supplied. A new fit is a separate realization; use its retained validation-selected checkpoint for deployment.

```bash
python tools/export_transport.py --checkpoint models/transport/network.npz --out runs/exported_models --verify-paper
```

This command reproduces the three frozen exports and checks their value/derivative agreement. To export a newly trained checkpoint, give its path and omit `--verify-paper`.

Fit adaptive Chebyshev polynomials:

```bash
python training/transport/fit_chebyshev.py --data data/fitting --out runs/chebyshev_fit --parity build-gpu/parity_coordinates
```

The frozen protocol fits three repetitions, each reaching 1,024 cubic leaves (65,536 coefficients), with saved stages at 64, 256 and 1,024 leaves. The workflow records fitting, transfer, validation and checkpoint time. Native value/derivative checks use identical floating-point input coordinates on CPU and GPU.

Fit the alternative fixed bases:

```bash
python training/transport/fit_spline_rbf.py --local-check --out runs/basis_check
python training/transport/fit_spline_rbf.py --data data/fitting --out runs/spline_rbf
```

The screen uses grids 17, 33, 65 and 97 for tensor cubic B-splines and normalized compact Wendland RBFs. Each fit has the recorded 2,000-iteration / 480-second limit. Saved coefficients and all eight outcomes are in `results/transport/spline_rbf/`. A limit stop reports an unfinished optimization, not a bound on the approximation family.

## Bratu

`data/bratu/` supplies the sampled lower-branch inputs and labels; `models/bratu/` supplies six retained supervised fitting stages. The selected analytic inverse is also available in `experiments/bratu/inverse.py`.

```bash
build-train/bratu_lm_trainer --train data/bratu/train.bin --validation data/bratu/validation.bin --run-dir runs/bratu_training --weight 0 --max-accepted 10000 --threads 16
```

The binary dataset starts with `BRATUD01` and a 64-bit sample count, followed by rows of four little-endian doubles: `a`, `beta`, root, and local root slope. `training/bratu/regenerate_targets.py` regenerates labels at those same inputs using `labels_cpu` or `labels_gpu`.

## Saved formats

Transport fitting `.npz` files contain `raw` (five native scalar inputs), `x` (three transformed network inputs), `y` (root targets), `logit` (root coordinate) and `endpoint` (sampling membership). Native raw inputs use `[alpha, beta, wetting exponent, nonwetting exponent, M]`; the paper fixes the wetting exponent to 2 and `M` to 0.4. The solver's label output has root, root coordinate, normalized residual and derivative columns.
