# Constructing and using an inverse

The supplied data contain independent elemental input/root pairs. Complete simulation trajectories are not used for training. Fitted models in `models/` allow all simulation examples to run without retraining.

## Generate targets

Build `labels_cpu` or `labels_gpu` as described in [BUILD.md](BUILD.md).

```bash
python training/transport/prepare_data.py --solver build-linux/labels_cpu --out runs/targets --from-saved data/fitting --smoke
```

This small example generates 256 inputs per split. Omit `--smoke` for the full dataset; add `--chebyshev` to generate the larger Chebyshev pool. `--from-saved data/fitting` uses the exact supplied inputs. To regenerate the seeded sampling design, omit that option and install the training dependencies (the Sobol sampler uses PyTorch).

## Neural fit

Install PyTorch with `python -m pip install -r requirements-training.txt`. A short CPU example:

```bash
python training/transport/train_impl.py --data runs/targets --out runs/network --device cpu --active 64 --seconds 20 --width 13 --method lm
python tools/export_transport.py --checkpoint runs/network/best.npz --out runs/exported_models
```

This short run introduces fitting and export. To train the reported architecture with the full dataset and 30-minute budget:

```bash
python training/transport/train_impl.py --data data/fitting --out runs/network_full --device cuda --active 65536 --seconds 1800 --width 13 --method br --growth
```

The six width-13 hidden layers give 976 parameters. Export folds the chosen constitutive exponent into the first layer and checks values and derivatives against the three-input checkpoint. The quadratic, intermediate and singular exports all derive from the same fit. Use `--verify-paper` with the bundled checkpoint to check the supplied exports.

## Chebyshev fit

A small adaptive fit on the supplied data:

```bash
python training/transport/fit_chebyshev.py --data data/fitting --samples 4096 --leaves 4 --out runs/chebyshev
python tools/export_chebyshev.py --tree runs/chebyshev/inverse.tree --exponent 2 --out runs/chebyshev/quadratic.model
```

The fitter selects the leaf with the largest training squared error and chooses the best median-coordinate split. Each leaf has a tensor-product cubic polynomial fitted by linear least squares. CPU fitting uses rank-revealing least squares; `--device cuda` uses GPU QR. For the full fitting run, omit `--samples` and use `--leaves 1024 --device cuda`. The fitted model used in the paper is supplied in `models/`.

## Evaluate and simulate

```bash
python experiments/transport/local_accuracy.py --network runs/network/best.npz --tree runs/chebyshev/inverse.tree --out runs/local_errors
python tools/run_transport.py --case column --method neural --model runs/exported_models/shared976_quadratic.model --steps 1 --out runs/new_network
python tools/run_transport.py --case column --method chebyshev --model runs/chebyshev/quadratic.model --steps 1 --out runs/new_chebyshev
```

Select the corresponding exponent when exporting and running a model. Local evaluation reports independent saturation errors; the simulation reports convergence and work. A short or coarse fit need not match the accuracy or convergence of the bundled models.

## Bratu

`training/bratu/regenerate_targets.py` generates labels at the supplied Bratu inputs using the native target solver. With the optional native trainer built:

```bash
build-train/bratu_lm_trainer --train data/bratu/train.bin --validation data/bratu/validation.bin --run-dir runs/bratu_training --weight 0 --max-accepted 10000 --threads 16
```
