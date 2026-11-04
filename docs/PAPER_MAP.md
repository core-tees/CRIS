# Paper-to-code map

Figure numbers follow the manuscript order; some artwork filenames retain their earlier numbers.

| Paper result | Numerical source / implementation | Reproduction |
|---|---|---|
| Fig. 2, Bratu branches, basins and field accuracy | `experiments/bratu/inverse.py`, `two_cell.py`, `field_accuracy.py`; `results/bratu/` | `saved-plots`; numerical recalculation: `bratu-two-cell`, `bratu-field` |
| Fig. 3, independent-step work and accuracy | 114 named single-step configurations; `results/transport/edfm_single_step.json` | `saved-plots`; configurations beginning `edfm_dt` |
| Fig. 4, fields and production histories | Direct completed-SRDM comparisons in `field_accuracy.json`; retained fields and `production.json` | `saved-plots`, `field-errors`; named trajectories |
| Fig. 5, runtime and work reduction | `results/transport/timing.json` | `saved-plots`; ten headline CRIS configurations and corresponding SRDM trajectories |
| Fig. 6, singular laws and completed/terminated trajectories | `singular_progress_ledger.json`; law evaluation in `transport.hpp` | `saved-plots`; singular configurations |
| Extended Data 1, Bratu local accuracy | `data/bratu/`, `models/bratu/`, `results/bratu/local_accuracy.csv` and `fold_error_bins.csv` | `saved-plots`; numerical recalculation: `bratu-local` |
| Extended Data 2, Newton paths and local-to-field error | `bratu_full_newton_trajectories.json`, `bratu_checkpoint_transfer.json` | `bratu-field`; saved result plots |
| Extended Data 3 / Table 1, full transport suite | `results/transport/overview.json` | `saved-plots`; trajectory configurations |
| Extended Data 4a,b, fitting stages and accumulated bound | `accuracy_tables.json`; `column_chebyshev.cpp`, `column_neural.cpp` | `checkpoints`; `saved-plots` |
| Extended Data 4c–f, refinement | 19 configurations in `experiments/transport/refinement.json` | `refinement` |
| SI time-refined Norne accuracy | `src/accuracy/norne_reference.cpp`; `results/transport/norne_reference/` | `norne-reference` |
| SI inverse construction and alternatives | `training/transport/`, saved network / Chebyshev / spline / RBF results | `transport-local --alternatives`; [fitting guide](FITTING.md) |
| SI amortization | Construction costs and headline simulation times | `python tools/construction_cost.py` recomputes the recovery workloads; `saved-plots` exports the runtime/work ratios |
| SI Norne solver sensitivity | `results/transport/norne_sensitivity/`; the current transport network with a seven-iteration linear allowance | Named configuration `norne_singular_nn_bcgs7` |

Use `python tools/reproduce.py --list` for complete command spellings. `python tools/run_transport.py --list` enumerates individual GPU configurations without submitting or running them.

The final publication artwork is retained in `Figures/paper/`. Numerical redraws use saved measurements and write to `runs/figures/`; they do not require new simulations. The fractured-media and Norne fields are plotted from saved coordinates and saturation values. The SPE-10 ParaView view is included as an unchanged image artifact. [The plot guide](PLOTS.md) lists every plotting script. The field-error command recomputes comparisons directly from saved fields.

## Data conventions

Transport graph files begin with five little-endian 64-bit integers: magic, cell count, connection count, scheduled-step count and graph law marker. They then contain cell storage, outgoing flow, injected flux, compressed upstream adjacency, edge rates and the timestep schedule; the loader in `src/transport/transport.hpp` is the format specification. `.initial` and `.field` files contain one little-endian double per active cell.

The graph's flow and storage are fixed across formulations. The three folded network exports have the same source checkpoint, and the three Chebyshev law wrappers have the same tree. Configuration files select the exponent and solver policy explicitly.
