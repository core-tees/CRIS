# Plotting the paper results

From the repository root:

```bash
python tools/reproduce.py saved-plots
```

This draws the numerical plots with Matplotlib and writes PDF and PNG files to `runs/figures/`. No training, PDE simulations, LaTeX, MATLAB, PDF cropping or external rendering program is needed. The scripts read the bundled measurements and arrays directly.

| Figure | Script in `Figures/` | Outputs in `runs/figures/` |
|---|---|---|
| Main Fig. 2: Bratu roots, Newton basins, fields and fitting stages | `bratu_panels.py` | `bratu_mechanism` |
| Main Fig. 3: independent-step work and field error | `transport_results.py` | `single_step` |
| Main Fig. 4: fields and production histories | `transport_results.py` | `transport_fields`; `production_and_field_accuracy` also shows all five headline cases |
| Main Fig. 5: runtime and work ratios | `transport_cost.py` | `fig4_cost` (historical output name) |
| Main Fig. 6: constitutive response and singular-case progress | `transport_results.py` | `singular_progress` |
| Extended Data Fig. 1a,b: branch geometry and sensitivity | `bratu_results.py` | `bratu_local_roots` |
| Extended Data Fig. 1c–f: independent local accuracy tests | `bratu_panels.py` | `bratu_independent_accuracy` |
| Extended Data Fig. 2a,b: Newton paths | `bratu_results.py` | `bratu_iteration_paths` |
| Extended Data Fig. 2c: maximum errors and amplification bound | `bratu_panels.py` | `bratu_checkpoint_max` |
| Extended Data Fig. 3: complete transport comparison | `transport_overview.py` | `ncs_ed3_transport_overview` |
| Extended Data Fig. 4: fitting stages, accumulated bound and refinement | `transport_accuracy.py` | `ed4_transport_accuracy` |
| SI continuation routes and control-volume diagram | `supplementary_diagrams.py` | `si_continuation`, `si_control_volume` |
| Spline/RBF accuracy-versus-size comparison | `transport_results.py` | `approximator_accuracy`, `alternative_fits.csv` |

Individual scripts can also be run directly, for example `python Figures/bratu_panels.py`.

## Inputs and static artwork

Bratu plots read `results/bratu/`; transport plots read `results/transport/`. The fractured-media and Norne field views use `data/transport/*_coordinates.npz` and the saved saturation fields. Curves and numerical surfaces are regenerated from data rather than extracted from publication PDFs.

Fig. 1 is a workflow illustration, supplied as `Figures/paper/ncs_fig1_construction.pdf`. The original SPE-10 ParaView view is supplied unchanged as `Figures/artifacts/spe10_paraview.jpg` and placed in the transport figure by Matplotlib. Neither is a numerical reproduction task. The final manuscript plates remain available in `Figures/paper/` as reference artwork.
