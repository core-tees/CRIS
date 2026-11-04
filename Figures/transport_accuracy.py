from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TEAL,GRAY,PLUM='#008C91','#666666','#8B5E83'
def clean_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="major", width=0.6, length=3, pad=2.5)
    ax.tick_params(which="minor", width=0.4, length=1.6)
    ax.grid(axis="y", which="major", color="#E7E7E7", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

def panel_label(ax, letter, text=""):
    # Position changes only; established type sizes and font are preserved.
    ax.text(-0.18, 1.055, letter, transform=ax.transAxes, fontsize=10,
            weight="bold", va="bottom", ha="left")
    if text:
        ax.text(0, 1.055, text, transform=ax.transAxes, fontsize=8,
                va="bottom", ha="left")

def draw_figure(tables):
    checkpoints, refinement, residual = [tables[k] for k in ("stages", "refinement", "residual")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.7,
                         "axes.labelsize": 7.8, "axes.linewidth": 0.6,
                         "xtick.labelsize": 7.1, "ytick.labelsize": 7.1,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "mathtext.fontset": "dejavusans", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 2, figsize=(180 / 25.4, 170 / 25.4))
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.118, top=0.95,
                        wspace=0.34, hspace=0.58)
    axa, axb, axc, axd, axe, axf = axes.ravel()
    recorded_series = []

    def plot_series(ax, panel, name, x, y, **kwargs):
        line, = ax.plot(x, y, **kwargs)
        assert np.array_equal(line.get_xdata(), np.asarray(x))
        assert np.array_equal(line.get_ydata(), np.asarray(y))
        recorded_series.append({"panel": panel, "metric": name, "x": list(x), "y": list(y),
                                "exported_artist_matches_source": True})
        return line

    # (a) Current three Chebyshev fitting stages, with exact source values.
    plot_series(axa, "a", "checkpoint_field_vs_local",
                [r["local_validation_rmse"] for r in checkpoints],
                [r["field_rmse"] for r in checkpoints],
                color="#28618C", marker="o", ms=3.5, lw=1.2)
    axa.set(xscale="log", yscale="log", xlim=(7e-7, 6e-4), ylim=(8e-6, 2e-3),
            xlabel="Local RMSE ($p=2$)", ylabel="Field RMSE")
    axa.xaxis.set_major_locator(FixedLocator([1e-6, 1e-5, 1e-4]))
    axa.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    for row in checkpoints:
        axa.annotate(str(row["checkpoint"]), (row["local_validation_rmse"], row["field_rmse"]),
                     xytext=(5, 3), textcoords="offset points", fontsize=7.2)
    panel_label(axa, "a", "Chebyshev fitting stages")

    # (b) Previously (f): accumulated original-balance error bound.
    axb.plot([1.5e-4, 2e-2], [1.5e-4, 2e-2], color=GRAY, ls="--", lw=0.9, zorder=1)
    plot_series(axb, "b", "residual_bound_vs_measured_mae",
                [r["measured_mae"] for r in residual], [r["residual_bound"] for r in residual],
                marker="o", color=TEAL, ls="none", ms=4, zorder=3)
    axb.set(xscale="log", yscale="log", xlim=(1.5e-4, 2e-2), ylim=(1.5e-4, 2e-2),
            xlabel="Measured MAE", ylabel="Residual bound")
    axb.xaxis.set_major_locator(FixedLocator([1e-3, 1e-2]))
    axb.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2]))
    for row, offset in zip(residual, [(-1, 9), (-2, 9), (-2, 9), (-7, 9)]):
        axb.annotate(f"{row['cells']:,}", (row["measured_mae"], row["residual_bound"]),
                     xytext=offset, textcoords="offset points", ha="center", fontsize=7)
    axb.text(0.97, 0.07, "Equality", transform=axb.transAxes, color=GRAY,
             ha="right", va="bottom", fontsize=7)
    panel_label(axb, "b", "Accumulated error")
    axb.text(0.97, 1.055, r"$\beta=1$", transform=axb.transAxes, fontsize=8,
             va="bottom", ha="right")

    # (c-f) Previously (b-e): preserve all 19 matched refinement pairs.
    metric_styles = [
        ("inverse_rmse", "Inverse contribution", TEAL, "o", "-"),
        ("discretization_rmse", "Discretization error", GRAY, "s", "--"),
        ("total_rmse", "Total CRIS error", PLUM, "^", "-.")]
    ticks = {1: [100, 400, 1600], 10: [2000, 4000, 8000, 16000],
             100: [8000, 16000, 32000], 1000: [20000, 40000, 80000]}
    for ax, letter, beta in ((axc, "c", 1), (axd, "d", 10), (axe, "e", 100), (axf, "f", 1000)):
        rows = [r for r in refinement if r["beta"] == beta]
        cells = [r["cells"] for r in rows]
        for metric, label, color, marker, style in metric_styles:
            plot_series(ax, letter, metric, cells, [r[metric] for r in rows],
                        color=color, marker=marker, ls=style, lw=1.1, ms=3.2,
                        mfc="white" if metric != "inverse_rmse" else color,
                        mew=0.9, label=label)
        ax.set(xscale="log", yscale="log", ylim=(4e-5, 5e-2),
               xlim=(cells[0] / 1.15, cells[-1] * 1.15), xlabel=r"Cells, $N_h$", ylabel="RMSE")
        ax.xaxis.set_major_locator(FixedLocator(ticks[beta]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k" if v >= 1000 else str(int(v))))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2]))
        panel_label(ax, letter, rf"$\beta={beta:,}$")

    for ax in axes.ravel():
        clean_axis(ax)
    handles = [Line2D([0], [0], color=color, marker=marker, linestyle=style,
                      lw=1.1, ms=3.2, mfc="white" if key != "inverse_rmse" else color,
                      mew=0.9, label=label) for key, label, color, marker, style in metric_styles]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.545, 0.016),
               ncol=3, frameon=False, handlelength=2.6, columnspacing=1.8,
               handletextpad=0.6, fontsize=7.7)
    return fig, recorded_series

if __name__=='__main__':
    out=ROOT/'runs/figures';out.mkdir(parents=True,exist_ok=True)
    fig,series=draw_figure(json.loads((ROOT/'results/transport/accuracy_tables.json').read_text()))
    fig.savefig(out/'ed4_transport_accuracy.pdf');fig.savefig(out/'ed4_transport_accuracy.png',dpi=180)
    (out/'ed4_series.json').write_text(json.dumps(series,indent=2))
