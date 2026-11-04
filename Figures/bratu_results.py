from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import lambertw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/figures';OUT.mkdir(parents=True,exist_ok=True)
BLUE,TEAL,ORANGE='#28618C','#008B8B','#C67421'
SOURCES={};REPORT={}
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def newton(start, fun, line_search):
    u=np.array(start,dtype=float);path=[u.copy()];status='iteration_limit'
    for _ in range(80):
        r,j=fun(u)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(j)):status='undefined';break
        if np.linalg.norm(r)<1e-10:status='converged';break
        try:step=np.linalg.solve(j,-r)
        except np.linalg.LinAlgError:status='singular';break
        alpha=1.
        if line_search:
            for backtrack in range(35):
                trial=u+alpha*step;rr,_=fun(trial)
                if np.all(np.isfinite(rr)) and np.dot(rr,rr)<=(1-1e-4*alpha)*np.dot(r,r):break
                alpha*=.5
            else:status='line_search_failed';break
        u=u+alpha*step;path.append(u.copy())
        if not np.all(np.isfinite(u)) or max(abs(u))>1e6:status='diverged';break
    return np.array(path),status

def bratu_functions():
    def physical(u):
        with np.errstate(over='ignore',invalid='ignore'):
            return .65+.25*u[::-1]-u+.1*np.exp(u),np.diag(-1+.1*np.exp(u))+np.array([[0,.25],[.25,0]])
    def cris(u):
        a=.65+.25*u[::-1];z=-.1*np.exp(a)
        if np.any(z < -1/np.e):return np.full(2,np.nan),np.full((2,2),np.nan)
        w=lambertw(z).real
        return u-a+w,np.array([[1,-.25/(1+w[0])],[-.25/(1+w[1]),1]])
    return physical,cris

def source(path):
    path = Path(path)
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def rows(path):
    path=source(path)
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except UnicodeDecodeError:
        with path.open(newline="", encoding="cp1252") as stream:
            return list(csv.DictReader(stream))

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220)
    REPORT.setdefault("figures", {})[name] = {"size_inches": fig.get_size_inches().tolist()}
    plt.close(fig)

def title(ax, letter, text):
    ax.set_title(r"$\mathbf{" + letter + r"}$  " + text, loc="left", pad=10)

def local_roots():
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.18), layout="constrained")
    ax = axes[0]
    betas = [.05, .1, .2]
    critical_a = np.linspace(-.5, 2.12, 200)
    ax.plot(critical_a, critical_a+1, color=".35", ls=":", lw=1.2, zorder=1)
    ax.annotate("Critical points", xy=(1.65, 2.65), xytext=(.9, 3.65),
                fontsize=8.5, ha="center", color=".25",
                arrowprops=dict(arrowstyle="-", color=".4", lw=.7))
    for beta, color in zip(betas, [BLUE, ORANGE, TEAL]):
        limit = -1 - np.log(beta)
        a = np.linspace(-.5, limit-1e-10, 801)
        lower = a - lambertw(-beta*np.exp(a), 0).real
        upper = a - lambertw(-beta*np.exp(a), -1).real
        ax.plot(np.r_[a,limit], np.r_[lower,limit+1], color=color, lw=1.8, label=rf"$\beta={beta:g}$")
        ax.plot(np.r_[a,limit], np.r_[upper,limit+1], "--", color=color, lw=1.6)
        ax.plot(limit, limit+1, "o", color=color, ms=4)
    ax.set(xlabel=r"$a$", ylabel=r"Local root $U$", xlim=(-.5, 2.12), ylim=(-.5, 5.3))
    ax.legend(frameon=False,loc='upper center',fontsize=8.5,ncol=3,handlelength=1.3,columnspacing=.8)
    title(ax, "a", "Local root branches")
    deficit = np.logspace(-8, 0, 500)
    slope = 1/(1+lambertw(-(1-deficit)/np.e).real)
    ax = axes[1]
    ax.loglog(deficit, slope, color=BLUE, lw=2)
    ax.set(xlabel="Distance to critical point", ylabel=r"Inverse sensitivity $\partial\phi/\partial a$", xlim=(1e-8, 1))
    ax.grid(alpha=.18)
    title(ax, "b", "Inverse sensitivity")
    save(fig, "bratu_local_roots")
    REPORT["local_roots"] = {"betas": betas, "distance_coordinate": "1-beta*exp(a+1)"}

def residuals(points, kind):
    a = .65+.25*points[..., ::-1]
    if kind == "srdm":
        return points-a-.1*np.exp(points)
    q = .1*np.exp(a)
    w = lambertw(np.where(q <= 1/np.e, -q, 0)).real
    result = points-a+w
    result[np.any(q > 1/np.e, axis=-1)] = np.nan
    return result

def trajectories():
    solver_source = source(Path(__file__))
    physical, exact = bratu_functions()
    starts = [(0.1, .1), (1.2, 1.8), (2.2, 2.2)]
    colors = [BLUE, TEAL, ORANGE]
    x = np.linspace(0, 3, 301)
    xx, yy = np.meshgrid(x, x)
    points = np.stack((xx, yy), axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.85))
    fig.subplots_adjust(left=.08, right=.98, top=.86, bottom=.17, wspace=.18)
    traces = []
    for ax, kind, name, letter, fun in zip(axes, ("srdm", "exact"), ("SRDM", "Exact CRIS"), "ab", (physical, exact)):
        rn = np.linalg.norm(residuals(points, kind), axis=-1)
        scale = np.linalg.norm(residuals(np.array([starts[0]]), kind))
        lognorm = np.log10(np.maximum(rn/scale, 1e-8))
        ax.contourf(xx, yy, lognorm, levels=np.linspace(-4, 2, 25), cmap="Greys", extend="both")
        ax.contour(xx, yy, lognorm, levels=[-3,-2,-1,0,1], colors=".55", linewidths=.5)
        for i, (start, color) in enumerate(zip(starts, colors)):
            path, status = newton(start, fun, line_search=False)
            physical_error = float(np.linalg.norm(physical(path[-1])[0], ord=np.inf))
            expected_root = 2.51415481292131 if kind == "srdm" and i == 2 else 1.41617904903837
            assert status == "converged" and physical_error < 1e-9
            assert np.max(np.abs(path[-1] - expected_root)) < 1e-8
            record = dict(method=name, start=start, status=status, trajectory=path.tolist(),
                          updates=len(path)-1, line_search=False, physical_residual_inf=physical_error)
            ax.plot(path[:,0], path[:,1], color=color, lw=1.9)
            ax.plot(*path[0], "o", color=color, mec="white", mew=.65, ms=6)
            ax.annotate(str(i+1), path[0], xytext=(6,4), textcoords="offset points", color=color, weight="bold", fontsize=10)
            ax.plot(*path[-1], "*", color=color, mec="black", mew=.4, ms=11)
            for p, q in zip(path[:-1], path[1:]):
                if np.linalg.norm(q-p) > .055:
                    ax.annotate("", xy=p+.75*(q-p), xytext=p+.40*(q-p), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, mutation_scale=10))
            traces.append(record)
        if kind == "exact":
            bound = (np.log(10)-1-.65)/.25
            ax.axvline(bound, color=".4", lw=.85, ls="--")
            ax.axhline(bound, color=".4", lw=.85, ls="--")
        title(ax, letter, name)
        ax.set(xlim=(0,3), ylim=(0,3), xlabel=r"$U_1$", ylabel=r"$U_2$", aspect="equal", xticks=[0,1,2,3], yticks=[0,1,2,3])
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    fig.legend(handles=[Line2D([],[],marker="o",color=".4",ls="none",label="Initial state"),Line2D([],[],marker="*",color=".4",ls="none",ms=10,label="Terminal root")], loc="lower center", ncol=2, frameon=False)
    save(fig, "bratu_iteration_paths")
    REPORT["trajectories"] = traces
    evidence = {"solver": "Full Newton for both formulations", "tolerance_l2": 1e-10,
                "maximum_updates": 80, "trajectories": traces,
                "source": solver_source.relative_to(ROOT).as_posix(),
                "source_sha256": hashlib.sha256(solver_source.read_bytes()).hexdigest()}
    (OUT / "bratu_newton_paths.json").write_text(
        json.dumps(evidence, indent=2)+"\n", encoding="utf-8")
def fields_and_errors():
    z=np.load(ROOT/'results/bratu/bratu_checkpoint_transfer_fields.npz')
    ref=z['reference'].reshape(9,19,19)[4];error=abs(z['checkpoint_5']-z['reference']).reshape(9,19,19)[4]
    fig,axs=plt.subplots(1,2,figsize=(7.05,3.3),layout='constrained')
    for ax,v,label in zip(axs,[ref,error],['Reference saturation','Absolute error']):
        im=ax.imshow(v,origin='lower',extent=(.05,.95,.05,.95),cmap='cividis');fig.colorbar(im,ax=ax,shrink=.8);ax.set(xlabel='x',ylabel='y',title=label.replace('saturation','solution'))
    save(fig,'bratu_field_slice')
    stages=json.loads((ROOT/'results/bratu/bratu_checkpoint_transfer.json').read_text())['results']
    fig,axs=plt.subplots(1,2,figsize=(7.05,3.2),layout='constrained')
    for ax,metric in zip(axs,['rmse','maximum']):
        for name,label,col in [('test_error','Independent test',BLUE),('reference_inputs_error','Field inputs',TEAL),('field_error','Field solution','#444')]:
            ax.semilogy(range(1,7),[r[name][metric] for r in stages],'-o',color=col,label=label,ms=3)
        ax.set(xlabel='Training stage',ylabel='RMSE' if metric=='rmse' else 'Maximum error')
    axs[1].semilogy(range(1,7),[r['numerical_max_error_bound'] for r in stages],'--',color=ORANGE,label='Field-error bound');axs[1].legend(frameon=False,fontsize=7)
    save(fig,'bratu_checkpoint_errors')
    records=rows(ROOT/'results/bratu/fold_error_bins.csv');records=[r for r in records if r['model']=='mse']
    fig,ax=plt.subplots(figsize=(4,3),layout='constrained')
    ax.loglog([float(r['deficit_median']) for r in records],[float(r['prediction_rmse']) for r in records],'-o',ms=3,color=BLUE)
    ax.set(xlabel='Distance to the critical curve',ylabel='Local RMSE');save(fig,'bratu_near_critical_accuracy')

if __name__=='__main__':local_roots();trajectories();fields_and_errors()
