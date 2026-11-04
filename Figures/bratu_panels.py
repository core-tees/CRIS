"""Main Bratu figure and Extended Data accuracy panels, drawn from numerical data."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import lambertw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/bratu'
OUT = ROOT/'runs/figures'
BLUE, TEAL, ORANGE = '#28618C', '#008B8B', '#C67421'
plt.rcParams.update({'font.size':8, 'pdf.fonttype':42, 'text.usetex':False,
                     'axes.spines.top':False, 'axes.spines.right':False})

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf','png','svg'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=220)
    plt.close(fig)

def csv_rows(name):
    with (DATA/name).open(encoding='utf8', newline='') as stream:
        return [r for r in csv.DictReader(stream) if r['model']=='mse']

def label(ax, letter, title):
    ax.set_title(f'{letter}  {title}', loc='left', fontweight='bold', fontsize=8, pad=8)

def roots(ax):
    critical = np.linspace(-.5, 2.12, 200)
    ax.plot(critical, critical+1, ':', color='.4', lw=1)
    for beta, color in zip((.05,.1,.2), (BLUE,ORANGE,TEAL)):
        limit = -1-np.log(beta)
        a = np.linspace(-.5, limit-1e-10, 801)
        for branch, style in ((0,'-'),(-1,'--')):
            u = a-lambertw(-beta*np.exp(a),branch).real
            ax.plot(np.r_[a,limit],np.r_[u,limit+1],style,color=color,lw=1.2,
                    label=rf'$\beta={beta:g}$' if branch==0 else None)
        ax.plot(limit,limit+1,'o',color=color,ms=3)
    ax.set(xlabel=r'$a$',ylabel=r'Local root $U$',xlim=(-.5,2.12),ylim=(-.5,5.3))
    ax.legend(frameon=False,fontsize=6.8,loc='upper left')

def main_figure():
    basin=np.load(DATA/'bratu_capture_inputs.npz')
    fields=np.load(DATA/'bratu_checkpoint_transfer_fields.npz')
    stages=json.loads((DATA/'bratu_checkpoint_transfer.json').read_text())['results']
    fig,axes=plt.subplots(2,3,figsize=(7.1,5.4),layout='constrained')
    roots(axes[0,0]);label(axes[0,0],'a','Local root branches')
    c,beta,a0=.25,.1,.65
    edge=(-1-np.log(beta)-a0)/c
    lower,upper=[a0/(1-c)-lambertw(-beta/(1-c)*np.exp(a0/(1-c)),k).real for k in (0,-1)]
    cmap=ListedColormap(['#DADDE0',BLUE,ORANGE,'white'])
    for ax,key,letter,title in zip(axes[0,1:],('srdm','exact'),'bc',('Original equations','Exact CRIS')):
        ax.imshow(basin[key],origin='lower',extent=(-1/60,3+1/60,-1/60,3+1/60),
                  cmap=cmap,norm=BoundaryNorm(np.arange(-.5,4),4),interpolation='nearest')
        ax.set(xlim=(0,edge),ylim=(0,edge),xlabel=r'Initial $U_1$',ylabel=r'Initial $U_2$')
        ax.plot(lower,lower,'*',ms=7,mfc='white',mec='.1')
        ax.plot(upper,upper,'*' if key=='srdm' else 'x',ms=7,mfc='white',mec='.1')
        ax.axvline(edge,ls='--',color='.4',lw=.7,clip_on=False)
        ax.axhline(edge,ls='--',color='.4',lw=.7,clip_on=False)
        if key=='exact':
            gx,gy=np.meshgrid(np.linspace(0,edge-1e-8,400),np.linspace(0,edge-1e-8,400))
            sx=1+lambertw(-beta*np.exp(a0+c*gx)).real
            sy=1+lambertw(-beta*np.exp(a0+c*gy)).real
            ax.contour(gx,gy,1-c*c/(sx*sy),levels=[0],colors=['#8B1E48'],linewidths=.9)
        label(ax,letter,title)
    reference=np.pad(fields['reference'].reshape(9,19,19)[4],1)
    error=np.pad(abs(fields['checkpoint_5']-fields['reference']).reshape(9,19,19)[4],1)
    for ax,v,title,letter in zip(axes[1,:2],(reference,error),('Reference solution','Absolute error'),'de'):
        image=ax.imshow(v,origin='lower',extent=(0,1,0,1),cmap='cividis')
        fig.colorbar(image,ax=ax,shrink=.72,pad=.02)
        ax.set(xlabel=r'$x$',ylabel=r'$y$');label(ax,letter,title)
    ax=axes[1,2]
    for key,title,color in [('test_error','Independent inputs',BLUE),('reference_inputs_error','Field inputs',TEAL),('field_error','Field solution','.25')]:
        ax.semilogy(range(1,7),[s[key]['rmse'] for s in stages],'-o',color=color,ms=3,label=title)
    ax.set(xlabel='Training stage',ylabel='RMSE',xticks=range(1,7));ax.legend(frameon=False,fontsize=6.5)
    label(ax,'f','Local and field errors')
    fig.legend(handles=[Line2D([],[],color='.3',label='Lower branch'),Line2D([],[],color='.3',ls='--',label='Upper branch'),
                        Patch(color=BLUE,label='Lower solution'),Patch(color=ORANGE,label='Upper solution'),Patch(color='#DADDE0',label='Stopped')],
               loc='outside lower center',ncol=3,frameon=False,fontsize=7)
    save(fig,'bratu_mechanism')

def independent_accuracy():
    bins=csv_rows('fold_error_bins.csv');sets=csv_rows('local_accuracy.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.1),layout='constrained')
    for ax,metric,letter,title,ylabel in zip(axes[0],('prediction_rmse','residual_rmse'),'cd',('Root accuracy','Equation residual'),('Root RMSE','Residual RMSE')):
        ax.loglog([float(r['deficit_median']) for r in bins],[float(r[metric]) for r in bins],'-o',color=BLUE,ms=3)
        ax.axvline(1e-8,color='.4',ls=':',label='Training boundary')
        ax.set(xlabel=r'Distance to critical point, $1-\beta e^{a+1}$',ylabel=ylabel)
        ax.legend(frameon=False,fontsize=7);label(ax,letter,title)
    names=['validation','test','test_near_fold'];sets={r['dataset']:r for r in sets}
    for ax,metric,letter,title,ylabel in zip(axes[1],('prediction_rmse','maximum_error'),'ef',('Average error','Maximum error'),('Root RMSE','Maximum error')):
        ax.semilogy(range(3),[float(sets[n][metric]) for n in names],'o',color=BLUE)
        ax.set(xticks=range(3),xticklabels=['Validation','Test','Near-critical test'],ylabel=ylabel,xlim=(-.5,2.5))
        label(ax,letter,title)
    save(fig,'bratu_independent_accuracy')
    stages=json.loads((DATA/'bratu_checkpoint_transfer.json').read_text())['results']
    fig,ax=plt.subplots(figsize=(4.4,3.1),layout='constrained')
    for key,title,color in [('test_error','Independent inputs',BLUE),('reference_inputs_error','Field inputs',TEAL),('field_error','Field solution','.25')]:
        ax.semilogy(range(1,7),[s[key]['maximum'] for s in stages],'-o',color=color,ms=3,label=title)
    ax.semilogy(range(1,7),[s['numerical_max_error_bound'] for s in stages],'--',color=ORANGE,label='Field-error bound')
    ax.set(xlabel='Training stage',ylabel='Maximum error',xticks=range(1,7));ax.legend(frameon=False,fontsize=7)
    label(ax,'c','Field-error amplification');save(fig,'bratu_checkpoint_max')

if __name__=='__main__':
    main_figure()
    independent_accuracy()
