"""Draw the supplementary mathematical diagrams with Matplotlib."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/figures'
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'text.usetex':False})

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=220)
    plt.close(fig)

def continuation():
    with (ROOT/'results/bratu/continuation_validation.csv').open(newline='') as stream:records=list(csv.DictReader(stream))
    target=max((r for r in records if r['schedule']=='alpha1_only'),key=lambda r:(-float(r['alpha2']),float(r['rho'])))
    a,rho,beta=-float(target['alpha2']),float(target['rho']),-float(target['alpha1'])
    s=np.linspace(0,1,81);curved_beta=s*rho*np.exp(-s*a-1)
    assert np.isclose(curved_beta[-1],beta)
    fig,ax=plt.subplots(figsize=(3.8,3.05),layout='constrained')
    for x,y,color,name in [(s*0+a,s*beta,'#28618C',r'Fixed $a$'),(s*a,curved_beta,'#C67421','Curved route')]:
        ax.plot(x,y,'--',color=color,lw=2,label=name)
        ax.annotate('',xy=(x[47],y[47]),xytext=(x[43],y[43]),arrowprops=dict(arrowstyle='-|>',color=color,lw=1.4))
    ax.plot(0,0,'o',color='.2');ax.plot(a,0,'o',mfc='white',mec='#28618C')
    ax.plot(a,beta,'*',color='.2',ms=12)
    ax.text(.05,.007,r'$U=0$');ax.text(a-.05,.007,r'$U=a_\star$',ha='right');ax.text(a-.07,beta-.015,'Target',ha='right')
    ax.set(xlabel=r'$a$',ylabel=r'$\beta$',xlim=(-.05,1.65),ylim=(-.009,.14))
    ax.legend(frameon=False,loc='upper left');save(fig,'si_continuation')
    (OUT/'continuation_series.json').write_text(json.dumps(dict(a=a,beta=beta,rho=rho,parameter=s.tolist(),fixed_beta=(s*beta).tolist(),curved_beta=curved_beta.tolist()),indent=2))

def control_volume():
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
    left=np.array([[0,0],[1.2,-.1],[1.5,.9],[.2,1.2]])
    right=np.array([[1.2,-.1],[2.6,.1],[2.8,1.2],[1.5,.9]])
    for ax in axes:
        ax.add_patch(Polygon(left,facecolor='#E5EDF2',edgecolor='.25'))
        ax.plot(.65,.55,'ko',ms=4);ax.text(.6,.72,r'$x_i$');ax.text(.18,.2,r'$K_i$')
        ax.plot(1.35,.4,'ko',ms=3);ax.set(xlim=(-.1,3),ylim=(-.3,1.4),aspect='equal');ax.axis('off')
    ax=axes[0];ax.add_patch(Polygon(right,facecolor='#F2ECE3',edgecolor='.25'))
    ax.plot(2.1,.65,'ko',ms=4);ax.text(2.1,.78,r'$x_j$');ax.text(2.35,.28,r'$K_j$')
    ax.plot([.65,1.35,2.1],[.55,.4,.65],ls='--',color='.4',lw=.8)
    ax.text(1.3,.18,r'$x_{ij}$');ax.text(.8,.37,r'$d_{ij}^{i}$');ax.text(1.68,.38,r'$d_{ij}^{j}$')
    ax.annotate('',xy=(1.95,.22),xytext=(1.35,.4),arrowprops=dict(arrowstyle='->',color='#28618C'))
    ax.text(1.94,.1,r'$\hat{n}_{ij}$');ax.text(1.55,.85,r'$A_{ij}$');ax.set_title('(a) Interior face',loc='left')
    ax=axes[1];ax.plot([.65,1.35],[.55,.4],ls='--',color='.4',lw=.8)
    ax.text(1.28,.18,r'$x_b$');ax.text(.8,.36,r'$d_i^b$');ax.text(1.48,.85,r'$A_i^b$')
    ax.annotate('',xy=(1.95,.22),xytext=(1.35,.4),arrowprops=dict(arrowstyle='->',color='#28618C'))
    ax.text(1.95,.1,r'$\hat{n}_i^b$');ax.set_title('(b) Boundary face',loc='left')
    save(fig,'si_control_volume')

if __name__=='__main__':continuation();control_volume()
