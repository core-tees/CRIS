"""Plot two-cell branch selection and the coupled Newton turning point.

Uses saved basin outcomes; analytic evaluations explain the Newton geometry.
No training or new basin sweep is performed.
"""
from pathlib import Path
import csv
import json
import os
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOC = ROOT
OUT=ROOT/'runs/figures';OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.special import lambertw

BLUE, ORANGE, GREY, BURGUNDY = '#28618C', '#C67421', '#DADDE0', '#8B1E48'
c, beta, a0 = .25, .1, .65
edge = (-1-np.log(beta)-a0)/c
lower, upper = [a0/(1-c)-lambertw(-beta/(1-c)*np.exp(a0/(1-c)), k).real for k in [0,-1]]
turn = (np.log((1-c)/beta)-(1-c)-a0)/c

def phi(a):
    return a-lambertw(-beta*np.exp(a)).real

def residual(s):
    return s-phi(a0+c*s)

def slope(s):
    return 1-c/(1+lambertw(-beta*np.exp(a0+c*s)).real)

data_path=ROOT/'results/bratu/bratu_capture_inputs.npz'
data = np.load(data_path)
assert np.count_nonzero(data['initial']) == 6241
assert np.count_nonzero(data['exact']==1) == 6028
assert np.count_nonzero(data['srdm']==1) == 4355
assert np.count_nonzero(data['srdm']==2) == 1826

# Check the feedback interpretation on the same initial-state lattice.
x,y=np.meshgrid(data['grid'],data['grid'])
u=np.column_stack([x.ravel(),y.ravel()])[data['initial'].ravel()]
a=a0+c*u[:,::-1]
z=lambertw(-beta*np.exp(a)).real
p,q=(c/(1+z)).T
g=u-(a-z)
det=1-p*q
delta=-np.column_stack([g[:,0]+p*g[:,1],q*g[:,0]+g[:,1]])/det[:,None]
beyond=det<0
assert beyond.sum()==207
assert np.all(delta[beyond]>0)
assert np.all(np.any(u[beyond]+delta[beyond]>edge,axis=1))
assert 1-beta*np.exp(lower)>0 and 1-beta*np.exp(upper)<0

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),gridspec_kw={'width_ratios':[1,1,1.12]})
fig.subplots_adjust(left=.07,right=.995,bottom=.29,top=.87,wspace=.36)
cmap=ListedColormap([GREY,BLUE,ORANGE,'white'])
for index,(key,title) in enumerate([('srdm','SRDM'),('exact','Exact CRIS')]):
    ax=axes[index]
    ax.imshow(data[key],origin='lower',extent=(-1/60,3+1/60,-1/60,3+1/60),
              cmap=cmap,norm=BoundaryNorm(np.arange(-.5,4),4),interpolation='nearest')
    ax.set(xlim=(0,edge),ylim=(0,edge),xticks=[0,1,2],yticks=[0,1,2],
           xlabel=r'Initial $U_1$',ylabel=r'Initial $U_2$' if index==0 else '')
    ax.set_title(('a  ' if index==0 else 'b  ')+title,loc='left',fontweight='bold')
    ax.plot(lower,lower,'*',ms=8,mfc='white',mec='.15',mew=.6)
    ax.plot(upper,upper,'*' if index==0 else 'x',ms=6,color='.1',
            markerfacecolor='white',markeredgewidth=.9)
    ax.axvline(edge,color='.4',ls='--',lw=.8,clip_on=False)
    ax.axhline(edge,color='.4',ls='--',lw=.8,clip_on=False)
    if index==1:
        ax.tick_params(labelleft=False)
        gx=np.linspace(0,edge-1e-8,500)
        xx,yy=np.meshgrid(gx,gx)
        sigx=1+lambertw(-beta*np.exp(a0+c*xx)).real
        sigy=1+lambertw(-beta*np.exp(a0+c*yy)).real
        ax.contour(xx,yy,1-c*c/(sigx*sigy),levels=[0],colors=[BURGUNDY],linewidths=1.2)
        ax.plot([0,edge],[0,edge],':',color='white',alpha=.8,lw=.8)
        ax.plot(turn,turn,'o',ms=3,color=BURGUNDY,mec='white',mew=.3)
fig.legend(handles=[Patch(color=BLUE,label='Lower solution'),Patch(color=ORANGE,label='Upper solution'),
                    Patch(color=GREY,label='Stopped')],loc='lower center',bbox_to_anchor=(.36,-.005),
            ncol=3,frameon=False,handlelength=1,columnspacing=1.2,fontsize=8)
ax=axes[2]
s=np.linspace(.9,edge-1e-9,1000)
ax.axvspan(edge,3.06,color=GREY,alpha=.55)
ax.axhline(0,color='.55',lw=.6)
ax.plot(s,residual(s),color='.2',lw=1.5)
ax.axvline(edge,color='.4',ls='--',lw=.8)
ax.plot(lower,0,'*',ms=8,color=BLUE)
ax.plot(turn,residual(turn),'o',ms=4,color=BURGUNDY)
for start,color in [(2.25,BLUE),(2.58,ORANGE)]:
    value=float(residual(start))
    end=float(start-value/slope(start))
    ax.plot([start,end],[value,0],'--',color=color,lw=.9)
    ax.plot(start,value,'o',color=color,ms=3)
    ax.plot(end,0,'v',color=color,ms=4)
ax.set(xlim=(.9,3.06),ylim=(-.2,.56),xlabel=r'$U_1=U_2$',
        ylabel='CRIS residual',xticks=[1,2,3],yticks=[0,.2,.4])
ax.set_title('c  Symmetric residual',loc='left',fontweight='bold')
for ext in ['pdf','png','svg']:
    fig.savefig(OUT/f'bratu_capture_geometry.{ext}',dpi=220)
plt.close(fig)
report={'lower_coupled_solution':float(lower),'upper_coupled_solution':float(upper),
        'upper_state_principal_inverse':float(phi(a0+c*upper)),
        'inverse_domain_edge':float(edge),'diagonal_turning_point':float(turn),
        'residual_at_turn':float(residual(turn)),
        'negative_determinant_starts':int(beyond.sum()),
        'positive_corrections_and_first_step_exits':int(beyond.sum())}
(OUT/'bratu_capture_geometry.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
