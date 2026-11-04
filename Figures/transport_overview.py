"""Archived transport overview. Postprocessing only; no new simulations."""
from pathlib import Path
import re, json, hashlib, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator

COLORS={'nn':'#0072B2','cheb':'#8B4F9F','srdm':'#D55E00'}
NAMES={'nn':'Neural CRIS','cheb':'Chebyshev CRIS','srdm':'SRDM'}
MODELS=['nn','cheb','srdm']
CASES=[('Column',2.),('Layer 50',2.),('Layer 75',2.),('Layer 84',2.),('EDFM',2.),('Norne',2.),('Full SPE10',2.),('EDFM',.8),('Norne',.8),('EDFM',.2),('Norne',.2)]
DISPLAY={'Column':'Column','Layer 50':'SPE-10 layer 50','Layer 75':'SPE-10 layer 75','Layer 84':'SPE-10 layer 84','EDFM':'Fractured media','Norne':'Norne','Full SPE10':'Full SPE-10'}
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'runs/figures';HERE.mkdir(parents=True,exist_ok=True)
data=json.loads((ROOT/'results/transport/overview.json').read_text())
selected=data['rows'];accuracy=data['accuracy']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8.5,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(8.9,8.35))
fig.subplots_adjust(left=.19,right=.985,bottom=.075,top=.87,hspace=.30,wspace=.30)
metrics=[('time','a','Elapsed time','Simulation time (s)','log'),('newton','b','Newton work','Newton updates','log'),('krylov','c','Krylov work','Krylov iterations','log'),('steps','d','Accepted timesteps','Accepted steps','linear'),('cuts','e','Timestep reductions','Rejected attempts','linear')]
for ax,(key,letter,title,xlabel,scale) in zip(axes.flat,metrics):
    for r in selected:
        i=CASES.index((r['case'],r['p']))
        y=i+{'nn':-.25,'cheb':0,'srdm':.25}[r['model']]+(-.055 if r['update']=='FS' else .055)
        marker='o' if r['update']=='FS' else 's'; color=COLORS[r['model']]
        ax.plot(r[key],y,marker=marker,ms=3.7,mec=color,mfc=color,mew=.85,ls='none',zorder=3)
        if key=='time' and r['time_low'] is not None:
            ax.plot([r['time_low'],r['time_high']],[y,y],color=color,lw=.8,zorder=2)
        if not r['complete']:
            ax.plot(r[key],y,marker='x',ms=5.4,color='#333',mew=.9,ls='none',zorder=4)
    ax.set_ylim(len(CASES)-.45,-.55)
    ax.set_yticks(range(len(CASES)))
    if key in ['time','steps']:
        labels=[DISPLAY[c]+r'  $p='+f'{p:g}'+r'$' for c,p in CASES]
        ax.set_yticklabels(labels)
    else: ax.set_yticklabels([])
    for yy in [6.5,8.5]:ax.axhline(yy,color='#9d9d9d',lw=.6)
    ax.set_xscale(scale);ax.set_xlabel(xlabel)
    ax.tick_params(axis='y',length=0,pad=7)
    ax.grid(axis='x',color='#e5e5e5',lw=.4,zorder=0)
    ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=8)
    if scale=='log':
        ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.set_xlim(-3,74 if key=='steps' else 69);ax.set_xticks([0,20,40,60])
    if key=='time':ax.set_xlim(.65,650)
    elif key=='newton':ax.set_xlim(100,1e5)
    elif key=='krylov':ax.set_xlim(450,4e5)

# Final field error uses the same case positions and shared left-hand labels.
ax=axes[1,2]
case_map={'column_quadratic':('Column',2.),'layer50_quadratic':('Layer 50',2.),'layer75_quadratic':('Layer 75',2.),'layer84_quadratic':('Layer 84',2.),'edfm_quadratic':('EDFM',2.),'norne_quadratic':('Norne',2.),'spe10_quadratic':('Full SPE10',2.),'edfm_p08':('EDFM',.8),'norne_p08':('Norne',.8)}
for r in accuracy:
    i=CASES.index(case_map[r['case']]); model=r['model']
    ax.plot(r['final_rmse'],i+{'nn':-.25,'cheb':0}[model],marker='D',ms=4.7,color=COLORS[model],ls='none')
ax.set(xscale='log',xlim=(3e-6,.02),ylim=(len(CASES)-.45,-.55),xlabel='Final saturation RMSE\nrelative to SRDM')
ax.set_yticks(range(len(CASES)));ax.set_yticklabels([])
ax.tick_params(axis='y',length=0)
for yy in [6.5,8.5]:ax.axhline(yy,color='#9d9d9d',lw=.6)
ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.xaxis.set_minor_locator(NullLocator())
ax.grid(axis='x',color='#e5e5e5',lw=.4,zorder=0)
ax.set_title('f  Field agreement',loc='left',fontweight='bold',pad=8)

method_handles=[Line2D([],[],color=COLORS[m],marker='o',ls='none',ms=5,label=NAMES[m]) for m in MODELS]
fig.legend(handles=method_handles,loc='upper center',bbox_to_anchor=(.56,.997),ncol=3,frameon=False,columnspacing=2)
style_handles=[Line2D([],[],color='#333',marker='o',ls='none',ms=4,label='Ordinary Newton'),Line2D([],[],color='#333',marker='s',mfc='#333',ls='none',ms=4,label='Backtracking Newton'),Line2D([],[],color='#333',marker='x',ls='none',ms=5,label='Incomplete trajectory')]
fig.legend(handles=style_handles,loc='upper center',bbox_to_anchor=(.56,.955),ncol=3,frameon=False,columnspacing=1.2,fontsize=8.5)
fig.savefig(HERE/'ncs_ed3_transport_overview.pdf')
fig.savefig(HERE/'ncs_ed3_transport_overview.png',dpi=190)
plt.close(fig)

