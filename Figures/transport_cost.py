"""Staged ratio presentation; existing evidence only, no simulations."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'results/transport'
OUT=ROOT/'runs/figures';OUT.mkdir(parents=True,exist_ok=True)
source=HERE/'timing.json'
rows=json.loads(source.read_text(encoding='utf-8'))['rows']
lookup={r['id']:r for r in rows}
cases=[('edfm_quadratic','Fractured media\n$p=2$'),('norne_quadratic','Norne\n$p=2$'),
       ('spe10_quadratic','Full SPE10\n$p=2$'),('edfm_p08','Fractured media\n$p=0.8$'),('norne_p08','Norne\n$p=0.8$')]
colors={'nn':'#0072B2','cheb':'#8B4F9F'}
metrics=[('median_seconds','a  Runtime speedup'),('newton','b  Newton work reduction'),('krylov','c  Krylov work reduction')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.6,
 'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
fig,axes=plt.subplots(1,3,figsize=(180/25.4,108/25.4),sharey=True)
fig.subplots_adjust(left=.19,right=.98,bottom=.14,top=.81,wspace=.17)
fig.legend(handles=[Patch(facecolor=colors['nn'],label='Neural CRIS'),Patch(facecolor=colors['cheb'],label='Chebyshev CRIS')],
 loc='upper center',bbox_to_anchor=(.58,.985),ncol=2,frameon=False,columnspacing=2.4,handlelength=1.6)
fig.text(.585,.875,'Larger ratios indicate less cost or work than SRDM',ha='center',fontsize=8)
series=[]
for j,(metric,title) in enumerate(metrics):
    ax=axes[j]
    for i,(case,label) in enumerate(cases):
        if i%2==0: ax.axhspan(i-.45,i+.45,color='#F4F5F6',lw=0,zorder=0)
        baseline=lookup[case+'_srdm']
        for method,offset in [('nn',-.17),('cheb',.17)]:
            row=lookup[case+'_'+method];ratio=baseline[metric]/row[metric]
            ax.barh(i+offset,ratio-1,left=1,height=.25,color=colors[method],edgecolor='none',zorder=3)
            ax.text(ratio*1.07,i+offset,f'{ratio:.1f}×',va='center',ha='left',fontsize=7.7,color='#222222')
            series.append(dict(case=case,method=method,metric=metric,srdm=baseline[metric],cris=row[metric],ratio=ratio,
                               source_policy=row['policy'],baseline_policy=baseline['policy'],
                               model_status='Frozen 976-parameter NN' if method=='nn' else 'Frozen Chebyshev inverse'))
    ax.set(xscale='log',xlim=(1,100),ylim=(4.55,-.55),xlabel='SRDM / CRIS')
    ax.xaxis.set_major_locator(FixedLocator([1,10,100]));ax.xaxis.set_major_formatter(FixedFormatter(['1','10','100']))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(1,color='#777777',lw=.7,zorder=2)
    ax.grid(axis='x',which='major',color='#DFE2E4',lw=.5,zorder=1)
    ax.tick_params(axis='y',length=0,pad=9)
    ax.spines['left'].set_visible(False)
    ax.set_title(title,loc='left',fontweight='bold',pad=10)
axes[0].set_yticks(range(5),[c[1] for c in cases])
fig.savefig(OUT/'fig4_cost.pdf');fig.savefig(OUT/'fig4_cost.png');plt.close(fig)
(OUT/'cost_ratios.json').write_text(json.dumps(series,indent=2))
