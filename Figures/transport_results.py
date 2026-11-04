"""Redraw transport numerical results from the retained paper measurements."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'results/transport';OUT=ROOT/'runs/figures'
C={'nn':'#0072B2','cheb':'#8B4F9F','srdm':'#D55E00'}
N={'nn':'Neural CRIS','cheb':'Chebyshev CRIS','srdm':'SRDM'}
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'text.usetex':False,'axes.spines.top':False,'axes.spines.right':False})
def read(n):return json.loads((DATA/(n+'.json')).read_text())
def save(fig,name):
 OUT.mkdir(parents=True,exist_ok=True)
 for ext in ('pdf','png','svg'):fig.savefig(OUT/(name+'.'+ext),dpi=220)
 plt.close(fig)
def key(fig):
 fig.legend(handles=[Line2D([],[],color=C[k],label=N[k],lw=2) for k in C],loc='upper center',ncol=3,frameon=False)
def steps():
 data=read('edfm_single_step');scale=read('advective_timescale')['T_adv']
 fig,axs=plt.subplots(1,3,figsize=(8.3,3.1));fig.subplots_adjust(top=.74,bottom=.2,wspace=.4);key(fig)
 for j,metric in enumerate(['newton','krylov']):
  for m in C:
   for p,style,mark in [('basic','-','o'),('bt','--','s')]:
    rows=sorted([r for r in data['rows'] if r['model']==m and r['policy']==p],key=lambda r:r['dt'])
    axs[j].loglog([r['dt']/scale for r in rows],[max(1,r[metric]) for r in rows],style,color=C[m],lw=1)
    for r in rows:axs[j].plot(r['dt']/scale,max(1,r[metric]),marker=mark if r['complete'] else 'x',color=C[m],ms=3,mfc=C[m] if p=='basic' else 'white')
 for m in ['nn','cheb']:
  rows=sorted([r for r in data['accuracy'] if r['model']==m],key=lambda r:r['dt'])
  axs[2].loglog([r['dt']/scale for r in rows],[r['rmse'] for r in rows],'-o',color=C[m],ms=3)
 for ax,title in zip(axs,['Newton updates','Krylov iterations','Saturation RMSE vs SRDM']):ax.set(xlabel=r'$\Delta\tilde t$',ylabel=title);ax.grid(alpha=.2)
 fig.legend(handles=[Line2D([],[],color='#333',marker='o',label='Ordinary Newton'),Line2D([],[],color='#333',ls='--',marker='s',label='Backtracking Newton'),Line2D([],[],color='#333',ls='none',marker='x',label='Incomplete')],loc='upper center',bbox_to_anchor=(.5,.9),ncol=3,frameon=False)
 save(fig,'single_step')
def histories():
 prod=read('production');errors=read('field_accuracy')['rows']
 cases=[('edfm_quadratic','Fractured media, p=2'),('norne_quadratic','Norne, p=2'),('spe10_quadratic','Full SPE-10, p=2'),('edfm_p08','Fractured media, p=0.8'),('norne_p08','Norne, p=0.8')]
 fig,axs=plt.subplots(5,2,figsize=(7.2,10));fig.subplots_adjust(top=.94,hspace=.5,wspace=.35);key(fig)
 for i,(case,label) in enumerate(cases):
  horizon=max(max(v['time']) for v in prod[case].values())
  for m,r in prod[case].items():axs[i,0].plot(np.array(r['time'])/horizon,r['water_cut'],color=C[m],label=N[m])
  for row in errors:
   if row['case']!=case:continue
   v=row['common_time_fields'];axs[i,1].semilogy([r['time']/horizon for r in v],[r['rmse'] for r in v],'-o',color=C[row['model']],ms=2)
  axs[i,0].set(title=label,ylabel='Water cut');axs[i,1].set(ylabel='Saturation RMSE vs SRDM')
  for ax in axs[i]:ax.set(xlabel=r'Normalized time, $t/t_{\rm end}$',xlim=(0,1));ax.grid(alpha=.2)
 save(fig,'production_and_field_accuracy')

def field_plate():
 prod=read('production');errors=read('field_accuracy')['rows']
 cases=[('edfm_quadratic','Fractured media, $p=2$'),('spe10_quadratic','Full SPE-10, $p=2$'),('norne_p08','Norne, $p=0.8$')]
 fieldmap=LinearSegmentedColormap.from_list('transport_field',[(0,(.65,.80,.82,.3)),(.25,(.86,.93,.93,.55)),(.5,(.99,.97,.93,.85)),(.75,(.84,.25,.14,.95)),(1,(.43,0,.01,1))])
 fig,axs=plt.subplots(3,3,figsize=(7.1,7.1),gridspec_kw={'width_ratios':[1,1.1,1]})
 fig.subplots_adjust(left=.035,right=.985,bottom=.15,top=.87,wspace=.43,hspace=.32)
 key(fig);series={}
 for i,(case,title) in enumerate(cases):
  horizon=max(max(v['time']) for v in prod[case].values());ax=axs[i,0]
  row=next(r for r in errors if r['case']==case and r['model']=='nn')
  if case=='spe10_quadratic':
   ax.imshow(plt.imread(ROOT/'Figures/artifacts/spe10_paraview.jpg'))
  else:
   geometry='edfm' if case.startswith('edfm') else 'norne'
   xyz=np.load(ROOT/f'data/transport/{geometry}_coordinates.npz')['xyz']
   values=np.fromfile(ROOT/row['common_time_fields'][-1]['prepared_field'],'<f8')
   assert len(values)==len(xyz)
   centered=xyz[:,:2]-xyz[:,:2].mean(0);_,_,vh=np.linalg.svd(centered,full_matrices=False)
   xy=centered@vh.T;xy/=max(np.ptp(xy[:,0]),np.ptp(xy[:,1]));order=np.argsort(xyz[:,2])[::-1]
   sc=ax.scatter(xy[order,0],xy[order,1],c=values[order],s=1.2,edgecolors='none',vmin=0,vmax=1,cmap=fieldmap,rasterized=True)
   ax.set_aspect('equal')
  ax.axis('off');ax.set_title('abc'[i]+'  '+title,fontsize=8.5,loc='left')
  series[case]={'production':prod[case],'errors':{}}
  for method,curve in prod[case].items():
   axs[i,1].plot(np.array(curve['time'])/horizon,curve['water_cut'],color=C[method],ls={'nn':'-','cheb':'--','srdm':':'}[method],marker={'nn':'o','cheb':'s','srdm':'^'}[method],ms=2.5)
  for method in ('nn','cheb'):
   row=next(r for r in errors if r['case']==case and r['model']==method);values=row['common_time_fields']
   axs[i,2].semilogy([r['time']/horizon for r in values],[r['rmse'] for r in values],color=C[method],ls='-' if method=='nn' else '--',marker='o' if method=='nn' else 's',ms=2.5)
   series[case]['errors'][method]={'time':[r['time'] for r in values],'rmse':[r['rmse'] for r in values]}
  axs[i,1].set(ylim=(-.02,1.02),ylabel='Water cut');axs[i,2].set(ylabel='Saturation RMSE')
  for ax in axs[i,1:]:
   ax.set_xlim(0,1);ax.grid(alpha=.15);ax.tick_params(labelbottom=i==2)
   if i==2:ax.set_xlabel(r'Normalized time, $t/t_{\rm end}$')
 for ax,heading in zip(axs[0],['Saturation field','Production response','Field difference']):
  pos=ax.get_position();fig.text(pos.x0+pos.width/2,.935,heading,ha='center',fontsize=9,fontweight='bold')
 colorbar=fig.add_axes([.055,.075,.22,.012]);fig.colorbar(sc,cax=colorbar,orientation='horizontal',label='Saturation')
 save(fig,'transport_fields');(OUT/'transport_fields_series.json').write_text(json.dumps(series,indent=2))
def singular():
 fig,axs=plt.subplots(2,2,figsize=(7.2,5.8));fig.subplots_adjust(top=.89,hspace=.4,wspace=.35);key(fig)
 u=np.linspace(0,1,2001);distance=np.logspace(-8,-1,500)
 for p,col in [(2,'#999999'),(.8,'#147D83'),(.2,'#8B4F9F')]:
  d=u*u+.4*(1-u)**p;f=u*u/d
  complement=.4*distance**p/((1-distance)**2+.4*distance**p)
  axs[0,0].plot(u,f,color=col,label=f'p={p:g}');axs[0,1].loglog(distance,complement,color=col)
 axs[0,0].set(xlabel='Saturation',ylabel='Fractional flux');axs[0,0].legend(frameon=False)
 axs[0,1].set(xlabel=r'Distance to full saturation, $1-U$',ylabel=r'Complementary flux, $1-f(U)$');axs[0,1].invert_xaxis()
 for ax,case in zip(axs[1],['EDFM','Norne']):
  for r in read('singular_progress_ledger')['rows']:
   if r['case']!=case:continue
   v=r['progress'];ls='-' if r['policy']=='full_step' else '--'
   x=[0]+[t['elapsed_attempt_seconds'] for t in v];y=[0]+[t['time']/r['horizon'] for t in v]
   ax.plot(x,y,ls,color=C[r['model']],lw=1.2)
   ax.plot(x[-1],y[-1],'o' if r['complete'] else 'x',color=C[r['model']],ms=4)
   failed=[t for t in v if not t['accepted']]
   ax.scatter([t['elapsed_attempt_seconds'] for t in failed],[t['time']/r['horizon'] for t in failed],marker='x',color=C[r['model']],s=10,lw=.6)
  ax.set(title='Fractured media' if case=='EDFM' else case,xlabel='Elapsed solve time (s)',ylabel='Normalized simulation time',yscale='symlog');ax.set_yscale('symlog',linthresh=1e-7);ax.set_xlim(left=0);ax.grid(alpha=.2)
 for ax,letter in zip(axs.ravel(),'abcd'):ax.text(-.13,1.03,letter,transform=ax.transAxes,fontweight='bold')
 fig.legend(handles=[Line2D([],[],color='.3',label='Ordinary Newton'),Line2D([],[],color='.3',ls='--',label='Backtracking Newton')],loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,frameon=False)
 save(fig,'singular_progress')
def alternatives():
 rows=read('spline_rbf/RUNS');fig,ax=plt.subplots(figsize=(5,3.3))
 for method,col,marker in [('spline','#147D83','s'),('rbf','#C47B35','o')]:
  r=[x for x in rows if x['method']==method and x['fit']['stop']=='converged'];ax.loglog([x['coefficients'] for x in r],[x['scores']['transport_validation']['rmse'] for x in r],'-'+marker,color=col,label=method.upper())
 for r in rows:
  if r['fit']['stop']!='converged':ax.loglog(r['coefficients'],r['scores']['transport_validation']['rmse'],'x',color='#777777',label='Iteration/time limit')
 ax.loglog(976,read('training/COMPLETE')['best_validation']['rmse'],'D',color=C['nn'],label=N['nn']);ax.loglog(65536,read('cheb_costs')['validation_rmse'],'D',color=C['cheb'],label=N['cheb'])
 ax.set(xlabel='Parameters / coefficients',ylabel='Validation RMSE');ax.legend(frameon=False,fontsize=7);fig.tight_layout();save(fig,'approximator_accuracy')
 with (OUT/'alternative_fits.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['method','grid','coefficients','stop','validation_rmse','validation_maximum'])
  for r in rows:w.writerow([r['method'],r['n'],r['coefficients'],r['fit']['stop'],r['scores']['transport_validation']['rmse'],r['scores']['transport_validation']['maximum']])
if __name__=='__main__':steps();histories();field_plate();singular();alternatives()
