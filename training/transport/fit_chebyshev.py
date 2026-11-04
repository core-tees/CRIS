"""Frozen adaptive cubic Chebyshev construction with GPU FP64 least squares."""
from pathlib import Path
import os,time,json,pickle,hashlib,traceback,subprocess,platform
import numpy as np
import torch
from scipy.linalg import lstsq
from numpy.polynomial.chebyshev import chebvander
import data
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=Path('data/fitting'))
parser.add_argument('--out',type=Path,default=Path('runs/chebyshev_fit'))
parser.add_argument('--parity',type=Path,default=Path('build/parity_coordinates'))
args=parser.parse_args();P=Path(__file__).resolve().parent;O=args.out.resolve()
def save(n,v):
    q=O/n; t=q.with_suffix(q.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(q)
def sync():torch.cuda.synchronize()
def clock():sync();return time.perf_counter()
def basis(x,lo,hi):
    z=2*(x-lo)/(hi-lo)-1
    a=torch.stack((torch.ones_like(z[:,0]),z[:,0],2*z[:,0]**2-1,4*z[:,0]**3-3*z[:,0]),1)
    for j in (1,2):
        v=torch.stack((torch.ones_like(z[:,j]),z[:,j],2*z[:,j]**2-1,4*z[:,j]**3-3*z[:,j]),1)
        a=(a[:,:,None]*v[:,None,:]).reshape(len(x),64 if j==2 else 16)
    return a
def cpu_basis(x,lo,hi):
    z=2*(x-lo)/(hi-lo)-1;a=chebvander(z[:,0],3)
    for j in (1,2):a=(a[:,:,None]*chebvander(z[:,j],3)[:,None,:]).reshape(len(x),-1)
    return a
def predict(root,x,derivative=False):
    out=np.empty(len(x))
    def visit(n,ids):
        if not len(ids):return
        if 'children' in n:
            mask=x[ids,n['axis']]<=n['cut'];visit(n['children'][0],ids[mask]);visit(n['children'][1],ids[~mask]);return
        if not derivative:out[ids]=cpu_basis(x[ids],n['lo'],n['hi'])@n['c'];return
        z=2*(x[ids]-n['lo'])/(n['hi']-n['lo'])-1
        a=np.stack([np.zeros(len(ids)),np.ones(len(ids)),4*z[:,0],12*z[:,0]**2-3],1)*2/(n['hi'][0]-n['lo'][0])
        for j in (1,2):a=(a[:,:,None]*chebvander(z[:,j],3)[:,None,:]).reshape(len(ids),-1)
        out[ids]=a@n['c']
    visit(root,np.arange(len(x)));return out
def stripped(n):
    v={k:n[k] for k in ('lo','hi','c','sse')}
    if 'children' in n:v.update(axis=n['axis'],cut=n['cut'],children=[stripped(c) for c in n['children']])
    return v
def native_parity(root,x,rep):
    rows=[]
    def visit(n):
        i=len(rows);r=np.zeros(74);rows.append(r);r[0]=-1;r[4:7]=n['lo'];r[7:10]=2/(n['hi']-n['lo']);r[10:]=n['c']
        if 'children' in n:r[0]=n['axis'];r[3]=n['cut'];r[1]=visit(n['children'][0]);r[2]=visit(n['children'][1])
        return i
    visit(root);tree=O/('repeat%d.tree'%rep);np.array(rows).tofile(tree)
    physical_x=(x[:,0]+1)/2;beta=np.expm1((x[:,1]+1)/2*data.HIGH[1]);exponent=.2+.9*(x[:,2]+1)
    # Reconstruct exactly the points passed through the native embedding.
    # Original normalized points can differ by ulps after the raw-query roundtrip.
    xn=np.column_stack((2*(physical_x/(1+beta)+beta/(1+beta)*physical_x)-1,2*np.log1p(beta)/np.log1p(2500000.)-1,2*(exponent-.2)/1.8-1))
    save('repeat%d_QUERY_ROUNDTRIP.json'%rep,dict(max_coordinate_difference=float(abs(xn-x).max()),threshold_unchanged=1e-9))
    q=np.column_stack((physical_x,beta,physical_x,exponent,predict(root,xn),predict(root,xn,True)*2*beta/(1+beta)))
    queries=O/('repeat%d.queries'%rep);q.tofile(queries)
    dump=O/('repeat%d.native_coordinates'%rep)
    p=subprocess.run([str(args.parity.resolve()),str(tree),str(queries),str(dump)],capture_output=True,text=True)
    assert p.returncode==0,(p.stdout,p.stderr)
    native=np.fromfile(dump,dtype=np.float64).reshape(2,len(x),10)
    tr=np.array(rows); checks={}
    for index,name in enumerate(('cpu','gpu')):
        a=native[index]; co=tr[a[:,2].astype(int),10:];z=a[:,6:9]
        v=[chebvander(z[:,j],3) for j in range(3)]
        dv=np.column_stack((np.zeros(len(z)),np.ones(len(z)),4*z[:,0],12*z[:,0]**2-3))
        b=(v[0][:,:,None,None]*v[1][:,None,:,None]*v[2][:,None,None,:]).reshape(len(z),64)
        db=(dv[:,:,None,None]*v[1][:,None,:,None]*v[2][:,None,None,:]).reshape(len(z),64)
        expected=np.sum(b*co,axis=1);dexpected=np.sum(db*co,axis=1)*a[:,9]
        checks[name]=dict(value_max=float(abs(a[:,0]-expected).max()),scaled_derivative_max=float((abs(a[:,1]-dexpected)/(1+abs(dexpected))).max()))
    checks['raw_cpu_gpu']=dict(value_max=float(abs(native[0,:,0]-native[1,:,0]).max()),scaled_derivative_max=float((abs(native[0,:,1]-native[1,:,1])/(1+abs(native[0,:,1]))).max()),normalized_coordinate_max=float(abs(native[0,:,3:6]-native[1,:,3:6]).max()),leaf_coordinate_max=float(abs(native[0,:,6:9]-native[1,:,6:9]).max()),different_leaves=int(np.count_nonzero(native[0,:,2]!=native[1,:,2])))
    checks['threshold']=1e-9
    checks['qualification']='Independent polynomial evaluation at each native backend exact leaf coordinates; raw cross-backend differences retained separately.'
    checks['passed']=all(np.isfinite(native).ravel()) and all(checks[k]['value_max']<1e-9 and checks[k]['scaled_derivative_max']<1e-9 for k in ('cpu','gpu')) and checks['raw_cpu_gpu']['different_leaves']==0
    save('repeat%d_NATIVE_PARITY.json'%rep,checks)
    assert checks['passed'],checks

def main():
    O.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('This fitting workflow requires a CUDA-enabled PyTorch installation and a CUDA device.')
    torch.set_num_threads(4)
    save('IDENTITY.json',dict(parameters=65536,leaves=1024,degree=3,samples=4194304,repeats=3,seed=181001,dtype='float64',gpu=torch.cuda.get_device_name(),torch=torch.__version__,host=platform.node(),host_threads=4,solver='CUDA gels least squares; full-column-rank QR; CPU gelsy comparison preflight',selection='largest training SSE leaf; best of median splits in three coordinates',timing='fit workflow includes transfer, adaptive partitioning, least squares, milestone validation and checkpoint output; labels separately'))
    sets={n:dict(np.load(args.data/(n+'.npz'))) for n in ('validation','quadratic','singular','endpoint')}
    # Qualify GPU QR values against the archived CPU rank-revealing least squares.
    sample=sets['validation']; xx=sample['x'][:4096]; yy=sample['y'][:4096]; lo=np.full(3,-1.);hi=np.ones(3)
    A=cpu_basis(xx,lo,hi);cc=lstsq(A,yy,cond=1e-11,lapack_driver='gelsy')[0]
    xt=torch.tensor(xx,device='cuda',dtype=torch.float64,requires_grad=True)
    ag=basis(xt,torch.tensor(lo,device='cuda'),torch.tensor(hi,device='cuda'))
    cg=torch.linalg.lstsq(ag.detach(),torch.tensor(yy,device='cuda'),driver='gels').solution
    pred=ag@cg; grad=torch.autograd.grad(pred.sum(),xt)[0][:,0].detach().cpu().numpy()
    node=dict(lo=lo,hi=hi,c=cg.cpu().numpy(),sse=0.)
    checks=dict(value_max=float(abs(pred.detach().cpu().numpy()-A@cc).max()),derivative_max=float(abs(grad-predict(node,xx,True)).max()),condition=float(np.linalg.cond(A)))
    assert checks['value_max']<1e-10 and checks['derivative_max']<1e-10,checks
    save('PREFLIGHT_PASSED.json',checks)
    del xt,ag,cg,pred
    save('progress.json',dict(stage='shared_training_labels'))
    t=time.perf_counter();z=np.load(args.data/'train_cheb.npz');X=z['x'];Y=z['y'];lab=time.perf_counter()-t
    assert len(Y)==4194304
    save('LABELS.json',dict(load_seconds=lab,x_sha256=hashlib.sha256(X.tobytes()).hexdigest(),y_sha256=hashlib.sha256(Y.tobytes()).hexdigest(),samples=len(X),source=str(args.data/'train_cheb.npz')))
    records=[]
    for rep in (1,2,3):
        start=clock();tx=torch.tensor(X,device='cuda');ty=torch.tensor(Y,device='cuda');transfer=clock()-start
        fit_seconds=0.; validation_seconds=0.; checkpoint_seconds=0.;fit_calls=0
        def fit(ids,lo,hi):
            nonlocal fit_calls
            ix=torch.tensor(ids,device='cuda');a=basis(tx[ix],torch.tensor(lo,device='cuda'),torch.tensor(hi,device='cuda'))
            c=torch.linalg.lstsq(a,ty[ix],driver='gels').solution;e=a@c-ty[ix];sse=float(e@e);coef=c.cpu().numpy();fit_calls+=1
            if not np.isfinite(sse) or not np.isfinite(coef).all():raise RuntimeError('Nonfinite least squares')
            return dict(ids=ids,lo=lo,hi=hi,c=coef,sse=sse)
        tick=clock();root=fit(np.arange(len(X)),np.full(3,-1.),np.ones(3));leaves=[root];fit_seconds+=clock()-tick
        while len(leaves)<1024:
            tick=clock();eligible=[n for n in leaves if len(n['ids'])>=512 and n['sse']>=0]
            if not eligible:raise RuntimeError('Insufficient samples for requested leaf count')
            node=max(eligible,key=lambda n:n['sse']);best=None
            for ax in range(3):
                vals=np.unique(X[node['ids'],ax]);cut=float(np.median(vals));mask=X[node['ids'],ax]<=cut
                if cut<=node['lo'][ax]+1e-12 or cut>=node['hi'][ax]-1e-12 or min(mask.sum(),(~mask).sum())<128:continue
                h=node['hi'].copy();h[ax]=cut;l=node['lo'].copy();l[ax]=cut
                kids=[fit(node['ids'][mask],node['lo'],h),fit(node['ids'][~mask],l,node['hi'])];err=sum(k['sse'] for k in kids)
                if best is None or err<best[0]:best=(err,ax,cut,kids)
            if best is None:node['sse']=-1;continue
            leaves=[n for n in leaves if n is not node];node.update(axis=best[1],cut=best[2],children=best[3]);leaves.extend(best[3]);fit_seconds+=clock()-tick
            if len(leaves)%32==0:save('progress.json',dict(stage='fitting',repeat=rep,leaves=len(leaves),seconds=clock()-start))
            if len(leaves) in (64,256,1024):
                tick=clock();scores={}
                for name,d in sets.items():
                    e=predict(root,d['x'])-d['y'];scores[name]=dict(rmse=float(np.sqrt(np.mean(e*e))),maximum=float(abs(e).max()),p99=float(np.quantile(abs(e),.99)))
                validation_seconds+=clock()-tick;tick=clock()
                # Keep the same training indices/parent fits as the archived checkpoint boundary.
                with (O/('repeat%d_leaves%d.pkl'%(rep,len(leaves)))).open('wb') as f:pickle.dump(root,f)
                checkpoint_seconds+=clock()-tick
                save('repeat%d_leaves%d_scores.json'%(rep,len(leaves)),scores)
        elapsed=clock()-start
        row=dict(repeat=rep,workflow_seconds=elapsed,transfer_seconds=transfer,adaptive_fit_seconds=fit_seconds,validation_seconds=validation_seconds,checkpoint_seconds=checkpoint_seconds,fit_calls=fit_calls,scores=scores,coefficients=65536,training_rmse=float(np.sqrt(sum(n['sse'] for n in leaves)/len(Y))))
        # Accuracy qualifies the construction cost; the deployed model remains untouched.
        row['accuracy_qualified']=scores['validation']['rmse']<=1.05*1.5942367564e-6
        records.append(row);save('RUNS.json',records);print(json.dumps(row),flush=True)
        assert row['accuracy_qualified'],row
        native_parity(root,sets['validation']['x'],rep)
        del tx,ty,root,leaves
    save('COMPLETE.json',dict(repeats=3,all_accuracy_qualified=True,median_workflow_seconds=float(np.median([r['workflow_seconds'] for r in records])),label_seconds=lab,native_deployment_parity='all CPU/CUDA value and derivative checks passed; deployed artifact unchanged'))

if __name__=='__main__':
    try:main()
    except Exception:
        O.mkdir(parents=True,exist_ok=True);save('FAILURE.json',dict(traceback=traceback.format_exc()));raise
