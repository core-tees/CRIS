"""Fixed-basis convex accuracy screen; no neural training or simulations."""
import os, sys, json, time, hashlib, traceback, gc, itertools
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from basis import coords, design
import torch

P=Path(__file__).resolve().parent
import argparse
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data',type=Path,default=Path('data/fitting'))
ap.add_argument('--out',type=Path,default=Path('runs/spline_rbf'))
ap.add_argument('--local-check',action='store_true')
args=ap.parse_args();OUT=args.out.resolve()
RIDGE=1e-12
BETA_MAX=2.5e6  # Fitting input domain, including endpoint datasets.

def save(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/name; q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(obj,indent=2));q.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()



def build(x,n,kind,progress=False):
    # Preallocate the maximum row support, then pack. Avoid a second full COO.
    limit=64 if kind=='spline' else 81
    dat=np.empty(len(x)*limit,dtype=np.float64);idx=np.empty(len(x)*limit,dtype=np.int32)
    ptr=np.empty(len(x)+1,dtype=np.int64);ptr[0]=0;pos=0
    for start in range(0,len(x),32768):
        a=design(x[start:start+32768],n,kind);k=a.nnz
        if pos+k>len(dat):raise RuntimeError('RBF support exceeded preallocated bound')
        dat[pos:pos+k]=a.data;idx[pos:pos+k]=a.indices
        ptr[start+1:start+a.shape[0]+1]=a.indptr[1:]+pos;pos+=k
        if progress and start%524288==0:save('progress.json',dict(stage='basis',method=kind,n=n,rows=start,total=len(x)))
    return sp.csr_matrix((dat[:pos],idx[:pos],ptr),shape=(len(x),n**3))

def tensor(a,device):
    return torch.sparse_csr_tensor(torch.from_numpy(a.indptr),torch.from_numpy(a.indices),torch.from_numpy(a.data),size=a.shape,device=device)

def solve(a,y,device,seconds=480,maxiter=2000,progress=None):
    # Diagonal change of variables; tiny fixed ridge in scaled coordinates.
    start=time.perf_counter();diag=np.asarray(a.power(2).sum(axis=0)).ravel()
    scale=np.zeros_like(diag);used=diag>0;scale[used]=1/np.sqrt(diag[used])
    a.data*=scale[a.indices]
    at=a.T.tocsr();ag=tensor(a,device);atg=tensor(at,device);del at
    yg=torch.tensor(y,dtype=torch.float64,device=device)
    def mul(m,v):return torch.sparse.mm(m,v[:,None]).ravel()
    def normal(v):return mul(atg,mul(ag,v))+RIDGE*v
    b=mul(atg,yg);q=torch.zeros_like(b);r=b.clone();p=r.clone();rr=torch.dot(r,r);bn=float(torch.linalg.vector_norm(b));reason='iteration_limit'
    if bn==0:reason='converged';iterations=0
    else:
        for iterations in range(1,maxiter+1):
            v=normal(p);den=torch.dot(p,v)
            if not torch.isfinite(den) or float(den)<=0:raise RuntimeError('Nonpositive/nonfinite CG curvature')
            alpha=rr/den;q+=alpha*p;r-=alpha*v;nr=torch.dot(r,r)
            # Recompute true gradient periodically to avoid a false recursive stop.
            if iterations%25==0 or float(torch.sqrt(nr))/bn<1e-9:
                r=b-normal(q);nr=torch.dot(r,r)
                rel=float(torch.sqrt(nr))/bn
                if progress:progress(iterations,rel,time.perf_counter()-start)
                if rel<1e-9:reason='converged';break
                if time.perf_counter()-start>=seconds:reason='time_limit';break
            p=r+(nr/rr)*p;rr=nr
    pred=mul(ag,q).cpu().numpy();true=b-normal(q)
    report=dict(iterations=iterations,stop=reason,relative_normal_gradient=float(torch.linalg.vector_norm(true))/max(bn,1e-300),train_rmse=float(np.sqrt(np.mean((pred-y)**2))),seconds=time.perf_counter()-start,unused_coefficients=int((~used).sum()),ridge_scaled=RIDGE)
    c=q.cpu().numpy()*scale
    # Restore matrix for caller's independent prediction check.
    a.data/=scale[a.indices]
    del ag,atg,yg,q,b,r,p
    if device=='cuda':torch.cuda.synchronize();torch.cuda.empty_cache()
    return c,report

def predict(x,c,n,kind):
    return np.concatenate([design(x[i:i+8192],n,kind)@c for i in range(0,len(x),8192)])

def metrics(pred,y):
    assert pred.shape==y.shape and np.isfinite(pred).all()
    e=pred-y;ae=abs(e)
    return dict(rmse=float(np.sqrt(np.mean(e*e))),maximum=float(ae.max()),p99=float(np.quantile(ae,.99)),median=float(np.median(ae)),count=len(y))

def preflight(device):
    raw=np.array(list(itertools.product([0.,.5,1.],[0.,1e6,2499994.923581116,BETA_MAX],[.2,.8,2.])))
    queries=np.column_stack((raw[:,0],raw[:,1],np.full(len(raw),2.),raw[:,2],np.full(len(raw),.4)))
    transformed=coords(queries)
    assert np.isfinite(transformed).all() and transformed.min()>=0 and transformed.max()<=1
    assert np.max(abs(np.expm1(transformed[:,1]*np.log1p(BETA_MAX))-queries[:,1]))<1e-8
    for kind in ('spline','rbf'):
        for n in (17,33,65,97):
            a=design(transformed,n,kind)
            assert np.max(abs(np.asarray(a.sum(axis=1)).ravel()-1))<1e-13
    save('COORDINATE_PREFLIGHT_PASSED.json',dict(beta_max=BETA_MAX,queries=len(queries),grids=[17,33,65,97],no_clipping=True))
    rng=np.random.default_rng(1144);x=rng.random((1500,3));x[:8]=np.array(list(itertools.product([0.,1.],repeat=3)))
    checks=[]
    for kind in ('spline','rbf'):
        n=5;a=build(x,n,kind);d=design(x,n,kind)
        assert np.max(abs(a.toarray()-d.toarray()))<1e-15
        rowerr=float(abs(np.asarray(a.sum(axis=1)).ravel()-1).max());assert rowerr<1e-13
        c0=rng.normal(size=n**3);y=a@c0
        c,report=solve(a,y,device,seconds=60,maxiter=2000)
        e=float(abs(a@c-y).max());assert e<2e-6,(kind,e,report)
        # Independent dense least squares, including endpoint rows.
        cref=np.linalg.lstsq(a.toarray(),y,rcond=1e-12)[0]
        parity=float(abs(a@c-a@cref).max());assert parity<2e-6
        continuity=0.
        for j in range(3):
            left=x[8:24].copy();right=left.copy();left[:,j]=.5-1e-8;right[:,j]=.5+1e-8
            continuity=max(continuity,float(abs(predict(left,c,n,kind)-predict(right,c,n,kind)).max()))
        assert continuity<1e-5
        checks.append(dict(method=kind,partition_error=rowerr,represented_function_error=e,dense_parity=parity,continuity_difference=continuity,solve=report))
    save('PREFLIGHT_PASSED.json',dict(device=device,checks=checks));print(json.dumps(checks),flush=True)

def load_sets():
    sets={};identity={}
    for name in ('train_cheb','validation','endpoint','quadratic','singular'):
        path=args.data/(name+'.npz');z=np.load(path)
        sets['transport_cheb_4m' if name=='train_cheb' else 'transport_'+name]=(coords(z['raw']),z['y'])
        identity[name]=dict(count=len(z['y']),sha256=sha(path))
    return sets,identity

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('This fitting workflow requires a CUDA-enabled PyTorch installation and a CUDA device; --local-check runs on the CPU.')
    torch.set_num_threads(8);preflight('cuda')
    sets,ident=load_sets();x,y=sets.pop('transport_cheb_4m')
    save('IDENTITY.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,datasets=ident,grids=[17,33,65,97],samples=len(y),coordinate='arccos(1-2*x)/pi; log1p(beta)/log1p(2.5e6); (p-.2)/1.8',spline='open uniform tensor cubic B-spline',rbf='normalized compact C2 Wendland; grid centers; radius2.5 spacings; fixed weights at inference',objective='full-data squared saturation error + 1e-12 scaled coefficient squared norm',fit_limit_seconds=480,iteration_limit=2000,withheld_test_used=False,no_prediction_clipping=True))
    records=[];start=time.perf_counter()
    for n in (17,33,65,97):
        for kind in ('spline','rbf'):
            tag=f'{kind}_{n}';begin=time.perf_counter()
            a=build(x,n,kind,True);buildtime=time.perf_counter()-begin
            def progress(it,g,s):save('progress.json',dict(stage='least_squares',method=kind,n=n,iterations=it,relative_normal_gradient=g,seconds=s))
            c,fit=solve(a,y,'cuda',progress=progress)
            nnz=int(a.nnz);del a;gc.collect()
            np.savez(OUT/(tag+'.npz'),coefficients=c,n=n,method=kind)
            scores={name:metrics(predict(xx,c,n,kind),yy) for name,(xx,yy) in sets.items()}
            row=dict(method=kind,n=n,coefficients=len(c),basis_nnz=nnz,basis_build_seconds=buildtime,fit=fit,scores=scores,workflow_seconds=time.perf_counter()-begin)
            records.append(row);save(tag+'.json',row);save('RUNS.json',records);print(json.dumps(row),flush=True)
            if time.perf_counter()-start>4920:
                save('RESOURCE_STOP.json',dict(reason='82minute workflow guard',completed=len(records),remaining=8-len(records)));return
    save('COMPLETE.json',dict(configurations=len(records),seconds=time.perf_counter()-start,all_optimizations_converged=all(r['fit']['stop']=='converged' for r in records),withheld_test_used=False))

if __name__=='__main__':
    try:
        if args.local_check:
            OUT.mkdir(parents=True,exist_ok=False);preflight('cpu')
        else:main()
    except Exception:
        save('FAILURE.json',dict(traceback=traceback.format_exc()));raise
