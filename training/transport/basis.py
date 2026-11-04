"""Fixed coordinates and bases for the spline/RBF comparison."""
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline

BETA_MAX = 2.5e6

def coords(raw):
    x=np.column_stack((raw[:,0],np.log1p(raw[:,1])/np.log1p(BETA_MAX),(raw[:,3]-.2)/1.8))
    assert np.isfinite(x).all() and x.min()>=-1e-14 and x.max()<=1+1e-14
    # Endpoint-focused coordinate, identical for both fixed representations.
    x[:,0]=np.arccos(1-2*x[:,0])/np.pi
    return x

def design(x,n,kind):
    """CSR fixed basis, each row forms a partition of unity."""
    if kind=='spline':
        knots=np.r_[np.zeros(4),np.arange(1,n-3)/(n-3),np.ones(4)]
        bs=[BSpline.design_matrix(x[:,j],knots,3,extrapolate=False) for j in range(3)]
        ids=[b.indices.reshape(-1,4) for b in bs]; vals=[b.data.reshape(-1,4) for b in bs]
        ix=((ids[0][:,:,None,None]*n+ids[1][:,None,:,None])*n+ids[2][:,None,None,:]).reshape(len(x),64)
        v=(vals[0][:,:,None,None]*vals[1][:,None,:,None]*vals[2][:,None,None,:]).reshape(len(x),64)
        a=sp.csr_matrix((v.ravel(),ix.ravel(),np.arange(len(x)+1,dtype=np.int64)*64),shape=(len(x),n**3))
        a.eliminate_zeros(); return a
    # Compact C2 Wendland RBFs at fixed Cartesian centers. Normalization is
    # independent of coefficients: the least-squares fit remains convex.
    z=x*(n-1); base=np.floor(z).astype(np.int32)
    rows=[];cols=[];vals=[]
    for offset in itertools.product(range(-2,4),repeat=3):
        center=base+offset; valid=np.all((center>=0)&(center<n),axis=1)
        r=np.sqrt(np.sum((z-center)**2,axis=1))/2.5;valid &= r<1
        ids=np.flatnonzero(valid)
        if len(ids):
            rr=r[ids];rows.append(ids);cols.append((center[ids,0]*n+center[ids,1])*n+center[ids,2]);vals.append((1-rr)**4*(4*rr+1))
    a=sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(x),n**3))
    sums=np.asarray(a.sum(axis=1)).ravel();assert np.all(sums>0)
    a.data/=np.repeat(sums,np.diff(a.indptr));return a
