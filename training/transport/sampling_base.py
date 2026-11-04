import math
import numpy as np
import torch
BETA_MAX=2.5e6
LOW=np.array([0.,0.,.2,.2,math.log(.1)])
HIGH=np.array([1.,math.log1p(BETA_MAX),3.,3.,math.log(10.)])

def sample(power,seed,kind='mixed'):
    u=torch.quasirandom.SobolEngine(9,scramble=True,seed=seed).draw_base2(power,dtype=torch.float64).numpy()
    raw=np.empty((len(u),5))
    raw[:,0]=u[:,0]; raw[:,1]=np.expm1(u[:,1]*HIGH[1])
    raw[:,2:4]=.2+2.8*u[:,2:4];raw[:,4]=np.exp(LOW[4]+u[:,4]*(HIGH[4]-LOW[4]))
    # Independent mixtures enrich low exponents without correlating the phases.
    for j in (2,3):
        m=u[:,j+4]<.2;raw[m,j]=.2+.8*u[m,j]
    endpoint=u[:,5]<.2
    if kind=='broad': endpoint[:]=False
    if kind=='endpoint': endpoint[:]=True
    distance=10**(-8+u[:,0]*(math.log10(.05)+8))
    raw[endpoint,0]=np.where(u[endpoint,8]<.5,distance[endpoint],1-distance[endpoint])
    return raw,endpoint
