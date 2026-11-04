"""Sample the paper input splits and label them with PETSc local solves.

Run from the repository root. The native executable accepts the same raw inputs
on CPU and GPU; no complete transport trajectory is used to generate a target.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
import data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--solver',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('data/fitting'))
    ap.add_argument('--chebyshev',action='store_true',help='Also generate the 2^22 fitting pool')
    ap.add_argument('--from-saved',type=Path,help='Use the exact saved input arrays instead of reconstructing the sampling design')
    ap.add_argument('--smoke',action='store_true',help='256 samples per split; not a paper dataset')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    specs=[('train',20,181001,'mixed'),('validation',16,181002,'mixed'),
           ('test',16,181003,'mixed'),('endpoint',13,181004,'endpoint'),
           ('quadratic',13,181005,'mixed'),('singular',13,181006,'mixed')]
    if a.chebyshev:specs.append(('train_cheb',22,181001,'mixed'))
    records=[]
    for name,power,seed,kind in specs:
        if a.from_saved:
            saved=np.load(a.from_saved/(name+'.npz'));raw=saved['raw'];mask=saved['endpoint']
            if a.smoke:raw=raw[:256];mask=mask[:256]
        else:
            raw,mask=data.sample(8 if a.smoke else power,seed,kind)
            if name=='quadratic':raw[:,3]=2
            if name=='singular':raw[:,3]=.2
        prefix=(a.out/name).resolve();rawpath=prefix.with_suffix('.raw')
        raw.astype('<f8').tofile(rawpath)
        command=[str(a.solver.resolve()),'0',str(rawpath),str(prefix),'1']
        with prefix.with_suffix('.log').open('w') as stream:
            subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
        labels=np.fromfile(str(prefix)+'.repeat1.labels',dtype='<f8').reshape(-1,4)
        if len(labels)!=len(raw) or not np.isfinite(labels).all():raise ValueError('Invalid label output')
        np.savez(prefix.with_suffix('.npz'),raw=raw,x=data.transform(raw),y=labels[:,0],logit=labels[:,1],endpoint=mask)
        records.append(dict(name=name,rows=len(raw),seed=seed,input_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest(),npz_sha256=hashlib.sha256(prefix.with_suffix('.npz').read_bytes()).hexdigest(),command=command))
    (a.out/'READY.json').write_text(json.dumps(dict(smoke=a.smoke,sets=records),indent=2))


if __name__=='__main__':main()
