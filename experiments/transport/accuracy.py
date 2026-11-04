"""CPU accuracy studies from the paper. No GPU timing measurements."""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def error(a,b):
    e=a-b
    return dict(rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),maximum=float(abs(e).max()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('study',choices=['checkpoints','refinement','norne-reference'])
    ap.add_argument('--build',type=Path,default=ROOT/'build')
    ap.add_argument('--out',type=Path)
    ap.add_argument('--limit',type=int,help='Only first N configurations, for a quick check')
    a=ap.parse_args();out=a.out or ROOT/'runs'/a.study
    out.mkdir(parents=True,exist_ok=False)
    def exe(name):
        for p in (a.build/name,a.build/(name+'.exe'),a.build/'Release'/(name+'.exe')):
            if p.is_file():return str(p.resolve())
        raise FileNotFoundError(f'Build {name} with CMake first')
    rows=[]
    if a.study=='checkpoints':
        stage=json.loads((ROOT/'models/transport/STAGES.json').read_text())
        ref=np.fromfile(ROOT/'data/refinement/checkpoint_reference.bin',dtype='<f8',offset=8)
        configs=[(s['leaves'],s) for s in stage]
        for n,s in configs[:a.limit]:
            prefix=out/f'leaves{n}'
            cmd=[exe('column_chebyshev'),str(ROOT/f'models/transport/cheb_{n}.tree'),'256','150',str(prefix),'1000','front']
            subprocess.run(cmd,check=True)
            summary=json.loads(prefix.with_suffix('.summary.json').read_text())
            summary.update(leaves=n,local=s['scores']['quadratic'],field=error(np.fromfile(str(prefix)+'.t50.field','<f8'),ref))
            rows.append(summary)
    elif a.study=='refinement':
        configs=json.loads((ROOT/'experiments/transport/refinement.json').read_text())
        for c in configs[:a.limit]:
            prefix=out/f"beta{c['beta']:g}_N{c['N']}"
            cmd=[exe('column_neural'),str(ROOT/'models/transport/shared976_quadratic.model'),str(c['N']),str(c['beta']),str(prefix),'1000']
            subprocess.run(cmd,check=True)
            summary=json.loads(prefix.with_suffix('.summary.json').read_text())
            z=np.load(ROOT/c['reference']);u=np.fromfile(str(prefix)+'.t200.field','<f8')
            summary.update(N=c['N'],beta=c['beta'],inverse=error(u,z['discrete_200']),total=error(u,z['analytic_200']),discretization=error(z['discrete_200'],z['analytic_200']))
            rows.append(summary)
    else:
        for cap in [1000,500][:a.limit]:
            prefix=out/f'dt{cap}'
            cmd=[exe('norne_reference'),str(ROOT/'data/transport/norne_singular.bin'),str(ROOT/'data/transport/norne_singular.initial'),str(ROOT/'models/transport/cheb_quadratic.model'),str(cap),str(prefix)]
            subprocess.run(cmd,check=True)
            rows.append(json.loads(prefix.with_suffix('.summary.json').read_text()))
    (out/'RESULTS.json').write_text(json.dumps(rows,indent=2))
    print(out/'RESULTS.json')


if __name__=='__main__':
    main()
