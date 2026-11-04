"""Check the curated package without training or running benchmark simulations."""
from pathlib import Path
import argparse
import hashlib
import json
import struct

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--write-manifest',action='store_true');a=ap.parse_args()
    configs=json.loads((ROOT/'experiments/transport/configurations.json').read_text())
    ids=[c['id'] for c in configs];assert len(ids)==len(set(ids))
    missing={}
    for c in configs:
        for option in ('-graph','-initial_file','-model'):
            i=c['command'].index(option)+1;p=ROOT/c['command'][i]
            assert p.resolve().is_relative_to(ROOT)
            if not p.is_file():missing.setdefault(p.relative_to(ROOT).as_posix(),[]).append(c['id'])
    graphs=[]
    for p in sorted((ROOT/'data/transport').glob('*.bin')):
        with p.open('rb') as f:magic,n,edges,steps,singular=struct.unpack('<5q',f.read(40))
        assert magic==0x4352495354505431 and n>0 and edges>=0 and steps>0
        expected=40+24*n+8*(n+1)+12*edges+8*steps
        assert p.stat().st_size==expected,(p.name,expected,p.stat().st_size)
        initial=p.with_suffix('.initial');assert initial.stat().st_size==8*n
        graphs.append(dict(name=p.name,cells=n,edges=edges,scheduled_steps=steps))
    exclusions={'runs','.git','.venv','__pycache__','verification','.mplconfig'}
    files=[p for p in ROOT.rglob('*') if p.is_file() and not (set(p.relative_to(ROOT).parts)&exclusions) and not p.relative_to(ROOT).parts[0].startswith('build') and p.name!='SHA256SUMS.json']
    inventory={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    manifest=ROOT/'SHA256SUMS.json'
    if a.write_manifest:manifest.write_text(json.dumps(inventory,indent=2)+'\n')
    elif manifest.exists():
        frozen=json.loads(manifest.read_text())
        assert inventory==frozen,'Package differs from SHA256SUMS.json'
    report=dict(files=len(inventory),bytes=sum(v['bytes'] for v in inventory.values()),experiment_commands=len(configs),graphs=graphs,missing_artifacts=missing)
    out=ROOT/'verification';out.mkdir(exist_ok=True)
    (out/'package.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(files=len(inventory),megabytes=round(report['bytes']/1e6,1),commands=len(configs),graphs=len(graphs),missing_artifacts={k:len(v) for k,v in missing.items()}),indent=2))
    if missing:raise SystemExit(2)


if __name__=='__main__':main()
