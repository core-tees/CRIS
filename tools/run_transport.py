"""Run one named paper configuration, or print its command without executing."""
from pathlib import Path
import argparse
import json
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('id', nargs='?')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--execute', action='store_true', help='Run the printed command')
    ap.add_argument('--executable', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    configs = json.loads((ROOT/'experiments/transport/configurations.json').read_text())
    if a.list:
        for c in configs:
            print(f"{c['id']:42} {c['suite']}")
        return
    matches = [c for c in configs if c['id'] == a.id]
    if len(matches) != 1:
        ap.error('Choose one experiment ID from --list')
    c = matches[0]
    cmd = list(c['command'])
    cmd[0] = str(a.executable.resolve() if a.executable else ROOT/cmd[0])
    out = a.out.resolve() if a.out else ROOT/'runs/transport'/c['id']
    for flag in ('-graph', '-initial_file', '-model'):
        if flag in cmd:
            i = cmd.index(flag)+1
            cmd[i] = str(ROOT/cmd[i])
    cmd[cmd.index('-output')+1] = str(out)
    if '-log_view' in cmd:
        cmd[cmd.index('-log_view')+1] = ':' + str(out) + '.petsc.log'
    print(shlex.join(cmd))
    if not a.execute:
        return
    needed = [Path(cmd[0])] + [Path(cmd[cmd.index(f)+1]) for f in ('-graph','-initial_file','-model') if f in cmd]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        ap.error('Missing required files:\n'+'\n'.join(missing))
    out.parent.mkdir(parents=True, exist_ok=True)
    if list(out.parent.glob(out.name + '*')):
        ap.error('Output prefix already exists; select a new --out to preserve previous results')
    Path(str(out)+'.command.json').write_text(json.dumps(cmd,indent=2))
    with Path(str(out)+'.log').open('w') as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
