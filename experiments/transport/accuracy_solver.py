"""Accuracy-study inputs for the common PETSc transport solver."""

from pathlib import Path
import json
import os
import struct
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def flux(u):
    u = np.clip(u, 0, 1)
    return u*u / (u*u + 0.4*(1-u)**2)


def solve(executable, graph, initial, model, learned, prefix, atol, newton, exponent=2, adaptive=False):
    options = {
        "-graph": graph, "-initial_file": initial, "-model": model,
        "-method": "learned" if learned else "srdm", "-output": prefix,
        "-flux_exponent": exponent, "-threads": 1,
        "-snes_type": "newtonls", "-snes_linesearch_type": newton,
        "-snes_atol": atol, "-snes_rtol": 0, "-snes_stol": 0,
        "-snes_max_it": 1000, "-snes_max_funcs": 100000,
        "-ksp_type": "gmres", "-ksp_gmres_restart": 80,
        "-pc_type": "ilu", "-pc_factor_levels": 0,
        "-ksp_rtol": 1e-6, "-ksp_max_it": 100,
        "-max_cuts": 20 if adaptive else 0,
    }
    if adaptive:
        # Same linear-failure policy as the Case C original-formulation benchmark.
        options.update({"-snes_divergence_tolerance": 1e100,
                        "-snes_max_linear_solve_fail": 100000,
                        "-ksp_converged_maxits": "true"})
    command = [str(Path(executable).resolve())]
    for flag, value in options.items():
        command.extend([flag, str(value)])
    Path(str(prefix)+".command.json").write_text(json.dumps(command, indent=2))
    with Path(str(prefix)+".log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=os.environ | {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    summary = json.loads(Path(str(prefix)+".summary.json").read_text())
    if not summary["complete_requested_steps"]:
        raise RuntimeError(f"Incomplete PETSc accuracy solve: {prefix}")
    return summary


def column(executable, model, n, beta, prefix, front=False):
    prefix = Path(prefix).resolve()
    h = 1.0 if front else 1000.0/n
    dt = 50.0 if front else beta*h
    steps = 1 if front else round(200.0/dt)
    if not front and (n % 4 or abs(steps*dt-200)>1e-8 or steps % 2):
        raise ValueError("Refinement grid does not align with the report times")
    left = 1.0 if front else 0.25
    velocity = beta*h/dt
    initial = np.zeros(n) if front else np.where(np.arange(n)<n//4, 0.25, 0.05)
    graph = Path(str(prefix)+".graph.bin")
    initial_path = Path(str(prefix)+".initial.field")
    initial.astype('<f8').tofile(initial_path)
    injection = np.zeros(n); injection[0] = velocity*flux(left)
    with graph.open('wb') as f:
        np.array([0x4352495354505431,n,n-1,steps,0],dtype='<i8').tofile(f)
        for values in (np.full(n,h),np.full(n,velocity),injection):
            values.astype('<f8').tofile(f)
        np.maximum(np.arange(n+1)-1,0).astype('<i8').tofile(f)
        np.arange(n-1,dtype='<i4').tofile(f)
        np.full(n-1,velocity,dtype='<f8').tofile(f)
        np.full(steps,dt,dtype='<f8').tofile(f)
    learned = str(model) != 'original'
    if not learned:
        model = ROOT/'models/transport/shared976_quadratic.model'
    elif Path(model).suffix == '.tree':
        raw = Path(model).read_bytes()
        if len(raw) % (74*8):
            raise ValueError('Invalid Chebyshev tree')
        model = Path(str(prefix)+'.model')
        model.write_bytes(struct.pack('<iid',-1024,len(raw)//(74*8),2.0)+raw)
    atol = 1e-6 if front else 1e-10
    summary = solve(executable,graph,initial_path,model,learned,prefix,atol,
                    'bt' if front or not learned else 'basic')
    events = [json.loads(line) for line in Path(str(prefix)+'.jsonl').read_text().splitlines()]
    assert len(events)==steps and all(e['accepted'] for e in events)
    old = initial; budget = 0.0
    for k,event in enumerate(events,1):
        current = np.fromfile(str(prefix)+f'.step{k}.field','<f8')
        defect = current-old+beta*(flux(current)-flux(np.r_[left,current[:-1]]))
        budget += float(np.mean(abs(defect)))
        if k==steps or (not front and k==steps//2):
            suffix = '.t50.field' if front else '.t200.field' if k==steps else '.t100.field'
            current.tofile(str(prefix)+suffix)
        old = current
    summary.update(complete=True,N=n,beta=beta,accuracy_only=True,atol=atol,
                   accumulated_mean_absolute_defect=budget)
    Path(str(prefix)+'.summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def norne(executable,cap,prefix):
    prefix = Path(prefix).resolve()
    source = ROOT/'data/transport/norne_singular.bin'
    raw = source.read_bytes()
    header = list(struct.unpack('<5q',raw[:40]))
    _,n,edges,count,_ = header
    offset = 40+24*n+8*(n+1)+12*edges
    intervals = np.frombuffer(raw,dtype='<f8',count=count,offset=offset)
    schedule = np.concatenate([np.full(int(np.ceil(dt/cap)),dt/np.ceil(dt/cap)) for dt in intervals])
    header[3] = len(schedule)
    graph = Path(str(prefix)+'.graph.bin')
    graph.write_bytes(struct.pack('<5q',*header)+raw[40:offset]+schedule.astype('<f8').tobytes())
    summary = solve(executable,graph,ROOT/'data/transport/norne_singular.initial',
                    ROOT/'models/transport/shared976_intermediate.model',False,prefix,
                    1e-6,'bt',exponent=0.8,adaptive=True)
    events = [json.loads(line) for line in Path(str(prefix)+'.jsonl').read_text().splitlines()]
    accepted = [e for e in events if e['accepted']]
    for j,t in enumerate(np.cumsum(intervals),1):
        matches = [i for i,e in enumerate(accepted,1) if abs(e['time_before']+e['dt']-t)<1e-7]
        assert len(matches)==1
        Path(str(prefix)+f'.report{j}.field').write_bytes(Path(str(prefix)+f'.step{matches[0]}.field').read_bytes())
    return summary
