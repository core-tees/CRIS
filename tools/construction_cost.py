"""Recompute construction costs and recovery workloads from saved measurements."""
from pathlib import Path
import json,math,statistics
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/transport'
def main():
 labels={r['name']:r for r in json.loads((D/'labels_runs.json').read_text())}
 fit=json.loads((D/'cheb_costs.json').read_text());nn=json.loads((D/'training/COMPLETE.json').read_text())
 timings={r['id']:r for r in json.loads((D/'timing.json').read_text())['rows']};components={};recovery=[]
 for method,pool,fitting in [('nn','transport_train',nn['seconds']),('cheb','transport_cheb_4m',fit['median_fit_workflow_seconds'])]:
  records=labels[pool]['native'];targets=statistics.median(r['workflow_seconds'] for r in records)+records[0]['setup_seconds']+records[0]['input_read_seconds']
  prep=targets+fitting;components[method]=dict(target_generation_seconds=targets,fitting_seconds=fitting,combined_seconds=prep)
  for case in ['edfm_quadratic','norne_quadratic','spe10_quadratic','edfm_p08','norne_p08']:
   saving=timings[case+'_srdm']['median_seconds']-timings[case+'_'+method]['median_seconds']
   recovery.append(dict(case=case,model=method,saved_seconds=saving,simulations=math.ceil(prep/saving) if saving>0 else None))
 result=dict(components=components,recovery=recovery);out=ROOT/'runs/construction_cost';out.mkdir(parents=True,exist_ok=True);(out/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
