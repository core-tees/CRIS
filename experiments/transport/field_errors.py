"""Recompute direct-SRDM field errors at common saved report times."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
 rows=json.loads((ROOT/'results/transport/field_accuracy.json').read_text())['rows'];out=[]
 for row in rows:
  for r in row.get('common_time_fields',[]):
   a=np.fromfile(ROOT/r['prepared_field'],'<f8');b=np.fromfile(ROOT/r['reference_field'],'<f8');assert a.shape==b.shape
   e=a-b;rmse=float(np.sqrt(np.mean(e*e)));maximum=float(abs(e).max())
   assert np.isclose(rmse,r['rmse'],rtol=1e-10,atol=1e-14),(row['case'],rmse,r['rmse'])
   assert np.isclose(maximum,r['max_error'],rtol=1e-10,atol=1e-14)
   out.append(dict(case=row['case'],model=row['model'],time=r['time'],cells=len(a),rmse=rmse,maximum=maximum))
 dest=ROOT/'runs/field_errors';dest.mkdir(parents=True,exist_ok=True);(dest/'RESULTS.json').write_text(json.dumps(out,indent=2));print(f'Checked {len(out)} field comparisons')
if __name__=='__main__':main()
