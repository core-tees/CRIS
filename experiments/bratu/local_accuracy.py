"""Evaluate the frozen Bratu inverse on independent saved input sets."""
from pathlib import Path
import json,numpy as np
from inverse import inverse
ROOT=Path(__file__).resolve().parents[2]
def main():
 out=ROOT/'runs/bratu_local';out.mkdir(parents=True,exist_ok=True);rows=[]
 for name in ['validation','test','test_near_fold']:
  data=np.fromfile(ROOT/'data/bratu'/(name+'.bin'),'<f8',offset=16).reshape(-1,4)
  a,b=data[:,:2].T;pred,_=inverse('mse',a,b);e=pred-data[:,2]
  rows.append(dict(split=name,count=len(a),rmse=float(np.sqrt(np.mean(e*e))),maximum=float(abs(e).max())))
 (out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
