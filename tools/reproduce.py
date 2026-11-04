"""Small entry point for the paper's plotting and CPU reproduction workflows."""
from pathlib import Path
import argparse, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
TASKS={
 'saved-plots':[['Figures/transport_accuracy.py'],['Figures/transport_overview.py'],['Figures/transport_cost.py'],['Figures/transport_results.py'],['Figures/bratu_results.py'],['Figures/bratu_capture.py'],['Figures/bratu_panels.py'],['Figures/supplementary_diagrams.py'],['tools/construction_cost.py']],
 'bratu-two-cell':[['experiments/bratu/two_cell.py']],
 'bratu-field':[['experiments/bratu/field_accuracy.py']],
 'bratu-local':[['experiments/bratu/local_accuracy.py']],
 'transport-local':[['experiments/transport/local_accuracy.py']],
 'checkpoints':[['experiments/transport/accuracy.py','checkpoints']],
 'refinement':[['experiments/transport/accuracy.py','refinement']],
 'norne-reference':[['experiments/transport/accuracy.py','norne-reference']],
 'field-errors':[['experiments/transport/field_errors.py']],
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',nargs='?',choices=TASKS);p.add_argument('--list',action='store_true');a,extra=p.parse_known_args()
 if a.list:
  for name,commands in TASKS.items():print(name+': '+'; '.join(' '.join(c) for c in commands))
  return
 if not a.task:p.error('Select a task or --list')
 for command in TASKS[a.task]:subprocess.run([sys.executable,*command,*extra],cwd=ROOT,check=True)
if __name__=='__main__':main()
