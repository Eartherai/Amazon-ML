"""One command to reproduce the core research measurements in a fresh run folder."""
import argparse,os,shutil,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--baseline-config',type=Path,required=True)
    a=p.parse_args()
    if a.run_dir.exists():raise FileExistsError('Choose a new immutable run directory')
    parent=a.run_dir.parent
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<18*2**30:raise RuntimeError('Need >=18 GiB free for database, scratch and 8 GiB reserve')
    a.run_dir.mkdir(parents=True);db=a.run_dir/'audit.duckdb'
    commands=[
      ['src.audit_data','--data-dir',str(a.data_dir),'--output-dir',str(a.run_dir/'audit'),'--database',str(db)],
      ['src.audit_pairs','--database',str(db),'--output-dir',str(a.run_dir/'pairs')],
      ['src.build_validation','--database',str(db),'--output-dir',str(a.run_dir/'validation')],
      ['src.audit_memory','--data-dir',str(a.data_dir),'--database',str(db),'--output',str(a.run_dir/'memory.json')],
      ['src.exact_baseline','--database',str(db),'--config',str(a.baseline_config),'--output-dir',str(a.run_dir/'EXP-001')]]
    for command in commands:
        print('Running',command[0],flush=True)
        with (a.run_dir/(command[0]+'.log')).open('x') as f:
            subprocess.run([sys.executable,'-m',*command],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('Core research measurements complete. No model training, test prediction or cloud execution.')
if __name__=='__main__':main()
