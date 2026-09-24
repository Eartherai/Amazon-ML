"""Fetch compact result/status objects and record EC2 termination without large downloads."""
import argparse,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from provision_worker import aws
p=argparse.ArgumentParser();p.add_argument('run_id');a=p.parse_args()
root=Path('artifacts/cloud/phase5')/a.run_id;ledger=json.loads((root/'ledger.json').read_text())
state=aws('ec2','describe-instances','--instance-ids',ledger['instance_id'])['Reservations'][0]['Instances'][0]
ledger['last_checked']=datetime.now(timezone.utc).isoformat();ledger['instance_state']=state['State']['Name']
if ledger['instance_state']=='terminated' and not ledger.get('termination_observed_at'):ledger['termination_observed_at']=ledger['last_checked']
bucket='aml2026-ber-08be19ac500747';prefix=ledger['s3_outputs']
objects=aws('s3api','list-objects-v2','--bucket',bucket,'--prefix',prefix+'/')
keys={r['Key']:r for r in objects.get('Contents',[])}
for suffix in ['results/metrics.json','results/retrieval-metrics-unlocked.json','results/COMPLETE.json','status/worker-status.json','receipts/results-receipt.json','status/aml-worker.log']:
 key=prefix+'/'+suffix
 if key in keys:
  dest=root/suffix.replace('/','-');aws('s3api','get-object','--bucket',bucket,'--key',key,str(dest))
  if suffix.endswith('worker-status.json'):
   status=json.loads(dest.read_text());ledger.update(exit_code=status['exit_code'],stop=status['stop'])
   seconds=(datetime.fromisoformat(status['stop'].replace('Z','+00:00'))-datetime.fromisoformat(ledger['start'])).total_seconds()
   ledger['wall_seconds']=seconds;ledger['compute_cost_estimate_usd']=seconds/3600*ledger['compute_usd_per_hour']
ledger['output_objects_seen']=len(keys);ledger['output_bytes_seen']=sum(v['Size'] for v in keys.values())
(root/'ledger.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(ledger))
