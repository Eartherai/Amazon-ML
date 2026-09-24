"""Versioned completed5k checkpoint to the existing private S3 bucket via CLI."""
from pathlib import Path
import json,tarfile,subprocess,shutil
from datetime import datetime,timezone
from s3_data_layer import STATE,PREFIX,aws,put_immutable,sha
root=Path('artifacts/cloud/phase4/checkpoint-A-001');root.mkdir(parents=True,exist_ok=False)
commit='ae5cfcb';code=root/'code.tar';subprocess.run(['git','archive','--format=tar','--output',str(code),commit],check=True)
files=[code]
for directory in ['outputs/oof/P4-A-001','outputs/candidates/P4-A-001','outputs/candidates/P4-RESCUE-001']:
 files += [p for p in sorted(Path(directory).rglob('*')) if p.is_file()]
files += sorted(Path('artifacts/validation/phase4-v001').glob('*.parquet'))+[Path('artifacts/validation/phase4-v001/manifest.json'),Path('artifacts/transliteration/TRANS-001/name_map.parquet')]
total=sum(p.stat().st_size for p in files)
if shutil.disk_usage('.').free-total<8*1024**3:raise RuntimeError('Disk reserve')
manifest={'created_at':datetime.now(timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse',commit],text=True).strip(),'scope':'Completed5k OOF/candidates/ablations/calibration/country-transfer and rescue;20k ongoing run excluded','files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'storage_monthly_usd_estimate':total/1024**3*.023,'put_requests_usd_estimate':2*.000005,'pricing_basis':'Previously verified us-east-1 S3 Standard .023/GiB-month and .005/1000PUT; excludes other request/storage bill; not credit balance'}
mp=root/'manifest.json';mp.write_text(json.dumps(manifest,indent=2));archive=root/'checkpoint.tar'
with tarfile.open(archive,'w') as tar:
 for p in files:tar.add(p,arcname=str(p),recursive=False)
 tar.add(mp,arcname='checkpoint-manifest.json',recursive=False)
state=json.loads(STATE.read_text());ident=aws('sts','get-caller-identity')
if ident['Account'][-4:]!=state['account_suffix']:raise RuntimeError('Account mismatch')
checks=aws('s3api','get-public-access-block','--bucket',state['bucket'])['PublicAccessBlockConfiguration']
if not all(checks.get(k) for k in ['BlockPublicAcls','IgnorePublicAcls','BlockPublicPolicy','RestrictPublicBuckets']):raise RuntimeError('Bucket privacy checks failed')
receipts=[put_immutable(p,PREFIX+'checkpoints/phase4/A-001/'+p.name,state) for p in [archive,mp]]
(root/'upload.json').write_text(json.dumps({'completed_at':datetime.now(timezone.utc).isoformat(),'receipts':receipts},indent=2));print(json.dumps({'verified_objects':len(receipts),'bytes':sum(r['bytes'] for r in receipts),'additional_monthly_storage_estimate':manifest['storage_monthly_usd_estimate']}))
