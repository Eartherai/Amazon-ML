"""Archive the confirmed20k development pipeline before scaling; immutable S3 keys."""
import json,tarfile,subprocess,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
from upload_verified import upload,sha
root=Path('artifacts/cloud/phase5/checkpoint-B-001');root.mkdir(parents=True,exist_ok=False)
files=[]
for folder in ['outputs/oof/P4-B-001','outputs/candidates/P4-B-002','outputs/experiments/P4-NUMERIC-B-001','artifacts/features/P4-NUMERIC-B-001']:
 files += sorted(p for p in Path(folder).rglob('*') if p.is_file())
files += [Path('artifacts/validation/phase4-v001/P4-SAMPLE-B.parquet'),Path('artifacts/transliteration/TRANS-001/name_map.parquet')]
size=sum(p.stat().st_size for p in files)
if shutil.disk_usage('.').free-size<8*1024**3:raise RuntimeError('Disk reserve')
archive=root/'checkpoint.tar'
manifest={'time':datetime.now(timezone.utc).isoformat(),'model_checkpoint_commit':'aaee676','files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in files],'scope':'20k candidates, OOF baseline, numeric features and confirmed best model; Fold4 closed'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(archive,'w') as tar:
 for p in files:tar.add(p,arcname=str(p),recursive=False)
 tar.add(root/'manifest.json',arcname='manifest.json')
receipts=[upload(p,'aml2026-ber-08be19ac500747','amazon-ml-2026/phase5/checkpoints/B-001/'+p.name) for p in [archive,root/'manifest.json']]
(root/'upload.json').write_text(json.dumps(receipts,indent=2));print(json.dumps({'objects':len(receipts),'bytes':sum(r['bytes'] for r in receipts)}))
