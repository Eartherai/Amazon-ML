"""Create immutable cloud inputs for frozen SUB-001 inference."""
from pathlib import Path
import json,shutil,hashlib
root=Path('artifacts/cloud/phase5/sub001-input-v001');root.mkdir(parents=True,exist_ok=False)
inputs={
 'queries.parquet':Path('artifacts/cloud/phase5/test-input-v001/queries.parquet'),
 'target2.parquet':Path('artifacts/cloud/phase5/test-input-v001/target2.parquet'),
 'target3.parquet':Path('artifacts/cloud/phase5/test-input-v001/target3.parquet'),
 'name_map.parquet':Path('artifacts/cloud/phase5/test-translit-v001/name_map.parquet'),
 'name_char3_idf.npz':Path('outputs/candidates/P4-B-002/name_char3_idf.npz'),
 'address_char3_idf.npz':Path('outputs/candidates/P4-B-002/address_char3_idf.npz'),
 'model.txt':Path('outputs/submissions/SUB-001/train/model.txt')}
files=[]
for name,original in inputs.items():
 dest=root/name;shutil.copy2(original,dest);h=hashlib.sha256(dest.read_bytes()).hexdigest();files.append({'name':name,'sha256':h,'bytes':dest.stat().st_size})
for name,original in [('submission-config.json',Path('configs/submissions/SUB-001.yaml')),('train-manifest.json',Path('outputs/submissions/SUB-001/train/manifest.json'))]:
 dest=root/name;shutil.copy2(original,dest);files.append({'name':name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size})
(root/'manifest.json').write_text(json.dumps({'scope':'Frozen SUB-001 test inference; all provided test records; no labels; Fold4 closed','files':files},indent=2));print(json.dumps({'files':len(files),'bytes':sum(f['bytes']for f in files)}))
