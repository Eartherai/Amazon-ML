"""Train frozen SUB-001 LightGBM on all currently materialized development entities."""
import hashlib,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,polars as pl,lightgbm as lgb
from src.models.phase4_oof import NAMES as BASE
from src.models.phase4_numeric import NAMES as EXTRA
from src.models.phase4_oof import fitting_mask


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 config=Path('configs/submissions/SUB-001.yaml');cfg=json.loads(config.read_text());out=Path('outputs/submissions/SUB-001/train');out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 names=BASE+EXTRA
 if len(names)!=51 or len(set(names))!=51:raise AssertionError('Frozen feature list changed')
 base=pl.read_parquet(cfg['train_features']).sort('source1_entity_id','target_id')
 numeric=pl.read_parquet(cfg['numeric_features'])
 frame=base.join(numeric,on=['source1_entity_id','target_id'],how='left',validate='1:1')
 if frame.select(pl.any_horizontal(pl.col(EXTRA).is_null()).any()).item():raise ValueError('Missing numeric features')
 if set(frame['fold'].unique())!={1,2,3}:raise ValueError('Unexpected training folds')
 owner=frame['owner_fold'].to_numpy();fold=frame['fold'].to_numpy();y=frame['label'].to_numpy()
 mask=fitting_mask(fold,owner,[1,2,3]);
 if np.any((y==1)&~mask):raise AssertionError('Positive excluded by owner filter')
 params=json.loads(Path(cfg['model_params_source']).read_text())['hyperparameters']
 model=lgb.LGBMClassifier(**params)
 model.fit(frame.filter(pl.Series(mask)).select(names).to_numpy(),y[mask],feature_name=names)
 model.booster_.save_model(str(out/'model.txt'))
 manifest={'timestamp':datetime.now(timezone.utc).isoformat(),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'config_sha256':sha(config),'baseline_sha256':sha(Path(cfg['model_params_source'])),'phase4_feature_source_sha256':sha(Path('code/business_entity_resolution/src/models/phase4_oof.py')),'numeric_source_sha256':sha(Path('code/business_entity_resolution/src/models/phase4_numeric.py')),'feature_names':names,'training_pairs':int(mask.sum()),'training_positives':int(y[mask].sum()),'training_entities':int(frame['source1_entity_id'].n_unique()),'folds':[1,2,3],'owner_folds_allowed':[-1,1,2,3],'fold4':'CLOSED','threshold':cfg['threshold'],'threshold_rule':cfg['decision'],'model_params':params,'lightgbm_version':lgb.__version__,'runtime_seconds':time.perf_counter()-start,'model_sha256':sha(out/'model.txt')}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
if __name__=='__main__':main()
