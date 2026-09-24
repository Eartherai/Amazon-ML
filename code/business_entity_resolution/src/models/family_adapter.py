"""Bounded CPU model-family probes with a shared probability/save interface."""
from pathlib import Path

def config(family):
    if family=='catboost':return dict(iterations=400,depth=6,learning_rate=.07,l2_leaf_reg=4,border_count=128,random_seed=20260925,thread_count=2,allow_writing_files=False,verbose=False,bootstrap_type='Bernoulli',subsample=.9,loss_function='Logloss')
    if family=='xgboost':return dict(n_estimators=400,max_depth=5,learning_rate=.06,min_child_weight=5,reg_lambda=3,max_bin=128,subsample=.9,colsample_bytree=.9,tree_method='hist',device='cpu',n_jobs=2,random_state=20260925,objective='binary:logistic',eval_metric='logloss')
    raise ValueError(family)

class Adapter:
    """Expose probability prediction where baseline code expects a booster."""
    def __init__(self,family):
        self.family=family
        if family=='catboost':
            from catboost import CatBoostClassifier
            self.model=CatBoostClassifier(**config(family))
        elif family=='xgboost':
            from xgboost import XGBClassifier
            self.model=XGBClassifier(**config(family))
        else:raise ValueError(family)
    def fit(self,x,y,feature_name=None):
        if feature_name is not None and len(feature_name)!=x.shape[1]:raise ValueError('Feature shape mismatch')
        self.model.fit(x,y);return self
    @property
    def booster_(self):return self
    def predict(self,x):return self.model.predict_proba(x)[:,1]
    def save_model(self,path):
        self.model.save_model(str(Path(path).with_suffix('.cbm' if self.family=='catboost' else '.ubj')))
