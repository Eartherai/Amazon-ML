"""Country balancing computed solely from eligible training S1 entities."""
import numpy as np
import polars as pl


def country_pair_weights(queries: pl.DataFrame, pair_ids: np.ndarray, allowed_folds: list[int]) -> np.ndarray:
    """Equalize training entity mass by country; preserve natural validation prevalence."""
    train=queries.filter(pl.col('fold').is_in(allowed_folds))
    if len(train)==0 or train['entity_id'].n_unique()!=len(train):
        raise ValueError('Empty or duplicate training queries')
    counts=train.group_by('country').len()
    weights={country:len(train)/(len(counts)*count) for country,count in counts.iter_rows()}
    by_id={qid:weights[country] for qid,country in train.select('entity_id','country').iter_rows()}
    if any(qid not in by_id for qid in pair_ids):raise ValueError('Pair outside eligible training queries')
    return np.array([by_id[qid] for qid in pair_ids],dtype=np.float32)
