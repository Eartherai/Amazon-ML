"""Check exact shard merging, ties and sparse zero handling."""
import numpy as np
from src.blocking.char_retrieval import top_k

def test_boundary_ties_and_zero():
    ids,scores=top_k(np.array(['z','c','a','b','x']),np.array([1.,.5,.5,.5,0.]),3)
    assert ids.tolist()==['z','a','b']
    assert scores.tolist()==[1,.5,.5]

def test_sharded_matches_full():
    ids=np.array([f'id{i:04}' for i in range(81)])
    scores=np.random.default_rng(42).integers(0,10,81).astype(float)
    got_i=np.array([],dtype='U16');got_s=np.array([])
    for start in range(0,81,7):
        i,s=top_k(ids[start:start+7],scores[start:start+7],9)
        got_i,got_s=top_k(np.concatenate([got_i,i]),np.concatenate([got_s,s]),9)
    expected_i,expected_s=top_k(ids,scores,9)
    assert got_i.tolist()==expected_i.tolist()
    assert got_s.tolist()==expected_s.tolist()
