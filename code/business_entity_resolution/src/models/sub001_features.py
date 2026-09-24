"""Frozen 51-feature SUB-001 inference adapter, with symmetric Unicode transliteration."""
import re
import numpy as np
from src.models.lexical_pilot import text_features
from src.models.phase4_oof import NAMES as BASE_NAMES
from src.models.phase4_numeric import NAMES as NUMERIC_NAMES,compare_numbers
from src.transliteration_features import compare_transliterated

NAMES=BASE_NAMES+NUMERIC_NAMES


def pair_features(query_name,query_address,query_country,target_name,target_address,target_country,target_id,name_score,address_score,name_reciprocal_rank,address_reciprocal_rank,route_count,transliteration_map):
    """Match Phase4's feature order and float32 casts exactly for ASCII training queries."""
    qn,qa,tn,ta=(v or '' for v in (query_name,query_address,target_name,target_address))
    n=text_features(qn,tn);a=text_features(qa,ta)
    na,nb=set(re.findall(r'\d+',qa)),set(re.findall(r'\d+',ta));shared=len(na&nb)
    firsta=re.search('[0-9]+',qa);firstb=re.search('[0-9]+',ta)
    fa=firsta.group() if firsta else '';fb=firstb.group() if firstb else ''
    base=n+a+[n[1]*a[1],float(shared>0),float(bool(na and nb) and not shared),shared/len(na|nb) if na or nb else 0.,float(not qa),float(not ta),float(query_country==target_country),float(target_id.startswith('S2-')),float(name_score),float(address_score),float(name_reciprocal_rank),float(address_reciprocal_rank),float(route_count)]
    extra=compare_transliterated(qn,tn,transliteration_map)+[float(bool(fa and fb) and fa==fb),float(bool(fa and fb) and fa!=fb)]
    values=base+extra+compare_numbers(qa,ta)
    if len(values)!=51 or len(NAMES)!=51:raise AssertionError('SUB-001 feature count changed')
    return np.asarray(values,dtype=np.float32)
