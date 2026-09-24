import numpy as np
import polars as pl
import pytest
from src.models.country_weights import country_pair_weights


def test_training_country_mass_and_holdout_exclusion():
    q=pl.DataFrame({'entity_id':['a','b','c','d','hold'],'country':['India','US','US','US','India'],'fold':[1,1,1,1,2]})
    weights=country_pair_weights(q,np.array(['a','b','c','d']),[1])
    assert weights[0]==pytest.approx(sum(weights[1:]))
    assert sum(weights)==pytest.approx(4)
    with pytest.raises(ValueError,match='outside'):country_pair_weights(q,np.array(['hold']),[1])
