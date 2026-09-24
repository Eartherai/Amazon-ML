"""Critical feature semantics for missing values and deceptive strings."""
import pytest
from src.analysis.pair_morphology import features, overlap, cosine_counts


def test_empty_is_not_matching_evidence():
    assert overlap(set(),set()) == (0,0,0)
    assert all(x==0 for x in features('','').values())


def test_token_subset_can_hide_conflicting_numbers():
    row=features('alpha 531 street','alpha 532 street')
    assert row['jw']>.9
    assert row['numeric_jaccard']==0
    assert features('alpha','alpha beta')['token_set']==1


def test_ngram_cosine_has_counts_and_normalization():
    assert cosine_counts('aaaa','aaaa',2)==pytest.approx(1)
    assert cosine_counts('a','a',2)==0
    assert cosine_counts('abcd','wxyz',2)==0
    assert features('name name other','other name name')['token_sort']==1
