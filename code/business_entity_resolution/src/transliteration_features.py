"""Symmetric transliteration comparisons with explicit, versioned text mappings.

Native text remains separate. Both query and target use the same generic backend;
no country-specific fallback or silent asymmetric comparison is allowed.
"""
from collections.abc import Mapping
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

def mapped_text(text: str, mapping: Mapping[str,str]) -> str:
    """ASCII is already in the comparison alphabet; non-ASCII must be cached."""
    if text.isascii():return text
    if text not in mapping:raise ValueError('Non-ASCII text missing from versioned transliteration cache')
    return mapping[text]

def compare_transliterated(source: str, target: str, mapping: Mapping[str,str]) -> list[float]:
    """Return JW, ratio, token-sort, exact, symmetrically transforming both sides."""
    left,right=mapped_text(source,mapping),mapped_text(target,mapping)
    if not left or not right:return [0.,0.,0.,0.]
    return [JaroWinkler.normalized_similarity(left,right),fuzz.ratio(left,right)/100,fuzz.token_sort_ratio(left,right)/100,float(left==right)]
