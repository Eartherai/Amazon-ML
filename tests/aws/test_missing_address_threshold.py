"""Check that a missing-address rule uses exact entity-level singleton scoring."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/experiments"))
from missing_address_threshold import score


def test_missing_address_threshold_keeps_singletons_and_counts_false_merge():
    queries = pl.DataFrame({"entity_id": ["S1-a", "S1-b"], "n_matches": [1, 0],
                            "country": ["India", "India"]})
    pairs = pl.DataFrame({"source1_entity_id": ["S1-a", "S1-b"],
                          "label": [1, 0], "query_address_missing": [1, 1],
                          "target_address_missing": [0, 0]})
    probabilities = np.array([0.7, 0.6])
    conservative = score(pairs, queries, probabilities, 0.8, 0.65)
    assert conservative["macro_f0_5"] == 1.0
    permissive = score(pairs, queries, probabilities, 0.8, 0.55)
    assert permissive["macro_f0_5"] == 0.5
    assert permissive["precision"] == 0.5
