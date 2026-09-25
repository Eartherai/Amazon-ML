"""Guard the held-out entity and singleton mechanics of the size experiment."""

import numpy as np
import polars as pl

from scripts.aws.train_learning_curve import predict_sets, select_fit_ids, target_sets


def test_fit_sample_excludes_held_fold_and_is_nested() -> None:
    queries = pl.DataFrame({"entity_id": [f"S1-{fold}-{index}" for fold in (1, 2, 3) for index in range(6)],
                            "fold": [fold for fold in (1, 2, 3) for _ in range(6)]})
    old_ids = {"S1-2-0", "S1-3-0"}
    small = select_fit_ids(queries, old_ids, held_fold=1, size=4)
    large = select_fit_ids(queries, old_ids, held_fold=1, size=8)
    assert len(small) == 4 and len(large) == 8
    assert small <= large
    assert old_ids <= small
    assert all(not item.startswith("S1-1-") for item in large)


def test_prediction_keeps_empty_singleton_and_threshold() -> None:
    queries = pl.DataFrame({"entity_id": ["S1-1", "S1-2"]})
    truth = pl.DataFrame({"source1_entity_id": ["S1-1"], "target_id": ["S2-1"]})
    candidates = pl.DataFrame({"source1_entity_id": ["S1-1", "S1-2"], "target_id": ["S2-1", "S3-2"]})
    expected = target_sets(queries, truth)
    predicted = predict_sets(candidates, np.array([.9, .2]), .83, queries["entity_id"].to_list())
    assert expected == {"S1-1": {"S2-1"}, "S1-2": set()}
    assert predicted == expected
