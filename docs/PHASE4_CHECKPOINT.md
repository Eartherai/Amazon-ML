# Phase4 measured OOF checkpoint

Run: outputs/oof/P4-A-001. 5,000 natural-prevalence entities, three original entity folds; fold4 CLOSED. Candidate retrieval used all10,320,219 training targets. IDF excluded all OOF-owned targets. Thresholds were selected from inner OOF only.

| Outer fold | S1 | Singleton rate | Threshold | Macro F0.5 | Precision | Recall | Singleton F0.5 | India | US |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1676 | 0.0513 | 0.83 | 0.925744 | 0.983360 | 0.830363 | 0.941860 | 0.913230 | 0.934764 |
| 2 | 1659 | 0.0555 | 0.82 | 0.920961 | 0.975373 | 0.836855 | 0.858696 | 0.895613 | 0.939190 |
| 3 | 1665 | 0.0529 | 0.81 | 0.923490 | 0.984007 | 0.836792 | 0.840909 | 0.897047 | 0.942250 |

Pooled macro: **0.923407**. Fold mean/std: 0.923398 / 0.002393; min/max: 0.920961 / 0.925744. S1 bootstrap95% interval: [0.9185769101763813, 0.9279348540650086]. The interval is conditional on the fitted procedure, not model-refit uncertainty.

Source-specific macro scores (all S1 entities retained):
- Fold1: S2 0.891991; S3 0.892177.
- Fold2: S2 0.893892; S3 0.890972.
- Fold3: S2 0.897095; S3 0.888649.

## Candidate coverage

```json
{
  "query_count": 5000,
  "true_links": 17345,
  "retrieved_links": 16764,
  "candidate_pairs": 985945,
  "target_pool_count": 10320219,
  "link_recall": 0.9665033150763909,
  "oracle_macro_f0_5": 0.9891492764644434,
  "oracle_non_singleton_f0_5": 0.9885395822395896,
  "oracle_singleton_f0_5": 1.0,
  "positive_entity_any_coverage": 0.9974651457541192,
  "positive_entity_all_coverage": 0.9034643008027038,
  "singleton_count": 266,
  "singleton_rate": 0.0532,
  "singleton_candidate_free_rate": 0.0,
  "average_candidates_per_query": 197.189,
  "candidate_quantiles": {
    "p50": 197.0,
    "p95": 200.0,
    "p99": 200.0
  },
  "max_candidates_per_query": 200,
  "candidate_reduction_ratio_vs_unrestricted_pool": 0.9999808929442292,
  "by_country_source": [
    {
      "country": "India",
      "target_source": "S2",
      "true_links": 3450,
      "retrieved_links": 3304,
      "link_recall": 0.9576811594202899
    },
    {
      "country": "India",
      "target_source": "S3",
      "true_links": 3721,
      "retrieved_links": 3474,
      "link_recall": 0.9336199946251008
    },
    {
      "country": "US",
      "target_source": "S2",
      "true_links": 4920,
      "retrieved_links": 4836,
      "link_recall": 0.9829268292682927
    },
    {
      "country": "US",
      "target_source": "S3",
      "true_links": 5254,
      "retrieved_links": 5150,
      "link_recall": 0.9802055576703465
    }
  ],
  "by_country": [
    {
      "country": "India",
      "queries": 2087,
      "link_recall": 0.9451959280435086,
      "oracle_macro_f0_5": 0.9806798982423558,
      "singleton_count": 124
    },
    {
      "country": "US",
      "queries": 2913,
      "link_recall": 0.9815215254570474,
      "oracle_macro_f0_5": 0.9952171076863784,
      "singleton_count": 142
    }
  ]
}
```

## Nested decision comparisons

| Method | Pooled macro | Precision | Recall | Singleton |
|---|---:|---:|---:|---:|
| raw | 0.923407 | 0.980893 | 0.834650 | 0.879699 |
| platt | 0.922785 | 0.979986 | 0.835630 | 0.875940 |
| isotonic | 0.923497 | 0.979566 | 0.837417 | 0.875940 |
| pair_plus_empty | 0.923113 | 0.981100 | 0.834996 | 0.906015 |

These are development OOF comparisons. Selecting a method consumes this evidence; fold4 remains the final one-time check after the whole decision procedure is frozen. No leaderboard submission is justified by this checkpoint alone. Larger coverage, candidate rescue, ablations, training-scale and final test inference are still pending.
