# Phase4 measured OOF checkpoint

Run: outputs/oof/P4-B-001. 20,000 natural-prevalence entities, three original entity folds; fold4 CLOSED. Candidate retrieval used all10,320,219 training targets. IDF excluded all OOF-owned targets. Thresholds were selected from inner OOF only.

| Outer fold | S1 | Singleton rate | Threshold | Macro F0.5 | Precision | Recall | Singleton F0.5 | India | US |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6732 | 0.0536 | 0.79 | 0.930941 | 0.979489 | 0.852309 | 0.925208 | 0.911228 | 0.944314 |
| 2 | 6613 | 0.0584 | 0.82 | 0.926093 | 0.979522 | 0.845239 | 0.909326 | 0.905212 | 0.940783 |
| 3 | 6655 | 0.0520 | 0.81 | 0.929201 | 0.982397 | 0.848191 | 0.927746 | 0.901667 | 0.947580 |

Pooled macro: **0.928759**. Fold mean/std: 0.928745 / 0.002456; min/max: 0.926093 / 0.930941. S1 bootstrap95% interval: [0.9264584347104737, 0.9310251784781759]. The interval is conditional on the fitted procedure, not model-refit uncertainty.

Source-specific macro scores (all S1 entities retained):
- Fold1: S2 0.899561; S3 0.902638.
- Fold2: S2 0.892067; S3 0.897335.
- Fold3: S2 0.895808; S3 0.899194.

## Candidate coverage

```json
{
  "query_count": 20000,
  "true_links": 69366,
  "retrieved_links": 67051,
  "candidate_pairs": 3943627,
  "target_pool_count": 10320219,
  "link_recall": 0.9666263010696883,
  "oracle_macro_f0_5": 0.9885314605341274,
  "oracle_non_singleton_f0_5": 0.9878684725595047,
  "oracle_singleton_f0_5": 1.0,
  "positive_entity_any_coverage": 0.996985243560586,
  "positive_entity_all_coverage": 0.9030517797641086,
  "singleton_count": 1093,
  "singleton_rate": 0.05465,
  "singleton_candidate_free_rate": 0.0,
  "average_candidates_per_query": 197.18135,
  "candidate_quantiles": {
    "p50": 197.0,
    "p95": 200.0,
    "p99": 200.0
  },
  "max_candidates_per_query": 200,
  "candidate_reduction_ratio_vs_unrestricted_pool": 0.9999808936854925,
  "by_country_source": [
    {
      "country": "India",
      "target_source": "S2",
      "true_links": 13576,
      "retrieved_links": 12926,
      "link_recall": 0.952121390689452
    },
    {
      "country": "India",
      "target_source": "S3",
      "true_links": 14525,
      "retrieved_links": 13543,
      "link_recall": 0.9323924268502581
    },
    {
      "country": "US",
      "target_source": "S2",
      "true_links": 19959,
      "retrieved_links": 19647,
      "link_recall": 0.984367954306328
    },
    {
      "country": "US",
      "target_source": "S3",
      "true_links": 21306,
      "retrieved_links": 20935,
      "link_recall": 0.9825870646766169
    }
  ],
  "by_country": [
    {
      "country": "India",
      "queries": 8116,
      "link_recall": 0.941923774954628,
      "oracle_macro_f0_5": 0.9789356903597326,
      "singleton_count": 441
    },
    {
      "country": "US",
      "queries": 11884,
      "link_recall": 0.9834484429904278,
      "oracle_macro_f0_5": 0.9950847482096066,
      "singleton_count": 652
    }
  ]
}
```

## Nested decision comparisons


These are development OOF comparisons. Selecting a method consumes this evidence; fold4 remains the final one-time check after the whole decision procedure is frozen. No leaderboard submission is justified by this checkpoint alone. Larger coverage, candidate rescue, ablations, training-scale and final test inference are still pending.
