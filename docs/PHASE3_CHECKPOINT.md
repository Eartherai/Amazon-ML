# Phase 3 checkpoint

All retrieval measurements use the same 1,000 balanced-country development S1 and all 10,320,219 training targets. Model check uses 505 development S1; calibration uses the other 495. Neither is final holdout or OOF model performance. Fold4 remains closed.

## Retrieval union

| Added route | Link recall | Complete entity recall | Oracle F0.5 | Avg candidates | P95 | P99 | New true links |
|---|---:|---:|---:|---:|---:|---:|---:|
| token_union | 77.33% | 57.26% | 0.86630 | 179.3 | 400 | 432 | 2667 |
| name_char3 | 88.66% | 73.16% | 0.95038 | 263.9 | 489 | 522 | 391 |
| address_char3 | 96.78% | 90.84% | 0.98892 | 348.7 | 569 | 603 | 280 |
| name_char4 | 96.98% | 91.26% | 0.98935 | 377.8 | 602 | 650 | 7 |
| address_char4 | 97.16% | 91.79% | 0.98984 | 400.6 | 624 | 670 | 6 |
| name_char5 | 97.22% | 92.00% | 0.99005 | 427.1 | 652 | 706 | 2 |
| address_char5 | 97.27% | 92.21% | 0.99028 | 450.7 | 670 | 734 | 2 |
| name_translit_char3 | 97.42% | 92.32% | 0.99065 | 459.4 | 679 | 742 | 5 |
| expanded_token_union | 97.62% | 93.05% | 0.99116 | 952.4 | 1406 | 1460 | 7 |

The broad union prioritizes recall and is not yet a frozen production configuration. Expanded-token retrieval has poor marginal efficiency; targeted rescue/reranking should be tested. Per-route runtime, RSS and fit statistics: outputs/candidates/CHAR-001/metrics.json and TRANS-002/metrics.json. Complete/any recall excludes singletons; oracle credits perfect empty predictions and is not a matcher score.

## First model tournament

| Model | Calibration F0.5 | Check F0.5 | Precision | Recall | Singleton | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| gbdt31 | 0.91227 | 0.89738 | 95.48% | 82.79% | 80.00% | 0.72 |
| gbdt15 | 0.90491 | 0.90578 | 95.17% | 83.93% | 84.00% | 0.65 |
| multiview | 0.91669 | 0.91056 | 95.35% | 86.26% | 80.00% | 0.58 |
| multiview_hardneg3 | 0.91464 | 0.90635 | 96.11% | 84.16% | 76.00% | 0.59 |

Selected by calibration only: **multiview**. India check F0.5 0.89486; US 0.92524. Model uses name/address char3 candidates, not the full broad retrieval union.

Inner two-fold entity cross-fitting identified 403 training hard negatives; held-out target owners were excluded from each inner fit. Weighting those negatives 3× increased check precision but reduced macro F0.5 and singleton accuracy, so this branch was rejected. The independent transliteration feature branch improved calibration and check scores.

## Limits and next work

Only 1,000 training S1 were used. The development check has been observed across experiments and architecture choices; do not present it as an untouched test set. No final test output or leaderboard submission exists. No GPU or EC2 launched. S3 stores 51 verified objects / 4,326,917,279 bytes; estimated storage run-rate $0.0927/month, actual billing unverified.

Next: expand training/query coverage, persist efficient retrieval indexes, evaluate cross-country and collision-group robustness, fit OOF singleton/calibration decisions, inspect false merges, then determine whether dense rescue adds value on the remaining hard pairs. Do not immediately scale the broad ~900-candidate union to every test entity.
