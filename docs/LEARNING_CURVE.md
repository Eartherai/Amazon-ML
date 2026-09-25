# Classical LightGBM learning curve

EXP-031 retrieved 200,000 deterministic fold-1–3 Source-1 entities against the
full 10,320,219-record training target pool. EXP-032 materialized the same
51 NUMERIC-V2 features for its final candidate union. EXP-033 fitted the same
LightGBM configuration at 20,000, 50,000 and 100,000 entities per outer fold.
Each size was evaluated on the same newly selected 15,000 Source-1 entities
(5,000 per fold) that were absent from the prior 20,000-entity development
sample. Target ownership masks prevent held-out positive targets from becoming
fit negatives. Fold 4 remains closed.

EXP-031 completed with all 256 route archives SHA256-verified. Its old20k
candidate comparison passed the documented numerical-tie audit, not byte-exact
rank parity: name has no changed candidate IDs and 16 near-tie rank changes;
address has 57 rank changes and one rank-100 exchange of two known negatives.
Shared-pair score deltas are at most 2.384e-7. See
`outputs/analysis/P5-LEARNING-PARITY-001/report.json`. EXP-032 and EXP-033
must cite this retrieval provenance when comparing with older 20k evidence.

EXP-032 completed successfully at 2026-09-25T05:25:15Z with EC2 terminated.
Its 200,000 S1 produced 39,440,694 final candidate pairs and 854 feature
parts; all parts were SHA256-verified by the uploading worker, and the exact
S3 name/size inventory matched the immutable receipt. Sampled first/last
objects retained the receipt's service SHA256 and version IDs. Link recall is
0.9660719915 (668,118 of 691,582 labeled links), complete positive-entity
recall 0.9010811555, mean candidates/S1 197.20347, p95/p99 200.
This retrieval result is not a matcher F0.5 score. Feature computation took
2,538.10 seconds; estimated EC2 compute $0.4011, final billed amount pending.
EXP-033 launched at 2026-09-25T05:27:58Z on a capped r8i.2xlarge after a
fresh cost guard and exited zero with EC2 terminated. All nine model and nine
fold-report checkpoints passed S3 service SHA256/version verification. The
pooled report has exactly 15,000 fixed evaluation entities and each pooled
macro score equals the mean of its three 5,000-entity fold scores.

The thresholds 0.83, 0.79 and 0.83 were selected on the earlier 20,000-entity
development sample, before selecting this new evaluation set. EXP-033 is a
fixed-threshold learning-curve diagnostic; these sizes do not have separately
nested threshold optimization. The earlier 0.9318965293 score is a reference
from a different OOF population, so it is not a point on the new curve.

| Train entities | F0.5 | Precision | Recall | Singleton | India | US | Delta from previous size |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | 0.93492235 | 0.98292872 | 0.85724506 | 0.93953488 | 0.91210692 | 0.95005253 | — |
| 50,000 | 0.93630527 | 0.98373066 | 0.85931397 | 0.94767442 | 0.91397438 | 0.95111413 | +0.00138292 |
| 100,000 | 0.93715814 | 0.98408734 | 0.86095750 | 0.94418605 | 0.91438146 | 0.95226261 | +0.00085286 |

The verified plot is [learning_curve.png](figures/learning_curve.png). To
regenerate it at a new path, run:

```sh
.venv/bin/python scripts/analysis/plot_learning_curve.py \
  --metrics artifacts/cloud/phase5/P5-LEARNING-FIT-001/results-metrics.json \
  --output docs/figures/learning_curve-regenerated.png
```

The size effect is +0.00223578 from 20k to 100k. All three folds improve over
that interval; fold 2 is nearly flat from 50k to 100k. India improves
+0.00227454, US +0.00221008. Singleton F0.5 peaks at 50k, so greater fit
size does not improve every subgroup monotonically. The positive slope supports
preparing 250k/500k/1M owner-safe fits from the existing full retrieval. A 1M
fit needs enough unlocked S1 across training folds and cannot be drawn from
only this 200k feature store. Adding fold-0 training for that scale changes the
fit population, so absolute scores from the next curve must be labeled as a
new protocol even when the same 15k evaluation IDs are retained.

EXP-033 total worker runtime was 506.54 seconds, with 427.73 seconds in its
training script and 7.55 GiB peak RSS. Estimated EC2 compute was $0.07819;
final billing remains pending. The observed slope does not imply unlimited
gain from adding entities. No public leaderboard score is yet known.
