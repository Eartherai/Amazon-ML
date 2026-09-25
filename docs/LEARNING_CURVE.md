# Classical LightGBM learning curve

EXP-031 retrieved 200,000 deterministic fold-1–3 Source-1 entities against the
full 10,320,219-record training target pool. EXP-032 materialized the same
51 NUMERIC-V2 features for its final candidate union. EXP-033 is fitting the same
LightGBM configuration at 20,000, 50,000 and 100,000 entities per outer fold.
Each size will be evaluated on the same newly selected 15,000 Source-1 entities
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
fresh cost guard. Its 20k/50k/100k held-out scores are pending.

The thresholds 0.83, 0.79 and 0.83 were selected on the earlier 20,000-entity
development sample, before selecting this new evaluation set. EXP-033 is a
fixed-threshold learning-curve diagnostic; these sizes do not have separately
nested threshold optimization. The earlier 0.9318965293 score is a reference
from a different OOF population, so it is not a point on the new curve.

| Train entities | F0.5 | Precision | Recall | Singleton | India | US | Delta from previous size |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,000 | Pending | Pending | Pending | Pending | Pending | Pending | — |
| 50,000 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| 100,000 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

No plot is shown before EXP-033 produces verified metrics. Once its worker has
exited successfully, every model/metric checkpoint and feature input has been
SHA256-verified, and the 15,000-entity report is downloaded, run:

```sh
.venv/bin/python scripts/analysis/plot_learning_curve.py \
  --metrics artifacts/cloud/phase5/P5-LEARNING-FIT-001/results-metrics.json \
  --output docs/figures/learning_curve.png
```

The generated two-panel figure shows training size versus overall and India
macro F0.5, with measured values labeled. Compare pooled and per-fold slopes,
country slices, runtime, RAM and estimated AWS cost before deciding whether to
prepare 250k/500k/1M. An apparent plateau across three sizes is a reason to
investigate feature/model diversity, not proof of saturation.

Planning priority before results: EXP-031/032/033 have high expected
information gain because the prior fixed-holdout 2k→13.3k diagnostic rose from
0.924221 to 0.932231. An improvement of at least 0.002 by 100k is plausible
but unmeasured. Confidence is moderate, sequential wall time is bounded by
three six-hour worker caps (EXP-031 is complete), and combined EXP-031–033
planning ceilings are $13.50 before the separate full-retrieval job. Do not
interpret this estimate as a result or an AWS bill.
