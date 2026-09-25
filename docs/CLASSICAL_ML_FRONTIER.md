# Classical ML solution frontier

This track excludes neural models. All scores below are on competition training
records with held-out Source-1 entities; Fold 4 remains closed. The user will
upload SUB-001 after full official validation. A public leaderboard result is
not yet available. Unknown cells are pending measurements, not zero scores.

| Experiment | Training size | Retrieval | Candidate recall | Model | Features | Macro F0.5 | Precision | Recall | Singleton F0.5 | India | US | Runtime | AWS cost | Delta / confidence |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| NUMERIC-V2 / SUB-001 | 20,000 S1 | name+address char3 top100 each | 0.966626 link | LightGBM | 51 | 0.9318965293 | 0.98158282 | 0.85593807 | 0.92131747 | 0.90835539 | 0.94797359 | 202.4 s OOF model evaluation | $0 local | +0.00313760 vs 45-feature model; paired 95% CI [0.00194174, 0.00431221] |
| EXP-027 diagnostic | 2k / 5k / 10k / ~13.3k fit entities per outer fold | same frozen candidates | same frozen candidates | LightGBM | 51 | 0.924221 / 0.929379 / 0.930829 / 0.932231 | see EXP-027 | see EXP-027 | see EXP-027 | see EXP-027 | see EXP-027 | see EXP-027 | $0 local | Fixed thresholds and held-out entities; informative slope only, not an unbiased nested learning curve |
| EXP-029 rejected | 20,000 S1 | same | same | LightGBM + missing-address threshold | 51 | 0.930533 | pending | pending | pending | pending | pending | see EXP-029 | $0 local | −0.001363 vs frozen baseline; paired 95% CI [−0.002074, −0.000677] |
| 50k | pending | full retrieval required | pending | LightGBM first | 51 initially | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 100k | pending | full retrieval required | pending | LightGBM first | 51 initially | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 250k | pending | full retrieval required | pending | LightGBM first | 51 initially | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 500k | pending | full retrieval required | pending | LightGBM first | 51 initially | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 1M / full permitted | pending | full retrieval required | pending | LightGBM first | 51 initially | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Current bottlenecks and next measured decisions

1. Complete frozen SUB-001 full test inference, official validation and package.
   Do not change its model, threshold, features or candidate routes.
2. Finish `P5-FULL-RETRIEVAL-001` on AWS and evaluate unlocked folds 0–3 only.
   Its 2,206,821 query records and full target pool allow larger owner-safe
   learning-curve samples; Fold 4 labels remain withheld.
3. Complete EXP-031 200k-query retrieval. EXP-032 is implemented and its
   200k labeled-query, 10,320,219-target, ownership and truth inputs are
   SHA256-uploaded. Launch its 51-feature store only after EXP-031 completes
   and terminates. EXP-033 then fits the same LightGBM at 20k, 50k and 100k
   entities per outer fold. Its 15k fixed OOF evaluation entities were selected
   outside the previous 20k before model fitting. Thresholds 0.83/0.79/0.83
   were frozen on that earlier sample; these are fixed-threshold diagnostics,
   not new nested threshold optimization. Both jobs have six-hour/$4.50 caps,
   per-artifact SHA256 checks and automatic termination. Neither is launched.
4. If the size curve still rises, prioritize 250k/500k over broad feature
   experiments. If it flattens, compare targeted retrieval rescue, rare-token
   features, hard-negative specialists, GBDT diversity and entity-level
   decisions on the same held-out entities.
5. Promote a change only when entity macro F0.5 and robust country/singleton
   slices improve by more than plausible split noise. Track runtime and cost.

Current measured full-pool 20k retrieval: 3,943,627 candidate pairs,
197.18135 per S1, p95/p99 200, link recall 0.96662630, complete-positive-
entity recall 0.90305178, oracle macro F0.5 0.98853146. India link recall
0.94192378 versus US 0.98344844. A top200 India pilot added 99,873
candidates for 11 links among 1,000 sampled S1, so universal expansion is
not selected. `CLASSICAL_ML_SATURATED = FALSE` because the training-size curve,
full retrieval, targeted rescue and entity-level models are unmeasured at scale.

EXP-031 input is a 200,000-S1 deterministic fold1–3 sample, including all of
the previous 20,000 entities. Each fold has about 66,667 queries, so the two
training folds offer more than 100,000 entities per outer fit. Its cloud
retrieval result will support 50k and 100k fit-size comparisons on a fixed
held-out population. The retrieval EC2 worker was launched at
2026-09-25T02:39:53Z and remained healthy at the 03:04 UTC check. A sampled
S3 route archive exactly reproduced 104 previous-development query route lists
and ranks (10,400 pairs; maximum route-score difference 1.788e-7). That is a
parity check, not a metric result. No 50k/100k score is claimed until the
complete candidate routes, feature store and OOF metric checks finish.
