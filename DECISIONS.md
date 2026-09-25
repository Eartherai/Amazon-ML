# Decisions

## D001: Research and measurement before training
Use the reference prompt as a checklist, not a mandate to launch all experiments. Finish phase-one artifacts and no-training diagnostic measurements first. Research can prioritize methods; no method is proven best until local validation.

## D002: Local bounded processing
Use DuckDB with 4 threads and 6 GiB memory, on-disk scratch capped at 6 GiB. Keep raw sources in place. No dense pair matrix; no redundant dataset copies. Reserve 8 GiB disk space. Initial free space about 24 GiB.

## D003: Official metric, stronger validation
Compute 1.25*TP/(1.25*TP+FP+0.25*FN) for each nonempty case and 1 for empty/empty; average over all S1. This follows beta squared = 0.25. The narrative '2x' is informal; FP coefficient is four times FN in this form. Official validator's optional ID check and warning-only membership check are insufficient as internal gates.

## D004: Budget and cloud
Only the user-reported $800 is a planning ceiling, not verified available credit. Later $100 award is conditional on Top 500 at 48h in pasted event copy. No new cloud resources during research. Colab CLI installation allowed; authentication may require user sign-in. AWS CLI already installed.

## D005: Preserve Unicode marks and retain normalization history
Initial audit normalization excluded combining marks; corrected to preserve Unicode M categories. Recomputed full audit and pair diagnostics as AUDIT-004/PAIR-002. Historical artifacts remain. Unit tests cover Hindi, Telugu, Bengali and French.

## D006: Research evidence prioritizes retrieval
EXP-001 has 1.0 measured micro precision but only 0.013143 recall and 0.083272 macro F0.5 on 441,103 dev entities. Do not submit. Generic transliteration helps a conditional non-ASCII sample but is not a candidate-recall measurement. Test lexical/address/transliteration union before neural full-corpus work.

## D007: Country and structural rules are hypotheses with scope
All train positive pairs have equal country and unique target ownership. This supports arbitrary-label country blocking and target-only conflict tests, not S1 one-to-one matching, a maximum-match cap, or assumptions about hidden France labels.

## 2026-09-25T01:54:17.719946+05:30: preprocessing and candidate evidence
Keep raw + light Unicode-safe representations; optional compatibility, Latin fold, sorted and transliteration features remain parallel. Disable mined abbreviation maps because training evidence contains cycles and ambiguity. Postal/numeric agreement is optional evidence, not a mandatory filter. Fit dictionaries/IDF on training-owned/unowned text only. Sample pair AP is not deployed precision. Full-pool candidate pilot recall is 77.33%, insufficient: prioritize char-ngram and cross-script rescue before GBDT. Budget currently plans on user-reported ~$200 one account, not earlier unverified $800. AWS authentication verified read-only; no paid compute justified.

## Phase3 measured selection
Multiview GBDT selected using calibration macroF0.5; separate dev check0.91056. 3x inner-OOF hardnegative weighting rejected (calibration/check both lower). Expandedtoken union adds little recall for ~500k pairs; keep only as diagnostic broad ceiling until targeted rescue validated. No fullpopulation or OOF score claimed. S3 uploads complete; compute remains local.

## Phase4 — freeze and stricter OOF boundary

Freeze BASELINE-P4-001 at 9e08d01. Preserve its exact 45 features, model and threshold. The old IDF includes folds1–3 text and cannot support training-only OOF across those folds. Phase4 therefore fits IDF only on a fixed fold0/unowned target sample; full target search remains allowed. This is an explicit prerequisite change, not an unreported baseline reproduction.

Use three original entity folds with nested two-fold threshold selection. Exclude every held-out owner's target from fit negatives. Natural samples use uniform deterministic hash ranking; diagnostic stratification is separate. Start with5k before paying for more compute. Do not interpret pooled threshold optimization on those same OOF scores as independent evaluation. Fold4 stays closed.

The Phase3 broad-union audit now orders name/address char3 first: these alone recall96.35%, while96.78% includes token_union. Expanded tokens add493,006 candidates for7 links at the end of the union. Universal expansion is not justified by this pilot.

## Phase4 — local fused retrieval benchmark

Reference5k retrieval took738.51s and1.927GiB peak RSS. Test sparse-dot-topn1.2.0 (Apache2.0) to remove dense intermediate selection cost; bounded subset gives2.59x/3.42x speedup and identical candidate sets. Use two threads, topK+1 and reference fallback for boundary near-ties. Run20k locally while5k OOF trains; compare nested5k candidates to reference before claiming parity. No AWS compute required by current resource measurements. Package versions pinned; no business information fetched.

## Phase4 — country robustness is not established

The mixed-country5k OOF result is0.923407, but India→US label transfer0.886261 and US→India0.767327. Training-country OOF alone chose thresholds0.72/0.81; no held-country labels were used. Fixed external IDF contains both known-country texts, so even this is weaker than a strict unseen-text-country check. Keep this failure prominent. Require sample-size-matched controls and domain stress results for promotions; never use fold4 as a tuning remedy.

## Phase4 — numeric alternative representation passes initial paired test

P4-NUMERIC-001 adds six generic comparison features: canonical Unicode decimal/leading-zero overlap, Jaccard, conflict, zero-format rescue, first-number equality/conflict. Raw strings and original numeric features remain. Nested5k macro0.927768 versus0.923407; paired delta+0.004361,95% CI[+0.001716,+0.006888]. Precision0.978401, recall0.851369; singleton0.872180. Provisional only until20k and country-transfer confirmation. No numeric equivalence is an automatic merge rule; postal leading zeros may be meaningful.

Now run one bounded CPU configuration each of CatBoost1.2.8 and XGBoost3.0.2 on the exact baseline45 features/candidates/folds. No sweeps, no early stopping against outer labels. Both libraries document Apache2.0 licenses; model-family comparisons remain development evidence. Do not combine numeric and family changes until individual effects are understood.

## Phase5: full-scale execution gate
Use R8i.2xlarge64GiB within current8vCPU quota to measure index throughput before planning full2.2M retrieval. Request quota increases rather than pretending GPU access exists. Full-training generation does not authorize premature Fold4 label evaluation: withhold its metrics until freeze. Preserve numeric-v2 and all prior outputs.

## Phase5 — SUB-001 release and capacity fallback

Freeze SUB-001 as 51-feature LightGBM NUMERIC-V2 trained on the currently materialized 20,000 natural-sample entities; this is an early calibration submission, not a full-data-trained final model. Keep Fold4 CLOSED and threshold0.83 from the median of three nested OOF outer thresholds. Final test candidates are the union of top100 name and top100 address char3 retrieval per source1 entity; score every final candidate, preserve candidate lists and validate before portal submission. The public reference0.964733 is for perspective only, not a threshold-tuning target.

Spot capacity failed across four 64GiB pools. Run frozen inference locally on the Mac using the checksum-verified v002 bundle. When complete, upload every shard with SHA256 verification and run the prepared validator-only 64GiB on-demand worker after training retrieval releases the 8-vCPU quota. This costs less than rerunning inference and preserves the earliest available result. Do not count any failed Spot attempt as a running instance or spent compute.

## Phase5 — use the independent SageMaker Processing quota for SUB-001 validation

The newly approved `ml.r5.2xlarge` Processing quota (2) permits a validator-only 64GiB job while the full-training EC2 worker uses the account's 8 on-demand vCPUs. Prepare its role and immutable code bundle now; launch only after all 384 complete Mac shards are SHA-verified in S3 and the fresh cost guard passes. The job has a six-hour hard stop and a $5 planning ceiling. The earlier EC2 validator remains a fallback if that worker has finished and EC2 is preferable. The unchanged official validator with `--check-ids` can materialize hundreds of millions of candidate IDs as Python sets, so 64GiB remote validation is prudent even though a local merge might fit in the Mac's disk reserve. No paid SageMaker processing job has launched yet.

## Phase5 — reject a separate missing-address threshold

EXP-029 kept the candidate set, 51 features and frozen outer models fixed, selected only the missing-address threshold on inner OOF, and evaluated on outer folds1–3. Every inner fold selected 0.90. Macro F0.5 declined from 0.931897 to 0.930533 (paired delta -0.001363; 95% CI [-0.002074, -0.000677]). The increased precision did not compensate for lost recall. Do not promote the threshold split; keep SUB-001 untouched. Future missing-address work requires a different evidence source and nested validation, not another blanket score cutoff.

## Phase5 — do not universally expand India's name route to top200

EXP-030 reused the complete 4.13M-record India target index and byte-identical frozen IDF on a deterministic 1,000-entity India development sample. The top100 route reproduced frozen candidates exactly. Raising only name topK to200 added 99,873 candidates for 11 recovered true links (9,079 candidates/link), with link recall0.945977→0.949172 and candidate oracle macro0.981488→0.982530. This is an upper bound before matcher errors and increases candidate volume about50% for the slice. Do not apply a universal top200 expansion to SUB-001 or a later model without a selective observable trigger and nested end-to-end score gain. Focus on complementary retrieval mechanisms and larger training data.

## Phase5 — measure data-scale slope before adding classical complexity

Run the dependency-gated EXP-031 retrieval, EXP-032 owner-safe 51-feature store
and EXP-033 20k/50k/100k LightGBM comparison on one new fixed 15k OOF
population. The prior 20k thresholds were frozen before choosing the new
evaluation entities; do not claim separately nested threshold selection at each
training size or compare its new 20k point directly to the older 0.9318965293
OOF population. No 50k/100k result exists yet. If the slope is meaningfully
positive across folds/countries, prioritize 250k/500k/1M scaling; if it
flattens, shift compute to retrieval, feature and model diversity. Preserve
Fold4 CLOSED and the frozen SUB-001 path regardless of this outcome.

## Phase5 — accept only audited float32 near ties for EXP-031 handoff

The initial exact old20k parity gate for EXP-031 failed: among 2,000,000 route pairs per field, name had zero changed candidate IDs but 16 changed ranks in seven queries; address had 57 rank changes in 24 queries and one rank-100 candidate exchange for S1-428740130. Both exchanged target IDs are absent from that entity's known ground truth. Same-pair scores differ by at most 2.384e-7; changed-rank score spans are below 9e-8. All 256 route archives passed SHA256 verification, every old20k query has 100 pairs per route, and no known positive is lost by the one boundary exchange. These are float32 near-tie ordering effects, not evidence of a different blocking configuration. Preserve the failed strict result and report `exact_id_rank_parity: false`. Permit EXP-032 only under the new `NUMERIC_TIE_PASS` audit, whose code rejects more than one boundary candidate exchange, a positive exchange, wider score/rank drift, incomplete coverage, or a changed archive hash. The full candidate sets and old20k route rank features are nearly, but not byte-for-byte, identical to P4-B-002; any later score comparison must cite this difference. Fold4 remains CLOSED.
