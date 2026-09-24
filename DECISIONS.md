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
