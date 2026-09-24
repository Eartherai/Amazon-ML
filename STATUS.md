# Status

Updated 2026-09-25T01:54:17.719946+05:30. **Phase 3 IN PROGRESS: research, full-target sparse retrieval and S3 multi-view foundation.** Small learned matcher pilots are complete; full-scale training has not started. No final test predictions or leaderboard submissions exist.

## Current understanding

Business entity resolution with multi-match output and exact macro F0.5. Every S1 needs a row. Preserve all target IDs, raw text, open country strings and candidate provenance. Deadline 27 September 23:59 IST; maximum 5 submissions/day.

## Dataset statistics

24,229,173 source rows; train S1 2,206,821; test S1 1,732,544; labeled links 7,638,365. Singletons 123,247 (5.58%); most S1 have 3+ links. Train countries US/India; test additionally 259,452 France S1. Full figures in DATA_AUDIT.md and docs/audit_evidence/.

## Important discoveries

- No invalid label references, duplicate labels or shared target owners. No cross-country positive links.
- Same-source train/test IDs and raw full records have zero overlap; this is not proof against all near-duplicate leakage.
- Strict exact name+address catches only 1.31% of development links, despite zero measured false positives.
- India has much harder textual variation. Generic transliteration raises high name similarity from 16.49% to 29.78% on the Indian portion of a non-ASCII-positive sample; it helps but is insufficient alone. This is not candidate recall.
- Postal-like overlap is only about 0.22% of Indian positives; not a mandatory block.
- Preserve Indic combining marks. Target-only ownership is a hypothesis worth testing; top-one S1 matching is wrong.

## Current best measured result

- Model: EXP-001 deterministic exact normalized name AND address AND country. **Diagnostic, not competitive.**
- Development entities: 441,103; all 10,320,219 training targets remain in retrieval pool.
- Macro F0.5: **0.08327203**.
- Micro precision / recall: **100.00% / 1.3143%**.
- Singleton / non-singleton F0.5: 1.00000000 / 0.02894529.
- Candidate recall: 1.3143%; 20,075 pairs; oracle macro ceiling 0.08327203.
- All-empty control: 0.05594612.
- Latest matcher experiment: EXP-001, code commit a5588f86bde3fa7495da58640b31d71720666e97.
- Latest retrieval experiment: EXP-002 / TOKEN-001, code commit 70fa711c6c5f020445e6d8d1bca3c82348b8444c.
- Locked fold 4: no model score computed.
- Best public leaderboard: none from this task.

## Potential leakage risks

Fitted vocabulary, noise dictionaries, hard-negative mining and calibration must respect fold ownership; IDs/row order must never be predictors. Train+test fitted normalization/IDF remains excluded pending rule clarity. Broader cross-source/near-duplicate group leakage is not disproven. Cross-country stress and collision-group stress are designed, not executed.

## Top 10 experiments

1. Completed exact diagnostic (EXP-001).
2. Character TF-IDF/rare-token union against realistic full pool.
3. Add address/numeric and token-order rescue.
4. ICU transliteration retrieval and collision audit.
5. First LightGBM with rich pair/retrieval features.
6. Hard negatives, OOF calibration and entity threshold selection.
7. Multilingual E5-small rescue and similarity feature pilot.
8. Target ownership and set/singleton decisions.
9. Cross-encoder on justified ambiguity subset.
10. Complementary ensemble, ablations and package rehearsal.

Next five unfinished experiments are items 2-6. Details and promotion gates in docs/RESEARCH.md.

## Expected compute / AWS plan

Local M5/24 GiB confirmed; audit peak RSS 3.14 GiB; actual per-file memory measured separately. DuckDB 4 threads/6 GB cap. At test K=50, ~86.6M pairs; a 100-feature float32 matrix alone is ~34.7 GB, so batches/shards are required. Local disk about 20 GiB free after phase one; reserve 8 GiB.

Task AWS spend **$0**; task Colab spend **0 units**. Latest user-reported budget **~$200 on the configured account**; verified balance unknown. Other accounts and future credits are excluded. Read-only AWS STS authentication succeeded in us-east-1; no cloud resources launched.

Colab official CLI 0.7.2 installed. Login verified after user sign-in; balance 0.00 compute units, rate 0.00/hr, zero assignments/sessions. Free GPU availability untested. Kaggle capacity not inspected or used.

## First learned model to build and why

LightGBM on blocked pairs, with Unicode/transliteration-aware lexical routes, name/address/numeric disagreement and candidate rank/rarity features. It is inexpensive, interpretable and well suited to structured pair features. It cannot overcome a poor candidate ceiling; improve and measure retrieval first. Model not trained yet.

## Verification

44 tests pass, including hand-calculated metric cases, Unicode marks, strict TSV/membership checks, deterministic merging, multiple targets, unseen countries and the official validator on a synthetic fixture. Full data counts agree between independent DuckDB and Polars reads. Final real-output validation is pending final predictions.

## Current blockers and next action

No blocker to local retrieval prototypes. Remote limitations: zero verified paid Colab balance/free allocation unknown; AWS credit eligibility/balance and GPU quota unverified. Authentication is working. Linux ICU parity and large-scale retrieval speed remain unmeasured. No guarantee of winning.

**Current next action:** add character-ngram retrieval and transliteration rescue to the full-target pilot, then measure marginal recall and candidate growth. The 77.33% lexical-union recall is inadequate. No learned matcher yet.

## Phase 2 discoveries and current configuration

- Full profiles: 24,229,173 records / 12 text fields / 30 country-field slices; 675 integrity checks passed. Full run 824.17s, peak process RSS 2.81 GiB; supplement 9.52s.
- 34 figures, searchable local HTML report, exact within-country train/test JSD/PSI/Wasserstein comparisons.
- PREP-001: 25k train-fold positive pairs; 100k-string benchmark; zero native parity mismatches. Primary preprocessing retains raw text, NFC/casefold and Unicode combining marks. Aggressive, sorted, Latin-folded and transliterated forms remain separate views. Learned maps disabled.
- MORPH-002: 48,130 pair comparisons from 2,000 dev queries; light normalization exact positive name agreement 4.18% raw → 20.48%; aggressive raises equality but worsens some similarity discrimination. These are sample diagnostics, not model scores.
- TOKEN-001/run-002: all 10,320,219 training targets, 1,000 dev queries. Union recall **77.33%**, 179,298 pairs; top-100 **73.56%**, 78,332 pairs. Oracle macro ceiling **0.86630** is not an achieved score. EXP-001 remains the only matcher score.
- Candidate provenance and ranking persisted. 47/50 pilot singletons have candidates; do not infer a match from candidate existence.
- Full corpus and sampled analysis scopes are explicit in docs/PREPROCESSING_ANALYSIS.md. No exhaustive all-pairs fuzzy deduplication or supervised France validation is claimed.

## Next ten experiments, in order

1. Full-target character 3/4/5gram candidate retrieval.
2. Selective token fanout relaxation; diagnose queries with no rare keys.
3. ICU transliterated rescue plus negative collision audit.
4. Address-only retrieval with numeric evidence and missing-field handling.
5. Per-route cap and rank-fusion ablations; retain broad union initially.
6. Expand query sample by country and nonzero match-count strata.
7. One-operation suffix/abbreviation map ablations; leave disabled otherwise.
8. Cross-country and collision-group stress validation.
9. Persistent index/cache and full-query batch throughput benchmark.
10. First GBDT only after sufficient candidate coverage; then OOF macro F0.5 calibration.

See docs/PREPROCESSING_VISUAL_REPORT.html, docs/PREPROCESSING_ANALYSIS.md and docs/CANDIDATE_PILOT.md.

## Latest Phase 3 checkpoint — 2026-09-24T21:09:35.946477+00:00

Broad candidate union link recall **97.62%**, complete-entity recall **93.05%**, oracle macro **0.99116**, average candidates **952.4**, P95 **1406**, P99 **1460**. Scope: 1,000 dev queries, full10.32M pool.

Best calibration-selected matcher: GBDT-004 multiview, threshold 0.58; 505-entity check macro **0.91056**, precision **95.35%**, recall **86.26%**, singleton **80.00%**. Not OOF/final holdout. First serious model exists; earlier no-model statements above describe the previous phase. Fold4 closed.

S3 raw + processed complete, 51 checksum-verified objects, 4.33GB. Storage now accrues nominal cost (~$0.0927/month), actual bill unknown; compute $0. Expanded-token volume and nonASCII/hard-pair coverage remain concerns. See docs/PHASE3_CHECKPOINT.md for tables and next work.
