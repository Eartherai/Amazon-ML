# Status

Updated 2026-09-25T01:08:47.520574+05:30. **Initial research/audit phase complete.** Learned-model training has not started. No final test predictions or leaderboard submissions exist.

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
- Latest successful experiment: EXP-001, code commit a5588f86bde3fa7495da58640b31d71720666e97.
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

Task AWS spend **$0**; task Colab spend **0 units**. User-reported initial budget **$800 across four accounts**, not verified; current verified dollar balance unknown. Extra credits budget **$0** until confirmed. Preserve $300 contingency. AWS profile exists but session expired and default region missing; no cloud resources launched.

Colab official CLI 0.7.2 installed. Login verified after user sign-in; balance 0.00 compute units, rate 0.00/hr, zero assignments/sessions. Free GPU availability untested. Kaggle capacity not inspected or used.

## First learned model to build and why

LightGBM on blocked pairs, with Unicode/transliteration-aware lexical routes, name/address/numeric disagreement and candidate rank/rarity features. It is inexpensive, interpretable and well suited to structured pair features. It cannot overcome a poor candidate ceiling; improve and measure retrieval first. Model not trained yet.

## Verification

23 tests pass, including hand-calculated metric cases, Unicode marks, strict TSV/membership checks, deterministic merging, multiple targets, unseen countries and the official validator on a synthetic fixture. Full data counts agree between independent DuckDB and Polars reads. Final real-output validation is pending final predictions.

## Current blockers and next action

No blocker to local retrieval prototypes. Remote blockers: zero verified paid Colab balance/free allocation unknown; expired AWS login, account/credit/region/GPU-quota confirmation outstanding. Linux ICU parity and large-scale retrieval speed remain unmeasured. No guarantee of winning.

**Next action:** implement EXP-002 high-recall lexical union, benchmark 4k-20k development queries against all training targets, then assess rescue from EXP-003/004 before fitting the matcher. Do not submit EXP-001.
