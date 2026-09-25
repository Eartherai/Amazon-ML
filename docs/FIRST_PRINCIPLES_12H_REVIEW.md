## 1. Chronological experiment history

Compiled 2026-09-26 from STATUS.md, TIMELINE.md, EXPERIMENTS.csv, SUBMISSIONS.csv, DECISIONS.md, AWS_SPEND.md, FINAL_SPRINT_BOARD.md, docs/*.md, `outputs/experiments/*` result JSONs, `outputs/submissions/*/provenance.json` and `git log --all` (HEAD 1a835e0). Reading only; no new measurement. Fold4 was never opened, so every local number is development OOF, not a lock-box score.

Local scores come from different populations. They can be compared within a population but not across populations:

- **P1k**: 1,000 balanced-country fold-0 dev queries (Phase 2/3 retrieval pilot). The matcher check used 505 of them and calibration used 495.
- **N5k / N20k**: natural samples from folds 1-3 with nested 3-outer/2-inner OOF.
- **F15k**: a fixed 15k OOF set (5k per fold 1-3), drawn outside the old 20k, with thresholds frozen from 20k.
- **O200k**: 200k strictly OOF S1 from folds 1-3 (EXP-032 feature store).
- **F6k**: a fixed 6,000 S1 from held folds 2/3 (the EXP-044 set). It was reused across all stage-2 work.
- **I194k**: stage-2 training S1 with grouped inner OOF.
- **Public**: a hidden test subset that includes France (15.0% of test S1) and has no train labels.

Public scores are user-reported. The VSAFE public 0.947 was supplied by the lead engineer and is not yet recorded in SUBMISSIONS.csv. In the table, "—" means not submitted or not measured. Only public scores are listed in the Public column; local candidates that never received a public score are marked "not uploaded" in the last two columns.

| Experiment id | Idea | Validation population | Local score | Public score | What improved / failed | Likely reason |
|---|---|---|---|---|---|---|
| EXP-001 | Exact normalized name AND address, same country | fold-0 dev (441,103 entities) | macro 0.08327203; P 1.0, R 0.01314345 | — | Diagnostic floor only | Exact equality almost never holds; fuzzy retrieval needed |
| EXP-002 / TOKEN-001 | Exact/rare-token and numeric-name routes, RRF | P1k | link recall 0.7733; oracle 0.86630 | — | Recall too low | Token equality misses spelling, format and script variation |
| EXP-003 / CHAR-001, TRANS-002, TOKEN-002 | Char 3/4/5-gram TF-IDF name/address routes, transliteration, expanded tokens | P1k | name+address char3 alone 96.35%; full union 97.62% recall, oracle 0.99116 | — | Char3 fixed recall. Char4/5 and translit each added 2-7 links; expanded tokens added 493,006 pairs for 7 links | Diminishing marginal routes; char3 top100 frozen for production |
| EXP-004 / GBDT-002 | LightGBM on lexical pair features | P1k (505 check) | 0.89738 | — | First matcher | — |
| EXP-005 / GBDT-004 | Multiview LightGBM with transliteration view; hard-neg 3x variant | P1k (505 check) | 0.91056 (hardneg3 0.90635) | — | Multiview selected; hard-neg weighting rejected | Dev-check was re-observed across choices, so optimistic |
| EXP-007 / P4-A-001 | 45-feature multiview LGBM, disjoint IDF, nested OOF | N5k | 0.9234065, CI [0.918577, 0.927935] | — | First credible OOF | — |
| EXP-008 | Platt / isotonic / pair+empty calibration | N5k | 0.9227854 / 0.9234973 / 0.9231134 vs raw 0.9234065 | — | No promotion | Differences within noise |
| EXP-009..016 | Feature-group ablations | N5k | deltas: numeric -0.018183, token -0.016118, char -0.016009, retrieval -0.005850, translit -0.005432, route count -0.001075 (CI crosses 0) | — | All groups except route count carry signal | — |
| EXP-017 | Query-context (max/gap/relative) features | N5k | 0.9222138 (-0.0011927) | — | Failed | No signal beyond pair features |
| EXP-018 / P4-NUMERIC-001 | +6 canonical numeric comparison features | N5k | 0.9277678 (+0.0043613, CI [+0.0017161, +0.0068877]) | — | Improved | Digit and leading-zero variation in addresses |
| EXP-019 / 020 / 021 | CatBoost / XGBoost / ownership-weighted LGBM | N5k | 0.9218153 / 0.9227954 / 0.9218733 | — | None beat LightGBM | Model family was not the bottleneck |
| Country transfer | Train on one known country, evaluate on the other | N5k by country | India->US 0.886261, US->India 0.767327 (singleton 0.516129); with numeric 0.892801 / 0.781921 | — | Large transfer loss | Country-specific text and format distributions; early warning for France |
| EXP-022 | 45-feature model at 20k | N20k | 0.9287589; candidate link recall 0.966626, oracle 0.988531 | — | Confirmed at scale | — |
| EXP-023 / NUMERIC-V2 | 51 features (numeric added) | N20k | 0.9318965 (P 0.9815828, R 0.8559381); new-15k delta +0.003299, CI [+0.001840, +0.004701] | (see SUB-001) | Became SUB-001 model | — |
| EXP-024 / EXP-025 | Persistent full-target index; full 2.2M-S1 char3 retrieval | Partial audit of 109,890 unlocked S1 | India link recall 0.940972 (S3 0.930220), US 0.982811; India oracle 0.978510 | — | Full-population table never finalized | India/S3 retrieval ceiling |
| EXP-026 | Country-balanced query weights | N20k | 0.9317121 (-0.000184, CI crosses 0); India +0.00196, US -0.00165 | — | Not promoted | Trades US for India |
| SUB-001 | Frozen EXP-023 (20k fit), threshold 0.83, char3 top100 union | N20k | 0.9318965 | 0.912967 (rank 308); first upload FAILED, retry scored | Gap -0.0188965 (vs 0.913) | Public includes unlabeled France; France ~0.838 is only an algebraic hypothesis (BASELINE_PUBLIC_0913) |
| EXP-027 | Fit-size curve, 2k to ~13.3k | fixed folds 1-3 of 20k | 0.924221 / 0.929379 / 0.930829 / 0.932231 | — | Rising | More data helps |
| EXP-028 | Error slices | N20k | 2,315 retrieval misses, 7,678 rejected; missing-address 684/3,045 predicted | — | Diagnostic | — |
| EXP-029 | Separate missing-address threshold | N20k | 0.930533 (-0.001363, CI [-0.002074, -0.000677]) | — | Failed | Recall lost exceeds precision gained |
| EXP-030 | India name top200 | 1k India | recall 0.945977 -> 0.949172; +99,873 candidates for 11 links | — | Rejected on cost | Untriggered cap increase has low yield |
| EXP-031 / 032 | 200k retrieval and 51-feature store | 200k folds 1-3 | link recall 0.9660720; complete-entity 0.9010812 | — | Infrastructure (NUMERIC_TIE_PASS) | — |
| EXP-033 | LGBM at 20k / 50k / 100k fit | F15k | 0.9349224 / 0.9363053 / 0.9371581 | — | +0.0022358 from 20k to 100k | Modest data-scale slope |
| SUB-002 (V001 / V002) | 200k LGBM final fit; V002 adds source completion and ownership | no OOF for the 200k fit | — | — | Not uploaded; later reused as first stage | — |
| EXP-034 | Max-score target ownership | O200k | 0.9362345 vs 0.9357645 (+0.00047) | — | Small gain | Train truth: every target has one owner |
| EXP-035 | Cross-fit missing-source completion + ownership | O200k | 0.9379423 (+0.0021779, CI [+0.0019933, +0.0023624]) | — | Small gain; never uploaded (V002) | — |
| EXP-038 | 29 entity set-decision rules | O200k | best 0.9366603 (+0.000896), threshold searched on same OOF | — | Shelved | Low gain, optimistic selection |
| EXP-039 | Hard-negative LGBM, 50% blend with base | F15k | 0.9388301 vs 0.9371581 (+0.00167, CI [+0.000718, +0.002626]) | — | Positive, never shipped | — |
| EXP-042/043/044 | Fold-1 fine-tuned mMARCO MiniLM cross-encoder logit blended with 51f LGBM (sidecar pairs) | F6k | 0.9431175 vs 0.9325591 (+0.0105584) | — | Improved locally | Independent text signal on hard pairs |
| EXP-045 | Larger fold-1 mMARCO (3k S1, 19,600 pairs) blend | F6k | 0.9474835 (+0.0149243); India +0.0256133 | — | Full-test neural TSV built; never uploaded | Superseded by CL-003 (see CL-004) |
| EXP-046 | Cardinality/set rules on the neural blend | F6k | 0.9431051 (-0.00001245) | — | Failed | — |
| EXP-047 | Conditional numeric top10 India retrieval rescue | 1k India | 14 links; oracle +0.00133724 | — | No promotion | Low yield |
| EXP-048 | France unlabeled shift audit | test France | non-ASCII names 15.67%, addresses 28.47% | — | Diagnostic | Train S1 names are all ASCII |
| EXP-049 | E5-small CE logit + 51f LGBM (friend reproduction) | F6k | 0.9406633; E5+mMARCO+base 0.9450201 | — | Below EXP-045 | 3k S1 / 70k model-mined pairs vs friend's ~18k / 420k lexical; friend 0.9626/0.9653 is on a different 20k set |
| FULL-OWNER-002 | Full 2.2M-S1 lexical top20 competitor search + fuzzy veto | F6k | 0.9475210 (+0.0000375 vs EXP-045); oracle veto ceiling +0.007258 | — | Failed | Fold-safe veto could not separate owners |
| CL-001 | Stage-2 LGBM over first-stage OOF scores plus new classical features (core names, IDF rarity, translit skeleton, digit edits, sibling corroboration) | F6k, cross-fit folds 2<->3 | sidecar 0.9456921 (+0.013146); top12 0.9503636 (+0.01782); with EXP-045 logit 0.9543238 / top12 0.9578128 | — | Large gain | Richer string evidence plus competitor context |
| CL-002 | Production stage-2 fit on the 6k only; test TSV variants | — | — | — | Not uploaded; superseded | — |
| CL-003 | Stage-2 trained on 194k OOF S1 | F6k (eval S1 and owned targets excluded) | sidecar 0.9581314; top12 0.9626062 | — | +own+dup upload (~0.9585) not uploaded | Stage-2 training scale |
| CL-004 | Stack CL-003 probability + EXP-045 logit | F6k | 0.9580124 (-0.000119, CI [-0.00160, +0.00144]) | — | Failed | Neural signal subsumed by stage-2 |
| CL-006 (v2 features) | Digit deletion vs substitution; address skeleton cover | F6k; I194k inner 60k | 0.9631106 (6k); inner 0.9653327 | — | Small gain over v1 top12 | — |
| SUB-003 (CL-005 + own + dup) | v2 top-12 stage-2 on SUB-002 200k first-stage test scores; thr 0.67; ownership; duplicate expansion | F6k; I194k | ~0.9635 (F6k, own+dup); 0.9652122 plain (I194k) | 0.944 (rank ~491) | +0.031 public vs SUB-001; gap still ~-0.019 | Stage-2 gain transferred; shared frozen retrieval suspected for the gap |
| CL-007 | Transliterated skeleton-key rescue outside top-12 | F6k | 0.9629090 vs 0.9626062 (+0.000303; 11 links, all true) | — | Not shipped | Too few links |
| CL-008 | Max-probability ownership on stage-2 | I194k | 0.9652122 -> 0.9655647 (+0.000352, CI [+0.000293, +0.000407]); 435 links removed | (inside SUB-003) | Small gain | — |
| CL-009 | Text-only stage-2 (no first-stage features) | F6k | 0.9561288 at 0.70 vs 0.9631106 | — | Weaker; reused only to score rescue pairs | Text-only override inside top-12 ~62% precise |
| ERR-MINE-001 | Loss decomposition of v2 stage-2 | I194k | loss 0.0348: rejected-in-top12 0.012965, outside-top12 0.012273, FP non-singleton 0.007776, singleton FP 0.001773; oracle_top12 0.987727, oracle_entity_decision 0.985323 | — | Diagnostic | — |
| CL-010 | Covariate-shift reweighting of 194k OOF to test near-text counts | I194k reweighted | no result recorded | — | Unknown | — |
| CL-011 | Test-fitted char3 TF-IDF retrieval probe; text-only stage-2 scores new pairs | test (unlabeled); small train-India sample | strong new pairs/S1: France 0.2-0.28, India 0.01, US 0.005, train India 0.03 (92% precise at p>=0.8, small sample) | — | Located France retrieval deficit | Train-fit IDF/vocabulary under-represents French text (hypothesis) |
| SUB-004 (CL-012 p>=0.85) | France-fitted retrieval (13.1M new pairs), 47,863 rescue links; 44,165 ownership removals | France unlabeled; India/US = SUB-003 | — | — | No public score recorded | — |
| CL-012 FINAL5 | Rescue at p>=0.90, no-steal, 42,728 links | same | — | — | No public score recorded | — |
| VSAFE (CL-012) | 39,470 rescue links (the `safe`-flag subset of the p>=0.90 no-steal set); France only | India/US rows identical to SUB-003 | — | 0.947 | +0.003 public from a France-only change | Confirms missed France retrieval was a real loss |
| P5-RESCUE-F1..F4 | AWS France routes: name char3, combo char3-5, address char3, combo word (top50) | — | no result recorded | — | Unknown | — |
| EXP-050 / CL-013 | Friend-scale E5-small CE (18k S1, 420,347 pairs) stacked on v2 stage-2 | F6k | 72,000 pair logits scored; stack result not recorded | — | Unknown | — |

The file does not record what the `safe` flag in VSAFE means. It has 42,728 rows, and the flag is set on 39,470 of them, which is the VSAFE row set.

### What actually worked

- **Char3 TF-IDF retrieval** (name and address, top100 each). It raised link recall from 77.33% (tokens) to 96.35%, and every submission is built on it.
- **Canonical numeric features.** They added +0.0043613 on N5k and +0.003299 on the new 15k, with both CIs above 0.
- **Classical stage-2 re-ranker over first-stage OOF scores** (top-12 scope, 194k-trained). This was the only step change. On F6k it moved 0.9325 to ~0.9635, and on the public board 0.912967 to 0.944. SUB-003 bundled the 200k first stage, stage-2, top-12 scope, ownership and dup expansion, so the public gain cannot be split among them.
- **France test-fitted retrieval rescue (VSAFE).** Public went from 0.944 to 0.947 with India/US rows unchanged. It is the only component whose public effect is isolated.
- **Single-owner target ownership.** The gain is small but consistent: +0.00047 on O200k and +0.000352 on I194k.

### What only looked good locally

- **Absolute known-country scores.** They overstated public by about 0.019 for both SUB-001 and SUB-003. The local-to-local gain and the public gain were similar in size, but the local scores were measured on different populations (N20k and F6k).
- **Neural cross-encoder blends** (EXP-044 +0.0106, EXP-045 +0.0149 on F6k). They were never uploaded, and CL-004 shows no added value once stage-2 exists.
- **Friend's 0.9626 / 0.9653.** These come from a different 20k population with no weights supplied. Our EXP-049 reproduction reached 0.9407.
- **Small positive OOF tweaks never isolated on public.** EXP-035 completion (+0.0022), EXP-039 hard-negative blend (+0.00167), EXP-038 set rules (+0.000896, threshold chosen on the same OOF) and CL-007 rescue (+0.000303).
- **F6k selection optimism.** F6k was the evaluation set for CL-001 through CL-006 scope and feature-version choices, so its ~0.9635 carries some selection optimism.

### What failed

- **Exact matching and token-only retrieval** (0.0833 macro; 77.33% recall).
- **Untriggered retrieval expansion.** char4/5, translit and expanded tokens added 2-7 links each. India top200 added 99,873 candidates for 11 links, and numeric top10 recovered 14 links.
- **Threshold and calibration tweaks.** The missing-address threshold lost -0.001363. Platt and isotonic calibration showed no gain, EXP-046 changed by -0.0000125, and hard-negative 3x weighting and ownership weighting were both rejected.
- **Model-family and weighting swaps.** CatBoost, XGBoost, query-context features (-0.0011927) and country-balanced weights (-0.000184) did not help.
- **E5 reproduction (0.9407).** It did not help.
- **Neural logit stacked on stage-2** (-0.000119). It did not help.
- **Full-corpus fuzzy owner veto** (+0.0000375). It did not help.
- **Text-only stage-2 as a replacement** (0.9561 vs 0.9631).
- **Cross-country label transfer** (US->India 0.767327). This warning was later borne out by the France-driven public gap.

### Never tested properly

- **France accuracy.** There are no labels. The ~0.838 and ~0.85 estimates are algebraic hypotheses from public minus known-country OOF.
- **Which France rescue is best.** SUB-004 (p>=0.85, 47,863 links), FINAL5 (p>=0.90 no-steal, 42,728) and VSAFE (39,470): only VSAFE has a public score. The F1-F4 route union results are not recorded, and text-only rescue precision on France was never measured.
- **Test-fitted retrieval for India/US.** It was only probed (0.01 and 0.005 strong new pairs/S1). The India/S3 retrieval ceiling (oracle 0.978510) is also unaddressed at scale.
- **Recovery of the two largest measured losses** (I194k). True links outside top-12 cost 0.012273, and rejected-in-top12 links cost 0.012965. No nested entity-level decision model was built on stage-2 outputs (entity-decision oracle 0.985323).
- **Friend-scale components.** The stack result for the friend-scale CE (EXP-050 / CL-013) is not recorded. The full-corpus BIENC competitor search with target-centric assignment is unmeasured.
- **Training scale.** Scale beyond 100k first-stage fits and 194k stage-2 is untested (250k/500k/1M pending). EXP-025 full-population retrieval metrics were never finalized.
- **SUB-003 component attribution.** The public effect of each SUB-003 component is unknown, and duplicate expansion has no recorded local metric.
- **CL-010 covariate-shift reweighting.** No result was recorded.
- **Fold4 lock-box.** It was never evaluated.

<!-- analysis sections appended below -->

## 2. First-principles findings (measured 2026-09-26 00:00–01:05 IST)

Validation populations are stated for every number. "194k OOF" = 194,000 held-fold OOF S1 (folds 1-3 of the 200k store, fixed-6k removed); "fold-3" = ~65k fold-3 S1 of that set; Fold4 CLOSED.

### 2.1 Loss decomposition (194k OOF, stage-2 top-12 v2, threshold 0.67): macro 0.9652
| Component | Loss mass |
|---|---:|
| True links outside top-12 (almost all retrieval misses) | 0.0123 |
| Rejected true links inside top-12 | 0.0130 |
| False-positive links on non-singletons | 0.0078 |
| Singleton false merges | 0.0018 |
Oracles: top-12 0.9877, full candidates ~0.9885, per-S1 decision oracle 0.9853. India 0.9530 vs US 0.9734; India S3 retrieval miss rate 7.0%.

### 2.2 Decision rules are not the bottleneck
Expected-F0.5 per-S1 set selection with isotonic calibration: 0.96523 vs global threshold 0.96521 (+0.00002). Ownership variants: best +0.00035 in-sample (larger on test where all S1 compete). Conclusion: gains must come from better pair probabilities and better candidates.

### 2.3 Public/local gap is a domain-transfer problem
Two independent pipelines (ours and the friend's CE pipeline) both lose 0.016-0.019 local->public. Unseen-country simulation with text-only stage-2: US->India 0.868 vs in-country 0.946; India->US 0.919 vs 0.968. Self-training on the unlabeled target country recovers only +0.0065/+0.0076. With India/US at local level, public 0.947 implies France ~0.855. A per-country public decomposition (France-empty / India-empty diagnostic files, built and validated) would confirm the split; awaiting user approval of 2 portal slots.

### 2.4 Two measured step changes
1. **Scaled cross-encoder** (multilingual-e5-base, MIT, trained on 739,590 owner-safe fold 1-2 band pairs, 1 epoch): stacked with stage-2 on fold-3 (~65k S1) macro 0.96466 -> **0.96927 (+0.0046, CI +0.0040..+0.0053)**; India +0.0111.
2. **Dense bi-encoder retrieval** (multilingual-e5-small, InfoNCE on fold 1-2 positives): India fold-3 dense link recall **99.77%**; of 5,554 India true links outside the sparse top-12, top-10 dense recovers 5,095 with 4.9 new candidates/S1. Adding dense pairs scored by text-only stage-2 at p>=0.7 (only unclaimed targets): India fold-3 macro 0.9526 -> **0.9663 (+0.0137)**, added-link precision 0.917.

### 2.5 Implication for 0.98 / 0.99
Retrieval is no longer the hard ceiling for India once dense rescue is used. Remaining path: CE-scored union of sparse top-12 and dense top-10, stacked, plus ownership; then France robustness (dense retrieval is vocabulary-free, which should also help France). 0.99 still requires near-perfect matching of the remaining ~2% hard pairs and is not supported by current evidence.
