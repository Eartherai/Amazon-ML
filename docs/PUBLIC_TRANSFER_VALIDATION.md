# Public-transfer validation (leaderboard-shift simulator)

Purpose: the public score has consistently sat ~0.019 below local validation (local 0.932 -> public 0.913; local ~0.963 -> public 0.944/0.947), and the implied France score is ~0.855. This document measures how each architecture survives distribution shift, so selection is not driven by in-domain OOF alone. All numbers are exact macro F0.5 on labeled train S1 with full truth (retrieval misses count); Fold4 CLOSED; thresholds are always chosen on the source domain only. Scripts: `scripts/classical/transfer_matrix.py`, `transfer_stack.py`, `retrieval_shift.py`, `fs_em.py`, `eval_pair_scorer.py`, `dense_ce_rescue.py`, `slice_report.py`.

## Stress tests 1/2 — cross-country transfer (fold-3 target S1)

Stage-2 GBDT alone (CL-030; train folds 1-2 of source country):

| variant | US->US | US->India | India->India | India->US |
|---|---:|---:|---:|---:|
| FULL (stage-1 context + pair features) | 0.9733 | 0.9178 | 0.9530 | 0.9511 |
| NO_RANK | 0.9734 | 0.9182 | 0.9534 | 0.9500 |
| NO_ABS_BASE (relative context only) | 0.9728 | 0.9150 | 0.9523 | 0.9497 |
| TEXT (pair features only) | 0.9687 | **0.8736** | 0.9466 | **0.9176** |
| TEXT_NO_SCRIPT | 0.9687 | 0.8736 | 0.9462 | 0.9185 |

- Raw rank / absolute score are not the shift driver (ablations change transfer by <0.003).
- Calibration is not the driver: the source-chosen threshold is within 0.0024 of the target-oracle threshold for FULL; TEXT loses up to 0.008 from threshold shift.
- The stage-1 context rows are optimistic: stage-1 was trained on both countries, so it carries target-domain knowledge that France never had. **TEXT is the honest unseen-country analogue; its US->India loss (-0.073) matches the implied public France gap.**

Full primary system (CL-034; stage-2 + e5-base CE stack + ownership + dense rescue, every learned part source-only):

| system | US in-domain | US->India | India in-domain | India->US | mean transfer |
|---|---:|---:|---:|---:|---:|
| FULL + CE | 0.9824 | 0.9235 | 0.9778 | 0.9254 | 0.9245 |
| TEXT + CE | 0.9783 | 0.9104 | 0.9762 | 0.9111 | **0.9108** |
| TEXT, no CE | 0.9716 | 0.8829 | 0.9603 | 0.9217 | 0.9023 |
| FULL, no CE | 0.9758 | 0.9217 | 0.9668 | 0.9484 | 0.9351 |
| TEXT + CE + zero-shot bge-reranker-v2-m3 | 0.9786 | 0.9007 | | | |

- The cross-encoder raises in-domain by +0.007 to +0.016. Its transfer effect is direction-dependent: +0.028 US->India but -0.011 India->US (TEXT path). A US-only CE generalizes; an India-only CE (trained with 41% Indic-script targets) does not.
- The production models train on both countries, which should transfer better than either single country. That cannot be simulated with only two labeled countries.

## Stress tests 3/4 — retrieval vocabulary shift and unsupervised adaptation (CL-032)

India fold-3 queries (3,000) against an India universe of 1.01M targets; char3 TF-IDF (name + address), link recall:

| K | fit on US only | fit on India targets (unlabeled) | both | India + accent/NFKC folding |
|---:|---:|---:|---:|---:|
| 5 | 0.8276 | 0.8349 | 0.8241 | **0.8381** |
| 10 | 0.8895 | 0.9014 | 0.8889 | **0.9055** |
| 20 | 0.9064 | 0.9201 | 0.9103 | **0.9243** |
| 100 | 0.9339 | 0.9492 | 0.9383 | **0.9533** |

Test-side adaptation (allowed: unsupervised test statistics) recovers +1.2 to +1.9 points of link recall. That is real but modest; vocabulary shift is not the main France loss, the matcher is.

## Stress test 5 — candidate-density shift

Rank features were ablated (above): removing raw rank or absolute scores barely changes transfer, so the matcher does not depend on absolute rank. Test candidate density: sparse top-12 is 12.0/S1 in every country, and the stage-1 band p>=0.02 is 5.96 India, 5.59 US but **7.83 France**. France has more mid-score candidates, which is consistent with weaker stage-1 separation there.

## Stress tests 6/7 — script, Unicode and missingness shift

Train S1 is effectively all ASCII (US 0.0%, India 0.06% non-ASCII) with no missing addresses. Test France S1 is **38.6% Latin-accented** (targets 37.9%). So France introduces a query-side script/diacritic pattern that no training S1 ever had. Train and test India targets have the same share of Indic script (41.4% / 41.7%), and missing-address rates match (2-4% of targets).

Per-slice fold-3 macro of the new primary system (CL-037): India 0.98377, US 0.98278; short names (<=6 chars, 0.5%) 0.966 is the worst slice; singletons 0.987; S1 with 5+ links 0.988. The ASCII / missing-address slices cannot be stress-tested in train because train has no such S1. **The accent pattern is the most concrete untested France shift. Accent/NFKC folding of France queries and targets before matching is the targeted fix; it can only be verified on the public leaderboard (A/B against the unfolded file).**

## Killed tracks (kill rule: no complementary recall, no OOF gain, no transfer gain)

| track | pilot | result |
|---|---|---|
| D Fellegi-Sunter EM (unsupervised) | 14 binned comparisons, EM per country | AUC 0.77-0.84, macro 0.53-0.57; conditional independence fails on correlated features |
| C zero-shot bge-reranker-v2-m3 (Apache-2.0) | standalone and as stack feature | standalone 0.80-0.82; stack transfer -0.010 (0.9104 -> 0.9007) |
| B byte-level CE from scratch (11M params) | US-only, 3 epochs | US 0.909, US->India 0.715 (-0.19, vs e5 CE -0.085) |
| e5-large CE | fold-3 stack | 0.97398 vs e5-base 0.97379 (+0.0002 for 3x cost) |

## Robust selection summary

| system | normal fold-3 | unseen-domain (honest TEXT analogue, mean of 2 directions) | worst country (fold-3) | cand/S1 |
|---|---:|---:|---:|---:|
| P-old: sparse top-12 + stack, text-only dense | 0.98059 | ~0.911 | India 0.9783 | 16.7 |
| P-new: compact band p>=0.02 + CE stack + CE-decided dense | **0.98314** (cross-fit) | ~0.911 (same components) | US 0.9828 | 5.9 |
| GBDT-only (no CE) | 0.9738 (stage-2 + ctx) | ~0.902 | India 0.952 | 12 |

Public-gap risk for P-new: MEDIUM-HIGH for France specifically (unseen-domain ~0.91 in simulation, ~0.855 implied by the public score), LOW for India/US.

## Transfer-improvement block (CL-038 .. CL-043)

| lever | test | in-domain | transfer | decision |
|---|---|---:|---:|---|
| QNORM: within-S1 z-scores of the 48 pair features (CL-039) | TEXT stage-2 | +0.003 (US 0.9687->0.9716, India 0.9466->0.9492) | US->India 0.8736->0.8794, India->US 0.9176->0.9341 (mean +0.011) | **promoted** |
| QNORM on FULL features | stage-2 | both-country fold-3 0.96361->0.96540 | US->India 0.9178->0.9236, India->US 0.9511->0.9619 | **promoted** |
| QNORM in the CE stack (fold 3) | stage-2 v2q + CE stack | 0.9739-0.9744 -> 0.9752-0.9757 over 8 salted stacker partitions | | **promoted** |
| relative gaps / within-S1 ranks only | TEXT | -0.05 | worse | killed |
| CE pseudo-label adaptation (positives only, 98.3% precise) | US-only CE -> India | | pair AUC 0.960->0.975, but stack transfer 0.9104->0.8941 (logit calibration shift) | killed |
| calibration-invariant CE features (within-S1 z-score) | stack | 0.9824->0.9801 | no gain | killed |
| France accent folding (CE) | label-free agreement with accent-invariant stage-2 | | France AUC 0.98366 raw vs 0.98326 folded; accented 0.98327 vs ASCII 0.98393 | killed: accents do not confuse the CE |

Deployment finding (CL-043): the stacker is sensitive to which stage-2 model produced p2. When the stacker's cross-fit
partition equals the stage-2 OOF partition, v2q scores 0.97336 instead of ~0.9755. Test p2 therefore uses the mean of
the three stage-2 fold models (the same distribution the stacker trained on), and the stacker threshold cross-fit uses a
salted partition.

France: CE-versus-stage-2 agreement on France (0.9837) is at US level (0.9861) and above India (0.9757), so the matcher
is internally consistent on France. The ~0.855 implied France score assumes India/US score publicly exactly as locally;
that assumption is untested. A per-country diagnostic upload (France rows empty, or India rows empty) is the only way to
measure where the public gap comes from.
