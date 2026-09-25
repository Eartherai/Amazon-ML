# Parallel competition war room — 2026-09-25

The user authorized ten concurrent research tracks with a $100 total AWS sprint ceiling and three portal uploads remaining. Agent runtime supports four concurrent agents, so the ten tracks are grouped across four isolated worktrees. The previous branch-local classical-only restriction remains on the SUB-002 production branch; the user's newer request authorizes a separate neural research track. Fold4 remains closed. No experiment changes the frozen SUB-002 test inference. Current public score reported by the user: 0.913 for SUB-001; exact uploaded bytes are not observable.

| Track | Owner | Method / fixed evaluation | Exact local macro F0.5 | Delta | Incremental AWS cap | Status |
|---|---|---|---:|---:|---:|---|
| 1 Retrieval | retrieval_shift | Conditional India rescue; 1k India S1 full-target pilot; report link recall/oracle before matcher score | pending | pending | $6 track | Running; no paid job |
| 2 Precision / F0.5 | root | EXP-038 29 set-decision rules on 200k strict OOF | 0.9366603031 best vs 0.9357644568 base | +0.0008958463 exploratory | $0.02392 estimated compute | Completed; shelved as low gain |
| 3 Hard negatives | hardneg_ensemble | EXP-039 hard-neg LightGBM + 50% base blend, 15k fixed OOF | 0.938830137 vs 0.937158136 base | +0.00167204, CI [+0.000718,+0.002626] | ~$0.082 compute incl failed first run | Completed; positive but not step-change |
| 4 Deep learning | root | EXP-044 frozen mMARCO 0.1B fold1 fit, 6k disjoint current 51-feature OOF; sidecar pairs only | 0.94311751 vs 0.93255913 fixed-threshold crossfit | +0.01055838, held directions +0.011255/+0.009495, both CI >0 | $0 AWS inference; capped G5 planned | Promoted for separate test GPU rerank; France transfer unmeasured |
| 5 Cross-encoder | root | EXP-044 LightGBM+neural logistic blend, production sidecar subset | India 0.92586660 vs 0.90919544; US 0.95461813 vs 0.94813493 | India +0.01667116; US +0.00648320 | $0 AWS calibration | Frozen 0.6875 threshold; standalone neural rejected |
| 6 Normalization | retrieval_shift | Conditional transliteration/numeric address rescue | retrieval-only | pending | included track 1 | Running |
| 7 France/domain shift | retrieval_shift | Unlabeled France distribution and robust normalization checks | no France labels | pending | included track 1 | Running |
| 8 Graph/global | root | EXP-034 max-score target ownership on 200k strict OOF | 0.9362345433 vs 0.9357644568 base | +0.0004700865 | $0.02174 used | Completed; combined with source completion +0.0021778540 EXP-035 |
| 9 Ensemble | hardneg_ensemble | Hard-negative/base score blend on fixed 15k | 0.938830137 | +0.00167204 | included track 3 | Completed, low priority for full inference |
| 10 Out-of-box | neural_rerank | Set/cardinality or other independent high-upside method after pilot | pending | pending | included track 4 | Prepared |

Baseline caveat: SUB-001 20k OOF 0.9318965293 and EXP-033 100k fixed 15k 0.9371581365 are different evaluation populations. Do not subtract them as an experimental delta. EXP-034/035 200k strict OOF baseline 0.9357644568 is likewise a separate protocol. Leaderboard 0.913 is a third, hidden test split; never tune directly to it.

Promotion requires exact entity macro F0.5, country/singleton slices and output provenance. France has no labels. A retrieval-only result must report candidate recall/oracle, never an invented matcher F0.5. Paid jobs require fresh cost guard and profile `amamzon_01_a1_0`. Official organizer default and `--check-ids` PASS with zero strict warnings gate every portal TSV; user uploads manually. No ZIP for the live portal.
