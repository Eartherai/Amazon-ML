# Competition execution roadmap

All times are Asia/Kolkata. Deadline from supplied guidelines: **27 September 2026, 23:59**. This is a plan, not an automation or a claim that future work has run. Research phase began shortly after the challenge opened. Leave six hours for final inference, packaging and portal recovery.

| Window | Deliverable | Exit condition | Suggested team owner |
|---|---|---|---|
| 25 Sep, initial phase | Full audit, generic research, validated metric/folds, CLI setup, exact diagnostic | Complete; see STATUS.md | Research/integration |
| 25 Sep 02:00-06:00 | Lexical/ICU union retrieval benchmark against realistic full target pool | Candidate recall, entity oracle ceiling and memory/runtime curves by country/source | Retrieval |
| 25 Sep 06:00-12:00 | Pair features and first LightGBM; error taxonomy | Leakage-safe dev score, complete candidates, no malformed outputs | Matcher |
| 25 Sep 12:00-18:00 | Hard negatives and threshold/singleton decisions; first meaningful submission if strong | Exact validator plus internal membership checks; hash/config/commit logged | Validation/integration |
| 25 Sep evening to 26 Sep morning | Cross-country tests and cross-script rescue/embedding prototype | Adds complementary recall or macro score; cost and full inference fit deadline | Neural/retrieval |
| 26 Sep daytime | Target ownership test, source calibration, focused cross-encoder if warranted | OOF gains plus independent slices, bounded candidate volume | Matcher/neural |
| 26 Sep evening | Complementary ensemble, ablations and full inference rehearsal | Best solution reproducible, runtime known, report draft ready | Integration |
| 27 Sep 00:00 onward | Check whether conditional extra credits were actually awarded | Use only for proven bottleneck; baseline budget sufficient | Account owners |
| 27 Sep 06:00-16:00 | Locked evaluation, final model choice, inference and candidate provenance | No new hypotheses after selection; package starts immediately | Whole team |
| 27 Sep 18:00 | Freeze code/model/config | Hashes, license manifest, pinned deps, final artifacts | Integration |
| 27 Sep 18:00-22:00 | Validate, reproduce key outputs, finish 1-2-page summary plus appendix | Both TSVs valid and final package complete | Validation |
| 27 Sep 22:00-23:00 | Final portal/package handling | Confirm uploaded file/score/status matches selected artifact | Team leader |
| 27 Sep 23:00-23:59 | Buffer | Avoid last-minute new model experiments | Team leader |

Times can move as measured runtime dictates. Never sacrifice final validity for a speculative experiment. Team roles are suggestions; no tasks were sent to people, no subagents launched, and no accounts allocated.

## Submission budget

Maximum five per day from the guideline PDF. Do not treat 15 as a target. On day one submit a validated competitive baseline only when useful; day two use genuinely different model hypotheses; day three reserve at least two slots for final confirmation/recovery. Portal reset semantics have not been independently verified; check the portal. All-empty and exact-only diagnostic outputs are not worth a leaderboard slot.

## Experiment acceptance

Each run gets an immutable EXP ID, config, code commit, dataset manifest, selected entity/pool manifest, compute benchmark, pair counts and score/error report. Keep the previous best model and candidate output. Compare entity-level paired results, not one cherry-picked aggregate. Raise the next priority from actual missed positives and false merges, not from model popularity.
