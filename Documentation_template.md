# ML Challenge 2026: Business Entity Resolution Solution

**Team name / members:** pending team details.  
**Status:** Phase5 development methodology,25September2026. Final architecture is not frozen; Fold4 is closed. No leaderboard submission.

## 1. Executive summary

The current best uses two character-trigram retrieval routes and a51-feature LightGBM matcher with canonical numeric comparisons. On20,000 naturally sampled S1 entities, nested three-fold entity OOF gives macroF0.5 **0.93189653**, pair precision **0.98158282**, pair recall **0.85593807**, singleton score **0.92131747**. These are development results, not full-training, Fold4, or leaderboard scores. Numeric features improve over the45-feature model by0.00313760; paired entity bootstrap95%CI[0.00194174,0.00431221]. On the15k entities outside the original5k pilot, improvement remains0.00329924,CI[0.00183971,0.00470053].

## 2. Data and preprocessing

All24,229,173 provided source records and7,638,365 labeled training links were audited. Training has2,206,821 S1 and10,320,219 target records. Test has1,732,544 S1, including259,452France records. Training S1 singleton prevalence is5.585%. S1 may have many targets; no target has multiple labeled S1 owners.

Raw fields are preserved. Materialized Unicode views include light NFC casefold, compatibility/accent variants, token sorting and numeric representations. The measured lexical pipeline uses the frozen legacy NFC/lowercase/punctuation-to-space representation from the audit DB; it is not silently replaced by newer casefold views. Generic transliteration supplies six additional baseline features together with first-number equality/conflict. Training target transliteration is a frozen Apple Foundation Any-Latin/Latin-ASCII map. Linux ICU equivalence and symmetric non-ASCII test-query handling must be verified before final inference.

NUMERIC-V2 adds six features: canonical digit overlap/Jaccard/conflict, leading-zero rescue, first-number canonical equality/conflict. Unicode decimal digits convert to ASCII and leading zeros normalize for comparisons while originals remain available. Numeric agreement neither automatically accepts nor rejects a pair.

## 3. Validation

Five deterministic SHA256 S1 folds preserve positive-target ownership. Fold4 stays locked until architecture, calibration and decision procedure freeze. Current20k OOF uses natural samples from folds1–3. Each outer model trains on the other two; inner entity folds choose thresholds without outer labels. Negatives owned by held-out entities are excluded from fitting. Full target pools remain available as retrieval distractors.

Frozen IDF fits only sampled unowned/fold0 targets, separate from OOF folds1–3 and lockedFold4. This external training-text basis is explicitly documented; country-transfer experiments test unseen country labels, not strictly unseen country text. MacroF0.5 is computed per S1 as1.25TP/(predicted_count+0.25truth_count), with empty/empty=1 and false-positive singleton=0. Confidence intervals resample entities, not pairs.

## 4. Candidate generation

Name and address char3 TF-IDF independently retrieve up to100 same-country targets, then union. Countries are arbitrary strings. Full targets from S2 and S3 are searched. Frozen IDF uses float32 L2 TF-IDF with at most200k features. Deterministic ID tie-breaking includes boundary ties. Current20k run yields3,943,627 candidates, average197.18135/S1; p95/p99=200. Link recall0.96662630, complete-entity recall0.90305178, oracle macroF0.5=0.98853146. India link recall0.94192378 versusUS0.98344844 indicates retrieval headroom.

Phase5 builds persistent country/source sparse shards and benchmarks reuse on EC2. Full-training metrics are pending, and Fold4 label metrics remain withheld until freeze. A targeted exact-name fanout rescue recovered5 links for830 extra candidates on5k; it is not yet part of the selected matcher. Dense retrieval is not implemented or selected.

## 5. Matcher and decision procedure

LightGBM4.7.0,250trees,31leaves,minimum child30,learning rate0.06,L2=2,seed20260925. Features compare name/address strings with exact, edit/Jaro-Winkler/fuzzy/token similarities, lengths and token boundaries; add missingness, numbers, source and retrieval ranks/scores/route agreement, transliteration and NUMERIC-V2. Inner OOF selects a threshold against entity macroF0.5. No learned singleton meta-model or global assignment is selected.

Measured ablations show both name and address, raw numeric signals, transliteration, token similarities and retrieval features matter. Small5k CatBoost/XGBoost and ownership-reweight probes did not beat the baseline reliably. Blind hard-negative upweighting previously hurt; no such weighting is promoted. Platt/isotonic and separate empty-threshold trials showed no compelling consistent gain. No dense encoder, cross-encoder, or ensemble is currently selected.

## 6. Robustness and limitations

Numeric-v2 country label-transfer: India→US0.89280089,US→India0.78192088. These are stress tests, not France forecasts; country robustness remains unresolved. False merges include common names, shared addresses, transliteration collisions, missing fields and numeric conflicts. Lexical misses concentrate in weak or absent addresses and script/alias variation. Large/full OOF, country-balanced training, targeted rescue and full unlabeled France confidence analysis remain pending.

## 7. Compute and reproducibility

Local development: M5MacBookAir24GB,Python3.12.13; pinned dependencies and git/config/data manifests. Phase5 launches a finite8vCPU/64GiBR8i.2xlarge index benchmark. Inputs/code use SHA256 checks, encrypted storage, scoped instance role, no inbound ports and90min automatic shutdown with terminate behavior. CPU/GPU quota increases are pending; SageMaker/GPU jobs have not run. Exact costs are tracked separately from estimates in AWS_SPEND.md.

See RUNBOOK.md,EXPERIMENTS.csv,configs/phase4/CURRENT-BEST.json and docs/NUMERIC_CONFIRMATION.md for current reproduction/provenance. End-to-end full-data training/test commands are not yet complete. Final candidate TSV must contain the actual scored candidates and all matches must be members. Both strict ID checks and the unchanged official validator must pass after final file placement. Final hashes, package, team details and leaderboard scores remain pending.
