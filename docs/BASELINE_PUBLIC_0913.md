# Public baseline and emergency gap audit

25 September 2026, 14:44 IST. The user reports public leaderboard F0.5 **0.913** for the first scored submission. The portal does not expose the uploaded bytes, so the link to the test-order file below is the best available provenance, not independent proof of its SHA256. Three portal uploads remain today as reported by the user. Fold 4 remains CLOSED.

## Frozen artifact

`outputs/submissions/SUB-001/test-order-v001/matching_results.tsv` has SHA256 `910c3e7c95a8ed7867dd6addfb6a9aad3d5477c5ede2422c23b081be1942440b`, 89,122,850 bytes and exactly 1,732,544 data rows. It contains the same S1-to-target predictions as the original validated file, in test Source 1 row order. The associated final-scored candidate file has SHA256 `89e95ad2f4cb0b0c4f082f9975c6549e8f073ceb71cbcb3ac1860247a590f4b6`. The organizer's exact two-file validator passed normally and with `--check-ids`, with zero strict warnings. The 51-feature LightGBM used 20,000 labeled S1, top-100 character 3-gram retrieval for name and address, and threshold 0.83.

| Slice | S1 entities | Predicted links | Empty | One | Two or more | Mean links |
|---|---:|---:|---:|---:|---:|---:|
| All | 1,732,544 | 5,172,588 | 122,166 | 197,862 | 1,412,516 | 2.9855 |
| US | 663,106 | 2,027,591 | 43,518 | 65,474 | 554,114 | 3.0577 |
| India | 809,986 | 2,330,172 | 62,187 | 102,641 | 645,158 | 2.8768 |
| France | 259,452 | 814,825 | 16,461 | 29,747 | 213,244 | 3.1406 |

Predicted links are 2,510,707 Source 2 and 2,661,881 Source 3. Predicted empty share is 7.05%; training truth is 5.585% empty and 3.461 links per S1. Those training marginals do not establish test truth. France has no labeled examples.

## Measured validation ceilings and errors

On 20,000 OOF training S1, exact entity macro F0.5 is 0.9318965293, micro precision 0.9815828194, micro recall 0.8559380676, singleton accuracy/F0.5 0.9213174748. Complete exact set accuracy is 0.61285. Candidate link recall is 0.96662630 and retrieval oracle macro F0.5 is 0.98853146. The gap from the measured candidate oracle to the matcher is 0.056635, so both retrieval and scoring matter. Of 9,993 missed true links, 2,315 were absent from candidates and 7,678 were scored but rejected. The model also made 1,114 false links. This is an exact OOF decomposition, not a test-label diagnosis.

| OOF slice | Entity macro F0.5 | Other evidence |
|---|---:|---|
| India | 0.90835539 | Link recall 0.82353; 4,959 FN, 686 FP; candidate link recall 0.94192 |
| US | 0.94797359 | Link recall 0.87801; 5,034 FN, 428 FP; candidate link recall 0.98345 |
| True singleton | 0.92131747 | 86/1,093 incorrectly given a link |
| True one match | 0.83690626 | 131/1,037 incorrectly predicted empty |
| True two matches | 0.91850317 | 951 FN |
| True three or more | 0.94278779 | 8,898 FN |
| Source 2 as an independent projected task | 0.90046612 | 4,929 FN |
| Source 3 as an independent projected task | 0.90576249 | 5,064 FN |

Among the 7,678 scored positive misses, 1,860 have missing target address, versus 684 of 59,373 true positives. Those positives are hard: mean name Jaro-Winkler 0.812 and address Jaro-Winkler 0.607, versus 0.897/0.845 for true positives. A naive missing-address threshold reduction *lost* macro F0.5 on exploratory OOF. A separate missing-address LightGBM specialist at threshold 0.83 added 107 TP and 43 FP, only +0.000124 exploratory OOF. Neither result warrants a public upload. Target-to-target anchor rescue's best 2k pilot was +0.000436 (16 TP, 2 FP), likewise low priority and unconfirmed.

The public-minus-local gap is -0.0188965. Test country mix differs from train: 15.0% France, 46.8% India, 38.3% US versus train's approximately 40.0% India and 60.0% US. If public sampling mirrors the full test mix and known-country OOF scores transfer, France F0.5 around 0.838 would account algebraically for the 0.913 total. Both assumptions are unverified. This calculation is a hypothesis, not a measured France score.

## Ranked next work

1. **200k final fit and new test inference**. EXP-033 on a fixed, new 15k OOF set rose 0.93492235 at 20k, 0.93630527 at 50k, 0.93715814 at 100k. Fit 200k on the verified feature store, retain old threshold and retrieval, and preserve test score sidecars to enable later decision policies without another retrieval pass. The 200k final fit has no independent OOF estimate. AWS training started at 09:26 UTC on `i-045185b5404fe3b7c`, two-hour/$2.50 cap.
2. **India retrieval rescue**. Full-retrieval worker `i-0609d158c38e96160` continues, 154 of 256 route archives as of 09:19 UTC. Confirm full-scale India recall and test conditional routes for weak/missing address, non-ASCII and transliteration. Do not run universal top-200, whose earlier pilot added only 11 links per 99,873 extra candidates.
3. **France robustness**. Compare unlabeled covariates, scores and prediction cardinality with known countries. Test country-held-out transfer and a country-agnostic scoring policy on known-country OOF. No France labels, external identity lookup or public threshold probing.
4. **Entity-level decision model**. Once per-pair OOF and test score sidecars exist, train a nested, owner-safe singleton/cardinality policy and evaluate exact macro F0.5. Require a meaningful gain and stable country slices before promotion.
5. **Full-data classical training**. If full retrieval finishes before useful deadline, build 250k/500k/1M feature/training scale using only unlocked folds and time-bounded AWS. Do not duplicate retrieval or assume this will bridge the leader gap.

The leaders near 0.98 are an external benchmark reported by the user. No available evidence explains their architecture or establishes a path to 0.98. The public score is a single partial-test measurement; private leaderboard robustness remains the promotion criterion.
