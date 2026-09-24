# ML Challenge 2026: Business Entity Resolution Solution

**Team name / members:** To be filled by the team.  
**Status:** Phase-one research draft, 25 September 2026. This is not a completed final-solution claim.

## 1. Executive summary

We audited all 24,229,173 provided source records and 7,638,365 labeled links, implemented the exact entity-level macro F0.5 metric, and froze entity-disjoint validation manifests. The proposed architecture combines several retrieval routes with a compact pair classifier, followed by calibrated entity decisions. Learned training and final test inference are pending.

## 2. Methodology

Training contains 2,206,821 S1 entities; only 5.58% have no matches, and most have three or more. Test contains 1,732,544 S1 entities, including 259,452 from unseen France. Training targets include substantial Indic-script variants despite ASCII reference names. All labeled targets have a single S1 owner; S1 may own many targets.

We preserve raw fields and compare multiple Unicode-safe views. Five SHA256-based S1 folds use seed 20260925: development fold 0, initial training folds 1-3 and locked fold 4. Positive target ownership is tracked so held-out targets cannot enter training negatives. Country-transfer and stricter collision-group stress evaluations are planned. Fitted preprocessing is train-only by default.

## 3. Candidate generation

The exact diagnostic considers country+normalized-name+normalized-address equality, excluding blank keys. On development fold 0 it produced 20,075 candidates against the complete 10,320,219-record training target pool. Link recall was only 1.3143%, establishing that stronger retrieval is essential.

Next planned routes: character n-grams, word/rare-token retrieval, address/numeric blocks and generic transliteration; embedding rescue only if measured incremental recall justifies cost. The final submitted candidate TSV will represent the actual scored set, with final matches a subset.

## 4. Matching model

No learned matcher has been trained yet. The first planned model is LightGBM using name/address similarities, numeric agreement/conflicts, missingness, token rarity and retrieval ranks. Hard-negative mining and calibration will use fold-safe predictions. Thresholds will maximize exact macro F0.5, not pairwise F1. Model/checkpoint licenses and parameters will be recorded before use.

## 5. Results and error analysis

The no-training exact diagnostic scores **0.08327203 macro F0.5**, with micro precision **1.0000** and recall **0.01314345** on 441,103 development S1 entities. Singleton F0.5 is 1.0; non-singleton F0.5 is 0.02894529. These are baseline diagnostics, not competitive final results. No leaderboard score is available.

Errors predominantly come from unretrieved variants: abbreviations, reordered/incomplete addresses, missing addresses, different scripts and severe name aliases. Same-name different-location negatives demonstrate why name similarity alone is unsafe. A 10,000-pair non-ASCII-positive probe shows generic transliteration improves some name similarities; retrieval recall and false-positive effects remain to be tested.

## 6. Next work and reproducibility

Prioritize high-recall union retrieval, then a compact classifier and calibration, before GPU-heavy methods. Current task cloud spend is zero. Code, pinned dependencies, data hashes, metric tests, split manifests and experiment provenance are recorded. See RUNBOOK.md, DATA_AUDIT.md, docs/RESEARCH.md and EXPERIMENTS.csv for the technical appendix. Final outputs, end-to-end learned inference, ablations, selected model, license manifest and final runtime will replace this draft as measured results arrive.
