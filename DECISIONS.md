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
