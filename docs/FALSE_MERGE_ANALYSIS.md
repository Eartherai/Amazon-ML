# False-merge analysis

Run: outputs/oof/P4-A-001. All 282 false-positive pairs are preserved; 162 have raw model score >=0.90. Scores are not calibrated probabilities. Predictions use each outer fold’s inner-selected threshold.

| Observable pattern | Pairs | Median score | India | US |
|---|---:|---:|---:|---:|
| different_known_owner | 113 | 0.9115650579499985 | 78 | 35 |
| name_token_containment | 91 | 0.9077805017924583 | 60 | 31 |
| near_identical_name | 51 | 0.9155246740619093 | 25 | 26 |
| missing_address | 26 | 0.8561695626413722 | 12 | 14 |
| exact_name_different_address | 23 | 0.9353984037170485 | 15 | 8 |
| exact_address_different_name | 17 | 0.91887166607439 | 8 | 9 |
| numeric_disagreement | 17 | 0.9353984037170485 | 5 | 12 |
| transliteration_collision | 4 | 0.9375297778230516 | 1 | 3 |

Patterns overlap. Different-known-owner means a target has a different labeled owner; it does not justify imposing a global ownership constraint without validation. Chain/franchise, generic business, and legal-suffix identity causes cannot be established reliably by these flags and remain unclassified.

Next controlled hypotheses: nested emptiness decision, numeric/name-address feature ablations, targeted negatives by observed pattern. No blanket rejection rules have been introduced.
