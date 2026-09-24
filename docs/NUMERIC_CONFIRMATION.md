# Numeric preprocessing confirmation

Development confirmation; new15k excludes previously inspected5k entities, but same country distribution and shared training pool. Not fold4 or France evidence.

| Population | Baseline macro | Numeric macro | Paired delta | 95% CI |
|---|---:|---:|---:|---|
| all20k | 0.928759 | 0.931897 | +0.003138 | [0.0019978133176212754, 0.004340151591059112] |
| new15k | 0.928786 | 0.932085 | +0.003299 | [0.0018397124901188102, 0.004700533768017748] |
| previous5k | 0.928679 | 0.931331 | +0.002653 | [0.0002660012396429485, 0.005081765667111886] |

Canonical digits are alternative comparison features. Raw values and original comparisons remain; numeric equality never automatically accepts a match. Thresholds are reselected inside each outer training partition. Country-transfer gains were modest and unseen-France robustness remains unresolved.
