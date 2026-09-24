# Missed-link analysis

Frozen Phase3 broad union misses 82 of 3,449 labeled links. This is the repeatedly inspected 1,000-query pilot, not fresh Phase4 OOF. Pattern flags overlap; they are not proven causes.

| Observable pattern | Missed links |
|---|---:|
| weak_name | 46 |
| weak_address | 76 |
| both_weak | 43 |
| numeric_disagreement | 8 |
| nonascii_target_name | 41 |
| missing_address | 12 |
| acronym_equal | 7 |
| accent_fold_equal | 0 |
| exact_name | 0 |
| exact_address | 0 |

Weak means Jaro-Winkler <0.70, a descriptive bin, not a decision rule. Numeric disagreement means both addresses contain numbers but no shared numeric token. Non-ASCII is an observable script flag, not proof of transliteration failure.

Cap truncation versus retrieval-score failure remains unresolved: top100 artifacts omit deeper rankings. No chain/franchise identity or business-name-change claim can be established from these features alone. Review raw competition text locally and test wider retrieval only on a controlled subset.

The name/address char3-first incremental table is in outputs/analysis/P4-PILOT-001/route_utility.csv. Ordering affects marginal attribution; compare route removals as well. Large-scale Phase4 misses will be appended separately.
