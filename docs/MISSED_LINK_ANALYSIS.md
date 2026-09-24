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

## Phase4 A: natural-prevalence candidate misses

Run outputs/oof/P4-A-001: 581 missed true links. All preserved in missed_links/all_missed_links.parquet.

| Observable pattern | Links |
|---|---:|
| weak_name | 272 |
| weak_address | 473 |
| both_weak | 228 |
| numeric_disagreement | 54 |
| nonascii_target_name | 257 |
| missing_address | 137 |
| acronym_equal | 108 |
| exact_name | 5 |
| exact_address | 0 |

The same pattern definitions and unresolved-cause cautions above apply. Next retrieval hypothesis: transliteration-aware names and address rescue, evaluated by marginal coverage on this development population; candidate-cap cause still requires deeper rankings.

## Exact-name cap experiment

All five missed exact-name links in the5k set had168–241 exact-name target records, above the name route's100 cap. A generic observable trigger (more than100 exact-name targets) activates address-WRatio reranking within the exact-name group, retaining20. It fired for126/5,000 queries, scored50,935 exact-name pairs, added830 unique candidates and recovered5 links/5 complete entities in2.36s. Average candidates increased197.189→197.355; recall96.6503%→96.6792%. This is development retrieval evidence, not an independently measured matcher improvement. No false-positive acceptance rule was added.

Of581 current misses,116 have non-ASCII target address text. Examples include untranslated regional suffixes plus heavy deletion/reordering; transliteration alone cannot be assumed to repair them. Preserve native and transformed views and benchmark marginal recovery before adding a universal route.
