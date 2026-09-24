# Phase4 feature ablations

5,000 natural-prevalence entities, same full-pool candidates, same three outer folds. Each ablation independently selects thresholds from its two inner folds. These are development comparisons of whole decision procedures.

| Removed group | Macro F0.5 | Delta | Paired95% CI |
|---|---:|---:|---|
| without_translit | 0.917974 | -0.005432 | [-0.00855701007410353, -0.0023781545123159192] |
| without_numeric | 0.905223 | -0.018183 | [-0.02249967888097747, -0.013824912088834022] |
| without_address | 0.828230 | -0.095176 | [-0.10210030203831313, -0.08860482202102307] |
| without_name | 0.836909 | -0.086498 | [-0.09271736400817748, -0.08026862037869104] |
| without_retrieval | 0.917556 | -0.005850 | [-0.009019769966779955, -0.0026211071409086876] |
| without_route_count | 0.922332 | -0.001075 | [-0.003143430467269873, 0.00090855219567418] |
| without_token_similarity | 0.907288 | -0.016118 | [-0.01981787759784005, -0.012866304211261483] |
| without_char_similarity | 0.907398 | -0.016009 | [-0.019411302142092034, -0.012606600260976365] |

No raw-text feature group or separate aggressive/accent/deduplicated preprocessing view exists in the frozen45-feature model. Their absence is not an ablation result. Token-sort/set comparisons are implemented and removed in the token-similarity ablation. Char-similarity removal affects pair similarities and retrieval-score features, while candidate generation remains identical.

Route-count removal has weak evidence if its interval crosses zero; retaining its negligible implementation cost is reasonable pending larger validation. Preserve beneficial groups. Do not interpret each conditional paired bootstrap as a multiple-comparison-corrected significance test.
