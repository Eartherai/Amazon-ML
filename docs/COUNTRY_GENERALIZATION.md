# Country generalization — Phase5

Current evidence:20k development uses natural India/US mixture. NUMERIC-V2 cross-country label transfer improves India→US from0.88626059 to0.89280089 and US→India from0.76732686 to0.78192088 on5k stress data. These are not France estimates. IDF includes disjoint fold0 text from both countries; describe this as label transfer.

The current country-equality feature is constant because retrieval is same-country, so removing it alone cannot remove domain dependence. Text similarity distributions, missingness, address formats, script and retrieval quality are plausible mechanisms. India retrieval recall is lower. Do not attribute classifier collapse solely to country ID.

Ranked next experiments on identical larger OOF candidates: (1) preserve numeric-v2 control; (2) fit country-balanced query weights using training entities only; (3) country/source-balanced hard negatives mined only from training folds; (4) remove country feature as sanity ablation; (5) compare matched training sample sizes in each country-transfer direction; (6) separate candidate-oracle gap from matcher gap; (7) evaluate generic transliteration rescue; (8) investigate pooled versus source-specific threshold stability. Every variant needs mixed OOF and country slices, no Fold4 selection.

France stress: compare unlabeled candidate counts, similarity/confidence quantiles, missingness, numeric and Unicode flags after frozen retrieval/inference exist. No external identities, surrogate labels or country-specific threshold tuning from invented labels. Arbitrary country strings, Unicode query+target symmetry, and library portability are release requirements.

## Measured20k country-balanced trial

P5-COUNTRY-BALANCE-001 / EXP-026 uses identical NUMERIC-V2 features/candidates, nested thresholds and folds. Training-only weights equalize total query mass per country; validation retains its natural prevalence. No held-out query contributes to fit weights.

| Slice | Current numeric model | Country-balanced | Paired delta95%CI |
|---|---:|---:|---|
| India | 0.908355395 | 0.910317038 | [0.000326714913508682, 0.003556256554250796] |
| US | 0.947973595 | 0.946323459 | [-0.002614161896730769, -0.000585285198377922] |

Overall macroF0.5=0.931712054, delta=-0.000184475, paired95%CI=[-0.0011007763667911373, 0.0007559698319234793]. Not promoted: India improves but US declines; pooled gain is unsupported and singleton score drops. Retain as a controlled hypothesis for larger OOF/domain routing, not a France improvement claim. Country-equality takes only1.0 across all20k candidates, confirming that its removal alone cannot explain domain differences. Code and exact source snapshot/provenance are stored with the run.
