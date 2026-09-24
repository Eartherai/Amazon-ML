# Controlled Phase4 model probes

All probes use the same5k natural-prevalence entities,45 baseline features, full-pool candidates, three entity outer folds and two inner folds for each threshold. No outer validation early stopping. These are one-configuration screening results, not a completed hyperparameter tournament.

| Model/decision | Macro F0.5 | Delta versus LightGBM | Paired95% CI |
|---|---:|---:|---|
| LightGBM baseline |0.923407|0|reference|
| CatBoost1.2.8,400 trees/depth6 |0.921815|−0.001591|[−0.004622,+0.001359]|
| XGBoost3.0.2,400 trees/depth5 |0.922795|−0.000611|[−0.003078,+0.001716]|
| LightGBM ownership-negative weighting |0.921873|−0.001533|[−0.003922,+0.000774]|
| LightGBM query-relative context |0.922214|−0.001193|[−0.003819,+0.001333]|

Retain the simpler existing LightGBM configuration. These intervals do not establish equivalence, but there is no clear improvement to promote. The ownership-weighting probe upweights eligible owned negative pairs by5/number_of_fit_folds to approximate missing target-owner strata; it is a structural sampling correction hypothesis, not blind hard-negative3x weighting. It did not improve the metric.

XGBoost's sklearn-wrapper save failed because its estimator-tag lookup was incompatible with installed sklearn. P4-XGB-001 is preserved as failed. P4-XGB-002 uses the documented native Booster serialization API; a round-trip test verifies exact probability preservation.

Primary references: [CatBoost classifier API](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier), [CatBoost1.2.8 license](https://github.com/catboost/catboost/blob/v1.2.8/LICENSE), [XGBoost sklearn API](https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html), [XGBoost3.0.2 license](https://github.com/dmlc/xgboost/blob/v3.0.2/LICENSE). Both exact library tags use Apache2.0. No pretrained weights or external business records used.
