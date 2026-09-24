# Country label-transfer stress test

The5k mixed-country nested OOF score0.923407 does not prove unseen-country robustness.

| Training labels | Evaluation labels | Entities | Threshold from training-country OOF | Macro F0.5 | Precision | Recall | Singleton |
|---|---|---:|---:|---:|---:|---:|---:|
| India | US |2,913|0.72|0.886261|0.943228|0.813249|0.760563|
| US | India |2,087|0.81|0.767327|0.876677|0.701855|0.516129|

Thresholds use three entity folds within the training country only. No evaluation-country labels select thresholds or fit models. All target candidates remain present when evaluating. Model fit excludes targets owned by folds0/4 or an inner-held fold. The fixed external vocabulary/IDF uses fold0/unowned training text from both known countries: this is a **country label-transfer** test, not a strictly held-country-text experiment. It must not be described as a France forecast.

The US-to-India decline is substantial. The model learns similarity/format distributions that transfer poorly even though country itself is only a constant equality feature within same-country candidates. Smaller training population and domain shift both contribute; their separate effects have not been isolated. False merges and singleton failures suggest that a threshold learned in one domain is not reliably portable.

Required follow-ups: sample-size-matched within-country controls; domain-shift feature/error comparisons; generic preprocessing augmentations evaluated within nested folds; multilingual features tested on the same stress regime; country-blind and worst-domain model selection. Preserve native text and avoid country-specific acceptance rules. Do not open fold4 to resolve this question.

Artifacts: outputs/oof/P4-A-001/country_transfer. No test labels or external business identity data used.
