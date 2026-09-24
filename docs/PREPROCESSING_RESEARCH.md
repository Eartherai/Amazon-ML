# Preprocessing research for this dataset

Research date: 2026-09-25. Generic methodology only; no competition business was looked up. Dataset-specific conclusions below are hypotheses tested in PREP-001, MORPH-001 and PROFILE-001, not claims from external sources.

## Evidence and design implications

**Unicode is data.** NFC makes canonically equivalent encodings comparable. NFKC additionally collapses compatibility distinctions and is therefore a separate view. Neither accent deletion nor ASCII conversion replaces original text. Our raw and light views preserve Indic combining marks; Latin accent folding is independently measured. [Unicode normalization specification](https://www.unicode.org/reports/tr15/).

**Transliteration is a retrieval aid.** ICU script conversion does not translate words, and ambiguous mappings need extra care. Store original-script and transliterated forms together, compare recovery and false collisions, and let numeric/address evidence disambiguate. A full-corpus transliteration cache should be justified by candidate recall on a representative probe. [ICU transforms and cautions](https://unicode-org.github.io/icu/userguide/transforms/general/).

**Containment scores can hide differences.** RapidFuzz token-set similarity can reach its maximum when one token set is contained in another. Keep token-set, order-sensitive, numeric-conflict and length features side by side. Do not use a high fuzzy score as an identity rule. [RapidFuzz scoring definitions and examples](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html).

**Rare agreement matters.** Agreement on a common word is weaker evidence than agreement on an unusual one. Preserve rare name and address tokens and weight their agreement by corpus frequency. Avoid autocorrecting rare tokens into common tokens without held-out evidence. [Splink term-frequency adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html).

**Character retrieval supports spelling variation.** Character ngrams provide overlap even when whole-word token matching fails; word TF-IDF contributes complementary information. Fit experimental IDF on permitted training text and transform held-out queries without refitting. Use chunked sparse products or inverted postings rather than dense all-pairs matrices. [scikit-learn text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction).

## Ranked preprocessing and retrieval experiments

| Priority | Experiment | Why / expected signal | Main risk | Local cost / decision |
|---|---|---|---|---|
| 1 | Raw + NFC/casefold/space-separated punctuation | Handles presentation variation while preserving letters, marks and numbers | Punctuation sometimes distinguishes entities | Implemented and tested; default light view |
| 2 | Exact name/address plus rare-token union | Fast full-pool coverage diagnosis, identifies what fuzzy retrieval must rescue | Common keys create huge candidate buckets | Full-target pilot; CPU, no AWS |
| 3 | Char 3–5gram name retrieval | Spelling noise and fragmented tokens | Very common grams create memory/runtime cost | Benchmark chunked sparse pilot before full run |
| 4 | Independent address retrieval | Can rescue badly corrupted names | Missing addresses, building-number conflicts | Keep numeric tokens; measure marginal recall |
| 5 | Optional Latin accent fold | Test S1 France has accents; US/India labels support some stress testing | Different names collapse | Auxiliary view only; France quality remains unmeasured |
| 6 | Generic ICU transliteration | Indian target scripts differ from mostly Latin S1 | Imperfect sound mapping and new collisions | Existing positive-only probe encouraging; expand negative audit |
| 7 | Token order and token-set views | Reordered legal names / addresses | Extra distinguishing tokens disappear under sets | Features or candidate routes, never sole merge decision |
| 8 | One-pass explicit abbreviation maps | Observed training replacements can align suffix/street forms | Cycles, polysemy, same abbreviation in different regions | Disabled by default; require dev collision and recall evidence |
| 9 | Number/punctuation alternate views | Recover number formatting changes | 1.5/15, hyphen/slash and house-number conflation | Optional feature only; strict original number features retained |
| 10 | Library-only phonetics / multilingual embeddings | Potential residual script and sound variation | Extra compute, uncertain improvement, license constraints | Only after retrieval-error evidence; no GPU now |

Library licenses are separate from final model-weight licenses. Existing pinned DuckDB/MIT, Polars/MIT, RapidFuzz/MIT and scikit-learn/BSD code dependencies are algorithm tools; any future model must separately satisfy the official model restriction (MIT/Apache-compatible, at most 8B). No model weights are introduced in this phase. Package license metadata should be included in the eventual reproducibility manifest; do not infer a model's license from its library.

## Acceptance criteria

A preprocessing change must be reversible through stored raw data; reproducible through a config hash; tested on Unicode, empty values, numerics and unseen countries; evaluated on held-out S1 entities; and compared on both positive recovery and hard-negative collisions. Descriptive distribution analysis may cover every unlabeled source. Learned maps use folds 1–3; development diagnostics use fold 0; fold 4 stays closed for model selection.

Pairwise AUC/AP/KS and thresholded feature rates explain feature behavior. They are not the competition macro F0.5 and do not select the final match threshold. Equal country/singleton sampling changes prevalence, so raw sampled precision or AP must never be presented as deployment precision.
