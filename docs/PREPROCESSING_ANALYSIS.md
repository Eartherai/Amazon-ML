# Preprocessing evidence and decisions

2026-09-25. This phase profiles all six sources (24,229,173 records), implements reusable normalization, analyzes sampled match morphology, and starts full-target candidate retrieval. No learned matcher, embedding model, paid compute or leaderboard submission was used.

Open [the visual report](PREPROCESSING_VISUAL_REPORT.html) to browse and filter the figures. The full corpus profile is [DATA_PROFILE.md](DATA_PROFILE.md); implementation details are [PREPROCESSING_IMPLEMENTATION.md](PREPROCESSING_IMPLEMENTATION.md); generic primary-source research is [PREPROCESSING_RESEARCH.md](PREPROCESSING_RESEARCH.md).

## What the data tells us

1. **Preserve original text and Unicode marks.** Train S1 names are ASCII, whereas 27.87% of Indian train S2 names contain non-ASCII text and 23.50% contain combining marks. Deleting all marks would damage written names. France test S1 has non-ASCII text in 15.72% of names and 28.27% of addresses. Keep NFC/casefold as a primary view; Latin accent folding and ICU transliteration are separate views.
2. **A name is not an entity key.** Within train S1, raw name duplicate excess is 319,570/883,188 India records and 347,366/1,323,633 US records. Light normalization collapses another 3,953 India and 14,508 US distinct name values. Name equality needs corroboration.
3. **Do not correct the vocabulary indiscriminately.** Distinct name tokens grow from 50,639/70,685 India/US in S1 to 293,570/549,005 in S2; single-occurrence vocabulary grows from 21.0%/28.7% to 75.2%/80.3%. Corpus sizes differ, so this is not a measured corruption rate. Character retrieval can tolerate unusual tokens without rewriting them.
4. **Abbreviation maps contain traps.** Training-only mining found 138 name and 264 address replacement evidence rows, with 54 and 150 ambiguous source phrases. Examples include directional cycles and geographic ambiguity. Proposals remain disabled; they are not verified universal equivalences.
5. **Preserve numeric disagreement.** Similar addresses can differ in a house/building number. In the development positive sample, 4.02% of Indian and 9.82% of US positive pairs have numbers on both sides with no shared numeric token. Thus numeric disagreement is useful evidence, but not a mandatory veto. Postal-shaped agreement is only 0.28% India versus 8.18% US in this sample; postal blocking must be optional.
6. **Singletons often have plausible candidates.** The retrieval pilot gives candidates to 47/50 true singletons. Candidate existence, a high name score, or token containment cannot decide whether to merge.

## Scope and sampling

| Artifact | Scope | Intended use |
|---|---|---|
| PROFILE-001 | All 24.23M rows, both name/address, country/source; exact histograms, Unicode counts, token DF and duplicate buckets | Full corpus descriptive statistics |
| PROFILE-001 ngrams | Deterministic up-to-10k rows per source/country, explicitly marked | Character ngram vocabulary diagnostics |
| PREP-001 | 25,000 positive pairs from folds 1–3; 6,250 per country/target-source; 100k strings | Mapping proposals and throughput/parity |
| MORPH-002 | 2,000 fold-0 S1: 500 per country × singleton status; all 3,638 positive links; 48,130 total diagnostic pair rows | Normalization and feature separability |
| MORPH-002 exact-name negatives | Top five address-similar nonmatches per sampled query, found against all training targets; 3,021 pairs | Deceptive same-name cases |
| MORPH-002 other negatives | Random, same-country, first-token, numeric, name/address TF-IDF; fixed 100k target pool | Compare negative difficulty, not estimate full retrieval recall |
| TOKEN-001/run-002 | 1,000 fold-0 queries, 500/country; all 10,320,219 training targets; 3,449 true links | Candidate ceiling and route coverage |

The morphology sample deliberately overrepresents singletons and balances countries. Its precision/AP and pooled rates are conditional, not deployment estimates. It is singleton-stratified, not balanced across every nonzero match count. France has no supervised score. Fold 4 stays closed for selection; its target records remain permissible members of the full inference pool. Training-owned/unowned text alone fits diagnostic TF-IDF and replacement proposals.

## Normalization ablation

Exact nonempty equality on the same 3,638 sampled positive pairs:

| View | Name agreement | Address agreement |
|---|---:|---:|
| Raw | 152 / 3,638 = 4.18% | 73 / 3,638 = 2.01% |
| Light NFC/casefold/punctuation spacing | 745 = 20.48% | 293 = 8.05% |
| Compatibility + ampersand | 738 = 20.29% | 293 = 8.05% |
| Optional Latin accent fold | 874 = 24.02% | 293 = 8.05% |
| Sorted tokens | 920 = 25.29% | 417 = 11.46% |
| Deduplicated tokens | 948 = 26.06% | 462 = 12.70% |
| Aggressive combined view | 1,044 = 28.70% | 461 = 12.67% |

Higher equality is not automatically better. Against 9,975 name-TF-IDF hard negatives, light name equality is 162/9,975 (1.62%) versus aggressive 181/9,975 (1.81%). More critically, aggressive name JW≥0.9 occurs for 1,724 negatives versus 1,357 under light normalization, while true-positive high-JW counts fall from 2,375 to 2,138. Sorting helps equality but can degrade an order-sensitive similarity. Therefore retain independent views and metric-specific features; do not choose one destructive normalization to replace everything.

Generic local ICU transliteration on non-ASCII development positives raises Indian name JW≥0.9 from 74/443 (16.70%) to 129/443 (29.12%), and Indian address JW≥0.9 from 122/403 (30.27%) to 160/403 (39.70%). This is useful recovery evidence but far from complete. Negative comparisons and worsening counts are saved in `transliteration.json`; this is not candidate recall or validated precision. Swift/Foundation processed 72,722 unique sampled strings in a 5.53-second complete diagnostic run. Linux/backend equivalence still needs verification.

## Signals and noise analysis

`feature_diagnostics.csv` measures ROC AUC, sample-dependent AP, KS separation, positive/negative means and feature rates at 0.9 by country and negative population. Features include edit distance, JW, token ratios, Jaccard/Dice/containment, count-based char 2–5gram cosine, word TF-IDF cosine, prefixes/suffixes/acronyms, numeric overlap and name×address agreement. Count-based char cosine is explicitly different from char TF-IDF retrieval.

`transliteration.json` records minimal insert/delete/replace edit operations, token reordering, case equivalence, missing sides and containment. These describe differences between strings, not a proven generative corruption process. Local TSV examples cover hard positives, aggressive collisions and missed retrieval links; they never leave the machine. Replacement evidence is fitted only on training folds and is not enabled automatically.

## Current preprocessing configuration

Recommended primary representation: immutable raw text + null/blank indicators + NFC → casefold → Unicode letter/mark/number-preserving punctuation spacing → whitespace normalization. Keep separate compatibility/ampersand, Latin-folded, compact, sorted, deduplicated and numeric-format variants. Preserve exact raw numeric tokens and punctuation-sensitive number forms as additional evidence. No fitted legal suffix/street map is active.

The earlier exact baseline and candidate pilot use their recorded NFC/lower implementation; the new Python module uses casefold. These are versioned separately and must not share caches silently. `config_hash` includes operations, explicit map contents, implementation version and Unicode version. Raw field values and IDs are unchanged.

## Performance

PREP-001: 100k strings, exact output parity across Python/DuckDB/Polars for supported operations. NFC + punctuation spacing + whitespace: about 361k / 550k / 954k strings/s respectively. This excludes casefold and the complete optional-view bundle. The complete Python view bundle measured about 27.5k strings/s. These are measured sample rates, not full-corpus throughput guarantees.

MORPH-002 completed in 71.64s, peak process RSS 2.38 GiB. TOKEN-001/run-002 completed in 34.65s, peak 1.26 GiB. Concurrent process RSS values must not be added and described as a measured whole-machine peak. Full-profile runtime/RSS is in PROFILE-001 metadata. No AWS or Colab runtime was launched.

## Retrieval result and next actions

The full-target lexical union recovers 2,667/3,449 links (77.33%) with 179,298 candidates; top-100 recovers 73.56%. The union's oracle macro F0.5 ceiling is 0.86630, **not a model score**. See [CANDIDATE_PILOT.md](CANDIDATE_PILOT.md). This recall is insufficient; do not train an elaborate matcher over an inadequate candidate set.

1. Add independent character 3/4/5gram retrieval using a full-target, memory-bounded index.
2. Relax rare-token fanout selectively; current threshold leaves 536/1,000 queries without rare-name keys.
3. Add transliterated retrieval and measure marginal recovery plus candidate growth.
4. Preserve wider unions; measure where route caps or fusion discard positives.
5. Add address-only rescue with number-aware ranking, never a mandatory postal/number filter.
6. Expand the query pilot across nonzero match-count strata and repeat cross-country diagnostics.
7. Evaluate suffix maps one operation at a time against hard-negative collision sets.
8. Stress unseen-country handling and normalization on unlabeled France without claiming supervised robustness.
9. Build persistent retrieval caches and deterministic sharding before full-scale scoring.
10. Only after sufficient retrieval coverage, build the first calibrated lexical GBDT and select decisions using exact entity macro F0.5.

## Boundaries of this phase

Full exact profiles are comprehensive; exhaustive all-pairs fuzzy deduplication is not feasible and was not claimed. Existing adjacent-neighbor duplicate scans are lower bounds. Formal collision-group CV, statistically weighted morphology estimates, full-corpus aggressive-view collision indexing, character TF-IDF full-pool recall, and a deployable final preprocessing/matching configuration remain follow-up work. No unseen test labels, external business enrichment, or hidden evaluation scores were used.
