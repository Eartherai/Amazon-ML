# Research and ranked experiment plan

Current branch scope (25 September 2026): **classical ML only**. Neural and
embedding ideas below are preserved as historical Phase-1 research and are
`DEFER_TO_ALT_ARCHITECTURE_AGENT`; do not implement them in this branch. The
current measured priorities and pending learning-curve gates are maintained in
`CLASSICAL_ML_FRONTIER.md` and `LEARNING_CURVE.md`.

Research date: 25 September 2026 IST. This is a dataset-specific working plan, not a claim that a method will win. Generic methods and primary project/paper sources only; no competition business was searched, uploaded to a research service, or enriched. Empirical facts below come from this workspace. No learned model has been trained in phase one.

## What changes the strategy

There are 24.23 million source rows, 2.207 million labeled S1 entities and 7.638 million positive links. Test inference searches 9.970 million S2/S3 records for 1.733 million S1 queries: about 17.27 trillion possible pairs. Brute force is excluded. Only 5.58% of training S1 are singletons; approximately 72% have at least three links. A top-one assignment is structurally wrong. A small top1/top2 margin is not inherently a nonmatch signal because both candidates can be correct.

Training S1 names are entirely ASCII; target records include substantial Indic-script and accented variants. India is much harder under raw string comparisons than US. France contributes 259,452 test S1 records, about 15%; test also shifts toward India. Country-transfer and transliteration are first-tier research, not optional polish. Postal-like tokens are too sparse for mandatory blocking. Preserve Unicode combining marks: deleting all non-letter characters can corrupt Indic words.

These observations motivate **union lexical and transliteration retrieval, address/numeric corroboration, a compact supervised pair model, and entity-level decisions**. Embeddings are a complementary retrieval route if they recover lexical misses at acceptable cost.

## Ranked top ten experiments

Estimates below are planning envelopes, not benchmarked runtimes or claimed score gains. Every promotion requires a frozen split, country slices, candidate recall, singleton errors and a recorded cost. Start with 4k-20k S1 queries against the full target pool; do not make retrieval artificially easy by retaining only known positives. A prototype may use a reduced pool only if explicitly labeled and then rechecked at full density.

| Priority / planned ID | Hypothesis and measurable test | Device / estimate | Promotion gate |
|---|---|---|---|
| 1 / EXP-001 | Exact normalized name AND address, country equal: establish precision and format/metric plumbing; all-empty control | Mac, minutes | Diagnostic only; never a competitive submission |
| 2 / EXP-002 | Char 3-5 gram name retrieval plus exact/rare-token blocks; compare K=20,50,100 and per-source allocation | Mac, 30-90 min prototype; full scale benchmark required | Plot link recall AND entity oracle macro ceiling versus candidate volume |
| 3 / EXP-003 | Add address retrieval, house-number+rare-token blocks, token order and Latin-accent views | Mac, 30-90 min | Unique rescued positives, especially difficult India pairs; bounded p99 fan-out |
| 4 / EXP-004 | ICU transliteration view for Indic script and French accents, preserving raw view | Mac prototype, 1-3 h incl. setup | Rescue rate on cross-script positives; collision/FP audit; library/revision recorded |
| 5 / EXP-005 | LightGBM pair classifier with name, address, numeric, retrieval-rank and rarity features | Mac, 30-120 min on bounded candidate training set | Group-disjoint dev macro F0.5 beats heuristic; retain precision under country transfer |
| 6 / EXP-006 | Hard negatives, sampling weights, OOF sigmoid calibration and entity thresholds | Mac, 30-120 min | Stable gains across held-out entities and country slices; no label-leaky mining |
| 7 / EXP-007 | Multilingual E5-small embeddings as rescue retrieval plus cosine features | Mac MPS small throughput probe; remote GPU only if needed | Adds candidates missing from lexical union; report gains per million pairs and GPU hour |
| 8 / EXP-008 | Target ownership conflict resolution, S1 singleton/cardinality features and expected set-F0.5 decisions | Mac, 30-90 min | Macro improves without suppressing valid multi-target matches; compare no-constraint control |
| 9 / EXP-009 | Fine-tuned multilingual cross-encoder on ambiguous candidates, including hard negatives | One eligible GPU, provisional 4-12 h after measured throughput | Gain on hard pairs AND end-to-end macro; bound inference volume before full run |
| 10 / EXP-010 | Complementary OOF blend, feature ablations, robustness and package rehearsal | Mac + already justified inference resources | Gain survives untouched fold and source/country stress; reproducible outputs and validator |

Priority is information gain per cost. If lexical recall is already near its useful ceiling, skip redundant retrieval work. If cross-script misses dominate, advance experiment 4 before training. Stop any experiment that cannot finish inference, validation and packaging before the freeze window.

## Methods assessed

All benefit statements are hypotheses specific to this dataset. Software licenses are separate from final model-checkpoint licenses. MIT/Apache model restriction is checked conservatively before selecting any learned checkpoint.

| Method | Why it may help / expected benefit | Cost and complexity | Risk | License / Mac / AWS / decision |
|---|---|---|---|---|
| Fellegi-Sunter / Splink concepts | Rarity-aware agreement and disagreement weights are interpretable; useful sanity baseline | Low-medium; DuckDB-based blocked features | Conditional-independence assumptions fail for correlated name metrics | Splink MIT; Mac yes, CPU cloud optional; P2 baseline/concepts |
| Dedupe-style linkage | Learned comparisons/block predicates; useful candidate and active-learning concepts | Medium integration | Active learning less valuable with 7.6M supplied positive links; do not collapse distinct target IDs | Dedupe MIT; Mac yes; P3 comparator, not first stack |
| Character TF-IDF 3/4/5 grams | Handles typos, punctuation and partial names; probably strongest cheap retrieval family | Medium; sparse postings/top-K with chunked queries | Common grams cause large products; different scripts share little | sklearn BSD-3, sparse_dot_topn Apache-2.0; Mac/CPU yes; P1 |
| Word TF-IDF / BM25 | Rewards rare business/address terms and robust word order | Low-medium; inverted index | Short names, misspellings, templated legal suffixes; BM25 implementations may materialize scores | BM25S MIT; Mac sharded/CPU yes; P1 complementary route |
| Exact/relaxed keys | Fast anchors, deterministic provenance | Low | Exact name alone collides across many businesses; blank keys must never match | Own code; Mac yes; P1 |
| Edit distance, Jaro/Winkler, RapidFuzz | Efficient rich pair features within candidates | Low per pair, total volume dependent | Token-set/partial ratios can be 100 for misleading subsets; JW overweights common prefixes | RapidFuzz MIT; Mac/CPU yes; P1 |
| Jaccard, Dice, token overlap / rarity | Explainable name/address agreement; exact uncommon token matches | Low | Generic terms and legal suffixes dominate without IDF | Own implementation; Mac yes; P1 |
| SoftTFIDF / Monge-Elkan | Soft token alignment for typographical variants | Medium; token-by-token comparisons | Higher per-pair cost and partial-match optimism | Implement equations or verify chosen library license; Mac on shortlist; P2 |
| Phonetic encodings | Might recover Romanized name variation | Low-medium | English-centric algorithms and noisy aliases can create false merges | Verify library; Mac yes; P3 limited ablation |
| ICU multilingual transliteration | Connects Indic target scripts to ASCII S1; separate accent view aids France | Medium setup and caching | Transliteration is not translation; phonetic forms differ; collisions; retain raw text | ICU Unicode license / PyICU MIT (not a learned model); Mac/CPU yes; P1 |
| LightGBM | Learns interactions such as strong name plus conflicting house number | Low-medium | Negative sampling distorts calibration; source/country shortcuts | MIT; CPU Mac feasible, OpenMP dependency check; P1 first learned model |
| CatBoost / XGBoost | Strong GBDT alternatives; test complementary error patterns | Medium | Redundant sweeps consume time; country category can overfit | Apache-2.0 for each, verify pinned releases; CPU Mac yes; P2 after LightGBM |
| Hard-negative mining | Exposes same-name different-location confusions | Medium incremental work | In-sample mining leaks if validation entities/targets enter train | Pipeline operation; Mac/CPU yes; P1 after baseline |
| Calibration / thresholds | Targets exact macro F0.5 and singleton penalty | Low | Pair calibration does not directly maximize entity utility; repeated tuning overfits | sklearn BSD-3 support; Mac yes; P1 |
| Multilingual E5 bi-encoder | Cross-script/semantic candidate rescue and complementary score | Medium-high at 24M records | Semantic similarity is not business identity; short names and numbers can blur | E5-small MIT, ~117.65M reported tensor elements; MPS pilot/GPU full; P2 |
| BGE-M3 | Dense plus sparse/late-interaction retrieval possibilities | High full-corpus footprint | Expensive and possibly redundant; checkpoint size must be independently checked before use | Model card MIT; GPU practical; P3 reserve |
| Supervised contrastive blocking (SC-Block) | Adapt retrieval to business identity with supplied positives | High training and index rebuild | Hard negative false negatives and country overfitting | Paper method; implementation/checkpoint license still gate; GPU; P3 conditional |
| Sudowoodo-style representation learning | Learn typo/field invariance without extra external records | High; augmentation study | Augmentations can change identity or erase decisive numbers | Method only; license of reused code/checkpoint must be checked; GPU; P3 |
| Ditto-style cross-encoder | Jointly reads both records and field boundaries; handles difficult aliases | High; only on blocked ambiguity region | Generic LM factual associations are unreliable; inference can dominate deadline | Use MIT XLM-R base plus trained head; GPU preferred; P2 after measured signal |
| Pretrained BGE reranker | Fast feasibility comparator on hard pairs | Medium-high | Relevance scores are not calibrated ER probabilities; require ER validation/fine-tuning | bge-reranker-v2-m3 Apache-2.0, ~567.76M; GPU; P3 comparator |
| Graph/ownership constraints | Training gives zero shared positive target ownership | Low-medium sparse graph | S1 may own many targets; one-to-one bipartite matching is wrong; transitive chains create false merges | Own sparse optimization; Mac yes; P2 validate target-only exclusivity |
| Routing/ensembles | Combine lexical/numeric strengths with cross-script models | Medium | High agreement means little benefit; complex routers overfit countries | Component licenses apply, aggregate parameter budget documented; P2 last |

## Validation protocol

Freeze five deterministic S1 folds using SHA256(seed|ID), seed 20260925. Fold 0 is development, folds 1-3 initial training, fold 4 locked. Positive target ownership travels with the S1 fold; training negatives cannot include held-out owned targets. Ground truth has no shared targets; split builder fails if this changes. Hashing is deterministic, approximately stratified; inspect country/cardinality balance rather than claim exact stratification.

For final model comparison, cross-fit within training/development entities, calibrate on out-of-fold scores, select thresholds there, then unlock fold 4 once. Report standard entity CV, train-US/validate-India and reverse, singleton slice, hard-negative slice, low-string-similarity slice, script slice, and strict same-name/address collision-group stress split. Country transfer tests robustness; it does not estimate France performance directly. A test-weighted US/India average must leave France uncertainty explicit.

Learn token frequencies, IDF, suffix dictionaries, noise transforms and embeddings from permitted training data only. Index held-out target strings for retrieval as unlabeled inference records without fitting learned transforms on them. Query realistic full target pools (including distractors); do not use true ownership to filter each validation query's candidate universe. For a disjoint-pool regime, report that it is easier and check full-pool results too. Keep held-out records out of training negatives even if they appear in the inference index.

Use exact per-entity score: empty/empty=1; otherwise 1.25*TP/(predicted_count+0.25*true_count). Average S1 scores, never pool links and call it macro. Report micro precision/recall separately. No-prediction micro precision is undefined, not 100%.

Candidate recall is both link recall and per-S1 coverage. Also compute oracle macro ceiling by predicting only true candidate links: singleton score 1; positive entity ceiling 1.25*r/(r+0.25*g), where r is retrieved positives and g is all positives. A missed entire entity matters more than one missed link from a large cluster. Bootstrap confidence intervals must resample entities/components, not pairs.

## Retrieval and features

Use raw, light Unicode, Latin-accent, transliterated, and suffix-relaxed views. Never strip combining marks from Indic text. Distinguish missing values from mismatch. Country-equal retrieval is empirically supported by 0 cross-country positives, but it must work for arbitrary strings and preserve fallback behavior for missing/unknown labels.

Union independent name, address, acronym, rare token and numeric routes. Shortlists are per route and source so one abundant source cannot starve the other; tune total K against recall. Numeric and postal keys require text corroboration and fan-out controls. Deduplicate candidate pairs by IDs, never collapse distinct S2/S3 IDs merely because text repeats. For duplicate strings, compute reusable features/embeddings once but expand back to every record ID.

Build name and address similarity families, informative-token overlap, digit token multiset/house-number conflict, postal-shape overlap, missing indicators, retrieval ranks, reciprocal-rank fusion, route indicators, target-name frequency and source interactions. Address numbers are not always the first numeric token; landmarks and inserted numbers occur. Keep weak parser outputs as features, not absolute vetoes. Start 50-100 float32 features in batches, not a single giant dataframe.

Use mostly retrieved hard negatives plus a documented random background mixture. Train only negatives attached to permitted train S1/targets. Preserve complete validation candidates. Weight/calibrate for the inference distribution, not the sampled training ratio. Evaluate negative-downsampling and high-cardinality entity weighting as ablations.

## Decisions beyond a pair threshold

A global OOF threshold is the reference. Compare source-aware thresholds with shrinkage toward global, per-entity empty-vs-nonempty utility, and target ownership competition. Because S1 commonly has several correct matches, top1/top2 margin alone is inappropriate. Investigate the boundary between selected and rejected scores, expected number of links and S1-level features from OOF candidate predictions.

Expected set-F0.5 selection is a later hypothesis: choose empty or a ranked prefix to maximize expected entity utility, using calibrated pair probabilities and a model for missed links/cardinality. A naive independence calculation is not exact for correlated duplicate records and must be compared to the global-threshold baseline. Never impose a fixed match count from the training histogram.

In a cascade, record candidate provenance at each stage. The submitted candidate set must correspond to the described final matching model's actual scored set. If a GBDT is a model in the final decision, do not hide its scored pairs by reporting only cross-encoder calls. Describe component/model boundaries and retain both sets; final matches must lie in the exported candidate set.

## Compute arithmetic and go/no-go rules

At test K=50, there are up to 86.63M pairs; K=100 gives 173.25M. A 100-feature float32 matrix alone is about 34.65/69.30 GB respectively, before metadata. Stream feature batches and write compressed numeric shards. A dense float32 all-pairs matrix exceeds 69 TB. For 9.97M targets, 384-dimensional float16 embeddings alone use about 7.66 GB; float32 doubles that. Names plus addresses as separate embeddings double again. A 1024-dimensional float16 target matrix alone is about 20.42 GB. Full embeddings/indexes therefore should not share the local 24 GB RAM/disk without sharding/compression.

Time estimates must use measured rates: retrieval_seconds=query_count/queries_per_second; feature_seconds=pair_count/pairs_per_second; embedding_seconds=records/records_per_second; cross_encoder_seconds=scored_pairs/pairs_per_second. Include parsing, I/O, index building, GPU transfer, and checkpoint time, not just model forward passes. With 100M pairs and 1,000 pair/s, a reranker takes 27.8 hours. Narrow its scope before renting a GPU.

Mac: audit, sparse prototypes, CPU GBDT, calibration and output logic. Use bounded workers, no concurrent large jobs, >=8 GiB disk reserve. MPS is planned but has not been benchmarked or even enabled in the current minimal environment. AWS: CPU/RAM overflow only if local timing justifies it; single T4/L4/A10G class GPU as first accelerator subject to account eligibility, region, quota and current price. Do not assume credits buy GPU access. Fresh accounts may have zero GPU quota or restricted eligible services.

Provisional $800 envelope: $40 prototypes, $140 retrieval/CPU storage work, $180 neural research, $140 final inference/reproducibility, $300 untouched contingency. Each account starts with only a reported $200; cap planned usage at $150 per account until real balances are confirmed. This is a ceiling, not a spending target. Actual hourly prices remain unquoted until region/profile/instance chosen and current AWS pricing verified. Storage, snapshots, public IP and transfer can outlive an instance and need cleanup. Stop on wall-time/cost cap, checkpoint, verify stop/termination, and write ledger rows. Additional credits after 48h are conditional and budgeted at zero.

Colab/Kaggle: optional legitimate per-person capacity; availability, runtime limits, memory and private-data handling must be checked. Do not rotate accounts to bypass quotas. Competition data must remain private. No dataset upload or remote session is needed for this research phase.

## Failure criteria and ablations

Reject models selected only on pairwise F1/public LB. Reject blockers tested on only a tiny easy distractor pool. Reject fragile country dummies, identity lookups, universal number-vetoes, arbitrary suffix removal, and transitive clustering without conflict tests. Required removals for a strong system: address, name, numeric, country, char retrieval, fuzzy metrics, embeddings, hard negatives and singleton calibration. Compare paired entity score deltas and FP/FN taxonomy. Keep the best validated pipeline immutable.

## Primary sources

1. [Splink: Fellegi-Sunter model](https://moj-analytical-services.github.io/splink/topic_guides/theory/fellegi_sunter.html) and [term-frequency adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html): probabilistic linkage and rare-value evidence.
2. [Dedupe repository](https://github.com/dedupeio/dedupe) and [MIT license](https://github.com/dedupeio/dedupe/blob/main/LICENSE).
3. [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn): sparse matrix product with top-N selection; [license](https://github.com/ing-bank/sparse_dot_topn/blob/master/LICENSE).
4. [BM25S](https://github.com/xhluca/bm25s): efficient lexical retrieval; [MIT license](https://github.com/xhluca/bm25s/blob/main/LICENSE).
5. [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) and [license](https://github.com/rapidfuzz/RapidFuzz/blob/main/LICENSE): string comparators.
6. [ICU transforms](https://unicode-org.github.io/icu/userguide/transforms/general/): transliteration rather than translation. [PyICU project](https://github.com/ovalhub/pyicu).
7. [LightGBM](https://github.com/lightgbm-org/LightGBM) and [MIT license](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE); [CatBoost](https://github.com/catboost/catboost) and [Apache license](https://github.com/catboost/catboost/blob/master/LICENSE).
8. [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html): fit calibrators on predictions independent of model fitting.
9. [Ditto paper](https://arxiv.org/abs/2004.00584): serialized record-pair transformer matching; paper benchmark scores are not evidence on this competition.
10. [Sudowoodo paper](https://arxiv.org/abs/2207.04122): contrastive self-supervised representations for integration tasks.
11. [SC-Block paper](https://arxiv.org/abs/2303.03132): supervised contrastive candidate blocking.
12. [Pretrained embeddings ER study](https://arxiv.org/abs/2304.12329): evidence that representation choices deserve ER-specific benchmarking.
13. [E5-small card](https://huggingface.co/intfloat/multilingual-e5-small), [BGE-M3 card](https://huggingface.co/BAAI/bge-m3), [BGE reranker card](https://huggingface.co/BAAI/bge-reranker-v2-m3), [XLM-R base](https://huggingface.co/FacebookAI/xlm-roberta-base). Revisions/licenses and available tensor counts are saved in MODEL_LICENSE_RESEARCH.json. Card metadata is a screening check; preserve actual license and count loaded model parameters at final selection.
14. [FAISS](https://github.com/facebookresearch/faiss): dense nearest-neighbor indexing; use CPU or suitable CUDA hosts, not an assumption of MPS support.
15. [Official Colab CLI](https://github.com/googlecolab/google-colab-cli) and [Colab FAQ](https://research.google.com/colaboratory/faq.html).
16. [AWS G6](https://aws.amazon.com/ec2/instance-types/g6/), [instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html), [EC2 pricing](https://aws.amazon.com/ec2/pricing/): hardware and region-specific cost verification before launch.

## Phase4 sparse retrieval implementation research

Primary source: https://github.com/ing-bank/sparse_dot_topn (v1.2.0 Apache2.0 license verified at tag). Fused sparse multiplication/topN is a compute optimization, not a new matching hypothesis. Local bounded benchmark:2.59x name /3.42x address speedup, zero candidate-set differences across1,000 queries x25,000 targets; reference boundary-tie fallback required. Full-pool nested-sample parity remains a gate. See docs/PHASE4_RETRIEVAL_ENGINE.md. No outside identities or enrichment used.
