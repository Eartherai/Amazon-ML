# State-of-the-art entity resolution: generic methodology survey and 12-hour sprint ranking

Research date: 2026-09-26. Scope: literature, repository and model-card research only. No model was trained, no job was launched, and no score in this file is newly measured. Every "expected upside" below is a planning hypothesis. It must be tested on the fixed 6,000-S1 fold 2/3 population before promotion. Fold4 remains CLOSED.

Compliance notes:

- Web research covered generic papers, repositories and model cards only. No competition business name, address or identifier was searched, fetched or sent to any service.
- Licenses were read on 2026-09-26 from the GitHub repository-license API and the Hugging Face model API (`license:` tag, safetensors parameter totals, and head revision). Re-pin the exact revision and re-audit through the `docs/MODEL_LICENSES.md` procedure before any checkpoint enters a final solution.
- Code licenses are not model licenses. The final-model rule (MIT or Apache-2.0, <=8B combined learned parameters, offline, fine-tuned only on provided data; see `docs/COMPETITION_RULES.md`) is applied conservatively.
- "Rules caveat" marks publisher checkpoints that were already supervised-fine-tuned on external ranking or transliteration data. Whether they are allowed is an open interpretation question recorded in `docs/MODEL_LICENSES.md`, not a license question.
- Test-fitted IDF or BM25 statistics (used by the SUB-004 France rescue) are also an open rules question (`docs/COMPETITION_RULES.md`, line 22). This file does not resolve it.

## 1. Our setting and where the remaining loss sits

Workspace facts come from STATUS.md (2026-09-25 entries), FINAL_SPRINT_BOARD.md, docs/FRIEND_METHOD_DIFF.md, docs/FRANCE_SHIFT.md, docs/MISSED_LINK_ANALYSIS.md and docs/MODEL_LICENSES.md. The sprint brief supplied the India S3 miss rate and the stacked E5 gain.

- **Scale.**
  - Training: 2.21M labeled S1, 7.64M positive links and 24.23M source rows.
  - Test: 1.733M S1 (India 809,986; US 663,106; France 259,452, about 15% and unseen in training) against about 9.97M S2/S3 targets.
  - About 72% of training S1 have three or more links. Only 5.6% are singletons.
- **Records.**
  - Short, noisy name and address fields.
  - France has 15.7% non-ASCII names.
  - India targets include Indic-script variants, while all training S1 names are ASCII.
  - Each target has at most one owner S1.
- **Retrieval.** Name and address char3 TF-IDF top-100 union reaches about 96.6% link recall. India S3 is the weakest slice, with about 7% miss.
- **Matchers.**
  - Stage 1 is a 51-feature LightGBM.
  - Stage 2 is a classical LightGBM trained on 194k held-fold OOF S1. It scores about 0.965 OOF and 0.962606 on the fixed 6k at top-12.
  - Stacking the EXP-050 multilingual-E5-small cross-encoder (117.65M parameters, 18k S1 / 420k pairs, fold1 owner-safe) on stage 2 adds +0.0024 on the fixed 6k (CL-013). This is a local measurement. It has not been run on test.
- **Public versus local.**
  - SUB-003 scored 0.944 public against about 0.9635 locally.
  - Two different matchers show the same gap, which points at shared components. France retrieval is the leading hypothesis, and SUB-004 isolates it.
  - EXP-045 G5 inference measured 3,350 scored pairs/s for a 117.6M cross-encoder.
- **Fixed-6k loss decomposition of CL-003** (macro points):

| Bucket | Loss | Families that address it (section 3 ids) |
|---|---:|---|
| Retrieval miss or true link outside top-12 | 0.0149 | B1 multi-view sparse blocking, B2 meta-blocking, B4 dense retrieval, E4/E5 bi-encoder training, F2 reverse retrieval, stage-2 depth |
| True link inside top-12 but outside sidecar | 0.0108 | stage-2 depth, E7 listwise GBDT, E1/E2 cross-encoder features |
| Rejected true links | 0.0079 | F1 expected-F decision, E1 stronger matcher, calibration |
| False-positive links | 0.0068 | F2 one-owner assignment, E1 CE features, C2 numeric tagging |
| Singleton false positives | 0.0015 | S1-empty model inside F1 |
| Oracle ceilings | sidecar 0.9744, top-12 0.9851 | |

### 1.1 What the metric implies (generic algebra, no data)

For one S1 with true set T and predicted set S:

- F0.5 = 1.25·TP / (0.25·|T| + |S|).
- Empty prediction with empty truth scores 1.
- Empty prediction with non-empty truth scores 0.
- Non-empty prediction with empty truth scores 0.

Take a non-empty prediction with `a` true positives and D = 0.25·|T| + |S|. Adding one candidate with calibrated match probability p gives expected F = 1.25·(a + p)/(D + 1). That beats the current 1.25·a/D exactly when **p > a/D = F_current / 1.25**. Consequences:

1. The expected-F-optimal acceptance threshold depends on the S1 and lies in (0, 0.8]. A single global threshold near 0.8-0.9 is right only for S1 whose current set is already near F = 1. For S1 with weak partial evidence, it rejects links that are worth adding.
2. The first link is a different decision. It flips empty versus non-empty, and its value depends on P(T is empty), which is the singleton bucket.
3. The rejected, false-positive and singleton buckets together hold 0.0162 of the 6k loss. A per-S1 expected-F decision (F1) is the cheapest untested lever against all three.

## 2. Summary catalog

Expected upside is a hypothesis in macro F0.5 on the fixed 6k unless marked "public". Implementation time assumes our existing codebase and stored OOF artifacts.

| Id | Method | Year | Code license; model license and parameters | Implementation | Expected upside here | Sprint verdict |
|---|---|---|---|---|---|---|
| A1 | Fellegi-Sunter / Splink | 1969; Splink 2020+ | MIT | 3-4 h | +0 to +0.0005 as a feature | Skip |
| A2 | Magellan / py_entitymatching | 2016 | BSD-3-Clause | n/a | about 0 (superseded by our GBDT stack) | Skip |
| A3 | dedupe (active learning) | 2014+ | MIT | 4 h+ | about 0 (labels are plentiful) | Skip |
| A4 | Zingg | 2021+ | AGPL-3.0 | n/a | about 0; copyleft license | Exclude |
| A5 | Learned string edit distance (Ristad-Yianilos; Bilenko-Mooney MARLIN) | 1998 / 2003 | own code | 4-5 h | +0 to +0.002 (India/France features) | Backlog |
| A6 | Auto-FuzzyJoin (reference-table fuzzy join) | 2021 | no license file in repo | idea only | covered by F2 reverse-margin features | Idea only |
| B1 | Sparkly top-k TF-IDF/BM25 blocking | 2023 | BSD-3-Clause | 2-4 h on top of ours | +0.002 to +0.008 public, mostly France | Recommended (R3) |
| B2 | JedAI / pyJedAI meta-blocking; SparkER | 2017-2023 | Apache-2.0; SparkER GPL-3.0 | 2 h (weighting idea only) | inside B1 | Use the idea |
| B3 | DeepBlocker | 2021 | BSD-3-Clause | 6 h+ | about 0 beyond B1/B4 | Skip |
| B4 | Sentence-BERT / SC-Block contrastive dense blocking | 2019 / 2023-24 | Apache-2.0 / BSD-3-Clause | 3 h + GPU | +0.001 to +0.004 | Recommended (R5) |
| B5 | Pre-trained embeddings for ER (Zeakis et al.) | 2023 | evidence only | n/a | evidence | Reference |
| C1 | DeepMatcher | 2018 | BSD-3-Clause | n/a | about 0 (RNN era) | Skip |
| C2 | Ditto (+ Rotom augmentation) | 2020 / 2021 | Apache-2.0 / BSD-3-Clause | 1 h (tricks folded into R4) | +0 to +0.001 over R4 via numeric tags | Fold into R4 |
| C3 | HierGAT | 2022 | MIT | 10 h+ | +0 to +0.002 | Skip |
| C4 | JointBERT | 2021 | BSD-3-Clause | 6 h | about 0 (test S1 are unseen entities) | Skip |
| C5 | R-SupCon supervised contrastive | 2022 | BSD-3-Clause | inside R5 | inside R5 | Fold into R5 |
| C6 | Sudowoodo | 2023 | BSD-3-Clause | 6 h+ | about 0 (label-rich setting) | Skip |
| C7 | Unicorn | 2023 | no license file in repo | n/a | about 0 | Exclude |
| C8 | DADER domain adaptation | 2022 | not checked | 8 h+ | 0 to +0.003 public (France) | Replace by leave-one-country-out check |
| D1 | LaBSE | 2020 / 2022 | Apache-2.0; 471M | 3 h | bi-encoder backup | Backup |
| D2 | multilingual-E5 small/base/large | 2023 / 2024 | MIT; 118M / 278M / 560M | in use | base CE +0.001 to +0.004 over small | R1, R4, R5 |
| D3 | mDeBERTa-v3-base | 2021 | MIT; about 276M | 3 h | CE alternative, slower attention | Ablation only |
| D4 | XLM-R base/large | 2019 | MIT; 279M / 561M | 3 h | raw MLM start; mE5 is the tuned descendant | Backup |
| D5 | mmBERT small/base | 2025 | MIT; about 140M / 307M | 4 h + environment risk | possible speed gain | Backlog |
| D6 | CANINE-s/c | 2021 | Apache-2.0; 132M | 6-8 h | +0 to +0.002 cross-script | Backlog |
| D7 | ByT5-small/base | 2021 | Apache-2.0 | 6-8 h | +0 to +0.002; slow on long byte sequences | Backlog |
| D8 | IndicXlit transliteration | 2022 | MIT (code and models) | 4-6 h | +0 to +0.002 India S3 | Backlog; rules caveat |
| D9 | Retrieval-tuned multilingual rerankers | 2021-2025 | Apache-2.0: bge-reranker-v2-m3 568M, gte-multilingual-reranker-base 306M, Qwen3-Reranker-0.6B 596M, mmarco-mMiniLMv2 118M | 3 h | alternatives inside R4 | Rules caveat |
| E1 | Cross-encoder reranker fine-tuning (sentence-transformers v4 recipe) | 2025 | Apache-2.0 | core of R4 | see R4 | R4 |
| E2 | Localized Contrastive Estimation (LCE) | 2021 | Apache-2.0 | 1-2 h | +0 to +0.002 over BCE | Ablation in R4 |
| E3 | Hard negatives with denoising (RocketQA) | 2021 | method | 1 h | enables R4/R5 | Use |
| E4 | In-batch negatives (MNRL) + GradCache | 2019 / 2021 | Apache-2.0 | inside R5 | inside R5 | R5 |
| E5 | Margin-MSE cross-architecture distillation | 2020 | Apache-2.0 | 4 h | +0 to +0.002 after R4/R5 | Backlog |
| E6 | ColBERT late interaction | 2020-2022 | MIT; colbertv2.0 110M (English), colbert-xm 853M MIT | 8 h+ | about 0 in sprint (index size) | Skip |
| E7 | LightGBM lambdarank / S1-context meta-features | 2017+ | MIT | 2 h | +0.0003 to +0.0015 | Filler |
| F1 | Expected-F decision (Ye et al.; Dembczynski et al. GFM) | 2011-2012 | own code | 2 h | +0.001 to +0.004 | Recommended (R2) |
| F2 | One-owner assignment / unique mapping / reverse competitor search | 2021-2023 | own code | 3-4 h | +0.0005 to +0.003 incremental | Inside R3/R5 |
| F3 | Collective ER (Bhattacharya-Getoor) | 2007 | method | 10 h+ | small beyond existing duplicate expansion | Skip |
| G1 | LLM matching (Peeters et al.; Steiner et al.; ComEM; AnyMatch) | 2023-2025 | only Apache/MIT <=8B bases eligible; Jellyfish CC-BY-NC excluded | 10 h+ | uncertain | Reference only |

## 3. Method notes

Each entry gives the source, core idea, relevance to our setting, compute, data needs, license, implementation estimate, expected upside and risks.

"Our setting" is shorthand for:

- millions of records with short, noisy name and address fields;
- three countries, with France unseen in training;
- precision-weighted per-S1 F0.5, where empty/empty scores 1;
- each target owned by at most one S1.

### A. Classical and probabilistic linkage

**A1. Fellegi-Sunter model / Splink**

- **Source.** Fellegi and Sunter, JASA 1969. Splink by the UK Ministry of Justice, `moj-analytical-services/splink`.
- **Core idea.**
  - The match weight is a sum of per-comparison-level log(m/u) ratios.
  - m and u are fit by EM without labels.
  - Term-frequency adjustments make agreement on rare values count more.
- **Relevance.**
  - With 7.6M labeled links, our supervised GBDT already learns interactions that Fellegi-Sunter cannot. Its conditional-independence assumption fails for correlated name and address similarities.
  - TF adjustment duplicates our IDF-rarity features.
  - The one differentiated angle is unsupervised EM fit on France comparisons, which would give France-specific m/u. That is subject to the same test-fitting rules question as test-fitted IDF.
- **Compute.** CPU. The DuckDB backend handles millions of records on one machine.
- **Data needs.** None for EM. Labels help estimate m.
- **License.** MIT.
- **Implementation.** 3-4 h to map our comparisons.
- **Expected upside.** +0 to +0.0005 as an extra feature.
- **Risks.** Redundant. EM can lock onto a wrong cluster inside blocked data.

**A2. Magellan / py_entitymatching**

- **Source.** Konda et al., PVLDB 9(12), 2016. `anhaidgroup/py_entitymatching`.
- **Core idea.** An end-to-end workflow: blocking, sampling, labeling, automatic string-similarity feature generation, then a classical classifier.
- **Relevance.** Our pipeline is already a larger-scale version: sparse top-k blocking, 51+ string/numeric features, GBDT and nested thresholds.
- **Compute.** Single-machine pandas. Does not scale to our pair counts.
- **Data needs.** Labels.
- **License.** BSD-3-Clause.
- **Implementation.** Not worth it.
- **Expected upside.** About 0.
- **Risks.** Time sink.

**A3. dedupe (dedupe.io)**

- **Source.** `dedupeio/dedupe`, based on Bilenko's thesis on learnable similarity functions.
- **Core idea.** Active learning selects informative pairs for a human to label. The library learns blocking predicates and a logistic matcher.
- **Relevance.** Active learning solves label scarcity, and we have millions of labels. Learned blocking predicates are subsumed by top-k sparse retrieval.
- **Compute.** Laptop scale.
- **Data needs.** Interactive human labels.
- **License.** MIT.
- **Implementation.** 4 h or more.
- **Expected upside.** About 0.
- **Risks.** Needs a human in the loop, which does not fit the setting.

**A4. Zingg**

- **Source.** `zinggAI/zingg`.
- **Core idea.** Spark-based active-learning ER with learned blocking trees.
- **Relevance.** Same as dedupe.
- **Compute.** Spark cluster.
- **Data needs.** Interactive human labels.
- **License.** AGPL-3.0. That is not MIT or Apache-2.0 and is copyleft, so it is unsuitable for the final solution.
- **Implementation.** Not applicable.
- **Expected upside.** About 0.
- **Risks.** License exclusion.

**A5. Learned string edit distance**

- **Source.** Ristad and Yianilos, 1998 (stochastic transducer). Bilenko and Mooney, KDD 2003 (MARLIN: learnable edit distance per field plus an SVM over similarity vectors).
- **Core idea.** Learn character substitution, insertion and deletion costs from matched pairs rather than using unit Levenshtein costs.
- **Relevance.**
  - Train positives contain systematic variation: vowel drops in romanized Indian names, abbreviations, transliteration of Indic-script targets after anyascii, and French accent and elision patterns.
  - A learned cost table, or a pair-HMM on anyascii-normalized strings, yields a cheap, portable feature and possibly a retrieval view.
  - Earlier notes say mined maps are disabled. This approach would learn weights for a feature rather than rewrite text, which keeps raw text intact.
- **Compute.** CPU. EM over about 1M sampled positive pairs takes minutes to hours.
- **Data needs.** Positive pairs (plentiful).
- **License.** Own code.
- **Implementation.** 4-5 h.
- **Expected upside.** +0 to +0.002, concentrated in India and France.
- **Risks.**
  - Overlaps with existing JW, token and skeleton features.
  - India/US-learned costs may not fit French patterns.

**A6. Auto-FuzzyJoin**

- **Source.** Li, Cheng, Chu, He and Chaudhuri, SIGMOD 2021. `chu-data-lab/AutomaticFuzzyJoin`.
- **Core idea.**
  - One input is a reference table, and each right-side record joins at most one reference record.
  - The distance geometry around each reference record programs precision-targeted joins without labels.
- **Relevance.**
  - Our S1 side acts as a reference table and each target has at most one owner. This is the same structure.
  - The transferable ideas are two features: the target's nearest-owner margin (distance to the best S1 versus the second-best S1) and a per-target precision target. Both belong in F2.
- **Compute.** CPU.
- **Data needs.** None (unsupervised).
- **License.** No license file in the repository, so treat the code as unavailable. The idea is usable.
- **Implementation.** Folded into F2.
- **Expected upside.** Via F2.
- **Risks.** None beyond F2.

### B. Blocking and candidate generation

**B1. Sparkly: top-k TF/IDF blocking**

- **Source.** Paulsen, Govind and Doan, PVLDB 16(6), 2023. `anhaidgroup/sparkly`.
- **Core idea.**
  - Lucene BM25 over 3-gram bags of concatenated identity attributes.
  - Top-k per query, run distributed and share-nothing.
  - "Sparkly Auto" selects blocking attributes and tokenizers automatically.
  - The paper reports that it outperforms 8 state-of-the-art blockers, including learned ones, and argues future blocking work should compare against top-k TF/IDF.
- **Relevance.**
  - Direct support for our char3 top-100 design.
  - Two differences are worth testing: BM25 length normalization (parameter b) versus our cosine TF-IDF, which matters for short versus long French and Indian addresses; and multiple tokenizer views per attribute.
  - Our test-fitted France char3 probe is the same family.
- **Compute.** CPU fleet. We already run sparse top-k on r8i instances.
- **Data needs.** None.
- **License.** BSD-3-Clause.
- **Implementation.** 2-4 h to add views (BM25 char3, accent-folded char3, word BM25 on canonicalized address, numeric keys) to the existing engine.
- **Expected upside.** +0.002 to +0.008 on public, mostly France. India/US local gain is +0 to +0.002.
- **Risks.**
  - France cannot be measured locally.
  - Rescued pairs outside the frozen candidate set need a text-only scorer, and the text-only override inside the top-12 was only about 62% precise.
  - Candidate fan-out.

**B2. JedAI / pyJedAI meta-blocking; SparkER**

- **Source.** Papadakis et al. (JedAI Toolkit, VLDB 2018 demo; EDBT 2020). pyJedAI is the Python port. SparkER (Gagliardelli et al., EDBT 2019) runs on Spark.
- **Core idea.** Schema-agnostic token blocking, then meta-blocking: build a blocking graph, weight each edge by co-occurrence evidence (common blocks, Jaccard, ARCS), and prune low-weight edges per node.
- **Relevance.**
  - The edge-weighting idea transfers: rank a union of views by how many views retrieved the pair and their reciprocal ranks, then keep top-N per S1 before feature computation.
  - This bounds the fan-out of multi-view unions in B1.
- **Compute.** CPU.
- **Data needs.** None.
- **License.** pyJedAI and JedAI are Apache-2.0. SparkER (`Gaglia88/sparker`) is GPL-3.0, so reference only.
- **Implementation.** 2 h, in our own code.
- **Expected upside.** Inside B1.
- **Risks.** Pruning can drop rare-view rescues. Protect single-view pairs that score high.

**B3. DeepBlocker**

- **Source.** Thirumuruganathan et al., PVLDB 14(11), 2021. `qcri/DeepBlocker`.
- **Core idea.** A design-space study of deep blocking. Autoencoder self-supervised tuple embeddings plus nearest-neighbor search worked best on structured and dirty data, and a hybrid worked best on text.
- **Relevance.**
  - Sparkly, from the same group, reports stronger results with top-k TF/IDF.
  - With abundant labels, supervised contrastive training (B4) dominates self-supervised autoencoders.
- **Compute.** GPU.
- **Data needs.** None.
- **License.** BSD-3-Clause.
- **Implementation.** 6 h or more.
- **Expected upside.** About 0 beyond B1/B4.
- **Risks.** Time.

**B4. Sentence-BERT blocking / SC-Block**

- **Source.** Reimers and Gurevych, EMNLP 2019 (siamese bi-encoders). Brinkmann, Shraga and Bizer, ESWC 2024 (arXiv 2303.03132), `wbsg-uni-mannheim/SC-Block`.
- **Core idea.**
  - Supervised contrastive learning places records of the same entity close together in embedding space.
  - Nearest-neighbor search builds the candidate set.
  - The paper reports smaller candidate sets and 1.5-8 times faster end-to-end pipelines at equal F1 compared with eight blockers.
- **Relevance.**
  - Aimed at the retrieval bucket (0.0149): India S3 misses, cross-script targets and France word-order or abbreviation variants that char3 misses.
  - Also yields a bi-encoder cosine feature and dense competitor search. The friend's TC pipeline used both.
  - Positives come from S1 groups: all targets owned by one S1, plus the S1 itself.
- **Compute.**
  - One GPU for training: about 20-40 min for mE5-small on 400k anchors (estimate).
  - Encoding about 14M texts: about 20-40 min on A10G (estimate).
  - FAISS search per country.
- **Data needs.** Positive links (plentiful) and hard negatives from the lexical retriever.
- **License.** sentence-transformers is Apache-2.0. SC-Block code is BSD-3-Clause. Model: mE5-small is MIT.
- **Implementation.** About 3 h of engineer time plus GPU time.
- **Expected upside.** +0.001 to +0.004.
- **Risks.**
  - Dense neighbors of short, generic names are noisy, so the stage-2 scorer must gate them.
  - The index must be rebuilt per model version.
  - GPU quota.

**B5. Pre-trained embeddings for ER (experimental analysis)**

- **Source.** Zeakis, Papadakis, Skoutas and Koubarakis, PVLDB 16(9), 2023. Extended in VLDB Journal 2024.
- **Core idea.** Benchmarks 12 language models for vectorization cost, blocking and matching on 17 datasets.
- **Relevance.** Supporting evidence that sentence-transformer-style models are the cost-effective embedding family for blocking. Their overhead matters at 10M+ records, so choose small models.
- **Use.** Evidence only.

### C. Pre-trained language model matchers

**C1. DeepMatcher**

- **Source.** Mudgal et al., SIGMOD 2018. `anhaidgroup/deepmatcher`.
- **Core idea.** An attribute-level RNN/attention design space for matching.
- **Relevance.** Transformer cross-encoders superseded it.
- **License.** BSD-3-Clause.
- **Expected upside.** About 0.
- **Decision.** Skip.

**C2. Ditto (with Rotom augmentation)**

- **Source.** Li et al., PVLDB 14(1), 2020, `megagonlabs/ditto`. Miao, Li and Wang, SIGMOD 2021 (Rotom), `megagonlabs/rotom`.
- **Core idea.**
  - Serialize each record as `[COL] attr [VAL] value ...` and fine-tune a pre-trained LM as a sequence-pair classifier.
  - Three optimizations: (1) domain-knowledge injection, which tags spans such as numbers and identifiers; (2) summarization, which keeps high-TF-IDF tokens to fit max length; (3) data augmentation, such as span deletion, attribute shuffling and entry swap.
  - Rotom meta-learns how to combine augmentation operators.
- **Relevance.**
  - Our E5 cross-encoder is already Ditto-shaped.
  - The transferable, cheap tricks: (a) append normalized digit tokens (house number, postcode, unit) to each side, so the CE sees numeric conflicts explicitly. Our largest FP driver is similar names with different numbers. (b) Entry-swap augmentation, since pair order must not matter. (c) Attribute-dropout augmentation for records with a missing address.
  - Summarization is not needed because records are short.
- **Compute.** Same as R4.
- **Data needs.** Existing pairs.
- **License.** Apache-2.0 (Ditto); BSD-3-Clause (Rotom).
- **Implementation.** About 1 h inside R4.
- **Expected upside.** +0 to +0.001 over R4 without tags.
- **Risks.** A tag format mismatch between training and inference silently degrades scores. Unit-test the serializer.

**C3. HierGAT**

- **Source.** Yao, Gu, Cong, Jin and Lv, SIGMOD 2022. `CGCL-codes/HierGAT`.
- **Core idea.** A hierarchical graph attention transformer over token, attribute and entity levels. It makes collective decisions across related pairs and reports up to +8.7 F1 over Ditto on its benchmarks.
- **Relevance.**
  - Collective decisions among the candidates of one S1 and the competing owners of one target match our structure.
  - Our GBDT stage 2 with sibling-corroboration and ownership features captures much of this cheaply.
- **Compute.** GPU. A graph per S1 neighborhood is slow at 20M pairs.
- **Data needs.** Labels.
- **License.** MIT.
- **Implementation.** 10 h or more.
- **Expected upside.** +0 to +0.002.
- **Risks.** Engineering time and inference cost. Skip in the sprint.

**C4. JointBERT (dual-objective fine-tuning)**

- **Source.** Peeters and Bizer, PVLDB 14(10), 2021. `wbsg-uni-mannheim/jointbert`.
- **Core idea.** Train the binary match head and a multi-class entity-ID head together.
- **Relevance.**
  - The paper reports gains only for entities seen in training. It underperforms on unseen entities.
  - All test S1 are unseen entities, so the entity-ID head does not transfer.
- **License.** BSD-3-Clause.
- **Expected upside.** About 0.
- **Decision.** Skip.

**C5. R-SupCon (supervised contrastive pre-training for matching)**

- **Source.** Peeters and Bizer, WWW 2022 Companion. `wbsg-uni-mannheim/contrastive-product-matching`.
- **Core idea.**
  - Pre-train the encoder with the SupCon loss: all records sharing an entity ID are positives, other batch members are negatives. Then fine-tune a pair classifier.
  - Source-aware sampling avoids inter-source label noise.
- **Relevance.**
  - The S1-owned target group is our entity ID, so SupCon fits R5 directly: several positives per anchor inside one batch.
  - Source-aware sampling corresponds to not mixing duplicate-text targets of other owners into one batch.
- **Compute.** GPU.
- **Data needs.** Entity groups (we have them).
- **License.** BSD-3-Clause.
- **Implementation.** Inside R5.
- **Expected upside.** Inside R5.
- **Risks.** Batches need several same-entity members, which calls for a custom sampler.

**C6. Sudowoodo**

- **Source.** Wang, Li and Wang, ICDE 2023. `megagonlabs/sudowoodo`.
- **Core idea.** Self-supervised contrastive representation learning with augmentation for multiple data-integration tasks. Few labels are needed.
- **Relevance.** Designed for label scarcity, which we do not have.
- **License.** BSD-3-Clause.
- **Expected upside.** About 0.
- **Decision.** Skip.

**C7. Unicorn**

- **Source.** Tu et al., SIGMOD 2023. `ruc-datalab/Unicorn`.
- **Core idea.** One multi-task encoder plus mixture-of-experts matcher across seven matching tasks, with zero-shot transfer.
- **Relevance.** Multi-task breadth brings no benefit for a single task with abundant labels.
- **License.** The repository declares no license, so treat the code as unusable.
- **Expected upside.** About 0.
- **Decision.** Exclude.

**C8. DADER (domain adaptation for deep ER)**

- **Source.** Tu et al., SIGMOD 2022; DADER demo, PVLDB 15(12).
- **Core idea.** A feature extractor, a matcher and a feature aligner (for example MMD or adversarial alignment). Features learned on a labeled source domain are aligned to an unlabeled target domain.
- **Relevance.**
  - France is an unlabeled target domain.
  - A full adversarial alignment is too costly and unsafe in a sprint, because its effect on France cannot be validated.
  - Cheap proxy: leave-one-country-out stage-2 validation (train US, test India, and the reverse), plus a variant without country-specific features. This measures transfer robustness before public spend.
- **Compute.** CPU for the proxy.
- **Data needs.** Unlabeled France text for full DADER.
- **License.** Not checked.
- **Implementation.** 2 h for the proxy.
- **Expected upside.** 0 to +0.003 public. Mostly risk reduction.
- **Risks.** Using France test text to adapt a model is the same unresolved rules question as test-fitted IDF.

### D. Cross-lingual and character-level models

License and parameter data come from Hugging Face API metadata on 2026-09-26. Parameters are safetensors totals where reported.

| Model id | License | Parameters | Revision (head, 12 chars) | Role for us |
|---|---|---:|---|---|
| intfloat/multilingual-e5-small | MIT | 117.65M | 614241f622f5 | Current CE (EXP-050); R2, R5 bi-encoder |
| intfloat/multilingual-e5-base | MIT | 278.04M | d12875059715 | R4 primary CE |
| intfloat/multilingual-e5-large | MIT | 559.89M | 3d7cfbdacd47 | L40S-only CE option; slow inference |
| sentence-transformers/LaBSE | Apache-2.0 | 470.93M | 836121a0533e | Bi-encoder backup (translation-pair pre-training) |
| FacebookAI/xlm-roberta-base / -large | MIT | 278.89M / 561.19M | e73636d4f797 / c23d21b0620b | Raw MLM starts |
| microsoft/mdeberta-v3-base | MIT | about 276M (86M backbone + 190M embeddings, model card) | a0484667b223 | CE ablation |
| jhu-clsp/mmBERT-small / -base | MIT | about 140M / 307M (model card) | abc32620dd4f / c5955035435e | Fast ModernBERT-style encoder; needs recent transformers |
| google/canine-s / canine-c | Apache-2.0 | 132.10M | 75d6d0b3f4d0 / dc0eaffdff3f | Character-level, tokenizer-free |
| google/byt5-small / -base | Apache-2.0 | about 300M / 580M (paper) | 68377bdc18a2 / 92d8c008d55c | Byte-level encoder-decoder |
| Alibaba-NLP/gte-multilingual-base | Apache-2.0 | 305.37M | 9bbca17d9273 | Bi-encoder alternative (custom code) |
| Alibaba-NLP/gte-multilingual-reranker-base | Apache-2.0 | 305.96M | 8215cf04918b | CE alternative; rules caveat |
| BAAI/bge-m3 | MIT | about 568M | 5617a9f61b02 | Dense/sparse/multi-vector; heavy |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | 567.76M | 953dc6f6f85a | Strong CE start; rules caveat; L40S |
| Qwen/Qwen3-Reranker-0.6B | Apache-2.0 | 595.78M | e61197ed4502 | Decoder reranker; slower per pair; rules caveat |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | Apache-2.0 | 117.64M | 1427fd652930 | EXP-045 base; upstream base license unverified |
| jinaai/jina-reranker-v2-base-multilingual | CC-BY-NC-4.0 | 278.44M | 9cfeff2df7d4 | **Excluded** (license) |
| jinaai/jina-colbert-v2 | CC-BY-NC-4.0 | 559.50M | a9dc5cd7293d | **Excluded** (license) |

**D1. LaBSE**

- **Source.** Feng et al., ACL 2022 (arXiv 2020).
- **Core idea.** A dual-encoder BERT trained on translation ranking with additive-margin softmax, covering 109 languages.
- **Relevance.**
  - Strong for translation-equivalent sentences.
  - Our cross-script cases are transliteration rather than translation, and a 471M model costs about 4 times mE5-small to encode 14M texts.
- **Compute.** GPU.
- **Data needs.** Positive pairs for fine-tuning.
- **License.** Apache-2.0.
- **Implementation.** 3 h as a swap-in for R5.
- **Expected upside.** Uncertain versus mE5.
- **Risks.** Slower. Its 501k-token vocabulary inflates memory.

**D2. multilingual-E5 (small/base/large)**

- **Source.** Wang et al., technical report 2024. Models from `microsoft/unilm` (MIT).
- **Core idea.** Contrastive pre-training on about 1B multilingual pairs, then supervised fine-tuning. Use the "query: " prefix on both sides for symmetric tasks.
- **Relevance.** Already our CE base, with the XLM-R tokenizer shared across sizes. Base is the natural next step for the CE (R4). Small is the right bi-encoder for throughput (R5).
- **Compute.** Base is roughly 2.4 times small's parameters. Expect about 2-3 times slower inference (estimate; benchmark first).
- **Data needs.** Our pairs.
- **License.** MIT.
- **Implementation.** In place.
- **Expected upside.** See R4/R5.
- **Risks.** Base inference cost on 20.8M test pairs.

**D3. mDeBERTa-v3-base**

- **Source.** He et al., 2021. `microsoft/DeBERTa`.
- **Core idea.** Disentangled attention with ELECTRA-style replaced-token pre-training on CC100. Strong on classification fine-tuning.
- **Relevance.**
  - A plausible CE backbone.
  - It has no retrieval pre-training, and disentangled attention is slower than BERT attention.
  - Worth an ablation only if R4 is saturated.
- **Compute.** GPU.
- **Data needs.** Our pairs.
- **License.** MIT.
- **Implementation.** 3 h.
- **Expected upside.** Uncertain.
- **Risks.** fp16 instability has been reported for DeBERTa-v3. Prefer bf16.

**D4. XLM-R base/large**

- **Source.** Conneau et al., 2019/2020. MIT via fairseq.
- **Core idea.** Multilingual masked-LM encoder.
- **Relevance.** mE5 is XLM-R further contrastively tuned, so it is usually a better starting point for similarity tasks.
- **License.** MIT.
- **Decision.** Backup only.

**D5. mmBERT**

- **Source.** JHU CLSP, 2025. ModernBERT architecture trained on 3T tokens covering 1,833 languages.
- **Core idea.** Modern encoder with FlashAttention 2 and unpadding. The authors report up to 4 times faster than older multilingual encoders.
- **Relevance.** A potential speed win for CE inference over 20.8M pairs.
- **Compute.** GPU, fast if the environment supports it.
- **Data needs.** Our pairs.
- **License.** MIT (model cards).
- **Implementation.** About 4 h plus environment risk: it needs a recent transformers and ideally flash-attn inside the SageMaker container.
- **Expected upside.** Mainly throughput.
- **Risks.** Container build time. Unproven on our data.

**D6. CANINE**

- **Source.** Clark et al., TACL 2022.
- **Core idea.** Tokenizer-free encoder over Unicode code points, with downsampling to keep sequences short.
- **Relevance.**
  - Character-level input suits misspellings and script variation.
  - Mixed Latin and Indic scripts are still different code points, so it does not transliterate by itself.
- **Compute.** GPU, moderate.
- **Data needs.** Pairs.
- **License.** Apache-2.0.
- **Implementation.** 6-8 h, including a new CE head and inference path.
- **Expected upside.** +0 to +0.002, cross-script slices only.
- **Risks.** Weaker general pre-training than mE5. Late in the sprint.

**D7. ByT5**

- **Source.** Xue et al., TACL 2022. `google-research/byt5`.
- **Core idea.** A byte-level T5, robust to noise and spelling variation.
- **Relevance.**
  - Robust to noisy text.
  - Indic UTF-8 costs 3 bytes per character, so sequences get long and slow. An encoder-only use needs a custom head.
- **Compute.** GPU, heavy.
- **Data needs.** Pairs.
- **License.** Apache-2.0.
- **Implementation.** 6-8 h.
- **Expected upside.** +0 to +0.002.
- **Risks.** Throughput.

**D8. Transliteration models (IndicXlit)**

- **Source.** Madhani et al., Findings of EMNLP 2023 (Aksharantar). `AI4Bharat/IndicXlit`.
- **Core idea.** A multilingual transformer transliterating between Roman script and 21 Indic languages.
- **Relevance.**
  - Could produce a native-to-roman view of Indic-script targets for retrieval and features.
  - Only 116 of 581 analyzed misses had non-ASCII target addresses (docs/MISSED_LINK_ANALYSIS.md), so most India misses are not cross-script.
- **Compute.** GPU or CPU seq2seq, slow for millions of strings. Cache per unique token.
- **Data needs.** None.
- **License.** MIT for code and models. The training data are external (Aksharantar, CC-BY / CC0), so this is a rules caveat as a pretrained model.
- **Implementation.** 4-6 h.
- **Expected upside.** +0 to +0.002 on the India S3 slice.
- **Risks.** Collisions. Relevance is limited given the miss mix.

**D9. Retrieval-tuned multilingual rerankers**

- **Source.** BAAI bge-reranker-v2-m3; Alibaba gte-multilingual-reranker-base; Qwen3-Reranker-0.6B (2025); mMARCO mMiniLMv2 cross-encoder.
- **Core idea.** Cross-encoders already fine-tuned for query-passage relevance, then fine-tuned further on our pairs.
- **Relevance.**
  - A better starting point for a short-text pair head than a raw encoder.
  - EXP-045, which started from mmarco, already measured gains.
  - All four are Apache-2.0 and at most 0.6B.
  - Each was supervised-fine-tuned by its publisher on external relevance data (rules caveat).
  - gte requires `trust_remote_code`, so the custom code must be pinned and audited.
  - Qwen3-Reranker is a decoder scored through yes/no logits and is slower per pair.
- **Compute.** GPU. The 568M models need about 4-5 times the inference of mE5-small.
- **Data needs.** Our pairs.
- **License.** Apache-2.0.
- **Implementation.** 3 h as an alternative checkpoint in R4.
- **Expected upside.** Uncertain versus mE5-base.
- **Risks.** Rules caveat and inference cost.

### E. Ranking and training recipes

**E1. Cross-encoder reranker fine-tuning (sentence-transformers v4 recipe)**

- **Source.** Hugging Face blog "Training and Finetuning Reranker Models with Sentence Transformers" (2025) and the sentence-transformers cross-encoder loss reference.
- **Core idea.**
  - Mine hard negatives from a retriever with `mine_hard_negatives`. The blog uses 5 negatives per positive, a range up to rank 100, `max_score` 0.8, a margin, and "top" sampling.
  - Train `BinaryCrossEntropyLoss` with `pos_weight` equal to the number of negatives.
  - Blog hyperparameters: batch 16, LR 2e-5, 1 epoch, warmup 0.1, bf16.
  - Use a reranking evaluator with `load_best_model_at_end`, because cross-encoders overfit quickly.
- **Guidance quoted from the blog.**
  - Plain BCE "remains a very strong option" against learning-to-rank losses.
  - Training only on hard negatives can hurt easier cases. Mixing in random negatives mitigates this.
  - The loss reference says LambdaLoss "anecdotally" performs best among its listwise losses.
- **Relevance.** This is the recipe for R4, adapted with negatives from our actual stage-1 retriever and model.
- **License.** Apache-2.0.

**E2. Localized Contrastive Estimation (LCE)**

- **Source.** Gao, Dai and Callan, ECIR 2021. `luyug/Reranker`.
- **Core idea.**
  - Train rerankers with a softmax over one positive plus a group of hard negatives, sampled from the same first-stage retriever used at inference.
  - The authors show that standard rerankers fail to exploit better retrievers unless trained this way.
- **Relevance.**
  - Our negatives should come from the char3 top-100 plus stage-1 model rank, not random targets.
  - Many S1 have several positives, so multi-positive softmax, ListNet or LambdaLoss is the listwise form.
  - Our end task is set thresholding, which needs calibrated pointwise scores. Treat LCE as an ablation behind BCE and let the stacker recalibrate.
- **Compute.** GPU, same as E1.
- **Data needs.** Groups of 1 positive plus 7-15 negatives.
- **License.** Apache-2.0.
- **Implementation.** 1-2 h as an R4 variant.
- **Expected upside.** +0 to +0.002 over BCE.
- **Risks.** Raw scores are less calibrated.

**E3. Hard-negative mining with denoising (RocketQA)**

- **Source.** Qu et al., NAACL 2021.
- **Core idea.** Hard negatives often contain unlabeled positives. Filter them with a stronger cross-encoder before training the bi-encoder. RocketQA also adds cross-batch negatives and data augmentation.
- **Relevance.**
  - Labels are near-complete, but duplicate-text targets and exact-name different-location branches create label-ambiguous negatives.
  - Drop negatives whose normalized name and address equal a positive of the same S1, or that EXP-050 scores at p >= 0.98.
- **Implementation.** 1 h.
- **Expected upside.** Enables R4/R5.
- **Risks.** Over-filtering removes exactly the confusable negatives that drive false positives. Filter only near-exact duplicates.

**E4. In-batch negatives (MultipleNegativesRankingLoss) and GradCache**

- **Source.** Sentence-BERT (EMNLP 2019). Gao et al., RepL4NLP 2021 (GradCache), `luyug/GradCache`, which ships as `CachedMultipleNegativesRankingLoss` in sentence-transformers.
- **Core idea.**
  - A softmax over all other in-batch positives acts as negatives.
  - GradCache decouples the contrastive loss from the encoder backward pass, so batch size is limited by compute rather than GPU memory.
- **Relevance.**
  - R5 needs large, same-country batches (512-1024) on 24GB GPUs.
  - Build batches country-homogeneous, so in-batch negatives are plausible confusers.
  - Use a no-duplicates batch sampler so the same S1 or a duplicate text never appears twice in one batch.
- **License.** Apache-2.0.
- **Implementation.** Inside R5.

**E5. Margin-MSE cross-architecture distillation**

- **Source.** Hofstätter et al., arXiv 2010.02666 (2020). `sebastian-hofstaetter/neural-ranking-kd`.
- **Core idea.** Train an efficient bi-encoder to match the teacher cross-encoder's score margin between a positive and a negative. Scale-invariant across architectures.
- **Relevance.** After R4 exists, distill its margins into the R5 bi-encoder to improve dense rescue precision.
- **Compute.** GPU. Needs teacher scores for the training triplets.
- **License.** Apache-2.0.
- **Implementation.** 4 h.
- **Expected upside.** +0 to +0.002.
- **Risks.** Depends on R4 and R5 both finishing. Post-sprint.

**E6. ColBERT late interaction**

- **Source.** Khattab and Zaharia, SIGIR 2020; ColBERTv2, NAACL 2022. `stanford-futuredata/ColBERT` (MIT).
- **Core idea.** Per-token embeddings with MaxSim scoring, giving near cross-encoder quality at retrieval cost.
- **Relevance.**
  - Storage for 10M targets times about 30 tokens is large.
  - The eligible multilingual checkpoint (colbert-xm) is 853M.
  - jina-colbert-v2 is CC-BY-NC and excluded.
- **License.** MIT.
- **Expected upside.** About 0 in the sprint.
- **Decision.** Skip.

**E7. LightGBM lambdarank and S1-context meta-features**

- **Source.** LightGBM documentation (`objective=lambdarank`, groups by query, `lambdarank_truncation_level`). The same library as stage 2 (MIT).
- **Core idea.** A listwise GBDT objective over each S1's candidates.
- **Relevance.**
  - Stage 2 is pointwise binary.
  - A cheap companion adds S1-context features: candidate count, max/sum/second probability, gap to best, count of candidates above thresholds, number of same-name siblings.
  - Alternatively, a lambdarank score can be stacked as one more feature.
- **Compute.** CPU.
- **Data needs.** The 194k OOF.
- **License.** MIT.
- **Implementation.** 2 h.
- **Expected upside.** +0.0003 to +0.0015.
- **Risks.** Lambdarank scores are not probabilities. They are only useful as stacked features.

### F. Set decisions and structural constraints

**F1. Expected-F decision rule**

- **Source.**
  - Ye, Chai, Lee and Chieu, ICML 2012, "Optimizing F-measures: a tale of two approaches".
  - Dembczynski, Waegeman, Cheng and Hüllermeier, NeurIPS 2011, "An exact algorithm for F-measure maximization" (GFM).
  - Waegeman et al., JMLR 2014.
- **Core idea.**
  - Predict the label subset that maximizes expected F given calibrated marginals, instead of thresholding each pair.
  - Under independence the optimum is a top-k prefix of the probability-sorted list, found in O(n²) with a Poisson-binomial dynamic program.
  - Ye et al. find the decision-theoretic approach better for rare classes and domain adaptation, while the empirical-threshold approach is more robust to model misspecification.
- **Relevance.**
  - Our metric is a per-S1 set F0.5, so this is the Bayes-style decision for it (section 1.1).
  - It targets the rejected, FP and singleton buckets together.
  - Its domain-shift robustness is relevant to France, where no threshold can be tuned on labels.
- **Compute.** Trivial CPU: at most 13 k-values times a 13x13 DP per S1.
- **Data needs.** Calibrated OOF probabilities (the 194k stage-2 OOF exists).
- **License.** Own code.
- **Implementation.** 2 h.
- **Expected upside.** +0.001 to +0.004.
- **Risks.**
  - Sibling links of one S1 are correlated, not independent.
  - Calibration drift on France.
  - Mitigations: tune a logit temperature and a minimum-gain margin on inner folds, and compare against the global threshold on the fixed 6k with the nested protocol.

**F2. One-owner assignment, unique-mapping clustering and reverse competitor search**

- **Source.**
  - Papadakis et al., "Bipartite graph matching algorithms for clean-clean ER", EDBT 2022 (arXiv 2112.14030), and the VLDB Journal 2023 follow-up. Unique Mapping Clustering sorts edges by weight and accepts an edge when both endpoints are free.
  - Auto-FuzzyJoin (A6) for the reference-table view.
  - The friend's target-centric best-owner, audited in docs/FRIEND_METHOD_DIFF.md, measured +0.00604 over its same-model pair rule on its own population.
- **Core idea.**
  - Here the constraint is one-to-many: an S1 may own many targets, and a target has at most one owner.
  - Resolve each target by choosing its best owner among all plausible S1, not only among the S1 that retrieved it.
  - Plausible owners are found by reverse top-k search (target to S1) in each lexical view and optionally the dense view.
- **Relevance.**
  - SUB-003/004 already apply ownership among retrieved pairs (44,165 removals in SUB-004).
  - The missing piece is competitors that never retrieved the target.
  - A soft variant feeds competitor evidence into stage 2 as features instead of hard deletion: best competitor probability, margin to it, this S1's rank among the target's owners.
- **Compute.** CPU. Reverse top-10 per view for targets with stage-2 p >= 0.02.
- **Data needs.** OOF scores.
- **License.** Own code.
- **Implementation.** 3-4 h.
- **Expected upside.** +0.0005 to +0.003 incremental over current ownership.
- **Risks.**
  - Competitor pairs need features. Rank-free features are required, because ranks are undefined for reverse-found pairs.
  - A hard rule can delete true links when two S1 are themselves duplicates of each other.

**F3. Collective ER**

- **Source.** Bhattacharya and Getoor, TKDD 2007.
- **Core idea.** Resolve co-occurring references jointly with relational clustering.
- **Relevance.**
  - Our relational signal is the S1 to many-target star, plus duplicate targets.
  - Duplicate expansion (identical-text targets inherit decisions) and sibling corroboration already implement the cheap part.
- **Compute.** Heavy for full relational clustering.
- **Data needs.** Relational structure.
- **Expected upside.** Small beyond current features.
- **Decision.** Skip.

### G. LLM-based matching (reference only)

- **Sources.**
  - Peeters, Steiner and Bizer, "Entity matching using large language models", EDBT 2025.
  - Steiner, Peeters and Bizer, "Fine-tuning large language models for entity matching" (arXiv 2409.08185; ICDE Workshops 2025). LoRA r = 64, alpha = 16, dropout 0.1, LR 2e-4, 10 epochs. Fine-tuning helped smaller models substantially, while larger models gave mixed results.
  - Zhang et al., AnyMatch (arXiv 2409.04073): a fine-tuned GPT-2-size model competitive in zero-shot EM.
  - Wang et al., COLING 2025, "Match, Compare, or Select?" (ComEM): choosing among candidates is better than independent binary matching, and a compound pipeline is cost-effective.
- **Relevance.**
  - The select-among-candidates framing matches our one-owner structure.
  - Throughput at our scale does not work: 20.8M test pairs at tens to low hundreds of pairs/s per GPU.
  - Only an ambiguous band of about 1-3% of pairs could be routed to an LLM.
- **License and size gates.**
  - Eligible bases (HF metadata): Qwen/Qwen2.5-7B-Instruct (Apache-2.0, 7.62B); Qwen/Qwen3-4B (Apache-2.0, 4.02B); Qwen/Qwen3-1.7B (Apache-2.0, 2.03B); microsoft/Phi-4-mini-instruct (MIT, 3.84B); mistralai/Mistral-7B-Instruct-v0.3 (Apache-2.0, 7.25B).
  - Qwen/Qwen3-8B is Apache-2.0 but reports 8.19B total parameters, which exceeds a strict 8B reading.
  - Jellyfish-7B/8B are CC-BY-NC-4.0 and excluded.
  - Llama and Gemma family licenses are neither MIT nor Apache-2.0.
  - A 7B model plus our encoders would nearly exhaust the <=8B combined-parameter budget.
- **Compute.** LoRA training takes 2-4 GPU-h on L40S. Band inference is 1-2 h.
- **Implementation.** 10 h or more, including prompt, serializer, calibration and stacking.
- **Expected upside.** Uncertain.
- **Risks.** Budget, parameter limit and calibration. Not a 12-hour item.

## 4. Ranking by expected gain per wall-clock hour (12-hour sprint)

Assumptions:

- One agent's critical path, with cloud GPU and CPU jobs running in parallel.
- "Critical h" is engineer attention plus blocking waits. "Elapsed h" includes background compute.
- Gains are fixed-6k macro hypotheses unless marked public. Rows that touch the same bucket are **not additive**.
- Confidence levels:
  - H: a direct local measurement exists.
  - M: the mechanism is well supported and cheap to measure locally.
  - L: unmeasurable locally, or high variance.

| Rank | Item (as applied to our stack) | Bucket | Expected gain (hypothesis) | Critical h | Elapsed h | Gain per critical h (midpoint) | Confidence |
|---:|---|---|---|---:|---:|---:|---|
| 1 | R1: score test top-12 with the existing EXP-050 E5-small CE and apply the CL-013 stack | rejected, FP | +0.0015 to +0.0025 (+0.0024 measured on 6k) | 1.5 | 3 | about 0.0013 | H for India/US, L for France |
| 2 | R2: expected-F0.5 per-S1 set decision with calibration and S1-empty model | rejected, FP, singleton | +0.001 to +0.004 | 2 | 2 | about 0.0012 | M |
| 3 | R3: France/India multi-view sparse rescue (B1+B2) with reverse-owner search (F2) | retrieval, France | +0.002 to +0.008 public; +0 to +0.002 local | 3 | 5 | about 0.0012 (wide) | L |
| 4 | R4: CE v2 on multilingual-E5-base, BCE with retriever hard negatives and numeric tags (E1, E3, C2) | rejected, FP, sidecar | +0.001 to +0.004 over rank 1 | 2.5 | 7 | about 0.0010 | M-L |
| 5 | R5: fine-tuned E5-small bi-encoder: dense rescue, cosine feature, dense competitor search (B4, C5, E4) | retrieval, FP | +0.001 to +0.004 | 3 | 7 | about 0.0008 | L-M |
| 6 | Stage-2 candidate depth sweep, top-12 to top-20/30 (skip if already measured) | outside-top-12 | 0 to +0.002 | 2 | 3 | about 0.0005 | M |
| 7 | Soft ownership features in stage 2 (competitor max p, margin, owner rank) | FP | +0.0005 to +0.002 | 2.5 | 3 | about 0.0005 | M |
| 8 | Leave-one-country-out stage-2 check and country-agnostic variant (C8 proxy) | France robustness | 0 to +0.003 public | 2 | 2 | about 0.0004 (risk reduction) | L |
| 9 | LightGBM lambdarank score plus S1-context features (E7) | sidecar, rejected | +0.0003 to +0.0015 | 2 | 2 | about 0.0004 | M |
| 10 | LCE / LambdaLoss CE ablation (E2), conditional on R4 infrastructure | rejected | 0 to +0.002 over BCE | 1 | 3 | conditional | L |
| 11 | Margin-MSE distillation CE to bi-encoder (E5) | retrieval | 0 to +0.002 | 3 | 5 | about 0.0003 | L |
| 12 | Learned edit distance / char-alignment features (A5) | India, France | 0 to +0.002 | 4.5 | 5 | about 0.0002 | L |
| 13 | Transliteration view with IndicXlit (D8) | India S3 retrieval | 0 to +0.002 | 5 | 6 | about 0.0002 | L, rules caveat |
| 14 | Char/byte CE with CANINE-s or ByT5-small (D6, D7) | cross-script | 0 to +0.002 | 6 | 10 | about 0.0001 | L |
| 15 | Fellegi-Sunter / Splink weights, France EM (A1) | France calibration | 0 to +0.001 | 3.5 | 4 | about 0.0001 | L |
| 16 | HierGAT or other collective GNN (C3, F3) | FP | 0 to +0.002 | 10 | 12+ | under 0.0001 | L |
| 17 | LLM <=8B listwise select on ambiguous band (G) | rejected, FP | uncertain | 10 | 14+ | under 0.0001 | L |
| 18 | DeepMatcher, Magellan, DeepBlocker, dedupe, Zingg, Unicorn, Sudowoodo, JointBERT, Auto-FuzzyJoin code | none | about 0 | n/a | n/a | about 0 | n/a |

How the numbers were set:

- Rank 1 reuses the CL-013 local measurement, discounted for France (15% of test) and for a production stacker trained on few S1.
- Rank 2 bounds F1 by the 0.0162 combined rejected, FP and singleton buckets. Decision-rule changes in the literature typically recover a small fraction of such losses.
- Rank 3 uses the public-local gap: 0.944 public implies France near 0.85 if India/US transfer at local level. A +0.02 to +0.05 France improvement is worth +0.003 to +0.0075 overall.
- Ranks 4-5 assume a larger or better-trained CE, or dense rescue, recovers 10-30% of the relevant bucket.

## 5. Recommended for this sprint (top 5)

Ordered by expected gain per critical-path hour. Items 1, 4 and 5 are single-GPU jobs. Items 2 and 3 are CPU jobs that run while the GPU jobs train or score.

Instance guidance (estimates to benchmark in the first 200-300 steps; abort if the projected finish passes the stop rule below):

- **ml.g5.2xlarge** (1x A10G 24GB, 8 vCPU, 32 GiB) is the default. The 8 vCPUs matter for tokenization.
- **ml.g6.2xlarge** (1x L4 24GB) is a fallback. L4 has lower memory bandwidth than A10G, so expect similar or slower throughput for these small models.
- **ml.g6e.xlarge/2xlarge** (1x L40S 48GB) is only for 560M-class models or larger batches.
- GPU quota: docs/AWS_CAPACITY.md records SageMaker GPU training quota as pending. EC2 G5 was used for EXP-045 and EXP-050. Use whichever path has approved quota.
- For every job:
  - Stage pinned model snapshots to S3 with SHA256.
  - Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
  - Pin sentence-transformers (Apache-2.0) and transformers (Apache-2.0) versions.
  - Use bf16 (A10G, L4 and L40S all support it).
  - Write checkpoints and receipts to the run's S3 prefix.
  - Set a hard `max_run`.

Combined learned parameters of all recommended models: about 118M (EXP-050 CE) + 278M (R4) + 118M (R5) + LightGBM. That is well under 8B, even with the EXP-045 118M CE retained.

### R1 (GPU inference, rank 1): deploy the measured EXP-050 E5 stack on test

1. **Model.** The EXP-050 `P5-E5-18K-001` checkpoint, fine-tuned from `intfloat/multilingual-e5-small` at revision 614241f622f5 (MIT). Verify its SHA256 against the training receipt first. docs/MODEL_LICENSES.md records the trained output as not yet inspected locally.
2. **Inputs.** Unchanged from training: raw NFC `name | address` against `s2:` or `s3:` plus raw target text, tokenizer pair, `max_length=128`. Use sigmoid probability, not the logit, where the stacker expects probability.
3. **Pairs.** Exactly the SUB-003 test top-12 sidecar (at most 20.8M pairs, sharded as the existing 192 shards), plus any France rescue pairs that will be submitted.
4. **Inference settings.**
   - fp16 or bf16 autocast, batch 1024.
   - Length-bucketed batches, DataLoader `num_workers=6`.
   - At the measured 3,350 pairs/s for a same-size CE on G5, 20.8M pairs take about 1.7 GPU-h. With 2-4 parallel single-GPU jobs, wall time is under 1 h.
5. **Stacker.**
   - The CL-013 LightGBM (5 features: logit of stage-2 p, E5 logit, and three stage-2 context columns; 300 trees, 15 leaves, LR 0.05, `min_child_samples` 40).
   - Before test use, score an extra 20-40k held fold 2/3 S1 (about 0.5M pairs, minutes of GPU). Train the production stacker on that larger set instead of only 6k.
   - Stacker training S1 must be disjoint from EXP-050 training S1 (fold1) and from fixed-6k evaluation S1.
   - Choose the threshold, or feed R1, with the nested inner-fold procedure.
6. **Gates.**
   - Reproduce +0.0024 on the fixed 6k with the production stacker.
   - Per-country slices must not regress by more than 0.001.
   - Official validator default and strict checks must PASS.
   - France diagnostic: compare the score distribution to India/US. Report only; no tuning.

### R2 (CPU, rank 2): expected-F0.5 per-S1 set decision

1. **Calibrate.** Fit isotonic regression per country on the 194k stage-2 (or stacked) OOF probabilities, using folds other than the evaluated one. Report reliability and ECE.
2. **S1-empty model.**
   - LightGBM on S1-level features: max p, sum p, second p, count of p above 0.5, candidate count, top1-top2 gap, name/address length, country.
   - Target: truth is empty.
   - Output: q0 = P(T is empty).
3. **Outside-candidate mass.** Per-country mean number of true links outside the scored top-12 (mu_out), from OOF. Optionally regress mu_out on S1 features.
4. **Decision.**
   - Sort each S1's candidates by p. For k = 1..min(K, 12), compute E[1.25·X_in / (0.25·(X_in + X_rest + O) + k)], where:
     - X_in is Poisson-binomial over the top-k;
     - X_rest is Poisson-binomial over the rest;
     - O is Poisson(mu_out).
     - A 13x13 dynamic program plus a short Poisson sum is enough.
   - Empty-set value: P(T is empty), blended from q0 and the independence estimate.
   - Pick the argmax. Accept k over empty only when the expected gain exceeds a margin delta.
5. **Tune on inner folds only.**
   - Logit temperature tau in {0.7, 0.85, 1.0, 1.2}, which absorbs sibling correlation.
   - delta in {0, 0.005, 0.01, 0.02}.
   - Test order both ways: ownership then decision, and decision then ownership.
6. **Evaluate.** Fixed 6k, nested, against the current global threshold. Report macro, precision, recall, singleton and per-country figures.
7. **Cost.** Seconds to minutes of CPU for all 1.733M test S1. Apply after R1's stack when available, and re-run on whatever final probability is chosen.

### R3 (CPU fleet, rank 3): France/India multi-view sparse rescue with reverse-owner search

This continues CL-012 and the P5-RESCUE-F1..F4 routes. It is conditional on SUB-004's public result confirming the France hypothesis.

1. **Views** per country (Sparkly/JedAI ideas):
   - (a) char3 BM25 over NFC-lowercased name and over address, with k1 = 1.2 and b in {0.3, 0.75} (b is compared on India/US labeled recall);
   - (b) char3 over an accent-folded copy (strip combining marks for Latin-script strings only; never for Indic scripts);
   - (c) word BM25 over addresses canonicalized with a small hand-written generic street-type abbreviation table. Write it from general linguistic knowledge, not an external gazetteer or libpostal dictionary, which are trained on OpenStreetMap/OpenAddresses data;
   - (d) numeric block: exact house number plus country postcode pattern, joined only when name char3 cosine clears a floor;
   - (e) anyascii view for non-Latin-script targets (India S3).
2. **Retrieval.** Top-100 per view. Meta-blocking weights: number of agreeing views plus summed reciprocal rank. Keep the top-30 union per S1, but always keep any single-view pair with rank 1.
3. **Reverse search.** For every target in the union with stage-2 p >= 0.02, retrieve the top-10 S1 per view within the country. Score competitor pairs with rank-free features. Feed best-competitor p, the margin, and this S1's owner rank into stage 2 (soft F2), then keep the hard best-owner rule as the final step.
4. **Scoring.**
   - Rescued pairs outside the frozen top-12 use the text-only stage-2 model trained on India/US OOF.
   - Accept at p >= 0.85 (the CL-012 operating point), then apply R2.
   - Measure precision of exactly this path on labeled India/US held S1 before trusting it on France.
5. **Gates.**
   - India/US held: link recall gain with rescue precision >= 0.9 at the operating threshold.
   - Fixed-6k macro not lower.
   - France: counts and score distributions only. Never tune on unlabeled France outcomes.

### R4 (single-GPU training plus inference, rank 4): CE v2 on multilingual-E5-base

1. **Model.**
   - `intfloat/multilingual-e5-base` at revision d12875059715 (MIT, 278M), loaded as `CrossEncoder(model_id, num_labels=1, max_length=128)`.
   - L40S alternative, rules caveat: `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M), `max_length=128`, batch 32, LR 1e-5.
   - A10G alternative, rules caveat and custom code: `Alibaba-NLP/gte-multilingual-reranker-base` (Apache-2.0, 306M).
2. **Input format.**
   - Query: `name: {raw_name} | addr: {raw_addr} # {digits}`.
   - Document: `{src}: name: {raw_name} | addr: {raw_addr} # {digits}`.
   - Keep raw NFC text and Indic marks.
   - `{digits}` is the space-joined list of Unicode-digit-normalized number tokens, following the Ditto domain-knowledge injection idea.
   - Augment 20% of pairs by swapping sides and 5% by dropping the address from one side.
   - Measure token lengths first. If p99 exceeds 128, use 160 for France/India only.
3. **Training data.** Owner-safe:
   - 50,000 training S1 from folds 0-1, excluding fixed-6k S1, their owned targets, and targets owned by fold 2/3/4 S1. Fold4 is never read.
   - About 2.8 times EXP-050's 18k S1.
4. **Positives and negatives per S1.**
   - Positives: all positives in the stage-1 top-100 union, capped at 6 (sampled if more).
   - 16 negatives:
     - 8 by lexical rank from the char3 union (the friend's lexical recipe);
     - 4 by highest stage-1 OOF LightGBM probability (the confusers our system actually gets wrong);
     - 4 random from top-100.
   - Denoise with E3: drop negatives whose normalized name and address equal a positive of the same S1.
   - Total: about 1.0-1.1M pairs.
5. **Loss.**
   - Primary: `BinaryCrossEntropyLoss` with `pos_weight` equal to the negative/positive ratio (about 4).
   - Ablation (rank 10): `LambdaLoss` on list format (query, 1-6 positives plus 16 negatives), LR 2e-5.
6. **Optimization.**
   - Batch 64 (A10G/L4), 128 (L40S).
   - LR 3e-5, `warmup_ratio` 0.1, linear decay, `weight_decay` 0.01, 1 epoch, bf16.
   - Evaluate every 2,000 steps with `CrossEncoderRerankingEvaluator` on 1,000 held fold-2/3 S1 that are not in the fixed 6k. Use `load_best_model_at_end`.
   - Estimated 1 epoch of about 17k steps: 45-75 min on A10G. Benchmark 300 steps first.
7. **Inference.**
   - Expected about 2-3 times slower than E5-small: roughly 1,100-1,600 pairs/s on A10G (estimate).
   - Test top-12 is about 4-5 GPU-h. Shard across 3-4 single-GPU jobs, or score only pairs whose stacked p lies in [0.01, 0.99]. Measure that fraction first.
8. **Stack and gate.**
   - Add the CE v2 probability to the R1 stacker. Promote only if the fixed-6k nested macro rises by at least 0.001 over R1 with no country regression beyond 0.001.
   - It overlaps R1. Keep one CE if they tie, favoring fewer parameters and faster inference.

### R5 (single-GPU training plus encoding, rank 5): fine-tuned E5-small bi-encoder

1. **Model.** `intfloat/multilingual-e5-small` at revision 614241f622f5 (MIT, 118M): SentenceTransformer, mean pooling, L2 normalization, `max_seq_length` 64 (96 if p95 tokens exceed 64). Use the text `query: {name} | {address}` on both sides, per the model card's symmetric usage.
2. **Data.**
   - 200,000 owner-safe training S1, following R4's exclusions.
   - Two positives sampled per S1, giving 400k anchor-positive pairs (the friend's BIENC scale).
   - One hard negative per anchor: the highest-ranked char3 top-100 target that the S1 does not own, is not a text duplicate of a positive, and comes from the same country.
3. **Loss.**
   - `CachedMultipleNegativesRankingLoss` (GradCache), scale 20 (temperature 0.05), `mini_batch_size` 128.
   - Batch 512 (friend's setting), or 1024 on L40S.
   - Country-homogeneous batches, with the NO_DUPLICATES batch sampler and at most one pair per S1 per batch.
   - Optional R-SupCon variant: 2 positives of the same S1 in one batch, with a SupCon-style multi-positive loss. Only if time.
4. **Optimization.** LR 5e-5, `warmup_ratio` 0.05, 1 epoch, bf16. Estimated 20-40 min on A10G.
5. **Encode and search.**
   - Encode all test S2/S3 targets (about 9.97M), the test S1 (1.733M), and the training S1 needed for evaluation. Use batch 1024 fp16 for 20-40 min on A10G (estimate).
   - Exact inner-product search with faiss (MIT) per country on GPU. At 384 dimensions fp16, about 5M vectors use under 4GB.
   - Retrieve top-50 targets per S1, and top-10 S1 per target for competitor search.
6. **Uses.**
   - (a) Dense-only rescue candidates enter the R3 text-only scoring path.
   - (b) The cosine becomes a stage-2 and stacker feature for all top-12 pairs.
   - (c) Dense competitors feed R3's soft ownership features.
7. **Gates.**
   - On 20k natural held fold-2/3 S1: at least +0.3 percentage points of link recall at no more than 20 new candidates per S1, or at least +1 point on the India S3 slice.
   - Then fixed-6k macro not lower after stage-2 rescoring.

### Suggested 12-hour schedule

| Hours | CPU track | GPU track |
|---|---|---|
| 0-0.5 | Verify EXP-050 checkpoint SHA; build R4/R5 training tables | Launch R1 test scoring shards; launch R4 training benchmark |
| 0.5-2.5 | R2 calibration, S1-empty model, nested 6k evaluation | R1 scoring continues; R4 training |
| 2.5-5 | R3 views, reverse search and soft ownership on India/US held, then France | R5 training and encoding after R1 frees a GPU; R4 held and 6k scoring |
| 5-8 | Stack R1 (+R4 if gated) into stage 2 → ownership → R2 → duplicate expansion; per-country reports | R4 test scoring (sharded) if gated; R5 FAISS and recall gate |
| 8-10 | Assemble candidate submission TSVs; official default and strict validation | Idle or cancelled |
| 10-12 | Freeze, provenance, write-up; buffer | none |

**Stop rules.**

- Cancel any GPU job whose projected finish (training, test scoring and official validation) passes hour 9.
- Promote nothing without a fixed-6k nested gain of at least 0.0005, or its stated recall gate.
- Never tune thresholds on unlabeled France outcomes.
- Fold4 stays CLOSED.
- Portal uploads remain the user's decision.

**Explicitly not recommended in this sprint:**

- Zingg (AGPL), SparkER (GPL) and any non-MIT/Apache checkpoint (jina CC-BY-NC, Jellyfish CC-BY-NC, Llama, Gemma).
- LLM routing, HierGAT or other GNN collective models, ByT5/CANINE CEs, JointBERT, Unicorn, Sudowoodo, DeepBlocker, DeepMatcher and Magellan.
- libpostal or any gazetteer, geocoder or external address dictionary (external geographic data risk).
- Pseudo-labeling on France test data (unresolved rules question and unvalidatable).

## Sources

Entity matching, blocking and linkage:
- [Ditto: Deep Entity Matching with Pre-Trained Language Models (arXiv 2004.00584)](https://arxiv.org/abs/2004.00584); [megagonlabs/ditto](https://github.com/megagonlabs/ditto)
- [Rotom (Megagon publication page)](https://megagon.ai/publications/rotom-a-meta-learned-data-augmentation-framework-for-entity-matching-data-cleaning-text-classification-and-beyond/)
- [Sudowoodo (arXiv 2207.04122)](https://arxiv.org/abs/2207.04122); [megagonlabs/sudowoodo](https://github.com/megagonlabs/sudowoodo)
- [Unicorn (SIGMOD 2023, ACM)](https://dl.acm.org/doi/abs/10.1145/3588938); [ruc-datalab/Unicorn](https://github.com/ruc-datalab/Unicorn)
- [HierGAT: Entity Resolution with Hierarchical Graph Attention Networks (ACM)](https://dl.acm.org/doi/10.1145/3514221.3517872)
- [JointBERT: Dual-Objective Fine-Tuning of BERT for Entity Matching (preprint)](https://www.uni-mannheim.de/media/Einrichtungen/dws/Files_Research/Web-based_Systems/pub/Peeters-Bizer-Dual-Objective-Fine-Tuning-VLDB2021-preprint.pdf); [wbsg-uni-mannheim/jointbert](https://github.com/wbsg-uni-mannheim/jointbert)
- [Supervised Contrastive Learning for Product Matching (R-SupCon)](https://dl.acm.org/doi/fullHtml/10.1145/3487553.3524254); [wbsg-uni-mannheim/contrastive-product-matching](https://github.com/wbsg-uni-mannheim/contrastive-product-matching)
- [SC-Block (arXiv 2303.03132)](https://arxiv.org/abs/2303.03132)
- [DeepBlocker: Deep Learning for Blocking in Entity Matching (PVLDB 14)](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)
- [Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker (PVLDB 16)](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)
- [pyJedAI overview](https://pyjedai.readthedocs.io/en/latest/intro.html); [JedAIToolkit](https://github.com/scify/JedAIToolkit); [SparkER](https://github.com/Gaglia88/sparker)
- [Pre-trained Embeddings for Entity Resolution: An Experimental Analysis (arXiv 2304.12329)](https://arxiv.org/abs/2304.12329)
- [DeepMatcher (SIGMOD 2018, ACM)](https://dl.acm.org/doi/10.1145/3183713.3196926); [anhaidgroup/deepmatcher](https://github.com/anhaidgroup/deepmatcher)
- [Magellan (PVLDB 9)](http://www.vldb.org/pvldb/vol9/p1581-konda.pdf)
- [Splink](https://github.com/moj-analytical-services/splink); [Splink term-frequency adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html); [Splink Fellegi-Sunter guide](https://moj-analytical-services.github.io/splink/topic_guides/theory/fellegi_sunter.html)
- [Zingg](https://github.com/zinggAI/zingg)
- [dedupe](https://github.com/dedupeio/dedupe); [dedupe docs](https://docs.dedupe.io/)
- [Bilenko and Mooney, Adaptive duplicate detection using learnable string similarity measures (KDD 2003)](https://dl.acm.org/doi/10.1145/956750.956759)
- [Collective Entity Resolution in Relational Data (Bhattacharya and Getoor)](https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf)
- [Auto-FuzzyJoin (arXiv 2103.04489)](https://arxiv.org/abs/2103.04489)
- [Bipartite Graph Matching Algorithms for Clean-Clean ER (arXiv 2112.14030)](https://arxiv.org/abs/2112.14030); [An analysis of one-to-one matching algorithms for ER (VLDB Journal)](https://dl.acm.org/doi/10.1007/s00778-023-00791-3)
- [DADER: Domain Adaptation for Deep Entity Resolution](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/entity-sigmod-2022.pdf)

Metric decision theory:
- [Optimizing F-measures: A Tale of Two Approaches (arXiv 1206.4625)](https://arxiv.org/pdf/1206.4625)
- [An Exact Algorithm for F-Measure Maximization (NeurIPS 2011)](https://proceedings.neurips.cc/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf); [On the Bayes-optimality of F-measure maximizers (JMLR)](https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf)

Ranking, reranking and contrastive training:
- [Training and Finetuning Reranker Models with Sentence Transformers (HF blog)](https://huggingface.co/blog/train-reranker); [Cross-encoder losses reference](https://sbert.net/docs/package_reference/cross_encoder/losses.html)
- [Rethink Training of BERT Rerankers (LCE, arXiv 2101.08751)](https://arxiv.org/abs/2101.08751)
- [RocketQA (arXiv 2010.08191)](https://arxiv.org/abs/2010.08191)
- [Sentence-BERT (ACL Anthology D19-1410)](https://aclanthology.org/D19-1410/)
- [GradCache (arXiv 2101.06983)](https://arxiv.org/abs/2101.06983); [luyug/GradCache](https://github.com/luyug/GradCache)
- [Margin-MSE cross-architecture distillation (arXiv 2010.02666)](https://arxiv.org/abs/2010.02666)
- [LightGBM parameters (lambdarank)](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

Multilingual and character models:
- [LaBSE (ACL 2022)](https://aclanthology.org/2022.acl-long.62/)
- [Multilingual E5 technical report](https://arxiv.org/html/2402.05672v1); [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
- [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base)
- [mmBERT (arXiv 2509.06888)](https://arxiv.org/pdf/2509.06888); [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)
- [CANINE (arXiv 2103.06874)](https://arxiv.org/abs/2103.06874); [google/canine-s](https://huggingface.co/google/canine-s)
- [ByT5 (arXiv 2105.13626)](https://arxiv.org/abs/2105.13626); [google/byt5-small](https://huggingface.co/google/byt5-small)
- [Aksharantar / IndicXlit (arXiv 2205.03018)](https://arxiv.org/abs/2205.03018); [AI4Bharat/IndicXlit](https://github.com/AI4Bharat/IndicXlit)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3); [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base); [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [Improving Address Matching using Siamese Transformer Networks (arXiv 2307.02300)](https://arxiv.org/abs/2307.02300) (address-matching background)
- [libpostal](https://github.com/openvenues/libpostal) (cited only to explain its exclusion: parser and dictionaries built from OpenStreetMap/OpenAddresses)

LLM-based matching (reference only):
- [Entity Matching using Large Language Models (EDBT 2025 preprint)](https://www.uni-mannheim.de/media/Einrichtungen/dws/DWS_News/Documents/Peeters-Entity-Matching-using-LLMs-EDBT2025.pdf)
- [Fine-tuning Large Language Models for Entity Matching (arXiv 2409.08185)](https://arxiv.org/abs/2409.08185)
- [AnyMatch (arXiv 2409.04073)](https://arxiv.org/abs/2409.04073)
- [Match, Compare, or Select? (COLING 2025)](https://aclanthology.org/2025.coling-main.8/); [tshu-w/ComEM](https://github.com/tshu-w/ComEM)

License metadata endpoints read on 2026-09-26: `https://api.github.com/repos/{owner}/{repo}/license` and `https://huggingface.co/api/models/{id}`. Repositories with no declared license: ruc-datalab/Unicorn, chu-data-lab/AutomaticFuzzyJoin, tshu-w/ComEM.
