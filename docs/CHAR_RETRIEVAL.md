# Full-pool character retrieval pilot

CHAR-001 retrieves the same 1,000 development queries as TOKEN-001, with the complete 10,320,219-record train S2/S3 target index. Query countries filter retrieval, and all same-country target rows are transformed and scored. This is an exact top-K calculation within each route's fitted vector representation, not approximate nearest-neighbor search. It remains a development pilot, not a full development-population evaluation.

## Design and evidence

[Scikit-learn's TfidfVectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) provides character n-gram analysis, L2 normalization and float32 vectors. L2-normalized sparse matrix products compute cosine similarity. Character 3-, 4- and 5-gram routes are evaluated independently for names and addresses. Whitespace participates in character n-grams. Unicode marks are retained through the existing audited light-normalized fields.

The vocabulary and IDF are fitted on training-eligible target records only (owner folds -1,1,2,3) with SHA256 ID prefix in 00,01,02,03, approximately 1/64 of eligible records. This sampling is for fitted statistics, **not a reduction of the retrieval target pool**. Min document frequency is 2 and maximum features 200,000. Exact fit counts, ID hashes, vocabulary sizes and persisted vocabulary/IDF arrays are saved for each completed route. No query labels participate in retrieval; development labels only measure candidate coverage.

Targets stream in 25,000-row shards; query products are limited to 50 rows, bounding the dense product to about 5 MB. Each shard's positive-score top 100 is merged into an exact global top 100, with target ID resolving all score ties including at the cutoff. Top K is across both target sources. Raw scores, route ranks, representation and target source are saved in route Parquet files. The union with TOKEN-001 preserves previous retrieval results. Zero-score records are not padded into the candidate set.

A 600-second runtime cap applies independently per route. An incomplete route is marked timeout and contributes no candidates. Routes checkpoint separately. No target vector index is persisted, avoiding disk expansion. Completed metrics are in `outputs/candidates/CHAR-001/metrics.json`.

## Reproduce

```sh
PYTHONPATH=code/business_entity_resolution OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m src.blocking.char_retrieval
```

The output directory must not exist; use `--output` for another immutable run.

## Tests

Boundary ties, omitted zero scores and exact equivalence between sharded merging and a full brute-force top K are tested in `tests/test_char_retrieval.py`. Oracle macro F0.5 is a retrieval ceiling; no learned model score is claimed. Balanced-country sampling is not population weighting. Cross-country links are excluded by this configuration; training evidence currently shows no such labeled links.
