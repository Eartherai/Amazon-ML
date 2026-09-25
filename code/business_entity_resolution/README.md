# Amazon ML Challenge 2026: SUB-001 business entity resolution

This folder reproduces the two TSVs in `output/` from the official unlabeled test
sources. It contains the exact frozen LightGBM model, character-trigram IDF
vocabularies, generic transliteration map, threshold, source code, and pinned
Python dependencies. No outside business data or identity service is used.

## Reproduce both TSVs

Use Python 3.12 on a machine with enough disk for the raw test data, derived
Parquet inputs, compressed shards, and the two uncompressed TSVs. The supplied
run used macOS M5 24 GB for sharded inference and a 64 GB validation machine
for the official full ID check. From this directory:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m src.submission.reproduce \
  --test-dir /path/to/student_resource/dataset/test \
  --work-dir /path/to/new-sub001-work \
  --output-dir /path/to/new-sub001-output \
  --official-validator utils/validate_submission.py \
  --threads 8
```

`--test-dir` must contain the original `test_source1.tsv`,
`test_source2.tsv`, and `test_source3.tsv`. `--work-dir` and `--output-dir`
must not already exist. The command:

1. Parses all three TSVs with every field as text, preserving empty strings;
   makes the frozen NFC/lowercase/punctuation-to-space view and checks record
   counts and ID uniqueness.
2. Retrieves up to 100 same-country targets per source entity through each of
   name and address character-trigram TF-IDF; the union is the final scored
   candidate set. Country labels are arbitrary strings.
3. Computes the frozen 51 comparison features, scores every candidate with the
   bundled LightGBM model, and predicts scores at or above 0.83.
4. Merges all 192 candidate and 192 matching shards, verifies complete S1
   coverage and match-to-candidate membership, and runs the unmodified official
   validator with and without `--check-ids` if supplied.

The resulting files are `matching_results.tsv` and `candidate_pairs.tsv` in
`--output-dir`. The full run is large; do not use a partial query cap for a
portal file. The run is deterministic for the pinned environment and frozen
artifacts. ZIP `output/` contains the separately validated final run; its
SHA256 values appear in `PACKAGE_MANIFEST.json`.

## Provenance and training

`artifacts/model.txt` was fitted using 2,713,116 owner-safe candidate pairs
from 20,000 Source-1 entities in development folds 1–3. Locked fold 4 was not
used. The exact model SHA256, feature order, hyperparameters, and threshold
rule are in `artifacts/train-manifest.json` and
`artifacts/submission-config.json`. The two IDF vocabularies were fitted only
on permitted training text. `artifacts/name_map.parquet` is a deterministic,
unlabeled transliteration cache from official test strings using the generic
Apple Foundation `Any-Latin; Latin-ASCII` transform; source is in
`src/submission/transliterate_probe.swift`. The raw test TSVs are not included.

`src/` also contains the normalization, retrieval, feature, validation, and
training research modules. `src/submission/run_sub001.py` is byte-identical to
the script used for the frozen full test inference. The included `tests/`
cover critical metric, parsing, blocking, and feature behavior. The frozen
model is included so reproducing the portal outputs does not depend on
recreating development caches or retraining.

The local development score for this architecture was macro per-Source-1
F0.5 = 0.9318965293 on 20,000 naturally sampled development entities with
nested threshold selection. This is not a leaderboard or hidden-test score.
