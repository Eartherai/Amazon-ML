# Business entity resolution research pipeline

Current implementation: full data audit, positive-pair and near-neighbor diagnostics, deterministic validation/ownership manifests, exact metric, strict TSV helpers, and an exact-match diagnostic baseline. Learned training, high-recall retrieval and final test inference are **not yet implemented**. This is research infrastructure, not the final submission solution.

Python 3.12. Install pinned requirements in a virtual environment. From this directory use `python -m src.<module>`; from the workspace root set `PYTHONPATH=code/business_entity_resolution` and use `.venv/bin/python`. Every input TSV uses an explicit tab separator. Original dataset remains in student_resource/dataset.

See the root RUNBOOK.md for exact commands and docs/RESEARCH.md for the ranked implementation plan. No external data lookup is part of any module. The original validator remains under student_resource/utils and is exercised by a synthetic test; final real-output validation has not occurred because final predictions do not exist yet.

## Source modules

- `audit_data`: full TSV integrity, distribution, duplicates, hashes and truth topology.
- `audit_pairs`: all labeled pair similarities; deterministic bounded hard-negative and adjacent-neighbor diagnostics.
- `audit_supplement`: Unicode script counts and sampled generic transliteration (macOS probe executable required).
- `audit_memory`: sequential per-file Polars buffer sizes and token-boundary suffix frequencies.
- `build_validation`: S1 folds plus target ownership and forbidden-training-target guard.
- `evaluation`: exact set/count macro F0.5.
- `normalization`: preserves Unicode combining marks, separate compatibility/accent views.
- `io_utils`: strict lists, deterministic shards/merge, coverage and membership checks.
- `exact_baseline`: development-only exact diagnostic against all training targets; no fitting.

Audit modules use a 6 GB DuckDB memory cap and four CPU threads. Reserve disk space for the derived database. Output directories are versioned; reuse a fresh database for a new data version. Source files are immutable. Parameterized data/output paths allow use outside this workspace; the final packaging wrapper is still future work.
