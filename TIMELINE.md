# Append-only timeline

All timestamps include timezone. Never replace historical entries.

## 2026-09-25T00:46:46.247511+05:30 | Codex | Initialization
- Action: inspected complete workspace; read source documents and supplied code.
- Commands: rg --files; find student_resource; pypdf extraction; sysctl; df.
- Config: M5, 10 CPU cores, 24 GiB unified RAM, 24 GiB available disk.
- Result: seven TSVs, two PDFs, README, template, validator, original ZIP. No existing Git repo.
- Interpretation: research-first local processing feasible; all-pairs comparison infeasible; storage constrained.
- Next: install isolated tools, audit every source, document discrepancies.

## 2026-09-25T01:01:23.120469+05:30 | Codex | Phase-one evidence
- Action: full six-source/ground-truth audit, full positive-pair comparison, near-neighbor diagnostics, entity folds, metric/TSV tests; official Colab CLI installed and login verified.
- Commands: src.audit_data; src.audit_pairs; src.build_validation; pytest; colab usage; colab sessions.
- Config: 4 CPU workers, DuckDB 6 GB memory limit, folds seed 20260925.
- Result: 24,229,173 source records, 7,638,365 labeled links, zero shared target owners or cross-country positive links; 22 tests pass. Colab 0 units, no sessions. AWS saved session expired.
- Interpretation: large candidate pool; multi-match output and cross-script recall are essential. No model training or cloud spend.
- Repairs: audit attempts 001/002 hit reserved SQL alias names and were retained as failed logs; 003 completed. Unicode audit normalization was then corrected to preserve combining marks; authoritative audit is AUDIT-004/PAIR-002, older reports preserved. Colab 0.7.1 unavailable; PyPI 0.7.2 installed.
- Next: finalize report, transliteration probe and EXP-001 diagnostic.
