# France test-only shift — unlabeled evidence

This note uses only provided test text and frozen SUB-001 inference outputs. No hidden labels or outside business identities were used. The first 14 of 64 deterministic SHA256 France query shards form a roughly random 56,710-entity sample; these figures remain provisional until all shards finish.

| Statistic | France test | India test | US test |
|---|---:|---:|---:|
| S1 records | 259,452 | 809,986 | 663,106 |
| Non-ASCII normalized names | 15.72% | 0.00% | 0.00% |
| Median name length | 19 | 27 | 22 |
| Median address length | 46 | 70 | 32 |
| Address contains ASCII digit | 99.58% | 91.27% | ~100% |

On the completed France subset, the frozen name-char3 and address-char3 top-100 union scores 198.70 candidates/S1 (p50 199, p95 200). It predicts an empty match list for 6.31% and 3.14 matches per S1 on average. For directional context, the known-country 20k OOF predictions use their original per-fold thresholds and yield empty rates 6.94% for India and 6.22% for US, with means 2.94 and 3.08 matches/S1. These distributions do not measure France correctness; similar aggregate rates can hide many false merges or misses. The threshold procedures differ, so do not interpret small differences as a calibrated transfer test.

The non-ASCII name and address-format shift motivates inspecting generic transliteration and numeric/string comparisons after SUB-001. Do not tune on an unlabeled France outcome proxy or infer hidden matches. Source: `artifacts/submissions/SUB-001/shift-partial-v002.json`; regenerate with `scripts/submissions/diagnose_test_shift.py` after more shards complete.
