# Phase5 persistent-index benchmark

P5-INDEX-001 / EXP-024 completed, exit0, EC2 termination verified. Code76a0db2. All10,320,219 training targets indexed, without labels.200 reference queries per country per route, three thread counts. No candidate-set mismatches.

| Route | Country | Build/save seconds | 2-thread q/s | 4-thread q/s | 8-thread q/s |
|---|---|---:|---:|---:|---:|
| name | India | 35.14 | 48.06 | 73.01 | 79.14 |
| name | US | 45.20 | 49.24 | 70.43 | 72.85 |
| address | India | 65.88 | 27.08 | 45.36 | 48.51 |
| address | US | 59.34 | 42.54 | 74.09 | 80.69 |

Persistent index bytes: 6,078,068,248. Peak observed process RSS: 2.564GiB (does not include all OS cache). Each country/route is partitioned by source and250k target chunks; transpose CSR is saved once. Raw labels are absent from the benchmark input.

Full2,206,821-query extrapolation:17.76hours query compute on this instance, approximately$9.87 compute. This is a small-query benchmark extrapolation, not a measured full run; write/upload/query-mix overhead may change it. P5-FULL-RETRIEVAL-001 caps24hours, plans$18 total, and checkpoints each deterministic query shard to S3.

Benchmark job wall time521.28seconds; compute estimate approximately$0.0805 through worker status timestamp, excluding termination lag/EBS/IP/storage. Actual billing remains pending.184S3 result/status/receipt objects,6,078,184,306bytes. Mac current-best20k checkpoint separately saved:591,802,307bytes in2verified objects.
