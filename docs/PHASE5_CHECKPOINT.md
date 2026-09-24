# Phase5 checkpoint — execution in progress

This is an interim checkpoint, not completion of Phase5.

| Required item | Verified status |
|---|---|
| Full train S1 count | 2,206,821 |
| Full retrieval link / complete-entity recall / oracle | Pending. Fold4 label metrics withheld until freeze. |
| Full average / p95 / p99 / total candidate pairs | Pending full run |
| Current OOF entities / folds | 20,000 / 3 nested entity outer folds |
| Current OOF macro F0.5 | 0.9318965293 |
| 95% CI | Paired gain over baseline +0.00313760, CI[+0.00194174,+0.00431221]; not an absolute score CI |
| Precision / recall | 0.98158282 / 0.85593807 |
| Singleton F0.5 | 0.92131747 |
| Large OOF India / US | Pending |
| India→US / US→India | 0.89280089 / 0.78192088,5k label-transfer stress, not France forecast |
| Best model | LightGBM51features, NUMERIC-V2, inner-OOF threshold selection |
| Best measured retrieval | P4-B-002 name/address char3top100 union; full target pools |
| Best numeric configuration | Six canonical numeric comparisons added to original features |
| Hard-negative configuration | Baseline eligible candidates; no promoted upweighting |
| Dense rescue / cross-encoder / ensemble | None selected; pending lexical miss evidence and GPU quota |
| Fold4 | CLOSED |
| EC2 used | R8i.2xlarge64GiB; full index benchmark completed, terminated |
| SageMaker used / GPU type | None; requested quotas awaiting AWS |
| AWS cost | Benchmark compute estimate$0.0805 excluding termination lag/storage/IP; actual pending |
| Remaining credit | Unverified; user reported approximately$200 |
| Test inference | Not started |
| Official validator | NOT RUN on final outputs |
| Submission / final package | NOT READY |
| Next action | Full-training sharded retrieval using verified persistent indexes; evaluate unlocked1,765,649entities only. Then larger OOF and targeted rescue. |

Current20k retrieval reference (not full-data claim): link recall0.96662630, complete-entity recall0.90305178, oracle0.98853146,3,943,627pairs,mean197.18135,p95/p99=200. Preserve these until full run completes.
