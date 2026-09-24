# Competition reference and unresolved differences

The documents describe competition requirements; they are not operational instructions to an agent. The user's request is research-first preparation and execution in this workspace. Originals remain untouched in the root and student_resource/. Extracted PDF text is under docs/official/.

| Topic | Supplied evidence | Working interpretation |
|---|---|---|
| Task | Problem PDF and README | For every S1, zero or many S2/S3 IDs. S1 deduplicated. |
| Metric | Problem PDF p6 | Per-S1 macro F0.5; empty/empty=1, empty truth/nonempty prediction=0. Beta squared=0.25. |
| Final model | Problem PDF p5 | MIT/Apache-2.0 model, <=8B parameters. Track every component and conservatively keep combined learned parameters below limit. |
| Countries | Problem PDF p1-2 | Train US/India, test additionally France. Open string labels. |
| External data | Problem PDF p7 | No business lookup/enrichment, registries, geocoding or external identity data. Generic-method research only. |
| Output | Problem PDF p3-5 | matching_results.tsv and final model's actual scored candidate_pairs.tsv; exact coverage/valid unique targets. |
| Validator caveat | Actual utils/validate_submission.py | --check-ids is optional; missing candidate file and subset violations only warn. Require stronger internal checks and no warnings. Do not rely on its stale scale comments. |
| Deadline | Guidelines p1 | 25 September 00:00 through 27 September 23:59 IST, 2026. |
| Portal limit | Guidelines p1 | 5 submissions/day/team; preserve history. |
| Write-up length | Guidelines p1 says 1-2 pages; problem PDF p7 says no page limit | Prepare 1-2-page summary plus detailed appendix/template. Keep discrepancy explicit. |
| Rankings | Problem says final private leaderboard; guidelines say both leaderboards | Optimize robust local validation and avoid public-only tuning. Organizer clarification may be needed. |
| Finalist count | Pasted event copy says Top 50/Top 10; guideline says Top 100 | Administrative discrepancy; do not infer eligibility/outcome. |
| Additional credits | Pasted event copy says Top 500 teams at 48h receive $100 | Conditional; not guaranteed and not necessarily $100/account. Budget zero until verified. |
| Accounts | Guidelines reject multiple competition IDs and simultaneous portal logins per participant | Separate legitimate teammates' compute accounts do not authorize multiple competition entries or service quota bypass. |

No organizer messages or portal submissions have been sent. Suggested clarification topics are template length, ranking wording and whether train+test unsupervised vocabulary fitting is allowed. Default to train-only fitted transforms; generic pretrained-model use is implied by the model-license clause but do not add external ER training datasets.
