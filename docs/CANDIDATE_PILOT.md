# Full-target candidate pilot: TOKEN-001

1,000 fixed fold-0 S1 queries (500 each India/US), all 10,320,219 training targets. The sample has 3,449 true links and 50 singletons. No model fitting. Frequency ranking excludes targets owned by folds 0/4; full-pool frequency is used only as an inference fanout guard.

| Final candidate cap | Pairs | Link recall | Oracle macro F0.5 ceiling |
|---|---:|---:|---:|
| total_top_20 | 17,546 | 65.58% | 0.79152 |
| total_top_50 | 41,449 | 70.51% | 0.82141 |
| total_top_100 | 78,332 | 73.56% | 0.84026 |
| route_capped_union | 179,298 | 77.33% | 0.86630 |

Runtime 34.65s; peak process RSS 1.26 GiB. CPU only.

The best recall here (77.33%) is inadequate for final candidate generation. The oracle is an impossible perfect matcher restricted to these candidates; it is **not an achieved model score**. Full-pool target coverage does not make 1,000 queries a complete validation run.

## Implications

- Only 464/1,000 queries have an eligible rare-name key and 705 have an eligible rare-address key under the current DF≤1,000 cap. Relax fanout selectively and add fuzzy character retrieval.
- The numeric route promotes already retrieved name hits; it does not independently rescue missing candidates.
- Top-100 fusion loses 130 true links compared with the route-capped union. Keep broader candidates until a validated ranker can reduce volume safely.
- 47/50 true singletons still receive candidates. Candidate existence is not evidence of a match.
- India union recall is 80.68%; US is 73.92%. This reflects this particular lexical blocker and balanced sample, not a claim that India is the easier overall task.

## Failure and repair

The first query plan exceeded its deliberate 1 GiB scratch cap when correlated UNNEST expanded the full target pool. Preserved failure metadata/source in `outputs/candidates/TOKEN-001/`. The successful `run-002` streams scalar UNNEST through country/source partitions and filters query keys before aggregation. It retained the full target pool and used 1.26 GiB peak RSS.

## Missed-link morphology

| Country | Retrieved | Links | Non-ASCII name | Missing address | Numeric overlap | Mean name JW | Mean address JW |
|---|---|---:|---:|---:|---:|---:|---:|
| India | False | 336 | 39.9% | 8.0% | 77.1% | 0.691 | 0.658 |
| India | True | 1403 | 17.9% | 3.8% | 79.8% | 0.847 | 0.801 |
| US | False | 446 | 8.5% | 8.1% | 71.7% | 0.899 | 0.771 |
| US | True | 1264 | 6.3% | 4.0% | 78.6% | 0.938 | 0.834 |

## Reproduce

```sh
PYTHONPATH=code/business_entity_resolution .venv/bin/python -m src.blocking.token_candidates --database artifacts/audit.duckdb --config configs/blocking/TOKEN-001.json --output-dir outputs/candidates/TOKEN-001/run-003
```

Always select a new output directory. The current CLI is a development-pilot scaffold; a configurable test/full-query streaming mode and persistent indexes remain to be built.
