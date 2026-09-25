# Full training retrieval analysis

EXP-025 is running. Its route archives and final metrics are not complete, so
the full-population values below remain pending. The worker retrieves all
2,206,821 training Source-1 queries against all 10,320,219 Source-2/3 targets,
using frozen name/address character-3-gram top-100 routes. It may generate
candidates for Fold 4 without reading or evaluating Fold 4 labels. Every
label-based result in this report must use only folds 0–3.

| Scope | Link recall | Complete positive entity recall | Any-match entity recall | Oracle macro F0.5 | Mean candidates | p50 | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All unlocked entities | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| India | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| US | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

The worker's `retrieval-metrics-unlocked.json` supplies exact overall,
country, match-count and full unlabeled candidate-distribution metrics after
completion. Do not copy older 20k-sample results into this full-population
table. Before publishing, require `collect_run.py P5-FULL-RETRIEVAL-001` to
show exit 0 and EC2 termination, `results-COMPLETE.json` to report all
2,206,821 queries/four country-route combinations, and all 256 route archives
plus their SHA256 receipts. Then verify the final metrics file and record its
hash, runtime, code commit and estimated AWS cost.

Additional requested slices need explicit denominators. Source-2/Source-3,
ASCII/non-ASCII target name, either-side missing name/address and first
numeric-token disagreement are **positive-pair slices**: report true links,
retrieved true links and link recall, with overlapping flags allowed.
Singleton, one, two, three-to-five and six-plus matches are **Source-1 entity
slices**: report entity count, candidate count, complete/any-match retrieval
and oracle macro F0.5. Short/long Source-1 names use explicit length bins
chosen before inspecting slice outcomes. Do not report pair-level recall for
singletons, which contain no positive links. Never infer hidden France labels.

Current smaller reference only, not EXP-025: the 20k natural-sample P4-B-002
union had link recall 0.96662630, complete-positive-entity recall 0.90305178,
oracle macro F0.5 0.98853146 and 197.18135 candidates per Source-1. India
link recall was 0.94192378 versus US 0.98344844. The full-population result
will determine whether these gaps persist and which targeted classical rescue
routes have sufficient marginal value.
