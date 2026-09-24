# Decisions

## D001: Research and measurement before training
Use the reference prompt as a checklist, not a mandate to launch all experiments. Finish phase-one artifacts and no-training diagnostic measurements first. Research can prioritize methods; no method is proven best until local validation.

## D002: Local bounded processing
Use DuckDB with 4 threads and 6 GiB memory, on-disk scratch capped at 6 GiB. Keep raw sources in place. No dense pair matrix; no redundant dataset copies. Reserve 8 GiB disk space. Initial free space about 24 GiB.

## D003: Official metric, stronger validation
Compute 1.25*TP/(1.25*TP+FP+0.25*FN) for each nonempty case and 1 for empty/empty; average over all S1. This follows beta squared = 0.25. The narrative '2x' is informal; FP coefficient is four times FN in this form. Official validator's optional ID check and warning-only membership check are insufficient as internal gates.

## D004: Budget and cloud
Only the user-reported $800 is a planning ceiling, not verified available credit. Later $100 award is conditional on Top 500 at 48h in pasted event copy. No new cloud resources during research. Colab CLI installation allowed; authentication may require user sign-in. AWS CLI already installed.
