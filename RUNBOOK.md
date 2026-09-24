# Runbook

Use the workspace .venv. All dataset reads use an explicit tab separator and preserve empty text. Commands below are updated as implementations become available.

```sh
source .venv/bin/activate
export PYTHONPATH="$PWD/code/business_entity_resolution"
python -m pytest code/business_entity_resolution/tests -q
```

Original supplied validator (when a real prediction set is available):

```sh
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

Never use original all-pairs Python dataframes. Avoid running concurrent heavy jobs on the 24 GiB Mac. Do not delete user files to free disk.
