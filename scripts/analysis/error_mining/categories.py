"""ERR-MINE-001 observable error categories (priority order, mutually exclusive).

Only stage2_features_v2 columns are used (no labels, IDs, row order or country), so the same rule
can segment every candidate link, not only errors. first_num_rel codes: 0 = no digits on one side,
1 exact, 2 one digit inserted/deleted, 3 one digit substituted, 4 containment, 5 unrelated.
"""
from __future__ import annotations

import polars as pl

NAME_STRONG = (pl.col("core_eq") == 1) | (pl.col("sk_eq") == 1) | (pl.col("core_jw") >= 0.95)
NAME_WEAK = (pl.col("core_jw") < 0.80) & (pl.col("core_tset") < 0.80) & (pl.col("sk_jw") < 0.85)
ADDR_MATCH = pl.col("addr_tset") >= 0.85

# (code, label, rule) evaluated top to bottom; first match wins.
RULES = [
    ("C01_nonlatin_target", "Target name in non-Latin script (transliteration)", pl.col("t_nonlatin") == 1),
    ("C02_handle_domain_initials", "Target name is a domain/handle/initials form", (pl.col("t_domain") == 1) | (pl.col("initials_match") == 1)),
    ("C03_nameonly_strong_name", "Target address missing, core name equal/near-equal", (pl.col("t_addr_missing") == 1) & NAME_STRONG),
    ("C04_nameonly_partial_name", "Target address missing, name only partially similar", pl.col("t_addr_missing") == 1),
    ("C05_renamed_same_address", "Name unrelated/garbled but street tokens match", NAME_WEAK & ADDR_MATCH),
    ("C06_renamed_other", "Name unrelated/garbled, street tokens partial", NAME_WEAK),
    ("C07_housenum_conflict", "First house numbers unrelated (code 5)", pl.col("first_num_rel") == 5),
    ("C08_digit_noise", "First house number one-digit edit or containment (codes 2-4)", pl.col("first_num_rel").is_in([2, 3, 4])),
    ("C09_target_no_number", "Target address present but one side has no digits (code 0)", pl.col("first_num_rel") == 0),
    ("C10_full_agreement", "Same first number, strong name, street tokens match", (pl.col("first_num_rel") == 1) & NAME_STRONG & ADDR_MATCH),
    ("C11_same_number_partial", "Same first number, name or street only partial", pl.col("first_num_rel") == 1),
]


def categorize(df: pl.DataFrame) -> pl.DataFrame:
    expr = pl.lit("C12_residual")
    for code, _, rule in reversed(RULES):
        expr = pl.when(rule).then(pl.lit(code)).otherwise(expr)
    return df.with_columns(expr.alias("cat"))


LABELS = {code: label for code, label, _ in RULES} | {"C12_residual": "Anything else"}
