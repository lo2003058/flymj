"""KC / MBON / PN label definitions.

These three definitions were confirmed against the printed output of
explore_labels.py (see data/doc/task.md):
  - KC   : `type` column starts_with('KC')                       (confirmed)
  - MBON : `class` column == 'MBON'                               (confirmed, 97 cells / 37 types)
  - PN   : `class` column == 'ALPN', excluding `type` starting with
           'M_' (multiglomerular)                                 (confirmed, 387 cells / 101 types)

387 is higher than the 100-200 originally expected in task.md — likely
because that expectation came from older single-hemisphere literature,
while this male CNS dataset covers both hemispheres (387 ≈ 190 per side
x2). Confirmed with the user to proceed with 387.

This module is the single source of truth, so explore_labels.py and
build_masks.py stay consistent instead of each maintaining a drifting copy.
"""

import polars as pl


def is_fragment(df: pl.DataFrame) -> pl.Series:
    """Rows whose `instance` contains 'fragment' (case-insensitive) are
    incomplete reconstructions and should be excluded."""
    return df["instance"].str.to_lowercase().str.contains("fragment").fill_null(False)


def kc_mask(ann: pl.DataFrame) -> pl.Series:
    return ann["type"].str.starts_with("KC")


def mbon_mask(ann: pl.DataFrame) -> pl.Series:
    return ann["class"] == "MBON"


def pn_mask(ann: pl.DataFrame) -> pl.Series:
    is_alpn = ann["class"] == "ALPN"
    is_multiglomerular = ann["type"].str.starts_with("M_")
    return is_alpn & ~is_multiglomerular


def get_kc(ann: pl.DataFrame) -> pl.DataFrame:
    df = ann.filter(kc_mask(ann))
    return df.filter(~is_fragment(df))


def get_mbon(ann: pl.DataFrame) -> pl.DataFrame:
    df = ann.filter(mbon_mask(ann))
    return df.filter(~is_fragment(df))


def get_pn(ann: pl.DataFrame) -> pl.DataFrame:
    df = ann.filter(pn_mask(ann))
    return df.filter(~is_fragment(df))
