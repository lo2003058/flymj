"""Step A: find the KC / MBON / PN label sets in annotations.feather.

Which column and condition to use for KC was already confirmed for the project.
MBON and PN weren't confirmed yet, so this script first explores (prints
every value in the class/superclass columns), then summarizes the
candidate definitions found so the user can eyeball-check them.

Run: uv run python src/explore_labels.py
"""

import polars as pl

from io_utils import load_feather
from labels import is_fragment, kc_mask, mbon_mask, pn_mask

ANNOTATIONS_PATH = "data/raw/annotations.feather"


def summarize_group(df: pl.DataFrame, name: str, column: str, condition: str, mask: pl.Series) -> pl.DataFrame:
    """Print a neuron label group's total count / type count / somaSide breakdown."""
    matched = df.filter(mask)
    frag_mask = is_fragment(matched)
    excluded = matched.filter(frag_mask)
    kept = matched.filter(~frag_mask)

    print(f"\n{'=' * 70}")
    print(f"=== {name} ===")
    print(f"{'=' * 70}")
    print(f"Column used: {column!r}   Condition: {condition}")
    print(f"Total before excluding fragments: {matched.height}")
    if excluded.height:
        print(f"Excluded {excluded.height} fragment neurons")
    print(f"Total after excluding fragments: {kept.height}")
    print(f"Distinct type count: {kept['type'].n_unique()}")

    print("\n-- type breakdown (sorted by count) --")
    print(kept.group_by("type").agg(pl.len().alias("count")).sort("count", descending=True))

    print("\n-- somaSide breakdown --")
    print(kept.group_by("somaSide").agg(pl.len().alias("count")).sort("somaSide"))

    return kept


def main() -> None:
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(50)

    ann = load_feather(ANNOTATIONS_PATH)
    print("=== annotations shape ===")
    print(ann.shape)

    # ------------------------------------------------------------------
    # KC: confirmed. `type` column starts_with "KC".
    # ------------------------------------------------------------------
    kc = summarize_group(
        ann, "KC (confirmed)", "type", "starts_with('KC')", kc_mask(ann)
    )

    # ------------------------------------------------------------------
    # Exploration: what values exist in class / superclass, to inform
    # picking MBON / PN definitions.
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("=== Exploration: `class` column value_counts (raw distribution before confirming MBON/PN) ===")
    print(f"{'=' * 70}")
    print(ann.get_column("class").value_counts(sort=True))

    print(f"\n{'=' * 70}")
    print("=== Exploration: `superclass` column value_counts ===")
    print(f"{'=' * 70}")
    print(ann.get_column("superclass").value_counts(sort=True))

    # ------------------------------------------------------------------
    # MBON: candidate definition = `class` column == "MBON"
    # ------------------------------------------------------------------
    mbon = summarize_group(
        ann, "MBON (candidate: class == 'MBON')", "class", "== 'MBON'", mbon_mask(ann)
    )

    # ------------------------------------------------------------------
    # PN: candidate definition = `class` column == "ALPN" (antennal lobe
    # projection neuron). Within ALPN, types starting with "M_" are
    # multiglomerular PNs; the rest (named like
    # "<glomerulus>_lPN/adPN/vPN/...") are uniglomerular PNs.
    # This split is done purely by eyeballing the printed type list, not
    # an automatic decision, so the multiglomerular group is also printed
    # below for the user to verify the split is correct.
    # ------------------------------------------------------------------
    alpn_all = summarize_group(
        ann, "PN exploration: all ALPN (not yet split uni/multi-glomerular)", "class", "== 'ALPN'", ann["class"] == "ALPN"
    )

    is_multiglomerular = alpn_all["type"].str.starts_with("M_")
    uni_mask = pn_mask(alpn_all)
    print(f"\n{'=' * 70}")
    print("=== PN candidate split: within ALPN, types starting with 'M_' are multiglomerular, excluded ===")
    print(f"{'=' * 70}")
    print(f"Multiglomerular (type starts_with 'M_'): {is_multiglomerular.sum()} cells")
    print(f"Uniglomerular candidates (the rest): {uni_mask.sum()} cells")

    pn = summarize_group(
        alpn_all,
        "PN (candidate: class == 'ALPN' AND NOT type.starts_with('M_'))",
        "class + type",
        "class == 'ALPN' AND NOT type.starts_with('M_')",
        uni_mask,
    )

    # ------------------------------------------------------------------
    # Sanity check: compare against the order-of-magnitude originally expected
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("=== Sanity check (vs. the originally expected order of magnitude) ===")
    print(f"{'=' * 70}")
    print(f"KC   : {kc.height} cells (expected ~4000)")
    print(f"MBON : {mbon.height} cells, {mbon['type'].n_unique()} types (expected 100-200 cells, 30-100 types)")
    print(f"PN   : {pn.height} cells, {pn['type'].n_unique()} types (expected 100-200 cells)")


if __name__ == "__main__":
    main()
