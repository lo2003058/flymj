"""Step A (task.md): 喺 annotations.feather 度揾 KC / MBON / PN 三組 label。

KC 用邊條欄、邊個條件已經喺 task.md 確認咗。
MBON 同 PN 未確認，呢個 script 會先做探索（印晒 class/superclass 欄有咩值），
再用揾到嘅候選定義去 summarize，等使用者肉眼核對。

跑法： uv run python src/explore_labels.py
"""

import polars as pl

from io_utils import load_feather
from labels import is_fragment, kc_mask, mbon_mask, pn_mask

ANNOTATIONS_PATH = "data/raw/annotations.feather"


def summarize_group(df: pl.DataFrame, name: str, column: str, condition: str, mask: pl.Series) -> pl.DataFrame:
    """印一組 neuron label 嘅 total count / type 數 / somaSide breakdown。"""
    matched = df.filter(mask)
    frag_mask = is_fragment(matched)
    excluded = matched.filter(frag_mask)
    kept = matched.filter(~frag_mask)

    print(f"\n{'=' * 70}")
    print(f"=== {name} ===")
    print(f"{'=' * 70}")
    print(f"用嘅欄: {column!r}   條件: {condition}")
    print(f"排除 fragment 前總數: {matched.height}")
    if excluded.height:
        print(f"排除咗 {excluded.height} 粒 fragment neuron")
    print(f"排除 fragment 後總數: {kept.height}")
    print(f"distinct type 數: {kept['type'].n_unique()}")

    print("\n-- type breakdown (依 count 排序) --")
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
    # KC：已確認。type 欄 starts_with "KC"。
    # ------------------------------------------------------------------
    kc = summarize_group(
        ann, "KC (已確認)", "type", "starts_with('KC')", kc_mask(ann)
    )

    # ------------------------------------------------------------------
    # 探索：class / superclass 欄有咩值，等 MBON / PN 揀有根據。
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("=== 探索: class 欄 value_counts (未確認 MBON/PN 定義前，先睇原始分佈) ===")
    print(f"{'=' * 70}")
    print(ann.get_column("class").value_counts(sort=True))

    print(f"\n{'=' * 70}")
    print("=== 探索: superclass 欄 value_counts ===")
    print(f"{'=' * 70}")
    print(ann.get_column("superclass").value_counts(sort=True))

    # ------------------------------------------------------------------
    # MBON：候選定義 = class 欄 == "MBON"
    # ------------------------------------------------------------------
    mbon = summarize_group(
        ann, "MBON (候選: class == 'MBON')", "class", "== 'MBON'", mbon_mask(ann)
    )

    # ------------------------------------------------------------------
    # PN：候選定義 = class 欄 == "ALPN"（antennal lobe projection neuron）。
    # ALPN 入面 type 開頭 "M_" 嘅係 multiglomerular PN，
    # 淨返嘅（"<glomerulus>_lPN/adPN/vPN/..." 呢種命名）先係 uniglomerular PN。
    # 呢個 split 純粹靠印出嚟嘅 type list 肉眼分辨，唔係自動判斷，
    # 所以下面連 multiglomerular 嗰組都印埋出嚟，等使用者自己核對呢個 split 啱唔啱。
    # ------------------------------------------------------------------
    alpn_all = summarize_group(
        ann, "PN 探索: 全部 ALPN (未分 uni/multi-glomerular)", "class", "== 'ALPN'", ann["class"] == "ALPN"
    )

    is_multiglomerular = alpn_all["type"].str.starts_with("M_")
    uni_mask = pn_mask(alpn_all)
    print(f"\n{'=' * 70}")
    print("=== PN 候選 split: ALPN 入面 type 開頭 'M_' 嘅係 multiglomerular，排除 ===")
    print(f"{'=' * 70}")
    print(f"Multiglomerular (type starts_with 'M_'): {is_multiglomerular.sum()} 粒")
    print(f"Uniglomerular 候選 (其餘): {uni_mask.sum()} 粒")

    pn = summarize_group(
        alpn_all,
        "PN (候選: class == 'ALPN' AND NOT type.starts_with('M_'))",
        "class + type",
        "class == 'ALPN' AND NOT type.starts_with('M_')",
        uni_mask,
    )

    # ------------------------------------------------------------------
    # Sanity check：同 task.md 預期數量級對比
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("=== Sanity check (對比 task.md 預期數量級) ===")
    print(f"{'=' * 70}")
    print(f"KC   : {kc.height} 粒 (預期約 4000)")
    print(f"MBON : {mbon.height} 粒, {mbon['type'].n_unique()} 個 type (預期 100-200 粒, 30-100 個 type)")
    print(f"PN   : {pn.height} 粒, {pn['type'].n_unique()} 個 type (預期 100-200 粒)")


if __name__ == "__main__":
    main()
