"""KC / MBON / PN label 定義。

呢三個定義已經喺 explore_labels.py 用 print 出嚟嘅結果同使用者核對過
（見 data/doc/task.md）：
  - KC   : type 欄 starts_with('KC')                          （已確認）
  - MBON : class 欄 == 'MBON'                                  （已確認，97 粒 / 37 type）
  - PN   : class 欄 == 'ALPN'，剔走 type 開頭 'M_'（multiglomerular） （已確認，387 粒 / 101 type）

387 呢個數字比 task.md 原先預期嘅 100-200 高，推測係因為 100-200 嗰個預期
嚟自單邊腦嘅舊文獻，而 male CNS 呢個 dataset 兩邊腦都有，387 大約等於單邊
~190 x2。已經同使用者確認用 387。

呢個 module 係單一嚟源，等 explore_labels.py 同 build_masks.py 兩個
script 用緊嘅定義保證一致，唔會各自維護一份走數。
"""

import polars as pl


def is_fragment(df: pl.DataFrame) -> pl.Series:
    """instance 欄含 'fragment'（大小寫不分）嘅係未完整重建，要排除。"""
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
