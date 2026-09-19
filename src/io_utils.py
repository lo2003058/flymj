import polars as pl
import pyarrow as pa
import pyarrow.feather as feather


def load_feather(path: str) -> pl.DataFrame:
    """讀 annotations.feather / edges.feather。

    polars 嘅 pl.read_ipc() 讀呢批檔會拋
    `ComputeError: The dictionary key must fit in a usize, but -1 does not`，
    因為檔案有 dictionary-encoded 欄用 -1 表示 null。
    呢個 loader 先用 pyarrow 讀，將 dictionary 欄解返做普通欄先轉去 polars。
    """
    tbl = feather.read_table(path)
    cols = []
    for field in tbl.schema:
        col = tbl.column(field.name)
        if pa.types.is_dictionary(field.type):
            col = col.cast(field.type.value_type)
        cols.append(col)
    return pl.from_arrow(pa.Table.from_arrays(cols, names=tbl.schema.names))
