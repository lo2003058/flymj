import polars as pl
import pyarrow as pa
import pyarrow.feather as feather


def load_feather(path: str) -> pl.DataFrame:
    """Read annotations.feather / edges.feather.

    polars' `pl.read_ipc()` raises
    `ComputeError: The dictionary key must fit in a usize, but -1 does not`
    on these files, because they have dictionary-encoded columns that use
    -1 to represent null. This loader reads via pyarrow first, decoding
    dictionary columns back to plain columns before handing off to polars.
    """
    tbl = feather.read_table(path)
    cols = []
    for field in tbl.schema:
        col = tbl.column(field.name)
        if pa.types.is_dictionary(field.type):
            col = col.cast(field.type.value_type)
        cols.append(col)
    return pl.from_arrow(pa.Table.from_arrays(cols, names=tbl.schema.names))
