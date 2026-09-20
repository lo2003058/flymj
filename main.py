import polars as pl
import pyarrow as pa
import pyarrow.feather as feather

pl.Config.set_tbl_cols(50)
pl.Config.set_tbl_rows(40)

tbl = feather.read_table("data/raw/annotations.feather")

# Decode dictionary-encoded columns back to plain strings, so polars
# doesn't hit the same conversion issue again.
cols = []
for field in tbl.schema:
    col = tbl.column(field.name)
    if pa.types.is_dictionary(field.type):
        col = col.cast(field.type.value_type)
    cols.append(col)
tbl = pa.Table.from_arrays(cols, names=tbl.schema.names)

ann = pl.from_arrow(tbl)

print("=== SHAPE ===")
print(ann.shape)

print("\n=== COLUMNS ===")
for c, d in zip(ann.columns, ann.dtypes):
    print(f"  {c:<30} {d}")

print("\n=== HEAD ===")
print(ann.head(5))

# Auto-detect which column holds the cell type
for col in ann.columns:
    if ann[col].dtype != pl.String:
        continue
    vals = ann[col].drop_nulls()
    if vals.len() == 0:
        continue
    hits = vals.str.starts_with("KC").sum()
    if hits > 0:
        print(f"\n=== column '{col}' has {hits} rows starting with KC ===")
        print(
            ann.filter(pl.col(col).str.starts_with("KC"))
            .get_column(col)
            .value_counts(sort=True)
            .head(20)
        )
