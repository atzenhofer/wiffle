import csv

import pyarrow
import pyarrow.parquet as parquet
import pytest

from wiffle.cli import read_records

ROWS = [
    {"key": "a1", "body": "The court heard both parties.", "summary": "A hearing."},
    {"key": "a2", "body": 'Rent is paid "in full" at Easter,\nin two parts.', "summary": ""},
]


def write(path, rows):
    """Write rows as Parquet, CSV or TSV by the suffix of the path."""
    if path.suffix == ".parquet":
        parquet.write_table(pyarrow.Table.from_pylist(rows), path)
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t" if path.suffix == ".tsv" else ",")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("suffix", [".parquet", ".csv", ".tsv"])
def test_records_come_from_the_named_columns(tmp_path, suffix):
    path = tmp_path / f"texts{suffix}"
    write(path, ROWS)
    records = read_records(path, "key", "body", "summary")
    assert [record["id"] for record in records] == ["a1", "a2"]
    assert [record["text"] for record in records] == [row["body"] for row in ROWS]
    assert records[0]["abstract"] == "A hearing."


def test_a_missing_column_stops_the_read(tmp_path):
    path = tmp_path / "texts.csv"
    write(path, ROWS)
    with pytest.raises(SystemExit, match="no column text"):
        read_records(path, "key", "text", "abstract")
