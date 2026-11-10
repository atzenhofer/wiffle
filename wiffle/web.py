"""wiffle in a browser worker: a run on texts given as JSON, its results as one ZIP."""

import csv
import json
import zipfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .cli import FIELDS, seed_labels, write_results
from .find import Options, find
from .labels import Seeds

TABLES = ("formulas", "words", "subwords", "texts", "concordance")


def _memory_bytes() -> int | None:
    """The size of the Pyodide runtime's memory, where there is one."""
    try:
        import pyodide_js
    except ImportError:
        return None
    return pyodide_js._module.HEAPU8.length


def _rows(path: Path) -> list[list[str]]:
    """The rows of a TSV file, the header first."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream, delimiter="\t"))


def run(input_json: str, options_json: str, directory: str) -> str:
    """Runs find on the texts of input_json in a new directory and returns, as JSON, the paths of the results ZIP and
    of the tables as JSON.

    input_json holds `texts`, each with `id` and `text` and optionally `abstract`, `year` and `url`; `seeds`, the seed
    phrases; `label_from`, which is text, abstract or both; and `record`, fields for run.json. options_json holds
    settings of `find.Options` by name, ranges as lists."""
    given = json.loads(input_json)
    settings = json.loads(options_json)
    options = Options(**{name: tuple(value) if isinstance(value, list) else value for name, value in settings.items()})
    records = given["texts"]
    seeds = Seeds.of(given["seeds"])
    label_from = given.get("label_from", "both")
    labels = seed_labels(records, seeds, FIELDS[label_from])
    result = find([record["text"] or "" for record in records], labels, seeds, options)

    out = Path(directory)
    out.mkdir(parents=True)
    write_results(out, records, result)
    record = {**given.get("record", {}), "version": version("wiffle"), "seeds": list(seeds.phrases),
              "label_from": label_from, "options": asdict(options), "counts": result.counts,
              "seconds": {key: round(value, 2) for key, value in result.seconds.items()},
              "memory_bytes": _memory_bytes()}
    archive = out / "results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name in TABLES:
            zipped.write(out / f"{name}.tsv", f"{name}.tsv")
        zipped.writestr("run.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    tables = out / "tables.json"
    tables.write_text(json.dumps({name: _rows(out / f"{name}.tsv") for name in TABLES}, ensure_ascii=False),
                      encoding="utf-8")
    return json.dumps({"archive": str(archive), "tables": str(tables)})
