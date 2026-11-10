"""wiffle in a browser worker: a run on texts given as JSON, its results as one ZIP."""

import csv
import json
import zipfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

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


def labels_of(records: list[dict], seeds: Seeds | None, label_from: str) -> tuple[np.ndarray, str]:
    """The labels given with the texts, else the labels the seeds give in the fields of label_from; and their origin."""
    given = [record.get("labelled") for record in records]
    if all(value is not None for value in given):
        return np.array(given, dtype=bool), "given"
    if any(value is not None for value in given):
        raise ValueError("labels are given for some texts and not for others")
    if seeds is None:
        raise ValueError("a run without labels given needs seeds")
    return seed_labels(records, seeds, FIELDS[label_from]), label_from


def run(input_json: str, options_json: str, directory: str) -> str:
    """Runs find on the texts of input_json in a new directory and returns, as JSON, the paths of the results ZIP and
    of the tables as JSON.

    input_json holds `texts`, each with `id` and `text` and optionally `abstract`, `name`, `year`, `url`, `source`
    and `labelled`; `seeds`, the seed phrases; `label_from`, which is text, abstract or both; and `record`, fields for
    run.json. When every text has `labelled`, these are the labels and the seeds, which may be none, are only masked;
    else a seed labels a text where it occurs in the fields of `label_from`. options_json holds settings of
    `find.Options` by name, ranges as lists."""
    given = json.loads(input_json)
    settings = json.loads(options_json)
    options = Options(**{name: tuple(value) if isinstance(value, list) else value for name, value in settings.items()})
    records = given["texts"]
    phrases = given.get("seeds") or []
    seeds = Seeds.of(phrases) if any(phrase.strip() for phrase in phrases) else None
    labels, label_from = labels_of(records, seeds, given.get("label_from", "both"))
    result = find([record["text"] or "" for record in records], labels, seeds, options)

    out = Path(directory)
    out.mkdir(parents=True)
    write_results(out, records, result)
    record = {**given.get("record", {}), "version": version("wiffle"), "seeds": list(seeds.phrases) if seeds else [],
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
