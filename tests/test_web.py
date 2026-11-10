import json
import zipfile

import pytest

from wiffle import web

LABELLED = [f"letter {number}: a leak under the sink and water came through the ceiling" for number in range(12)]
UNSEEDED = [f"note {number}: water came through the ceiling after the storm" for number in range(4)]
OTHERS = [f"note {number}: the rent was paid on time and the hall light was fixed" for number in range(24)]


def test_run_writes_one_zip_with_the_tables_and_the_record(tmp_path):
    texts = [{"id": f"t{position}", "text": text} for position, text in enumerate(LABELLED + UNSEEDED + OTHERS)]
    given = {"texts": texts, "seeds": ["leak"], "label_from": "text", "record": {"app": "wiffle", "basket": "letters"}}
    paths = json.loads(web.run(json.dumps(given), json.dumps({"refits": 2, "formula_grams": [3, 5]}),
                               str(tmp_path / "run")))
    with zipfile.ZipFile(paths["archive"]) as zipped:
        names = set(zipped.namelist())
        record = json.loads(zipped.read("run.json"))
    tables = json.loads((tmp_path / "run" / "tables.json").read_text(encoding="utf-8"))
    assert names == {"formulas.tsv", "words.tsv", "subwords.tsv", "texts.tsv", "concordance.tsv", "run.json"}
    assert record["basket"] == "letters" and record["counts"]["labelled"] == 12
    assert record["options"]["refits"] == 2
    assert "water came through the ceiling" in tables["formulas"][1][1]
    ranked = [row[1] for row in tables["texts"][1:]]
    assert len(ranked) == 40 and set(ranked[12:16]) == {"t12", "t13", "t14", "t15"}


def test_run_takes_the_labels_given_and_only_masks_the_seeds(tmp_path):
    bodies = LABELLED + UNSEEDED + OTHERS
    texts = [{"id": f"t{position}", "text": text, "labelled": position < 16, "source": "edited" if position % 2 else "",
              "name": f"Letter {position}"} for position, text in enumerate(bodies)]
    given = {"texts": texts, "seeds": ["leak"], "record": {}}
    paths = json.loads(web.run(json.dumps(given), json.dumps({"refits": 2}), str(tmp_path / "run")))
    with zipfile.ZipFile(paths["archive"]) as zipped:
        record = json.loads(zipped.read("run.json"))
    tables = json.loads((tmp_path / "run" / "tables.json").read_text(encoding="utf-8"))
    assert record["label_from"] == "given" and record["counts"]["labelled"] == 16
    assert not any("leak" in cell for row in tables["formulas"][1:] for cell in row)
    header, *rows = tables["texts"]
    source = {row[header.index("id")]: row[header.index("source")] for row in rows}
    name = {row[header.index("id")]: row[header.index("name")] for row in rows}
    assert source["t1"] == "edited" and source["t2"] == "" and name["t2"] == "Letter 2"


def test_run_takes_the_labels_given_without_seeds(tmp_path):
    texts = [{"id": f"t{position}", "text": text, "labelled": position < 12}
             for position, text in enumerate(LABELLED + OTHERS)]
    given = {"texts": texts, "seeds": [], "record": {}}
    paths = json.loads(web.run(json.dumps(given), json.dumps({"refits": 2}), str(tmp_path / "run")))
    with zipfile.ZipFile(paths["archive"]) as zipped:
        record = json.loads(zipped.read("run.json"))
    assert record["seeds"] == [] and record["counts"]["labelled"] == 12


def test_run_without_labels_given_needs_seeds(tmp_path):
    given = {"texts": [{"id": "t0", "text": LABELLED[0]}], "seeds": [" "], "record": {}}
    with pytest.raises(ValueError, match="needs seeds"):
        web.run(json.dumps(given), json.dumps({}), str(tmp_path / "run"))
