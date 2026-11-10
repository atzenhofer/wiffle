import json
import zipfile

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
