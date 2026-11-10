"""The wiffle command: count what seeds select, and find the formulas and texts of their class."""

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .find import Options, Result, find
from .labels import Seeds, concordance

FIELDS = {"text": ("text",), "abstract": ("abstract",), "both": ("text", "abstract")}
CONCORDANCE_FORMULAS = 50
CONCORDANCE_LINES = 3
DELIMITERS = {".csv": ",", ".tsv": "\t"}
LONGEST_FIELD = 2**31 - 1


def read_table(path: Path) -> tuple[list[str], list[dict]]:
    """The column names and the rows of a Parquet, CSV or TSV file, read by its suffix."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise SystemExit("reading Parquet needs pyarrow: install wiffle[parquet]") from error
        table = parquet.read_table(path)
        return table.column_names, table.to_pylist()
    if suffix not in DELIMITERS:
        raise SystemExit(f"{path}: the input must end in .parquet, .csv or .tsv")
    # csv refuses fields longer than 131,072 characters by default
    csv.field_size_limit(LONGEST_FIELD)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=DELIMITERS[suffix])
        rows = list(reader)
        return list(reader.fieldnames or ()), rows


def read_records(path: Path, id_column: str, text_column: str, abstract_column: str) -> list[dict]:
    """Records with `id`, `text` and `abstract` from the named columns, and `year` and `url` if the input has them."""
    names, rows = read_table(path)
    missing = [name for name in (id_column, text_column) if name not in names]
    if missing:
        raise SystemExit(f"{path}: no column {' or '.join(missing)}; the columns are {', '.join(names)}")
    return [{"id": row[id_column], "text": row[text_column], "abstract": row.get(abstract_column),
             "year": row.get("year"), "url": row.get("url")} for row in rows]


def seed_labels(records: list[dict], seeds: Seeds, fields: tuple[str, ...]) -> np.ndarray:
    """True for each record in which a seed occurs in one of the fields."""
    return np.array([any(seeds.match(record.get(name)) for name in fields) for record in records])


def file_digest(path: Path) -> str:
    """The SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count(arguments: argparse.Namespace) -> None:
    """Print how many records the seeds select, by field."""
    records = read_records(arguments.data, arguments.id_column, arguments.text_column, arguments.abstract_column)
    seeds = Seeds.read(arguments.seeds)
    lowered = {name: [(record.get(name) or "").lower() for record in records] for name in FIELDS["both"]}
    print(f"records\t{len(records)}")
    for name, fields in FIELDS.items():
        selected = sum(any(phrase in lowered[field][position] for field in fields for phrase in seeds.phrases)
                       for position in range(len(records)))
        print(f"selected by {name}\t{selected}")
    for phrase in seeds.phrases:
        print(f"  {phrase}\ttext {sum(phrase in text for text in lowered['text'])}"
              f"\tabstract {sum(phrase in text for text in lowered['abstract'])}")


def write_tsv(path: Path, header: list[str], rows) -> None:
    """Write rows as tab-separated values under a header line."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_results(out: Path, records: list[dict], result: Result) -> None:
    """Write formulas.tsv, words.tsv, subwords.tsv, texts.tsv and concordance.tsv of a run."""
    write_tsv(out / "formulas.tsv",
              ["rank", "span", "best", "weight", "stability", "labelled", "others", "members"],
              ([rank, formula.span, formula.best, f"{formula.weight:.4f}", f"{formula.stability:.2f}",
                formula.labelled, formula.others, " | ".join(formula.members)]
               for rank, formula in enumerate(result.formulas, 1)))
    write_tsv(out / "words.tsv", ["rank", "family", "weight", "labelled", "others", "words"],
              ([rank, family.subword, f"{family.weight:.4f}", family.labelled, family.others,
                " | ".join(f"{word.word} {word.labelled}/{word.others}" for word in family.members)]
               for rank, family in enumerate(result.families, 1)))
    write_tsv(out / "subwords.tsv", ["rank", "subword", "weight", "labelled", "others", "pieces", "examples"],
              ([rank, subword.subword, f"{subword.weight:.4f}", subword.labelled, subword.others,
                " | ".join(subword.pieces), " | ".join(subword.examples)]
               for rank, subword in enumerate(result.subwords, 1)))

    order = np.argsort(-result.scores)
    write_tsv(out / "texts.tsv", ["rank", "id", "score", "labelled", "year", "url", "reasons"],
              ([rank, records[row]["id"], f"{result.scores[row]:.4f}", int(result.labels[row]),
                records[row].get("year") or "", records[row].get("url") or "", " | ".join(result.reasons(row))]
               for rank, row in enumerate(order, 1)))

    rank_of = np.empty(len(order), dtype=np.int64)
    rank_of[order] = np.arange(len(order))
    presence = result.matrix.presence.tocsc()
    lines = []
    for rank, formula in enumerate(result.formulas[:CONCORDANCE_FORMULAS], 1):
        phrase = formula.best.split(" ")
        holders = presence.indices[presence.indptr[formula.column]:presence.indptr[formula.column + 1]]
        holders = holders[np.argsort(rank_of[holders])]
        for labelled in (True, False):
            chosen = 0
            for row in holders[result.labels[holders] == labelled]:
                if chosen == CONCORDANCE_LINES:
                    break
                line = concordance(records[row]["text"], phrase)
                if line:
                    lines.append([rank, formula.best, records[row]["id"], int(labelled), int(rank_of[row]) + 1, line])
                    chosen += 1
    write_tsv(out / "concordance.tsv", ["formula", "best", "id", "labelled", "text_rank", "line"], lines)


def add_run_options(command: argparse.ArgumentParser) -> None:
    """The settings of a run as command-line options, with the defaults of `Options`."""
    defaults = Options()
    command.add_argument("--rank-characters", type=int, nargs=2, default=defaults.rank_characters,
                         metavar=("SHORTEST", "LONGEST"), help="character n-gram lengths of the fit that ranks the texts")
    command.add_argument("--two-step", action=argparse.BooleanOptionalAction, default=defaults.two_step,
                         help="fit the ranking again without the best-scored unlabelled texts among the negatives")
    command.add_argument("--formula-grams", type=int, nargs=2, default=defaults.formula_grams,
                         metavar=("SHORTEST", "LONGEST"), help="word n-gram orders of the fit that names formulas")
    command.add_argument("--formula-frames", type=int, nargs=2, default=defaults.formula_frames,
                         metavar=("SHORTEST", "LONGEST"), help="lengths of the phrase frames with one open word")
    command.add_argument("--no-formula-frames", dest="formula_frames", action="store_const", const=None,
                         help="leave the phrase frames out of the formula fit")
    command.add_argument("--formula-min-labelled", type=int, default=defaults.formula_min_labelled,
                         help="fewest labelled texts a reported formula occurs in")
    command.add_argument("--min-df", type=int, default=defaults.min_df, help="fewest texts an n-gram occurs in")


def options_from(arguments: argparse.Namespace, **fixed) -> Options:
    """The settings of parsed command-line options; the fixed settings take precedence."""
    frames = tuple(arguments.formula_frames) if arguments.formula_frames else None
    settings = {"rank_characters": tuple(arguments.rank_characters), "two_step": arguments.two_step,
                "formula_grams": tuple(arguments.formula_grams), "formula_frames": frames,
                "formula_min_labelled": arguments.formula_min_labelled, "min_df": arguments.min_df}
    return Options(**(settings | fixed))


def find_command(arguments: argparse.Namespace) -> None:
    """Run find and write its results and run.json into the output folder."""
    records = read_records(arguments.data, arguments.id_column, arguments.text_column, arguments.abstract_column)
    seeds = Seeds.read(arguments.seeds)
    labels = seed_labels(records, seeds, FIELDS[arguments.label_from])
    options = options_from(arguments, refits=arguments.refits, top=arguments.top)
    result = find([record["text"] or "" for record in records], labels, seeds, options)
    arguments.out.mkdir(parents=True, exist_ok=True)
    write_results(arguments.out, records, result)
    run = {
        "app": "wiffle",
        "version": version("wiffle"),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"path": str(arguments.data), "sha256": file_digest(arguments.data)},
        "columns": {"id": arguments.id_column, "text": arguments.text_column, "abstract": arguments.abstract_column},
        "seeds": list(seeds.phrases),
        "label_from": arguments.label_from,
        "options": asdict(options),
        "counts": result.counts,
        "seconds": {key: round(value, 2) for key, value in result.seconds.items()},
    }
    (arguments.out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{result.counts['labelled']} of {result.counts['texts']} texts labelled, "
          f"{len(result.formulas)} formulas, {len(result.families)} word families, {len(result.subwords)} subwords; "
          f"results in {arguments.out}", file=sys.stderr)


def add_input_options(command: argparse.ArgumentParser) -> None:
    """The input file, its columns and the seeds file as command-line arguments."""
    command.add_argument("data", type=Path, help="Parquet, CSV or TSV file, read by its suffix")
    command.add_argument("--seeds", type=Path, required=True, help="one seed phrase per line")
    command.add_argument("--id-column", default="id", help="column of the text ids")
    command.add_argument("--text-column", default="text", help="column of the texts")
    command.add_argument("--abstract-column", default="abstract",
                         help="column of a second text such as an abstract, if the input has one")


def parser() -> argparse.ArgumentParser:
    """The command line of wiffle with its two commands."""
    command = argparse.ArgumentParser(prog="wiffle", description=__doc__)
    commands = command.add_subparsers(dest="command", required=True)

    counting = commands.add_parser("count", help="how many records the seeds select")
    add_input_options(counting)
    counting.set_defaults(handler=count)

    finding = commands.add_parser("find", help="formulas and texts of the class the seeds select")
    add_input_options(finding)
    finding.add_argument("--out", type=Path, required=True, help="folder for the results")
    finding.add_argument("--label-from", choices=FIELDS, default="both",
                         help="where a seed labels a text: the text column, the abstract column or both")
    add_run_options(finding)
    defaults = Options()
    finding.add_argument("--refits", type=int, default=defaults.refits, help="refits on subsamples for stability")
    finding.add_argument("--top", type=int, default=defaults.top, help="n-grams kept per fit")
    finding.set_defaults(handler=find_command)
    return command


def main() -> None:
    """Run the command given on the command line."""
    arguments = parser().parse_args()
    arguments.handler(arguments)
