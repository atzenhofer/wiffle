# Changelog

## Unreleased

### Added

- `wiffle.web.run`: a run on texts given as JSON, for a browser worker under Pyodide. The labels come from the seeds,
  or are given with each text, and then the seeds, which may be none, are only masked. It writes the results as one
  ZIP with the TSV files and `run.json`, and the tables as JSON.
- `texts.tsv` holds the `name` and the `source` of each text, where the input has these columns.

### Fixed

- `wiffle find` on platforms with 32-bit array indices, WebAssembly among them, where it stopped with a TypeError
  in `numpy.repeat`.
- A run in which no n-gram occurs in `--min-df` texts no longer stops with a ValueError: it names no formulas and
  still ranks the texts. When no subword occurs in `--min-df` texts either, the run stops with `NoFeatures`, and
  `wiffle find` says to lower `--min-df` or add texts.

## 0.1.0

### Added

- `wiffle count`: how many texts the seeds select, in the text, in the abstract or in either, and per seed.
- `wiffle find`: two classifiers trained on the texts the seeds select, with the seeds masked. One ranks the texts
  on character n-grams inside words; the other names the formulas, on word n-grams and phrase frames with one open
  word. Writes the formulas, word families, subwords, all texts ranked by score with the words behind each score,
  concordance lines and a run record.
- Input as Parquet, CSV or TSV, read by the file suffix. `--id-column`, `--text-column` and `--abstract-column`
  name the columns; the defaults are `id`, `text` and `abstract`.
