# Changelog

## Unreleased

### Added

- `wiffle.web.run`: a run on texts and seeds given as JSON, for a browser worker under Pyodide. It writes the results
  as one ZIP with the TSV files and `run.json`, and the tables as JSON.

## 0.1.0

### Added

- `wiffle count`: how many texts the seeds select, in the text, in the abstract or in either, and per seed.
- `wiffle find`: two classifiers trained on the texts the seeds select, with the seeds masked. One ranks the texts
  on character n-grams inside words; the other names the formulas, on word n-grams and phrase frames with one open
  word. Writes the formulas, word families, subwords, all texts ranked by score with the words behind each score,
  concordance lines and a run record.
- Input as Parquet, CSV or TSV, read by the file suffix. `--id-column`, `--text-column` and `--abstract-column`
  name the columns; the defaults are `id`, `text` and `abstract`.
