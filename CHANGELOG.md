# Changelog

## Unreleased

### Added

- `wiffle count`: how many texts the seeds select, in the text, in the abstract or in either, and per seed.
- `wiffle find`: two classifiers trained on the texts the seeds select, with the seeds masked. One ranks the texts
  on character n-grams inside words; the other names the formulas, on word n-grams and phrase frames with one open
  word. Writes the formulas, word families, subwords, all texts ranked by score with the words behind each score,
  concordance lines and a run record.
- Parquet input with the columns `id`, `text` and optionally `abstract`.
