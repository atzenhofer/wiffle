# wiffle

wiffle finds the language that sets a group of texts apart from other texts. It labels the texts in which one of a
few seed phrases occurs, masks the seeds and trains two linear classifiers on these labels. It then writes the
subwords, word families and formulas that weigh most for the labelled texts, and ranks all texts by how much they
read like them, also the texts that hold no seed.

## Install

wiffle needs Python 3.12 or later and [uv](https://docs.astral.sh/uv/). To install the `wiffle` command:

```sh
uv tool install "wiffle[parquet] @ git+https://github.com/atzenhofer/wiffle"
```

To work in a clone:

```sh
git clone https://github.com/atzenhofer/wiffle
cd wiffle
uv sync
uv run wiffle --help
uv run pytest -q
```

The `parquet` extra installs pyarrow, which wiffle needs for Parquet input only.

## Input

The texts are a table with one row per text, in Parquet (`.parquet`), CSV (`.csv`) or TSV (`.tsv`). The file
suffix decides how wiffle reads the table. CSV and TSV files are UTF-8 and start with a header line. wiffle reads
these columns:

- `id`: the id of the text, copied to the output (`--id-column`).
- `text`: the text (`--text-column`). The classifiers learn from this column only.
- `abstract`: an optional second text, such as an abstract or a summary (`--abstract-column`). `--label-from`
  says where a seed labels a text: in `text`, in `abstract` or in `both` (the default).
- `name`, `year`, `url` and `source`: optional, copied to `texts.tsv`; `name` is how a reader knows the text, such
  as a shelfmark, and `source` says where the text comes from, such as an edition or a machine reading.

The seeds file is UTF-8 text with one seed phrase per line. Empty lines and lines that start with `#` are skipped.
A seed matches anywhere in the lowercased text, also inside a word: the seed `leak` matches `Leaking`.
Before training, every word that a seed touches is masked, and no n-gram spans a masked word. The classifiers
learn from the words around the seeds.

## Commands

`wiffle count DATA --seeds SEEDS` prints how many texts the seeds select: in the text column, in the abstract
column, in either, and for each seed.

`wiffle find DATA --seeds SEEDS --out FOLDER` trains the two classifiers and writes the results into FOLDER.
`wiffle find --help` lists all options. Both classifiers are L2-regularized logistic regression with balanced
class weights.

- The ranking fit scores every text. Its features are the character n-grams of three to five characters inside
  words (`--rank-characters`); these are the subwords. A word weighs as much as the sum of its subwords, so
  spelling variants of a word, which share most of their subwords, get similar weights. The fit runs twice: the
  second fit leaves the best-scored unlabelled texts, twice as many as there are labelled texts, out of the
  negatives (`--no-two-step` fits once).
- The formula fit names the formulas. Its features are word n-grams of three to five words (`--formula-grams`)
  and phrase frames of four and five words with one open word, such as `in the _ of` (`--formula-frames`,
  `--no-formula-frames`). Refits on subsamples of 80 percent of the texts (`--refits`, 10 by default) measure
  how stable each n-gram is.

An n-gram or a subword counts when it occurs in at least `--min-df` texts (3 by default). A formula is
reported when it occurs in at least `--formula-min-labelled` labelled texts (5 by default) and at most half of
its words are numbers.

## Output

`wiffle find` writes six files into the output folder. In n-grams, `_` stands for the open word of a phrase frame
and `<num>` for a number. In subwords, a hyphen marks where the word goes on: `ceil-` begins a word, `-ing` ends
one, `-eili-` is inside one. `labelled` and `others` count the labelled texts and the other texts.

`formulas.tsv` holds the strongest n-grams of the formula fit, grouped where they overlap in the labelled texts.

- `rank`, `span`: the group as the merged span that covers most words in the labelled texts.
- `best`: the n-gram of the group with the highest weight; `weight` is its weight.
- `stability`: the largest share of refits in which an n-gram of the group is among the `--top` strongest.
- `labelled`, `others`: the texts that contain an n-gram of the group.
- `members`: the n-grams of the group, strongest first.

`words.tsv` holds word families: the 300 strongest words of the ranking fit, grouped by a shared stem. Spellings are
folded first (u and v, i, j and y, ae and oe, ci and ti, np and mp, tz and z, doubled letters, accents), and two
words share a stem when they share a prefix of five letters or more and of half the shorter word or more. A compound
whose modifier is a word of the texts joins the family of its stem, as zuspruch joins spruch. A family needs
`--min-df` labelled texts. The families are candidates for new seeds.

- `rank`, `family`: the stem that names the family, with a hyphen where it is no word of the family.
- `weight`: the weight of the best word of the family.
- `labelled`, `others`: the texts that contain a word of the family.
- `words`: the words with their counts as `labelled/others`, most labelled texts first.

`subwords.tsv` holds the 100 strongest subwords of the ranking fit. Subwords that overlap in the word that holds
them in most labelled texts are merged into one span.

- `rank`, `subword`: the merged span.
- `weight`, `labelled`, `others`: the weight of the strongest piece, and the texts that contain it.
- `pieces`: the subwords of the span, strongest first.
- `examples`: up to three words that hold the strongest piece, most labelled texts first.

`texts.tsv` holds every text, ranked by the score of the ranking fit. The texts without a seed that rank high read
like the labelled texts.

- `rank`, `id`, `name`, `score`: the score is the decision value of the ranking fit.
- `labelled`: 1 when a seed labels the text, else 0.
- `name`, `year`, `url`, `source`: from the input, when it has these columns.
- `reasons`: up to three words of the text that raise its score most.

`concordance.tsv` holds context lines for the 50 strongest formulas: up to three from labelled texts and three
from other texts, those with the best rank first.

- `formula`, `best`: the rank of the formula in `formulas.tsv` and its best n-gram.
- `id`, `labelled`, `text_rank`: the text and its rank in `texts.tsv`.
- `line`: the first occurrence in brackets, with up to 60 characters on each side.

`run.json` records the input file and its SHA-256, the columns, the seeds, `--label-from`, all settings, the
counts and the time of each step in seconds.

## Example

`letters.csv`:

```
id,text
1,There is a leak under the sink and water came through the ceiling.
2,The leak is back: water came through the ceiling of the kitchen.
3,A small leak in the bathroom pipe. Water came through the ceiling again.
4,Last night water came through the ceiling after the storm.
5,The garden gate is broken and the lock is stiff.
6,Please send a new key for the bicycle room.
7,The hall light is out and the stairs are dark.
8,The rent for March was paid on time.
9,Noise from the street goes on late into the night.
10,The bins were not collected this week.
```

`seeds.txt` holds one line, `leak`. The example has three labelled texts, so it lowers `--formula-min-labelled`
and `--refits`:

```
$ wiffle count letters.csv --seeds seeds.txt
records             10
selected by text     3
selected by abstract 0
selected by both     3
  leak  text 3  abstract 0
$ wiffle find letters.csv --seeds seeds.txt --out results --formula-min-labelled 2 --refits 2
3 of 10 texts labelled, 1 formulas, 4 word families, 5 subwords; results in results
```

`results/formulas.tsv` names the formula `water came through the ceiling`, found in the three labelled texts and
in one other. In `results/texts.tsv`, text 4 holds no seed and ranks right after the three labelled texts:

```
rank  id  name  score    labelled  year  url  source  reasons
1     3          3.8533  1                            ceiling | through | water
2     2          3.6293  1                            ceiling | through | water
3     1          3.0739  1                            ceiling | through | water
4     4          2.7761  0                            ceiling | through | water
5     8         -0.8542  0                            time | was
```

## License

Apache-2.0, see `LICENSE`.
