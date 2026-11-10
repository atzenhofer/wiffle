"""Find formulas: classifiers trained on seed labels, their strongest n-grams merged into spans, and the texts they rank.

One run fits twice. The ranking fit on subwords (character n-grams inside words) scores the texts; a word weighs as
much as its subwords there, and joins the family of its strongest subword. The formula fit on word n-grams and
phrase frames names the formulas.
"""

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from . import characters, ngrams
from .labels import MASK, NUMBER, Seeds, masked_streams, tokens

UNLABELLED_DROPPED_PER_LABEL = 2
FAMILY_LETTERS = 4


@dataclass(frozen=True)
class Options:
    """Settings of one run. Ranges are (shortest, longest): characters for the ranking fit, words for the formulas."""

    rank_characters: tuple[int, int] = (3, 5)
    two_step: bool = True
    formula_grams: tuple[int, int] = (3, 5)
    formula_frames: tuple[int, int] | None = (4, 5)
    formula_min_labelled: int = 5
    min_df: int = 3
    regularization: float = 1.0
    refits: int = 10
    fraction: float = 0.8
    top: int = 200
    words: int = 300
    subwords: int = 100
    random_state: int = 42


@dataclass
class Formula:
    """N-grams that overlap in the labelled texts, merged into one span."""

    span: str
    best: str
    column: int
    weight: float
    stability: float
    labelled: int
    others: int
    members: list[str]


@dataclass
class Word:
    """A word that raises the score of the ranking fit, weighed by its character n-grams."""

    word: str
    weight: float
    labelled: int
    others: int


@dataclass
class Subword:
    """Overlapping character n-grams of one word that raise the score of the ranking fit, merged into one span."""

    subword: str
    weight: float
    labelled: int
    others: int
    pieces: list[str]
    examples: list[str]


@dataclass
class Family:
    """Words that share their strongest subword, or hold the subword of another family; the best word names it."""

    subword: str
    weight: float
    labelled: int
    others: int
    members: list[Word]


@dataclass
class Result:
    """What a run found, and what is needed to explain a score."""

    formulas: list[Formula]
    words: list[Word]
    families: list[Family]
    subwords: list[Subword]
    scores: np.ndarray
    labels: np.ndarray
    word_scores: np.ndarray
    formula_weights: np.ndarray
    stability: np.ndarray
    matrix: ngrams.NgramMatrix
    counts: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)

    def reasons(self, row: int, count: int = 3) -> list[str]:
        """The words of a text that raise its score the most."""
        ids = np.unique(self.matrix.token_ids[self.matrix.text_starts[row]:self.matrix.text_starts[row + 1]]).astype(np.int64)
        raising = ids[self.word_scores[ids] > 0]
        strongest = raising[np.argsort(self.word_scores[raising])[::-1][:count]]
        return [self.matrix.words[token] for token in strongest]


def classifier(options: Options, random_state: int) -> LogisticRegression:
    """The linear model of a run: L2 logistic regression with balanced class weights."""
    return LogisticRegression(
        C=options.regularization, class_weight="balanced", solver="liblinear", dual=True,
        max_iter=2000, random_state=random_state,
    )


def strongest(weights: np.ndarray, count: int) -> np.ndarray:
    """The columns with the highest positive weights, best first."""
    positive = np.flatnonzero(weights > 0)
    return positive[np.argsort(weights[positive])[::-1][:count]]


def stratified_sample(labels: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Row indices drawn without replacement, the same fraction from each class."""
    chosen = []
    for value in (True, False):
        rows = np.flatnonzero(labels == value)
        chosen.append(rng.choice(rows, size=max(1, round(fraction * len(rows))), replace=False))
    return np.sort(np.concatenate(chosen))


def selection_frequency(presence, labels: np.ndarray, options: Options) -> np.ndarray:
    """For each column, the share of refits on subsamples in which it is among the top columns."""
    rng = np.random.default_rng(options.random_state)
    selected = np.zeros(presence.shape[1], dtype=np.float64)
    for refit in range(options.refits):
        rows = stratified_sample(labels, options.fraction, rng)
        model = classifier(options, options.random_state + refit + 1).fit(presence[rows], labels[rows])
        selected[strongest(model.coef_.ravel(), options.top)] += 1
    return selected / max(1, options.refits)


def holder_counts(presence, columns: list[int], labels: np.ndarray) -> tuple[int, int]:
    """How many labelled and how many other texts contain at least one of the columns."""
    present = presence[:, columns].max(axis=1).toarray().ravel() > 0
    return int((present & labels).sum()), int((present & ~labels).sum())


def fit_ranking(presence, labels: np.ndarray, options: Options) -> LogisticRegression:
    """The ranking model; with `two_step`, fitted again without the best-scored unlabelled texts among the negatives.

    The second fit drops at most as many texts as leave the negatives as many as the labelled texts; when none
    can be dropped, the first fit is the model.
    """
    model = classifier(options, options.random_state).fit(presence, labels)
    if not options.two_step:
        return model
    unlabelled = np.flatnonzero(~labels)
    labelled = int(labels.sum())
    count = min(UNLABELLED_DROPPED_PER_LABEL * labelled, len(unlabelled) - labelled)
    if count <= 0:
        return model
    first = model.decision_function(presence)
    dropped = unlabelled[np.argsort(-first[unlabelled])[:count]]
    kept = np.ones(len(labels), dtype=bool)
    kept[dropped] = False
    return classifier(options, options.random_state).fit(presence[kept], labels[kept])


class Groups:
    """Union-find over columns."""

    def __init__(self):
        self.parent: dict[int, int] = {}

    def root(self, column: int) -> int:
        """The representative column of the group a column belongs to."""
        self.parent.setdefault(column, column)
        while self.parent[column] != column:
            self.parent[column] = self.parent[self.parent[column]]
            column = self.parent[column]
        return column

    def join(self, first: int, second: int) -> None:
        """Put the groups of two columns together."""
        self.parent[self.root(second)] = self.root(first)


def merge(matrix: ngrams.NgramMatrix, columns: np.ndarray, rows: np.ndarray) -> tuple[dict[int, int], dict[int, Counter]]:
    """Group the columns whose n-grams overlap in the given texts, and count the merged spans of each group."""
    lookup = {matrix.key(int(column)): int(column) for column in columns}
    used = {index: matrix.patterns[index] for index in sorted({int(matrix.column_patterns[column]) for column in columns})}
    groups = Groups()
    runs = []
    for row in rows:
        ids = matrix.text_ids(row)
        found = sorted((start, end, lookup[key]) for start, end, key in ngrams.occurrences(ids, used) if key in lookup)
        run_start, run_end, run_columns = None, -1, []
        for start, end, column in found + [(len(ids) + 1, len(ids) + 1, None)]:
            if start < run_end:
                groups.join(run_columns[0], column)
                run_columns.append(column)
                run_end = max(run_end, end)
                continue
            if run_columns:
                runs.append((run_columns[0], " ".join(matrix.words[token] for token in ids[run_start:run_end])))
            run_start, run_end, run_columns = start, end, [column]
    group_of = {int(column): groups.root(int(column)) for column in columns}
    spans: dict[int, Counter] = {}
    for column, text in runs:
        spans.setdefault(groups.root(column), Counter())[text] += 1
    return group_of, spans


def widest(spans: Counter) -> str:
    """The span that covers most words in the labelled texts (occurrences times length), the shorter one on a tie."""
    return min(spans.items(), key=lambda item: (-item[1] * len(item[0].split()), len(item[0])))[0]


def reportable(matrix: ngrams.NgramMatrix, labels: np.ndarray, min_labelled: int) -> np.ndarray:
    """Columns that may be reported as formulas: in min_labelled labelled texts, at most half of their words numbers."""
    labelled_counts = np.asarray(matrix.presence[labels].sum(axis=0)).ravel()
    numbers = np.zeros(matrix.presence.shape[1], dtype=np.int64)
    if NUMBER in matrix.words:
        number_id = matrix.words.index(NUMBER)
        for index, pattern in enumerate(matrix.patterns):
            columns = np.flatnonzero(matrix.column_patterns == index)
            starts = matrix.column_starts[columns]
            for offset in pattern:
                numbers[columns] += matrix.token_ids[starts + offset] == number_id
    return (labelled_counts >= min_labelled) & (2 * numbers <= matrix.column_orders)


def formulas_of(matrix: ngrams.NgramMatrix, weights: np.ndarray, stability: np.ndarray, labels: np.ndarray,
                count: int) -> list[Formula]:
    """The strongest n-grams of a fit, merged into formulas, best first."""
    top = strongest(weights, count)
    group_of, spans = merge(matrix, top, np.flatnonzero(labels))
    members: dict[int, list[int]] = {}
    for column in top:
        members.setdefault(group_of[int(column)], []).append(int(column))
    top_presence = matrix.presence[:, top].tocsc()
    position_of = {int(column): position for position, column in enumerate(top)}
    formulas = []
    for root, columns in members.items():
        best = max(columns, key=lambda column: weights[column])
        labelled, others = holder_counts(top_presence, [position_of[column] for column in columns], labels)
        formulas.append(Formula(
            span=widest(spans[root]) if root in spans else " ".join(matrix.gram(best)),
            best=" ".join(matrix.gram(best)),
            column=best,
            weight=float(weights[best]),
            stability=float(stability[columns].max()),
            labelled=labelled,
            others=others,
            members=[" ".join(matrix.gram(column)) for column in sorted(columns, key=lambda column: -weights[column])],
        ))
    formulas.sort(key=lambda formula: -formula.weight)
    return formulas


def word_presence(matrix: ngrams.NgramMatrix) -> sparse.csc_matrix:
    """Texts by the words of the vocabulary, 1 where a text contains the word."""
    lengths = np.diff(matrix.text_starts)
    rows = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    present = sparse.csc_matrix((np.ones(len(rows), dtype=np.float32), (rows, matrix.token_ids.astype(np.int64))),
                                shape=(len(lengths), len(matrix.words)))
    present.data[:] = 1
    return present


def words_of(matrix: ngrams.NgramMatrix, word_scores: np.ndarray, present: sparse.csc_matrix, labels: np.ndarray,
             count: int, min_df: int) -> tuple[list[Word], np.ndarray]:
    """The words with the highest positive scores among those found in at least min_df texts, best first."""
    frequent = np.diff(present.indptr) >= min_df
    top = strongest(np.where(frequent, word_scores, 0.0), count)
    words = []
    for token in top:
        holders = present.indices[present.indptr[token]:present.indptr[token + 1]]
        labelled = int(labels[holders].sum())
        words.append(Word(matrix.words[token], float(word_scores[token]), labelled, len(holders) - labelled))
    return words, top


def naming_subword(columns: np.ndarray, weights: np.ndarray, long_enough: np.ndarray) -> int:
    """The strongest of the subwords of FAMILY_LETTERS letters or more among the columns, else the strongest of all."""
    candidates = columns[long_enough[columns]]
    if not len(candidates):
        candidates = columns
    return int(candidates[np.argmax(weights[candidates])])


def families_of(words: list[Word], tokens_of_words: np.ndarray, word_subwords: sparse.csr_matrix,
                weights: np.ndarray, names: np.ndarray, present: sparse.csc_matrix, labels: np.ndarray,
                min_df: int) -> list[Family]:
    """The words grouped by their naming subword, best family first.

    A word is named by its strongest subword of FAMILY_LETTERS letters or more, or by its strongest subword when it
    has none of that length. Two groups merge when the best word of one holds the subword of FAMILY_LETTERS letters
    or more that names the other. A family needs min_df labelled texts; its members come in the order of their
    labelled texts.
    """
    long_enough = np.fromiter((len(name.strip()) >= FAMILY_LETTERS for name in names), dtype=bool, count=len(names))
    strongest_of: dict[int, int] = {}
    held: dict[int, set[int]] = {}
    for token in map(int, tokens_of_words):
        columns = word_subwords.indices[word_subwords.indptr[token]:word_subwords.indptr[token + 1]]
        if len(columns):
            strongest_of[token] = naming_subword(columns, weights, long_enough)
            held[token] = set(columns.tolist())
    best_of: dict[int, int] = {}
    for token in map(int, tokens_of_words):
        if token in strongest_of:
            best_of.setdefault(strongest_of[token], token)
    groups = Groups()
    for column, token in best_of.items():
        for other in best_of.keys() & held[token]:
            if other != column and long_enough[other]:
                groups.join(column, other)
    grouped: dict[int, list[tuple[Word, int]]] = {}
    for word, token in zip(words, map(int, tokens_of_words), strict=True):
        if token in strongest_of:
            grouped.setdefault(groups.root(strongest_of[token]), []).append((word, token))
    families = []
    for members in grouped.values():
        holders = np.unique(np.concatenate([present.indices[present.indptr[token]:present.indptr[token + 1]]
                                            for _, token in members]))
        labelled = int(labels[holders].sum())
        if labelled < min_df:
            continue
        best, best_token = max(members, key=lambda member: member[0].weight)
        ordered = sorted((word for word, _ in members), key=lambda word: (-word.labelled, -word.weight))
        families.append(Family(characters.spelled(names[strongest_of[best_token]]), best.weight, labelled,
                               len(holders) - labelled, ordered))
    families.sort(key=lambda family: -family.weight)
    return families


def overlapping_runs(pieces: list[tuple[int, int, int]]) -> list[tuple[int, int, list[int]]]:
    """Pieces given as (start, end, column) in one word, joined where they overlap, as (start, end, columns)."""
    runs: list[tuple[int, int, list[int]]] = []
    for start, end, column in sorted(pieces):
        if runs and start < runs[-1][1]:
            run_start, run_end, columns = runs[-1]
            runs[-1] = (run_start, max(run_end, end), columns + [column])
        else:
            runs.append((start, end, [column]))
    return runs


def subwords_of(weights: np.ndarray, names: np.ndarray, presence: sparse.csr_matrix, word_subwords: sparse.csr_matrix,
                words: list[str], word_labelled: np.ndarray, labels: np.ndarray, count: int) -> list[Subword]:
    """The strongest subwords, best first; pieces that overlap in their main word are merged into one span.

    The main word of a piece is the word that holds it in most labelled texts.
    """
    labelled_counts = np.asarray(presence[labels].sum(axis=0)).ravel()
    totals = np.bincount(presence.indices, minlength=presence.shape[1])
    holders = word_subwords.tocsc()
    examples_of: dict[int, list[int]] = {}
    pieces_by_word: dict[int, list[tuple[int, int, int]]] = {}
    for column in map(int, strongest(weights, count)):
        carriers = holders.indices[holders.indptr[column]:holders.indptr[column + 1]]
        examples_of[column] = carriers[np.argsort(-word_labelled[carriers], kind="stable")[:3]].tolist()
        if examples_of[column]:
            main = examples_of[column][0]
            start = f" {words[main]} ".find(names[column])
            pieces_by_word.setdefault(main, []).append((start, start + len(names[column]), column))
    subwords = []
    for main, pieces in pieces_by_word.items():
        padded = f" {words[main]} "
        for start, end, columns in overlapping_runs(pieces):
            best = max(columns, key=lambda column: weights[column])
            ordered = sorted(columns, key=lambda column: -weights[column])
            subwords.append(Subword(characters.spelled(padded[start:end]), float(weights[best]),
                                    int(labelled_counts[best]), int(totals[best] - labelled_counts[best]),
                                    [characters.spelled(names[column]) for column in ordered],
                                    [words[token] for token in examples_of[best]]))
    subwords.sort(key=lambda subword: -subword.weight)
    return subwords


def pattern_name(pattern: ngrams.Pattern) -> str:
    """A pattern spelled with w for a word and the gap mark for an open position."""
    present = set(pattern)
    return " ".join("w" if offset in present else ngrams.GAP for offset in range(pattern[-1] + 1))


def find(texts: list[str], labels: np.ndarray, mask: Seeds | None, options: Options = Options(),
         plain: list[list[str]] | None = None) -> Result:
    """Train on the labels with the masked seeds removed, then rank the texts and name words and formulas.

    `plain` are the unmasked token streams of the texts, if already at hand.
    """
    labels = np.asarray(labels, dtype=bool)
    if labels.all() or not labels.any():
        raise ValueError("the labels need texts on both sides")
    seconds = {}
    clock = time.perf_counter()
    streams = [tokens(text, mask) for text in texts] if plain is None else masked_streams(texts, plain, mask)
    patterns = ngrams.contiguous(range(options.formula_grams[0], options.formula_grams[1] + 1))
    if options.formula_frames:
        patterns += ngrams.frames(range(options.formula_frames[0], options.formula_frames[1] + 1))
    matrix = ngrams.build(streams, patterns, options.min_df, MASK)
    present = word_presence(matrix)
    vocabulary = characters.Subwords(options.rank_characters, options.min_df)
    rank_presence = vocabulary.fit(matrix.words, present.tocsr())
    seconds["features"] = time.perf_counter() - clock

    clock = time.perf_counter()
    ranking = fit_ranking(rank_presence, labels, options)
    scores = ranking.decision_function(rank_presence)
    rank_weights = ranking.coef_.ravel()
    word_subwords = vocabulary.of_words(matrix.words)
    word_scores = np.asarray(word_subwords @ rank_weights).ravel()
    for marker in (MASK, NUMBER):
        if marker in matrix.words:
            word_scores[matrix.words.index(marker)] = 0.0
    seconds["rank_fit"] = time.perf_counter() - clock

    clock = time.perf_counter()
    formula_weights = classifier(options, options.random_state).fit(matrix.presence, labels).coef_.ravel()
    seconds["formula_fit"] = time.perf_counter() - clock

    clock = time.perf_counter()
    stability = selection_frequency(matrix.presence, labels, options)
    seconds["refits"] = time.perf_counter() - clock

    clock = time.perf_counter()
    allowed = reportable(matrix, labels, options.formula_min_labelled)
    formulas = formulas_of(matrix, np.where(allowed, formula_weights, 0.0), stability, labels, options.top)
    words, tokens_of_words = words_of(matrix, word_scores, present, labels, options.words, options.min_df)
    families = families_of(words, tokens_of_words, word_subwords, rank_weights, vocabulary.names, present, labels,
                           options.min_df)
    word_labelled = np.asarray(present.T @ labels.astype(np.float32)).ravel()
    subwords = subwords_of(rank_weights, vocabulary.names, rank_presence, word_subwords, matrix.words, word_labelled,
                           labels, options.subwords)
    seconds["merge"] = time.perf_counter() - clock

    counts = {
        "texts": len(texts),
        "labelled": int(labels.sum()),
        "tokens": int(matrix.text_starts[-1]),
        "subwords": int(rank_presence.shape[1]),
        "ngrams": int(matrix.presence.shape[1]),
        "ngrams_by_pattern": {pattern_name(pattern): int((matrix.column_patterns == index).sum())
                              for index, pattern in enumerate(matrix.patterns)},
    }
    return Result(formulas, words, families, subwords, scores, labels, word_scores, formula_weights, stability, matrix,
                  counts, seconds)
