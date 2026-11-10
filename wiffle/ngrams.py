"""Word n-grams of a corpus as a sparse presence matrix, counted by integer hashes of token ids.

An n-gram is read through a pattern of offsets: (0, 1, 2) is a contiguous trigram, and (0, 1, 3) a phrase frame of
four words whose third word is open.
"""

from dataclasses import dataclass
from itertools import chain

import numpy as np
from scipy import sparse

PRIME = np.uint64(1_000_003)
GAP = "_"

Pattern = tuple[int, ...]


def contiguous(orders: range) -> list[Pattern]:
    """The patterns of the contiguous n-grams of the given orders."""
    return [tuple(range(order)) for order in orders]


def frames(lengths: range) -> list[Pattern]:
    """The patterns of the phrase frames of the given lengths, each with one open word inside."""
    return [tuple(offset for offset in range(length) if offset != slot)
            for length in lengths for slot in range(1, length - 1)]


@dataclass(frozen=True)
class NgramMatrix:
    """Texts by n-grams, 1 where a text contains the n-gram, with what is needed to spell each column."""

    presence: sparse.csr_matrix
    token_ids: np.ndarray
    text_starts: np.ndarray
    words: list[str]
    patterns: list[Pattern]
    column_starts: np.ndarray
    column_patterns: np.ndarray
    column_orders: np.ndarray

    def pattern(self, column: int) -> Pattern:
        """The offset pattern of a column."""
        return self.patterns[self.column_patterns[column]]

    def gram(self, column: int) -> tuple[str, ...]:
        """The n-gram of a column as words, with the gap mark at each open position."""
        start = self.column_starts[column]
        pattern = self.pattern(column)
        present = set(pattern)
        return tuple(self.words[self.token_ids[start + offset]] if offset in present else GAP
                     for offset in range(pattern[-1] + 1))

    def key(self, column: int) -> tuple[int, tuple[int, ...]]:
        """The pattern index and the token ids of a column, as `occurrences` finds them in a text."""
        start = self.column_starts[column]
        ids = tuple(int(self.token_ids[start + offset]) for offset in self.pattern(column))
        return int(self.column_patterns[column]), ids

    def text_ids(self, row: int) -> list[int]:
        """The token ids of one text."""
        return self.token_ids[self.text_starts[row]:self.text_starts[row + 1]].tolist()


def occurrences(ids: list[int], patterns: dict[int, Pattern]):
    """Every window of a text under the given patterns: start, end and the key of `NgramMatrix.key`."""
    for index, pattern in patterns.items():
        span = pattern[-1] + 1
        for start in range(len(ids) - span + 1):
            yield start, start + span, (index, tuple(ids[start + offset] for offset in pattern))


def first_of_runs(values: np.ndarray, *more: np.ndarray) -> np.ndarray:
    """True at each position where a sorted sequence (of one or more aligned arrays) changes."""
    changes = np.ones(len(values), dtype=bool)
    changes[1:] = values[1:] != values[:-1]
    for other in more:
        changes[1:] |= other[1:] != other[:-1]
    return changes


def build(streams: list[list[str]], patterns: list[Pattern], min_df: int, blocked: str | None = None) -> NgramMatrix:
    """The presence matrix of all n-grams of the given patterns found in at least min_df texts.

    An n-gram never crosses the end of a text, and no window that holds the blocked token counts,
    an open position included.
    """
    vocabulary: dict[str, int] = {}
    per_text = [[vocabulary.setdefault(token, len(vocabulary)) for token in stream] for stream in streams]
    lengths = np.fromiter(map(len, per_text), dtype=np.int64, count=len(per_text))
    text_starts = np.concatenate(([0], np.cumsum(lengths)))
    token_ids = np.fromiter(chain.from_iterable(per_text), dtype=np.uint64, count=int(text_starts[-1]))
    text_of = np.repeat(np.arange(len(per_text), dtype=np.int64), lengths)
    blocked_id = vocabulary.get(blocked) if blocked else None
    blocked_before = None
    if blocked_id is not None:
        blocked_before = np.concatenate(([0], np.cumsum(token_ids == blocked_id)))

    rows, columns, column_starts, column_patterns = [], [], [], []
    width = 0
    for index, pattern in enumerate(patterns):
        span = pattern[-1] + 1
        count = len(token_ids) - span + 1
        if count <= 0:
            continue
        starts = np.arange(count, dtype=np.int64)
        inside = text_of[starts] == text_of[starts + span - 1]
        if blocked_before is not None:
            inside &= blocked_before[starts + span] == blocked_before[starts]
        starts = starts[inside]

        keys = token_ids[starts + pattern[0]].copy()
        for offset in pattern[1:]:
            keys = keys * PRIME + token_ids[starts + offset]
        keys = keys * PRIME + np.uint64(index + 1)
        texts = text_of[starts]

        ordering = np.lexsort((texts, keys))
        keys, texts, starts = keys[ordering], texts[ordering], starts[ordering]
        unique_pairs = first_of_runs(keys, texts)
        keys, texts, starts = keys[unique_pairs], texts[unique_pairs], starts[unique_pairs]

        new_gram = first_of_runs(keys)
        gram_of_pair = np.cumsum(new_gram) - 1
        frequent = np.bincount(gram_of_pair) >= min_df
        column_of_gram = np.full(len(frequent), -1, dtype=np.int64)
        column_of_gram[frequent] = np.arange(width, width + int(frequent.sum()))
        column_of_pair = column_of_gram[gram_of_pair]
        kept = column_of_pair >= 0

        rows.append(texts[kept])
        columns.append(column_of_pair[kept])
        column_starts.append(starts[new_gram][frequent])
        column_patterns.append(np.full(int(frequent.sum()), index, dtype=np.int64))
        width += int(frequent.sum())

    empty = np.zeros(0, dtype=np.int64)
    row_index = np.concatenate(rows) if rows else empty
    column_index = np.concatenate(columns) if columns else empty
    presence = sparse.csr_matrix(
        (np.ones(len(row_index), dtype=np.float32), (row_index, column_index)), shape=(len(streams), width)
    )
    pattern_of_column = np.concatenate(column_patterns) if column_patterns else empty
    order_of_pattern = np.array([len(pattern) for pattern in patterns], dtype=np.int64)
    return NgramMatrix(
        presence=presence,
        token_ids=token_ids,
        text_starts=text_starts,
        words=list(vocabulary),
        patterns=list(patterns),
        column_starts=np.concatenate(column_starts) if column_starts else empty,
        column_patterns=pattern_of_column,
        column_orders=order_of_pattern[pattern_of_column] if len(pattern_of_column) else empty,
    )
