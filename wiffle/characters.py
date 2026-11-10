"""Subwords: character n-grams inside words, padded at the word ends, over a vocabulary of those found in min_df texts.

The subwords of a text are the subwords of its distinct words, so they are computed once per word and joined to
the texts by a sparse product.
"""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .labels import MASK


def texts_by_words(streams: list[list[str]]) -> tuple[list[str], sparse.csr_matrix]:
    """The distinct words of the streams, and the texts by those words, 1 where a text contains a word."""
    vocabulary: dict[str, int] = {}
    rows, columns = [], []
    for row, stream in enumerate(streams):
        for word in set(stream):
            rows.append(row)
            columns.append(vocabulary.setdefault(word, len(vocabulary)))
    presence = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)),
                                 shape=(len(streams), len(vocabulary)))
    return list(vocabulary), presence


def binary(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """The matrix with every stored value set to one."""
    matrix = matrix.tocsr()
    matrix.data[:] = 1
    return matrix


def spelled(subword: str) -> str:
    """A subword with its word boundaries as hyphens: arb- begins a word, -ibus ends one, -rbitr- sits inside."""
    return f"{'' if subword.startswith(' ') else '-'}{subword.strip()}{'' if subword.endswith(' ') else '-'}"


class Subwords:
    """The subword vocabulary of a corpus and the presence of its subwords in words and texts."""

    def __init__(self, grams: tuple[int, int], min_df: int):
        self.grams = grams
        self.min_df = min_df
        self.names: np.ndarray = np.array([], dtype=object)
        self.counter: CountVectorizer | None = None

    def vectorizer(self, vocabulary=None) -> CountVectorizer:
        """Character n-grams inside words, padded at the word ends, as presence."""
        return CountVectorizer(analyzer="char_wb", ngram_range=self.grams, binary=True, lowercase=False,
                               dtype=np.float32, vocabulary=vocabulary)

    def of_words(self, words: list[str]) -> sparse.csr_matrix:
        """Words by the subwords of the vocabulary; the mask word holds none."""
        if self.counter is None:
            raise ValueError("the subword vocabulary is not fitted")
        return self.counter.transform(["" if word == MASK else word for word in words]).tocsr()

    def fit(self, words: list[str], texts: sparse.csr_matrix) -> sparse.csr_matrix:
        """Learn the subwords found in at least min_df texts, and return the texts by those subwords."""
        counter = self.vectorizer()
        word_subwords = counter.fit_transform(["" if word == MASK else word for word in words]).tocsr()
        presence = binary(texts @ word_subwords)
        frequent = np.flatnonzero(np.bincount(presence.indices, minlength=presence.shape[1]) >= self.min_df)
        self.names = counter.get_feature_names_out()[frequent]
        self.counter = self.vectorizer(vocabulary=self.names)
        return presence[:, frequent].tocsr()

    def presence(self, words: list[str], texts: sparse.csr_matrix) -> sparse.csr_matrix:
        """The texts by the subwords of the vocabulary, for texts given by their words."""
        return binary(texts @ self.of_words(words))
