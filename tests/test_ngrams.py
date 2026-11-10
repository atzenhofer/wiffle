from sklearn.feature_extraction.text import CountVectorizer

from wiffle import ngrams
from wiffle.labels import MASK

STREAMS = [
    "ex una parte et ex altera parte".split(),
    "de alto et basso compromiserunt ex una parte".split(),
    ["in", MASK, "seu", "amicabiles", "compositores", "ex", "una", "parte"],
    "arbitratores seu amicabiles compositores de alto et basso".split(),
    "ex una".split(),
    [],
]


def reference(streams, orders, min_df):
    def analyzer(stream):
        return [" ".join(stream[start:start + order]) for order in orders
                for start in range(len(stream) - order + 1) if MASK not in stream[start:start + order]]
    vectorizer = CountVectorizer(analyzer=analyzer, min_df=min_df, binary=True).fit(streams)
    return vectorizer, vectorizer.transform(streams)


def test_presence_equals_count_vectorizer():
    matrix = ngrams.build(STREAMS, ngrams.contiguous(range(2, 5)), 2, MASK)
    vectorizer, expected = reference(STREAMS, range(2, 5), 2)
    names = [" ".join(matrix.gram(column)) for column in range(matrix.presence.shape[1])]
    assert sorted(names) == sorted(vectorizer.vocabulary_)
    columns = [vectorizer.vocabulary_[name] for name in names]
    assert (abs(expected[:, columns] - matrix.presence) > 0).nnz == 0


def test_no_ngram_crosses_a_text_or_the_mask():
    matrix = ngrams.build(STREAMS, ngrams.contiguous(range(2, 3)), 1, MASK)
    names = {" ".join(matrix.gram(column)) for column in range(matrix.presence.shape[1])}
    assert "parte de" not in names
    assert not any(MASK in name for name in names)


def test_frames_equal_a_reference_with_the_gap_mark():
    patterns = ngrams.frames(range(3, 5))
    assert patterns == [(0, 2), (0, 2, 3), (0, 1, 3)]
    matrix = ngrams.build(STREAMS, patterns, 2, MASK)

    def analyzer(stream):
        grams = []
        for pattern in patterns:
            span = pattern[-1] + 1
            for start in range(len(stream) - span + 1):
                window = stream[start:start + span]
                if MASK not in window:
                    grams.append(" ".join(word if offset in pattern else ngrams.GAP for offset, word in enumerate(window)))
        return grams

    vectorizer = CountVectorizer(analyzer=analyzer, min_df=2, binary=True).fit(STREAMS)
    names = [" ".join(matrix.gram(column)) for column in range(matrix.presence.shape[1])]
    assert sorted(names) == sorted(vectorizer.vocabulary_)
    assert "de alto _ basso" in names
    expected = vectorizer.transform(STREAMS)[:, [vectorizer.vocabulary_[name] for name in names]]
    assert (abs(expected - matrix.presence) > 0).nnz == 0
