import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from wiffle import characters
from wiffle.labels import MASK

STREAMS = [["de", "alto", "et", "basso"], ["arbitros", "et", "arbitri", MASK], ["basso", "arbitri"]]


def test_subwords_of_words_equal_char_wb_on_the_joined_text():
    vocabulary = characters.Subwords((3, 5), min_df=1)
    presence = vocabulary.fit(*characters.texts_by_words(STREAMS))
    joined = [" ".join(word for word in stream if word != MASK) for stream in STREAMS]
    reference = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), binary=True, lowercase=False).fit(joined)
    assert sorted(vocabulary.names) == sorted(reference.vocabulary_)
    expected = reference.transform(joined)[:, [reference.vocabulary_[name] for name in vocabulary.names]]
    assert (abs(expected - presence) > 0).nnz == 0


def test_min_df_keeps_subwords_of_enough_texts():
    vocabulary = characters.Subwords((3, 5), min_df=2)
    vocabulary.fit(*characters.texts_by_words(STREAMS))
    assert " ba" in vocabulary.names and " de" not in vocabulary.names


def test_word_weight_is_the_sum_of_its_subwords_and_the_mask_holds_none():
    vocabulary = characters.Subwords((3, 5), min_df=1)
    vocabulary.fit(*characters.texts_by_words(STREAMS))
    weights = np.random.default_rng(3).normal(size=len(vocabulary.names))
    subwords_of_basso = vocabulary.of_words(["basso", MASK])
    assert np.isclose((subwords_of_basso @ weights)[0], weights[subwords_of_basso[0].indices].sum())
    assert subwords_of_basso[1].nnz == 0


def test_spelled_marks_the_word_boundaries():
    assert [characters.spelled(name) for name in (" arb", "ibus ", "rbitr", " et ")] == ["arb-", "-ibus", "-rbitr-", "et"]
