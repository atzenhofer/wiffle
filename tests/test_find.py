import numpy as np
import pytest

from wiffle.find import Options, find
from wiffle.labels import Seeds

FILLER = ("dominus abbas conventus ecclesie bona curia vinea pratum census annuatim solvendum testes sigillum anno "
          "domini datum feria sancti martini item quod predicti fratres").split()


def corpus(seed: int = 7) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Background texts, labelled texts with the seed and a planted formula, and unlabelled texts with the formula."""
    rng = np.random.default_rng(seed)

    def filler(size: int) -> str:
        return " ".join(rng.choice(FILLER, size=size))

    texts, labelled, planted = [], [], []
    for _ in range(80):
        texts.append(filler(40)); labelled.append(False); planted.append(False)
    for _ in range(20):
        texts.append(f"{filler(15)} in arbitros de alto et basso compromiserunt {filler(15)}")
        labelled.append(True); planted.append(True)
    for _ in range(10):
        texts.append(f"{filler(15)} de alto et basso compromiserunt {filler(15)}")
        labelled.append(False); planted.append(True)
    return texts, np.array(labelled), np.array(planted)


def test_find_names_the_planted_formula_and_never_the_seed():
    texts, labelled, planted = corpus()
    seeds = Seeds.of(["arbitr"])
    result = find(texts, labelled, seeds, Options(min_df=2, refits=2, top=20, words=10))
    named = [result.formulas[0].span, *(member for formula in result.formulas for member in formula.members),
             *(word.word for word in result.words),
             *(word.word for family in result.families for word in family.members),
             *(example for subword in result.subwords for example in subword.examples)]
    assert not any("arbitr" in text for text in named)
    assert "alto et basso" in result.formulas[0].span
    unlabelled = ~labelled
    assert result.scores[unlabelled & planted].min() > np.median(result.scores[unlabelled & ~planted])


def test_two_step_keeps_as_many_negatives_as_labelled_texts():
    texts, labelled, _ = corpus()
    few_negatives = np.concatenate([np.flatnonzero(labelled), np.flatnonzero(~labelled)[:30]])
    result = find([texts[row] for row in few_negatives], labelled[few_negatives], Seeds.of(["arbitr"]),
                  Options(min_df=2, refits=1, top=10, words=5))
    assert len(result.scores) == len(few_negatives)


def test_overlapping_pieces_join_into_one_run():
    from wiffle.find import overlapping_runs
    assert overlapping_runs([(4, 8, 2), (0, 4, 1), (2, 6, 3), (9, 12, 4)]) == [(0, 8, [1, 3, 2]), (9, 12, [4])]


def test_families_have_labelled_support_and_list_members_by_evidence():
    texts, labelled, _ = corpus()
    options = Options(min_df=2, refits=1, top=20, words=30)
    result = find(texts, labelled, Seeds.of(["arbitr"]), options)
    assert result.families
    for family in result.families:
        assert family.labelled >= options.min_df
        counts = [word.labelled for word in family.members]
        assert counts == sorted(counts, reverse=True)


def test_reportable_needs_labelled_support_and_words_beyond_numbers():
    from wiffle import ngrams
    from wiffle.find import reportable
    from wiffle.labels import MASK, NUMBER
    streams = [["de", "alto", "et", "basso", NUMBER, "die", NUMBER]] * 6 + [["ex", "una", "parte"]] * 6
    labels = np.array([True] * 5 + [False] * 7)
    matrix = ngrams.build(streams, ngrams.contiguous(range(3, 4)), 2, MASK)
    allowed = {" ".join(matrix.gram(column)): bool(flag) for column, flag in enumerate(reportable(matrix, labels, 5))}
    assert allowed["de alto et"] and allowed["basso <num> die"]
    assert not allowed["<num> die <num>"]
    assert not allowed["ex una parte"]


def test_a_stem_is_a_long_shared_prefix_of_the_folded_spellings():
    from wiffle.characters import folded
    from wiffle.find import same_stem
    assert folded("Conpositores") == folded("compositores") and folded("spruchlewt") == folded("spruchleut")
    assert same_stem(folded("arbitrium"), folded("arbitrando"))
    assert same_stem(folded("arbitrium"), folded("arbiter"))
    assert not same_stem(folded("concordiam"), folded("concessa"))
    assert not same_stem(folded("transkription"), folded("tractatus"))


def test_a_compound_needs_a_modifier_found_in_the_texts():
    from wiffle.find import compound_of
    attested = {"zu", "rechts", "recht"}
    assert compound_of("zuspruch", "spruch", attested)
    assert compound_of("rechtsspruch", "spruch", attested)
    assert not compound_of("waarbii", "arbi", attested)


def test_families_join_the_forms_and_compounds_of_a_stem_and_nothing_else():
    from scipy import sparse
    from wiffle.find import Word, families_of
    vocabulary = ["arbitrium", "arbitros", "waarbij", "spruch", "zuspruch", "zu", "tamquam"]
    words = [Word(word, 1.0 - position / 10, 3, 0) for position, word in enumerate(vocabulary) if word != "zu"]
    tokens = np.array([vocabulary.index(word.word) for word in words])
    present = sparse.csc_matrix(np.ones((4, len(vocabulary)), dtype=np.float32))
    families = families_of(words, tokens, vocabulary, present, np.array([True] * 3 + [False]), 3)
    assert {family.subword: [word.word for word in family.members] for family in families} == {
        "arbitr-": ["arbitrium", "arbitros"], "waarbij": ["waarbij"], "spruch": ["spruch", "zuspruch"],
        "tamquam": ["tamquam"]}


def test_a_fit_without_ngrams_names_no_formula_and_still_ranks():
    rng = np.random.default_rng(3)
    texts = [f"arbitri {word}" for word in rng.choice(FILLER, size=12)] + list(rng.choice(FILLER, size=30))
    labels = np.array([True] * 12 + [False] * 30)
    result = find(texts, labels, Seeds.of(["arbitri"]), Options(min_df=2, refits=2))
    assert result.counts["ngrams"] == 0 and result.formulas == []
    assert len(result.scores) == len(texts)


def test_a_fit_without_any_feature_stops_with_no_features():
    from wiffle.find import NoFeatures
    texts, labelled, _ = corpus()
    with pytest.raises(NoFeatures, match="no subword occurs in 1000 texts"):
        find(texts, labelled, Seeds.of(["arbitr"]), Options(min_df=1000))
    with pytest.raises(NoFeatures, match="no word that is not masked"):
        find(["arbitri", "arbitrium", "", ""], np.array([True, True, False, False]), Seeds.of(["arbitr"]))
