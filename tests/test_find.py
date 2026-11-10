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


def test_naming_subword_prefers_four_letters_or_more():
    from wiffle.find import naming_subword
    columns = np.array([0, 1, 2])
    weights = np.array([0.9, 0.5, 0.2])
    long_enough = np.array([False, True, True])
    assert naming_subword(columns, weights, long_enough) == 1
    assert naming_subword(np.array([0]), weights, long_enough) == 0


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
