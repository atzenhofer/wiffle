from wiffle.labels import MASK, NUMBER, Seeds, concordance, masked_streams, tokens


def test_seed_matches_inside_words():
    seeds = Seeds.of(["Arbitrator", "dem andern tail"])
    assert seeds.match("tamquam arbitratores et amicabiles")
    assert seeds.match("an dem andern tail")
    assert not seeds.match("arbiter")


def test_tokens_mask_every_word_a_seed_touches():
    seeds = Seeds.of(["arbitr", "andern tail"])
    assert tokens("Ad arbitros, et an dem andern taile 1369", seeds) == ["ad", MASK, "et", "an", "dem", MASK, MASK, NUMBER]


def test_masked_streams_change_only_texts_with_seeds():
    texts = ["in arbitros", "ex una parte"]
    plain = [tokens(text) for text in texts]
    streams = masked_streams(texts, plain, Seeds.of(["arbitr"]))
    assert streams[0] == ["in", MASK]
    assert streams[1] is plain[1]


def test_concordance_marks_the_phrase():
    line = concordance("Quod, de Alto et Basso, compromiserunt", ["alto", "et", "basso"], width=10)
    assert line == "Quod, de [Alto et Basso], compromi"
