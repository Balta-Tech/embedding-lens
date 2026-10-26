import numpy as np
import pytest

from lens.readout import METHODS, Space, sparse_decompose, unit


@pytest.fixture
def space():
    rng = np.random.default_rng(0)
    text = unit(rng.normal(size=(500, 32)))
    # Videos share a common offset, like the modality gap
    reference = unit(rng.normal(size=(80, 32)) + 3 * unit(rng.normal(size=32)))
    neutral = unit(rng.normal(size=32))
    return Space([f"p{i}" for i in range(500)], text, reference, neutral=neutral)


def test_neutral_baseline_does_not_change_the_ranking(space):
    v = space.reference[0]
    assert [p for p, _ in space.top(v, 10, "raw")] == [p for p, _ in space.top(v, 10, "neutral")]


@pytest.mark.parametrize("method", METHODS)
def test_rank_of_matches_the_vocabulary_ranking(space, method):
    v = space.reference[3]
    phrase, _ = space.top(v, 5, method)[2]
    index = space.phrases.index(phrase)
    _, ranks, percentiles = space.rank_of(v, space.vocab.text[[index]], method)
    # A vocabulary phrase scored as an "extra" phrase lands exactly at its own place
    assert ranks[0] == 3
    assert 99 < percentiles[0] <= 100


def test_sparse_decompose_recovers_a_planted_mixture(space):
    planted = [10, 200, 333]
    centered = space.vocab.centered
    v = unit(space.video_mean + unit(0.6 * centered[10] + 0.3 * centered[200] + 0.2 * centered[333]))
    terms, fit = sparse_decompose(v, space, n_terms=3)
    assert {p for p, _ in terms} == {f"p{i}" for i in planted}
    assert terms[0][0] == "p10"
    assert fit > 0.99
