import numpy as np
import pytest

from wordviz.similarity import compute_distances, embedding_distance, n_most_similar


@pytest.mark.parametrize(
    "dist",
    [
        "cosine",
        "euclidean",
        "manhattan",
        "braycurtis",
        "canberra",
        "chebyshev",
        "dot",
        "pearson",
        "spearman",
    ],
)
def test_n_most_similar(loader_static, dist):
    target_word = loader_static.tokens[0]
    n = 10
    words, vectors, distances = n_most_similar(
        loader_static, target_word, dist=dist, n=n
    )

    assert len(words) == n
    assert len(vectors) == n
    assert len(distances) == n

    for j, word in enumerate(words):
        assert word in loader_static.tokens
        assert word != target_word
        assert np.array_equal(vectors[j], loader_static.get_embedding(word))

    assert distances == sorted(distances)


def test_n_most_similar_invalid_word(loader_static):
    with pytest.raises(ValueError):
        n_most_similar(loader_static, "nonexistent_word", dist="cosine", n=10)


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "cosine",
        "manhattan",
        "braycurtis",
        "canberra",
        "chebyshev",
        "dot",
        "pearson",
    ],
)
def test_compute_distances(metric):
    X = np.array([[1, 0, 2], [0, 1, 3], [2, 1, 0]])

    D = compute_distances(X, metric=metric)

    assert D.shape == (3, 3)
    if metric != "dot":
        assert np.allclose(np.diag(D), 0)
    assert np.allclose(D, D.T)


def test_compute_distances_spearman():
    X = np.array([[1, 2, 3], [3, 2, 1]])

    D = compute_distances(X, metric="spearman")

    assert D.shape == (2, 2)
    assert D[0, 1] == D[1, 0]


def test_compute_distances_invalid_metric():
    X = np.array([[1, 2]])

    with pytest.raises(ValueError):
        compute_distances(X, metric="unknown")


def test_embedding_distance(loader_static):
    word1 = loader_static.tokens[0]
    word2 = loader_static.tokens[1]

    dist_cosine = embedding_distance(loader_static, word1, word2, dist="cosine")

    assert isinstance(dist_cosine, float)


def test_embedding_distance_invalid_word(loader_static):
    with pytest.raises(ValueError):
        embedding_distance(
            loader_static, "nonexistent_word1", "nonexistent_word2", dist="cosine"
        )
