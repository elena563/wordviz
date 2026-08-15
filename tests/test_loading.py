import numpy as np
from pytest import FixtureRequest
import pytest
from wordviz import EmbeddingLoader


@pytest.mark.slow
def test_load_pretrained():
    loader = EmbeddingLoader()
    embeddings = loader.load_pretrained("glove", "en", "wiki", "50d")
    assert embeddings is not None
    assert isinstance(embeddings, np.ndarray)


@pytest.mark.parametrize(
    "file, format",
    [
        ("glove_txt_file", "glove"),
        ("fasttext_bin_file", "fasttext"),
        ("word2vec_bin_file", "word2vec"),
    ],
)
def test_load_from_file(file, format, request: FixtureRequest):
    loader = EmbeddingLoader()
    file_path = request.getfixturevalue(file)
    embeddings = loader.load_from_file(str(file_path), format)
    assert isinstance(embeddings, np.ndarray)
