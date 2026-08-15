import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
from gensim.models import KeyedVectors, FastText
from gensim.models.fasttext import save_facebook_model

import pytest
from wordviz import EmbeddingLoader, Visualizer, Visualizer3D


@pytest.fixture(scope="session")
def sample_embeddings():
    np.random.seed(42)
    return np.random.randn(50, 384)


@pytest.fixture(scope="session")
def loader_static():
    loader_static = EmbeddingLoader()
    # let pytest find file
    file_path = os.path.join(os.path.dirname(__file__), "embedding_100words.txt")
    loader_static.load_from_file(file_path, "word2vec")
    return loader_static


@pytest.fixture(scope="session")
def vis_static():
    return Visualizer(loader_static)


@pytest.fixture(scope="session")
def loader():
    loader = EmbeddingLoader()
    np.random.seed(42)
    fake_embeddings = np.random.randn(50, 384)
    fake_labels = [f"This is test sentence number {i}." for i in range(50)]
    fake_classes = ["class1" for _ in range(len(fake_embeddings) // 2)] + [
        "class2" for _ in range(len(fake_embeddings) // 2)
    ]
    loader.load_contextual(fake_embeddings, fake_labels, "sentence", fake_classes)
    return loader


@pytest.fixture(scope="session")
def vis(loader):
    return Visualizer(loader)


@pytest.fixture(scope="session")
def vis3d(loader):
    return Visualizer3D(loader)


@pytest.fixture(scope="session")
def word2vec_bin_file(tmp_path_factory):
    kv = KeyedVectors(vector_size=5)
    kv.add_vectors(
        ["hello", "world", "embedding", "test"],
        np.random.default_rng(42).normal(size=(4, 5)).astype(np.float32),
    )
    path = tmp_path_factory.mktemp("word2vec") / "model.bin"
    kv.save_word2vec_format(str(path), binary=True)
    return path


@pytest.fixture(scope="session")
def glove_txt_file(tmp_path_factory):
    rng = np.random.default_rng(42)
    words = ["hello", "world", "embedding", "test"]
    path = tmp_path_factory.mktemp("glove") / "glove.txt"
    with open(path, "w") as f:
        for word in words:
            f.write(
                word + " " + " ".join(f"{x:.6f}" for x in rng.normal(size=5)) + "\n"
            )
    return path


@pytest.fixture(scope="session")
def fasttext_bin_file(tmp_path_factory):
    sentences = [
        ["hello", "world", "embedding", "test"],
        ["another", "sentence", "hello", "world"],
    ]
    model = FastText(vector_size=5, min_count=1, window=3, seed=42)
    model.build_vocab(corpus_iterable=sentences)
    model.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=5)
    path = tmp_path_factory.mktemp("fasttext") / "model.bin"
    save_facebook_model(model, str(path))
    return path
