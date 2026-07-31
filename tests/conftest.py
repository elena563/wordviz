import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pytest
from wordviz.loading import EmbeddingLoader
from wordviz.plotting import Visualizer

@pytest.fixture(scope='session')
def sample_embeddings():
    np.random.seed(42)
    return np.random.randn(50, 384)

@pytest.fixture(scope='session')
def loader_static():
    loader_static = EmbeddingLoader()
    # let pytest find file
    file_path = os.path.join(os.path.dirname(__file__), 'embedding_100words.txt')
    loader_static.load_from_file(file_path, 'word2vec')
    return loader_static

@pytest.fixture(scope='session')
def vis_static():
    return Visualizer(loader_static)

@pytest.fixture(scope='session')
def loader():
    loader = EmbeddingLoader()
    np.random.seed(42)
    fake_embeddings = np.random.randn(50, 384)
    fake_labels = [f"This is test sentence number {i}." for i in range(50)]
    fake_classes = ['class1' for _ in range(len(fake_embeddings)//2)] + ['class2' for _ in range(len(fake_embeddings)//2)]
    loader.load_contextual(fake_embeddings, fake_labels, 'sentence', fake_classes)
    return loader

@pytest.fixture(scope='session')
def vis(loader):
    return Visualizer(loader)