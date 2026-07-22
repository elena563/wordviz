import numpy as np
import pytest
from wordviz.loading import EmbeddingLoader
from wordviz.plotting import Visualizer

@pytest.fixture(scope='session')
def sample_embeddings():
    np.random.seed(42)
    return np.random.randn(50, 384)

'''
@pytest.fixture(scope='module')
def vis():
    loader = EmbeddingLoader()
    # let pytest find file
    file_path = os.path.join(os.path.dirname(__file__), 'embedding_100words.txt')
    loader.load_from_file(file_path, 'word2vec')
    return Visualizer(loader)
'''

@pytest.fixture(scope='session')
def vis():
    loader = EmbeddingLoader()
    np.random.seed(42)
    fake_embeddings = np.random.randn(50, 384)
    fake_labels = [f"This is test sentence number {i}." for i in range(50)]
    loader.load_contextual(fake_embeddings, fake_labels, 'sentence')
    return Visualizer(loader)