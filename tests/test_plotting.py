from wordviz.loading import EmbeddingLoader
from wordviz.plotting import Visualizer
import os
import pytest

@pytest.fixture(scope='module')
def vis():
    loader = EmbeddingLoader()
    # let pytest find file
    file_path = os.path.join(os.path.dirname(__file__), 'embedding_100words.txt')
    loader.load_from_file(file_path, 'word2vec')
    return Visualizer(loader)

@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_embeddings(vis, red_method):
    fig, ax = vis.plot_embeddings(red_method=red_method)
    assert fig is not None
    assert ax is not None

@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_clusters(vis, red_method):
    fig, ax = vis.plot_clusters(red_method=red_method)
    assert fig is not None
    assert ax is not None

@pytest.mark.parametrize("dist", [
    "cosine", "euclidean", "manhattan", "chebyshev", "dot", "pearson", "spearman"
])
@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_similarity(vis, dist, red_method):
    fig, ax = vis.plot_similarity("cat", dist=dist, red_method=red_method)
    assert fig is not None
    assert ax is not None

@pytest.mark.parametrize("dist", [
    "cosine", "euclidean", "manhattan", "chebyshev", "dot", "pearson", "spearman"
])
@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_topography(vis, dist, red_method):
    fig = vis.plot_topography(dist=dist, red_method=red_method)
    assert fig is not None

@pytest.mark.parametrize("dist", [
    "cosine", "euclidean", "manhattan", "chebyshev", "dot", "pearson", "spearman"
])
def test_similarity_heatmap(vis, dist):
    fig = vis.similarity_heatmap(dist=dist)
    assert fig is not None

@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_interactive_embeddings(vis, red_method):
    fig = vis.interactive_embeddings(red_method=red_method)
    assert fig is not None
