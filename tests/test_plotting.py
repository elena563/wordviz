import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import pytest


@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_embeddings(vis, red_method):
    fig, ax = vis.plot_embeddings(red_method=red_method)
    assert 'pca' in vis.reduced
    assert vis.reduced['pca'].shape[1] == 2
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

def test_plot_cache_reuse(vis, mocker):
    vis.plot_embeddings(red_method='pca')
    cached = vis.reduced.get('pca')
    assert cached is not None
    
    vis.plot_embeddings(red_method='pca')
    assert vis.reduced.get('pca') is cached

@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_clusters(vis, red_method):
    fig, ax = vis.plot_clusters(red_method=red_method)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

@pytest.fixture
def target(vis):
    return vis.loader.tokens[1]

@pytest.mark.parametrize("dist", [
    "cosine", "euclidean", "manhattan", "chebyshev", "dot", "pearson", "spearman"
])
@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_similarity(vis, target, dist, red_method):
    fig, ax = vis.plot_similarity(target, dist=dist, n=10, red_method=red_method)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_topography(vis, red_method):
    fig = vis.plot_topography(red_method=red_method)
    assert isinstance(fig, go.Figure)

@pytest.mark.parametrize("dist", [
    "cosine", "euclidean", "manhattan", "chebyshev", "dot", "pearson", "spearman"
])
def test_plot_similarity_heatmap(vis, dist):
    fig = vis.plot_similarity_heatmap(dist=dist)
    assert isinstance(fig, go.Figure)

@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_interactive(vis, red_method):
    fig = vis.plot_interactive(red_method=red_method)
    assert isinstance(fig, go.Figure)

def test_plot_dendrogram(vis):
    fig, ax = vis.plot_dendrogram()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)