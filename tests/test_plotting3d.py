from matplotlib import pyplot as plt
import plotly.graph_objects as go
import pytest


@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
@pytest.mark.parametrize("color_by_class", [True, False])
def test_plot_static(vis3d, red_method, color_by_class):
    fig, ax = vis3d.plot_static(red_method=red_method, color_by_class=color_by_class)
    assert 'pca' in vis3d.reduced
    assert vis3d.reduced['pca'].shape[1] == 3
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_embeddings_3d(vis3d, red_method):
    fig = vis3d.plot_embeddings(red_method=red_method)
    assert isinstance(fig, go.Figure)

@pytest.mark.parametrize("dist", [
    "cosine", "euclidean", "manhattan", "chebyshev", "dot", "pearson", "spearman"
])
@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_similarity_3d(vis3d, dist, red_method):
    target = vis3d.loader.tokens[1]
    fig = vis3d.plot_similarity(target, dist=dist, n=10, red_method=red_method)
    assert isinstance(fig, go.Figure)

@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_clusters_3d(vis3d, red_method):
    fig = vis3d.plot_clusters(red_method=red_method)
    assert isinstance(fig, go.Figure)