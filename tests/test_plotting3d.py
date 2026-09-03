import plotly.graph_objects as go
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from wordviz import Visualizer3D


@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
@pytest.mark.parametrize("color_by_class", [True, False])
def test_plot_static(
    vis3d: Visualizer3D, red_method: str, color_by_class: bool
) -> None:
    fig, ax = vis3d.plot_static(red_method=red_method, color_by_class=color_by_class)
    assert "pca" in vis3d.reduced
    assert vis3d.reduced["pca"].shape[1] == 3
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_embeddings_3d(vis3d: Visualizer3D, red_method: str) -> None:
    fig = vis3d.plot_embeddings(red_method=red_method)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    "dist",
    ["cosine", "euclidean", "manhattan", "chebyshev", "dot", "pearson", "spearman"],
)
@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_similarity_3d(vis3d: Visualizer3D, dist: str, red_method: str) -> None:
    assert vis3d.loader.tokens is not None
    target = vis3d.loader.tokens[1]
    fig = vis3d.plot_similarity(target, dist=dist, n=10, red_method=red_method)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
@pytest.mark.parametrize(
    "method", ["kmeans", "dbscan", "hdbscan", "hierarchical", "gmm"]
)
def test_plot_clusters_3d(vis3d: Visualizer3D, red_method: str, method: str) -> None:
    fig = vis3d.plot_clusters(red_method=red_method, method=method)
    assert isinstance(fig, go.Figure)
