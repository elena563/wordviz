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
    n_tokens = len(vis3d.tokens)
    fig, ax = vis3d.plot_static(red_method=red_method, color_by_class=color_by_class)
    assert red_method in vis3d.reduced
    assert vis3d.reduced[red_method].shape[1] == 3
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    assert ax.name == "3d"
    assert len(ax.collections) >= 1
    offsets = ax.collections[0].get_offsets()
    assert offsets.shape[0] == n_tokens


@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_embeddings_3d(vis3d: Visualizer3D, red_method: str) -> None:
    n_tokens = len(vis3d.tokens)
    fig = vis3d.plot_embeddings(red_method=red_method)
    assert isinstance(fig, go.Figure)

    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter3d)
    assert fig.data[0].mode == "markers"
    assert len(fig.data[0].x) == n_tokens


@pytest.mark.parametrize(
    "dist",
    ["cosine", "euclidean", "manhattan", "chebyshev", "dot", "pearson", "spearman"],
)
@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
def test_plot_similarity_3d(vis3d: Visualizer3D, dist: str, red_method: str) -> None:
    assert vis3d.loader.tokens is not None
    target = vis3d.loader.tokens[1]
    n = 10
    fig = vis3d.plot_similarity(target, dist=dist, n=n, red_method=red_method)
    assert isinstance(fig, go.Figure)

    assert len(fig.data) == 2
    similar_trace = fig.data[0]
    target_trace = fig.data[1]
    assert isinstance(similar_trace, go.Scatter3d)
    assert isinstance(target_trace, go.Scatter3d)
    assert similar_trace.mode == "markers"
    assert target_trace.mode == "markers"
    assert len(similar_trace.x) == n


@pytest.mark.parametrize("red_method", ["pca", "tsne", "umap", "isomap", "mds"])
@pytest.mark.parametrize(
    "method", ["kmeans", "dbscan", "hdbscan", "hierarchical", "gmm"]
)
def test_plot_clusters_3d(vis3d: Visualizer3D, red_method: str, method: str) -> None:
    n_tokens = len(vis3d.tokens)
    fig = vis3d.plot_clusters(red_method=red_method, method=method)
    assert isinstance(fig, go.Figure)

    if method not in ["dbscan", "hdbscan"]:
        n_clusters = 5
        assert len(fig.data) == n_clusters
    total_points = sum(len(t.x) for t in fig.data)
    assert total_points == n_tokens
    for trace in fig.data:
        assert isinstance(trace, go.Scatter3d)
        assert trace.mode == "markers"
