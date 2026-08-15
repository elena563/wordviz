from scipy.cluster.hierarchy import linkage, to_tree
from unittest.mock import Mock
import numpy as np

from wordviz.helpers.dendrogram_helpers import compute_positions, draw_tree

X = np.array([[0, 0], [1, 0], [5, 0], [6, 0]])
Z = linkage(X, method="complete")
tree = to_tree(Z)
n_leaves = len(X)
node_positions = {}
node_leaves = {}


def test_compute_positions():
    leaf_counter = [0]
    max_dist = np.max(Z[:, 2])
    leaf_angles = {}
    leaf_order = {}

    leaves = compute_positions(
        tree,
        leaf_counter,
        n_leaves,
        max_dist,
        leaf_angles,
        node_positions,
        leaf_order,
        node_leaves,
    )

    leaf_radii = [node_positions[leaf][1] for leaf in leaves]
    leaf_angle_values = [node_positions[leaf][0] for leaf in leaves]
    radix_radius = node_positions[tree.id][1]
    diffs = np.diff(sorted(leaf_angle_values))

    np.testing.assert_allclose(leaf_radii, 0.95)
    np.testing.assert_allclose(radix_radius, 0.0)
    assert len(node_positions) == 2 * n_leaves - 1
    np.testing.assert_allclose(diffs, 2 * np.pi / n_leaves, atol=1e-10)


def test_draw_tree():
    ax = Mock()
    n_leaves = len(X)
    draw_tree(tree, ax, node_positions, line_color="black", linewidth=1.0)

    n_internal = n_leaves - 1
    assert ax.plot.call_count == 3 * n_internal
