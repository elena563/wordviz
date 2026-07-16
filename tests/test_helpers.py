from scipy.cluster.hierarchy import linkage, to_tree
import numpy as np

from wordviz.dendrogram_helpers import compute_positions

X = np.array([[0, 0], [1, 0], [5, 0], [6, 0]])
Z = linkage(X, method='complete')
tree = to_tree(Z)

def test_compute_positions():
    leaf_counter = [0]
    n_leaves = len(X)
    max_dist = np.max(Z[:, 2])
    leaf_angles = {}
    node_positions = {}

    leaves = compute_positions(tree, leaf_counter, n_leaves, max_dist, leaf_angles, node_positions)
    
    leaf_radii = [node_positions[leaf][1] for leaf in leaves]
    leaf_angle_values = [node_positions[leaf][0] for leaf in leaves]
    radix_radius = node_positions[tree.id][1]
    diffs = np.diff(sorted(leaf_angle_values))

    np.testing.assert_allclose(leaf_radii, 1.0)
    np.testing.assert_allclose(radix_radius, 0.0)
    assert len(node_positions) == 2 * n_leaves - 1
    np.testing.assert_allclose(diffs, 2*np.pi/n_leaves, atol=1e-10)