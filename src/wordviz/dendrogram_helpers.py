import numpy as np
from scipy.cluster.hierarchy import ClusterNode
from matplotlib import pyplot as plt
from collections.abc import Mapping

def compute_positions(node: ClusterNode, leaf_counter: list[int], n_leaves: int, max_dist: float, leaf_angles: dict[int, float], node_positions: dict[int, tuple[float, float]], leaf_order: dict[int, int]):
    """
    Recursively computes the angular and radial positions of each node in the dendrogram.
    Updates leaf_angles and node_positions dictionaries with the computed values.
    """
    if node.is_leaf():
        angle = 2 * np.pi * leaf_counter[0] / n_leaves
        leaf_angles[node.id] = angle
        node_positions[node.id] = (angle, 0.95)
        leaf_order[node.id] = leaf_counter[0]
        leaf_counter[0] += 1
        return [node.id]
    else:
        left_leaves = compute_positions(node.left, leaf_counter, n_leaves, max_dist, leaf_angles, node_positions, leaf_order)
        right_leaves = compute_positions(node.right, leaf_counter, n_leaves, max_dist, leaf_angles, node_positions, leaf_order)

        all_leaves = left_leaves + right_leaves
        angles = [leaf_angles[lid] for lid in all_leaves]
        
        angles_sorted = sorted(angles)
        if angles_sorted[-1] - angles_sorted[0] > np.pi:
            gaps = [(angles_sorted[i+1] - angles_sorted[i], i) 
                    for i in range(len(angles_sorted)-1)]
            gaps.append((2*np.pi - angles_sorted[-1] + angles_sorted[0], len(angles_sorted)-1))
            max_gap, gap_idx = max(gaps)
            
            threshold = (angles_sorted[gap_idx] + max_gap/2) % (2*np.pi)
            angles_adjusted = [(a + 2*np.pi if a < threshold else a) for a in angles]
            mean_angle = np.mean(angles_adjusted) % (2 * np.pi)
        else:
            mean_angle = np.mean(angles)
        
        radius = 1.0 - (node.dist / max_dist if max_dist > 0 else 0) # radius is inversely proportional to distance from root
        node_positions[node.id] = (mean_angle, radius)
        return all_leaves
    

def draw_tree(node: ClusterNode, ax: plt.Axes, node_positions: Mapping[int, tuple[float, float]], line_color: str='black', linewidth: float=1.0):
    """Recursively draws the branches of the dendrogram."""
    if not node.is_leaf():
        _, parent_radius = node_positions[node.id]
        
        left_angle, left_radius = node_positions[node.left.id]
        right_angle, right_radius = node_positions[node.right.id]
        
        for child_angle, child_radius in [(left_angle, left_radius), (right_angle, right_radius)]:
            ax.plot([child_angle, child_angle], [parent_radius, child_radius],
                    color=line_color, linewidth=linewidth, zorder=1)
        
        if left_angle != right_angle:
            if abs(right_angle - left_angle) > np.pi:
                greater_angle = max(left_angle, right_angle)
                smaller_angle = min(left_angle, right_angle)
                arc_angles_1 = np.linspace(smaller_angle, 2*np.pi, 30)
                arc_angles_2 = np.linspace(0, greater_angle, 30)

                arc = np.concatenate([arc_angles_1, [np.nan], arc_angles_2])
            else:
                start_angle = min(left_angle, right_angle)
                end_angle = max(left_angle, right_angle)
                arc = np.linspace(start_angle, end_angle, 50)

            ax.plot(arc, [parent_radius]*len(arc), color=line_color, linewidth=linewidth, zorder=1)
        
        draw_tree(node.left, ax, node_positions, line_color, linewidth)
        draw_tree(node.right, ax, node_positions, line_color, linewidth)