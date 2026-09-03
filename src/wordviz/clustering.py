from typing import Any, cast

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture


def create_clusters(
    vectors: np.ndarray,
    n_clusters: int = 5,
    method: str = "kmeans",
    metric: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Performs clustering on embeddings for visualization purposes.

    Parameters:
    -----------
    vectors : np.ndarray
        Array of embeddings to cluster.
    n_clusters : int, default=5
        Number of clusters to generate (used only for k-means).
    method : str, default='kmeans'
        Clustering method to use. Options are:
        - 'kmeans': K-Means clustering
        - 'dbscan': Density-Based Spatial Clustering of Applications with Noise
        - 'hdbscan': Hierarchical Density-Based Spatial Clustering of Applications with Noise
        - 'hierarchical': Hierarchical clustering
        - 'gmm': Gaussian Mixture Model clustering
    metric : str | None, default=None
        Distance metric to use for clustering (used only for methods that support it).

    Returns:
    --------
    labels : np.ndarray
        Cluster labels assigned to each vector.
    centers : np.ndarray or None
        Coordinates of cluster centers (only for k-means; None for dbscan).
    reduced_emb : np.ndarray
        2D reduced embeddings used for clustering and plotting.
    """

    if method not in ["kmeans", "dbscan", "hdbscan", "hierarchical", "gmm"]:
        raise ValueError(
            f"Invalid clustering method: {method}. Choose from 'kmeans', 'dbscan', 'hdbscan', 'hierarchical', or 'gmm'."
        )

    if metric is None:
        metric = "euclidean" if method in ["kmeans", "gmm"] else "cosine"

    if method in ["kmeans", "gmm"] and metric != "euclidean":
        raise ValueError(
            f"{method} only support 'euclidean' metric. Please use 'euclidean' for these methods."
        )

    match method:
        case "hierarchical":
            Z = linkage(vectors, method="single", metric=cast(Any, metric))
            labels = fcluster(Z, t=n_clusters, criterion="maxclust")
            centers = None
        case "kmeans":
            clustering = KMeans(
                n_clusters=n_clusters, random_state=0, n_init="auto"
            ).fit(vectors)
            labels = clustering.labels_
            centers = clustering.cluster_centers_
        case "dbscan":
            clustering = DBSCAN(eps=0.5, metric=metric).fit(vectors)
            labels = clustering.labels_
            centers = None
        case "hdbscan":
            clustering = HDBSCAN(metric=metric).fit(vectors)
            labels = clustering.labels_
            centers = None
        case "gmm":
            clustering = GaussianMixture(n_components=n_clusters, random_state=0).fit(
                vectors
            )
            labels = clustering.predict(vectors)
            centers = clustering.means_

    return labels, centers, vectors
