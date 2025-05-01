import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from typing import Tuple, Optional
from .dim_reduction import reduce_dim


def create_clusters(vectors: np.ndarray, n_clusters: int = 5, method: str = 'kmeans') -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    '''
    Performs clustering on embeddings for visualization purposes.
    If the input vectors have more than 2 dimensions, dimensionality reduction is applied first.

    Parameters:
    -----------
    vectors : np.ndarray
        Array of embeddings to cluster.
    n_clusters : int, default=5
        Number of clusters to generate (used only for k-means).
    method : str, default='kmeans'
        Clustering method to use ('kmeans' or 'dbscan').

    Returns:
    --------
    labels : np.ndarray
        Cluster labels assigned to each vector.
    centers : np.ndarray or None
        Coordinates of cluster centers (only for k-means; None for dbscan).
    reduced_emb : np.ndarray
        2D reduced embeddings used for clustering and plotting.
    '''

    if vectors.shape[1] > 2:
        reduced_emb = reduce_dim(vectors)  
    else:
        reduced_emb = vectors  

    match method:
        case 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(reduced_emb)
            centers = clustering.cluster_centers_
        case 'dbscan':
            clustering = DBSCAN(eps=0.5, min_samples=n_clusters).fit(reduced_emb)
            centers = None

    return clustering.labels_, centers, reduced_emb