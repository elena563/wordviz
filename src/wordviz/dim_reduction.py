import umap
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
import warnings

def reduce_dim(vectors: np.ndarray, method: str = 'pca', n_dimensions: int = 2, dist: str = 'euclidean', **kwargs) -> np.ndarray:
    ''' 
    Applies dimensionality reduction for visualization to input vectors using a specified method.

    Parameters:
    -----------
    vectors: array
        Input high-dimensional data to reduce.
    method: str, default:'pca'
        Dimensionality reduction algorithm to apply. Options are:
        - 'pca': Principal Component Analysis
        - 'tsne': t-Distributed Stochastic Neighbor Embedding
        - 'umap': Uniformed Manifold Approximation and Projection
        - 'isomap': Isometric Mapping
        - 'mds': Multidimensional Scaling
    n_dimensions: int, default:2
        Number of dimensions for output. Must be 2 or 3
     dist : str, default='euclidean'
        Distance metric for methods that require a distance matrix (MDS). Ignored for other methods.
    **kwargs : dict
        Additional parameters passed to the selected dimensionality reduction algorithm.

    Returns:
    --------
    np.ndarray
        Embedding reduced at specified dimensions.
    '''
    
    if n_dimensions not in [2, 3]:
        raise ValueError('n_dimensions has to be 2 or 3')
    
    if (method in ['isomap', 'tsne', 'mds']) and vectors.shape[0] > 5000:
        warnings.warn(f"Warning: {method} is a computational heavy method, loading very large embeddings without subsetting may result in execution times so long that the process may never complete.")

    default_params = {
        'pca': {},
        'tsne': {'random_state': 42, 'perplexity': 30},
        'umap': {'n_neighbors': 15, 'min_dist': 0.1, 'random_state': 42},
        'isomap': {'n_neighbors': 5},
        'mds': {'metric':True, 'n_init':3, 'max_iter':300, 'random_state':42, 'dissimilarity': 'euclidean'}
    }

    if method in default_params:
        params = {**default_params[method.lower()], **kwargs}
    else:
        raise ValueError(f"Method {method} not supported. Choose between 'pca', 'tsne', 'umap' or 'isomap'")

    match method.lower():
        case 'pca':
            reducer = PCA(n_components=n_dimensions, **params)
        case 'tsne':    
            reducer = TSNE(n_components=n_dimensions, **params)
        case 'umap':
            reducer = umap.UMAP(n_components=n_dimensions, **params)
        case 'isomap': 
            reducer = Isomap(n_components=n_dimensions, **params)
        case 'mds':
            reducer = MDS(n_components=n_dimensions,
                        dissimilarity='euclidean' if dist == 'euclidean' else 'precomputed', **params)
            if dist != 'precomputed' and dist != 'euclidean':
                from sklearn.metrics import pairwise_distances
                vectors = pairwise_distances(vectors, metric=dist)

    reduced_emb = reducer.fit_transform(vectors)
    
    return reduced_emb
        
    
