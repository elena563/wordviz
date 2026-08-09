import numpy as np
from scipy.spatial.distance import cityblock, euclidean, cosine, chebyshev, canberra, braycurtis
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
from typing import List, Tuple
import warnings
from wordviz.loading import EmbeddingLoader

def embedding_distance(loader: EmbeddingLoader, word1: str, word2: str, dist: str = 'cosine') -> float:
    '''
    Computes distance between two words given by user. Also supports sentence distance.

    Parameters
    -----------
    loader: EmbeddingLoader
        Object used to load embeddings
    word1, word2: str
        Word to compute distance between
    dist: str, default='cosine'
        Type of distance to use:
        - 'braycurtis'
        - 'canberra'                      
        - 'chebyshev'
        - 'cosine'
        - 'dot'
        - 'euclidean'
        - 'manhattan'
        - 'pearson'
        - 'pearson'

    Returns
    --------
    distance: float
    '''
    warnings.warn(
        "The parameter names word1/word2 will be renamed to item1/item2 in a future release. "
        "Please update your code accordingly.",
        FutureWarning
    )
    words = loader.tokens

    missing = [w for w in (word1, word2) if w not in words]
    if missing:
        raise ValueError(f"Item(s) not in vocabulary: {', '.join(missing)}")

    vec1 = loader.get_embedding(word1)      
    vec2 = loader.get_embedding(word2)
    X = np.vstack([vec1, vec2])
    D = compute_distances(X, metric=dist)
    distance = D[0, 1].item()

    return distance

def word_distance(loader: EmbeddingLoader, word1: str, word2: str, dist: str = 'cosine') -> float:
    warnings.warn(
        "word_distance is deprecated and will be renamed to embedding_distance in a future release. "
        "Please update your code accordingly.",
        FutureWarning
    )
    return embedding_distance(loader, word1, word2, dist=dist)
 

def n_most_similar(loader: EmbeddingLoader, target_word: str, dist: str = 'cosine', n: int = 10) -> tuple[list[str], np.ndarray, list[float]]:
    '''
    Finds pairwise the n most similar words to a given target word using a specified distance metric.
    
    Parameters
    -----------
    loader : EmbeddingLoader
        An instance of the embedding loader containing word vectors.
    target_word : str
        The word for which to find the most similar neighbors.
    dist : str, default='cosine'
        The distance metric to use. Options include 'cosine', 'euclidean', etc.
    n : int, default=10
        The number of most similar words to retrieve.
    
    Returns
    --------
    words : list of str
        The most similar words found.
    vectors : np.ndarray
        Embedding vectors corresponding to the most similar words.
    distances : list of float
        Distances from the target word to each of the most similar words.
    '''
    warnings.warn(
        "The parameter names target_word will be renamed to target in a future release. "
        "Please update your code accordingly.",
        FutureWarning
    )
    print('function in local with changes')
    words = loader.tokens
    embeddings = loader.embeddings
    
    if target_word not in words:
        raise ValueError(f'{target_word} is not in vocabulary')
    
    target_vector = loader.get_embedding(target_word)
    target_index = words.index(target_word)
    
    mask = np.ones(len(words), dtype=bool)
    mask[target_index] = False
    
    filtered_words = [words[i] for i in np.where(mask)[0]]
    filtered_embeddings = embeddings[mask]
    
    # process in batch
    batch_size = 10000
    all_distances = []
    all_indices = []
    
    for i in range(0, len(filtered_embeddings), batch_size):
        batch_words = filtered_words[i:i+batch_size]
        batch_vectors = filtered_embeddings[i:i+batch_size]
        
        distances = compute_distances(batch_vectors, metric=dist, target=target_vector)
        
        all_distances.extend(distances)
        all_indices.extend(range(i, min(i+batch_size, len(filtered_words))))
    
    # select indices
    if len(all_distances) <= n:
        top_n_indices = np.argsort(all_distances)
    else:
        top_n_indices = np.argpartition(all_distances, n-1)[:n]
        # sort by distance
        top_n_indices = top_n_indices[np.argsort(np.array(all_distances)[top_n_indices])]
    
    result_words = [filtered_words[all_indices[i]] for i in top_n_indices]
    result_distances = [all_distances[i] for i in top_n_indices]
    result_vectors = result_vectors = filtered_embeddings[top_n_indices]
    
    return result_words, result_vectors, result_distances



def compute_distances(X: np.ndarray, metric: str='euclidean', target: np.ndarray = None) -> np.ndarray:
    '''
    Computes pairwise distances between rows of a matrix X using the specified metric.

    Parameters
    -----------
    X : np.ndarray
        A 2D array where each row is a vector for which distances will be computed.
    metric : str, default='euclidean'
        The distance metric to use.
        Options include 'euclidean', 'cosine', 'manhattan', 'braycurtis', 'canberra', 'chebyshev', 'dot', 'pearson', and 'spearman'.
    target : np.ndarray, optional
        If provided, computes distances from the target vector to each row in X instead of pairwise distances among rows of X.
    
    Returns
    --------
    distances : np.ndarray
        A 2D array of distances. If target is provided, returns a 1D array of distances from the target to each row in X.
    '''

    if metric in ['euclidean', 'cosine', 'manhattan', 'braycurtis', 'canberra', 'chebyshev']:
        if target is not None:
            X = np.vstack([target, X])
            distances = pairwise_distances(X, metric=metric, Y=target.reshape(1, -1))
            return distances[0, 1:]
        return pairwise_distances(X, metric=metric)
    
    elif metric == 'dot':
        if target is not None:
            return 1 - (X @ target)
        return 1 - (X @ X.T)
    
    elif metric == 'pearson':
        if target is not None:
            combined = np.vstack([target, X])
        else:
            combined = X
        
        stds = combined.std(axis=1, keepdims=True)
        stds = np.where(stds == 0, np.finfo(float).tiny, stds)      # avoid division by zero
        combined = (combined - combined.mean(axis=1, keepdims=True)) / stds
        corr = np.corrcoef(combined) if target is None else (combined @ combined.T) / combined.shape[1]
        corr = 1 - corr
        
        if target is not None:
            return corr[0, 1:]
        return corr
    
    elif metric == 'spearman':
        if target is not None:
            distances = np.array([1 - spearmanr(target, X[i])[0] for i in range(X.shape[0])])
            return distances
        n = X.shape[0]
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                r, _ = spearmanr(X[i], X[j])
                dist_mat[i, j] = dist_mat[j, i] = 1 - r
        return dist_mat
    else:
        raise ValueError(f"Unknown metric: {metric}")