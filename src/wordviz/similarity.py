import numpy as np
from scipy.spatial.distance import cityblock, euclidean, cosine, chebyshev, canberra, braycurtis
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
from typing import List, Tuple
import warnings
from wordviz.loading import EmbeddingLoader

def word_distance(loader: EmbeddingLoader, word1: str, word2: str, dist: str = 'cosine') -> float:
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
        raise ValueError(f"Word(s) not in vocabulary: {', '.join(missing)}")

    vec1 = loader.get_embedding(word1)      
    vec2 = loader.get_embedding(word2)
    emb_matrix = loader.embeddings

    match dist:  
        case 'braycurtis':
            distance = braycurtis(vec1, vec2) 
        case 'canberra':                        
            distance = canberra(vec1, vec2)   
        case 'chebyshev':
            distance = chebyshev(vec1, vec2)                              
        case 'cosine':
            distance = cosine(vec1, vec2)  
        case 'dot':
            distance = -np.dot(vec1, vec2)     
        case 'euclidean':
            distance = euclidean(vec1, vec2)  
        case 'manhattan':
            distance = cityblock(vec1, vec2)  
        case 'pearson':
            pearson_corr, _ = pearsonr(vec1, vec2)
            distance = 1 - pearson_corr 
        case 'spearman':
            spearman_corr, _ = spearmanr(vec1, vec2)
            distance = 1 - spearman_corr   
    
    return distance
 

def n_most_similar(loader: EmbeddingLoader, target_word: str, dist: str = 'cosine', n: int = 10) -> Tuple[List[str], np.ndarray, List[float]]:
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
    words = loader.tokens
    
    if target_word not in words:
        raise ValueError(f'{target_word} is not in vocabulary')
    
    target_vector = loader.get_embedding(target_word)
    target_index = words.index(target_word)
    
    word_indices = list(range(len(words)))
    word_indices.remove(target_index)
    
    filtered_words = [words[i] for i in word_indices]
    
    # process in batch
    batch_size = 10000
    all_distances = []
    all_indices = []
    
    for i in range(0, len(filtered_words), batch_size):
        batch_words = filtered_words[i:i+batch_size]
        batch_vectors = np.array([loader.get_embedding(word) for word in batch_words])
        
        X = np.vstack([target_vector, batch_vectors])
        D = compute_distances(X, metric=dist)
        distances = D[0, 1:] 
        
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
    result_vectors = np.array([loader.get_embedding(word) for word in result_words])
    
    return result_words, result_vectors, result_distances



def compute_distances(X, metric='euclidean'):
    if metric in ['euclidean', 'cosine', 'manhattan', 'braycurtis', 'canberra', 'chebyshev']:
        return pairwise_distances(X, metric=metric)
    elif metric == 'dot':
        # dot
        return 1 - (X @ X.T)
    elif metric == 'pearson':
        # pearson
        corr = np.corrcoef(X)
        return 1 - corr
    elif metric == 'spearman':
        # spearman
        n = X.shape[0]
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                r, _ = spearmanr(X[i], X[j])
                dist_mat[i, j] = dist_mat[j, i] = 1 - r
        return dist_mat
    else:
        raise ValueError(f"Unknown metric: {metric}")