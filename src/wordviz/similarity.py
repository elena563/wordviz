import numpy as np
from scipy.spatial.distance import cityblock, euclidean, cosine, chebyshev, canberra, braycurtis
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
from typing import List, Tuple
import warnings
from wordviz.loading import EmbeddingLoader

def word_distance(loader: EmbeddingLoader, word1: str, word2: str, dist: str = 'cosine') -> float:
    '''
    Computes distance between two words given by user.

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


def n_most_similar2(loader: EmbeddingLoader, target_word: str, dist: str = 'cosine', n: int = 10) -> Tuple[List[str], np.ndarray, List[float]]:
    '''
    Finds the n most similar words to a given target word using a specified distance metric.

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
    words = loader.tokens

    if target_word not in words:
        raise ValueError(f'{target_word} is not in vocabulary')
    
    topn_vectors = {}

    for word in words:
        if word == target_word:
            continue

        distance = word_distance(loader, word, target_word, dist)
        if len(topn_vectors) < n:
            topn_vectors[word] = distance
        else:
            max_dist_word = max(topn_vectors, key=topn_vectors.get)
            if distance < topn_vectors[max_dist_word]:
                del topn_vectors[max_dist_word]
                topn_vectors[word] = distance
    
    vectors = np.array([loader.get_embedding(word) for word in topn_vectors])
    words = list(topn_vectors.keys())
    distances = list(topn_vectors.values())

    return words, vectors, distances  

def n_most_similar(loader: EmbeddingLoader, target_word: str, dist: str = 'cosine', n: int = 10) -> Tuple[List[str], np.ndarray, List[float]]:
    '''
    Findsspairwise the n most similar words to a given target word using a specified distance metric.
    
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
    words = loader.tokens
    
    if target_word not in words:
        raise ValueError(f'{target_word} is not in vocabulary')
    
    # Ottieni il vettore target
    target_vector = loader.get_embedding(target_word)
    target_index = words.index(target_word)
    
    # Prepara gli indici di tutte le parole tranne la parola target
    word_indices = list(range(len(words)))
    word_indices.remove(target_index)
    
    # Sottoinsiemi di parole e vettori corrispondenti
    filtered_words = [words[i] for i in word_indices]
    
    # Per gestire vocabolari molto grandi, possiamo elaborare per batch
    batch_size = 10000
    all_distances = []
    all_indices = []
    
    for i in range(0, len(filtered_words), batch_size):
        batch_words = filtered_words[i:i+batch_size]
        batch_vectors = np.array([loader.get_embedding(word) for word in batch_words])
        
        # Calcola le distanze pairwise tra il vettore target e tutti i vettori batch
        # Reshape target_vector per ottenere una matrice 1 x dimensione
        distances = pairwise_distances(
            target_vector.reshape(1, -1), 
            batch_vectors, 
            metric=dist
        ).flatten()
        
        all_distances.extend(distances)
        all_indices.extend(range(i, min(i+batch_size, len(filtered_words))))
    
    # Seleziona gli indici delle n distanze minori
    if len(all_distances) <= n:
        top_n_indices = np.argsort(all_distances)
    else:
        top_n_indices = np.argpartition(all_distances, n-1)[:n]
        # Ordina questi n indici per distanza
        top_n_indices = top_n_indices[np.argsort(np.array(all_distances)[top_n_indices])]
    
    # Ottieni le parole, le distanze e i vettori corrispondenti
    result_words = [filtered_words[all_indices[i]] for i in top_n_indices]
    result_distances = [all_distances[i] for i in top_n_indices]
    result_vectors = np.array([loader.get_embedding(word) for word in result_words])
    
    return result_words, result_vectors, result_distances