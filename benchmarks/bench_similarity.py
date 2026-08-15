import time
import numpy as np
from wordviz import EmbeddingLoader
from wordviz.similarity import compute_distances, n_most_similar
import wordviz.similarity


# old function
def _n_most_similar_old(
    loader: EmbeddingLoader, target_word: str, dist: str = "cosine", n: int = 10
) -> tuple[list[str], np.ndarray, list[float]]:
    words = loader.tokens

    if target_word not in words:
        raise ValueError(f"{target_word} is not in vocabulary")

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
        batch_words = filtered_words[i : i + batch_size]
        batch_vectors = np.array([loader.get_embedding(word) for word in batch_words])

        X = np.vstack([target_vector, batch_vectors])
        D = compute_distances(X, metric=dist)
        distances = D[0, 1:]

        all_distances.extend(distances)
        all_indices.extend(range(i, min(i + batch_size, len(filtered_words))))

    # select indices
    if len(all_distances) <= n:
        top_n_indices = np.argsort(all_distances)
    else:
        top_n_indices = np.argpartition(all_distances, n - 1)[:n]
        # sort by distance
        top_n_indices = top_n_indices[
            np.argsort(np.array(all_distances)[top_n_indices])
        ]

    result_words = [filtered_words[all_indices[i]] for i in top_n_indices]
    result_distances = [all_distances[i] for i in top_n_indices]
    result_vectors = np.array([loader.get_embedding(word) for word in result_words])

    return result_words, result_vectors, result_distances


loader = EmbeddingLoader()
loader.load_pretrained("glove", "en", "wiki", "50d")
print(f"Loaded {len(loader.tokens)} tokens from GloVe embeddings.")
word = "example"
dist = "cosine"
n = 10
RUNS = 1


def timeit(fn, *args, runs=RUNS):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
        print(f"Run completed in {times[-1] * 1000:.2f} ms")
    return result, times


print(f"Benchmarking n_most_similar for word '{word}' with n={n} over {RUNS} runs...")
_, legacy_times = timeit(_n_most_similar_old, loader, word, dist, n)
print(
    f"Legacy:  {np.mean(legacy_times) * 1000:.1f} ms ± {np.std(legacy_times) * 1000:.1f}"
)
_, new_times = timeit(n_most_similar, loader, word, dist, n)
print(f"New:     {np.mean(new_times) * 1000:.1f} ms ± {np.std(new_times) * 1000:.1f}")
print(f"Speedup: {np.mean(legacy_times) / np.mean(new_times):.1f}x")
