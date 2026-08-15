import time
import numpy as np
from wordviz import EmbeddingLoader


def _to_matrix_loop(kv) -> np.ndarray:
    """old implementation: python loop over get_vector"""
    words = kv.index_to_key
    return np.array([kv.get_vector(word) for word in words])


def _to_matrix_vectors(kv) -> np.ndarray:
    """new implementation: direct numpy access"""
    return kv.vectors


loader = EmbeddingLoader()
loader.load_pretrained("glove", "en", "wiki", "50d")
print(f"Loaded {len(loader.tokens)} tokens from GloVe embeddings.")

kv = loader.embeddings_raw
RUNS = 1


def timeit(fn, *args, runs=RUNS):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    return result, times


print("Benchmarking embeddings matrix extraction...")
_, legacy_times = timeit(_to_matrix_loop, kv)
print(f"Loop (old):   {np.mean(legacy_times) * 1000:.1f} ms")
_, new_times = timeit(_to_matrix_vectors, kv)
print(f"Vectors (new): {np.mean(new_times) * 1000:.1f} ms")
print(f"Speedup: {np.mean(legacy_times) / np.mean(new_times):.1f}x")
