import pytest
import numpy as np

from wordviz.dim_reduction import reduce_dim

def test_reduce_dim_output_shape(sample_embeddings):
    reduced = reduce_dim(sample_embeddings, method='pca', n_dimensions=2)
    assert reduced.shape == (len(sample_embeddings), 2)

def test_reduce_dim_all_methods(sample_embeddings):
    for method in ['pca', 'tsne', 'umap', 'isomap', 'mds']:
        reduced = reduce_dim(sample_embeddings, method=method, n_dimensions=2)
        assert reduced.shape[1] == 2
        assert not np.isnan(reduced).any()
        assert not np.isinf(reduced).any()

def test_reduce_dim_invalid_method(sample_embeddings):
    with pytest.raises(ValueError):
        reduce_dim(sample_embeddings, method='nonexistent', n_dimensions=2)

def test_reduce_dim_preserves_samples(sample_embeddings):
    reduced = reduce_dim(sample_embeddings, method='pca', n_dimensions=2)
    assert reduced.shape[0] == sample_embeddings.shape[0]