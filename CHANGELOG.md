# Changelog

## [0.6.1] - 11/08/2026
### Fixed
- Added validation on zip file download to ensure the file is not corrupted before extraction.
- Moved internal files related functions outside of the `EmbeddingLoader` class to improve modularity and maintainability. Put the new file in `helpers/` together with other helper functions. 
- Removed internal deprecated glove2word2vec conversion, as GloVe files are now directly loaded in word2vec format.
- Fixed error in FastText loading when using binary files.
- Substituted list comprehension in `load_from_file` with gensim .vectors to improve performance.
- Removed broken GoogleNews pretrained embeddings from the package, as it is no longer available for download. Added a warning to inform users about this change.

### Added
- Tests for loading module, including tests for GloVe, Word2Vec, and FastText loading.

## [0.6.0] - 10/08/2026
### Added
- Added HDBSCAN and Gaussian Mixture Model (GMM) clustering methods to the `plot_clusters` function.
- New `metric` parameter in `plot_clusters` to select the distance metric for clustering, with default values based on the chosen method.
- New `ReducedCache` class to manage reduced embeddings for both 2D and 3D visualizations, allowing for more efficient caching, validation and retrieval of reduced embeddings.
- Added `custom` parameter for `red_method` to use embeddings reduced by external methods, with validation of shape and dimensionality. 

### Fixed
- Better error handling in the clustering functions.
- Removed automatic dimensionality reduction before clustering, reducing coupling between clustering and reduction.

### Updated
- `compute_distances` now can handle distances between a target embedding and a set of embeddings, reducing complexity from n^3 to n^2. A benchmark test was added to validate the performance improvement.
- `map_colors` has now a fallback to a default color palette if the number of unique labels exceeds the available colors in the selected theme, ensuring that all labels are assigned a different color.
- `plot_dendrogram` now supports coloring by cluster if a `n_clusters` to visualize is provided. Together with the threshold line, now all the branches are colored according to the cluster they belong to. 

## [0.5.0] - 01/08/2026
### Updated
- Adapted `map_colors` function to handle both class labels and cluster labels, with appropriate legend labeling. Changed color format to hex for better compatibility with Plotly.
- Moved optional `classes` parameter to the end of the `load_contextual` method signature for better clarity and usability.

### Added
- Added getter and setter for the `classes` property to use it also with static embeddings.
- Added tests for `map colors` function.
- All scatter plots now support coloring by class labels if available, with a new `color_by_class` parameter in the plotting functions.

### Fixed
- Added default n_neighbors parameter to UMAP dimensionality reduction to avoid errors when the number of samples is less than 15.
- Fixed `plot_similarity` 3d function, which displayed incorrect target word position.

## [0.4.1] - 27/07/2026
### Updated
- Edit `embedding_distance` (old `word_distance`) to use `compute_distances` function for flexibility in distance calculations.
- Little optimizations in `n_most_similar` function to avoid redundant computations and improve performance. 

### Added
- Added tests for `similarity` module.

### Deprecated
- From `/similarity`:
  * `word_distance` will change name to `embedding_distance` (FutureWarning added)

## [0.4.0] - 22/07/2026
### Updated
- Changed Visualizers attribute `reduced` type to dict instead of a single numpy array, to allow for multiple reduced embeddings to be stored and accessed by their reduction method name.
- Changed default value of `red_method` parameter in plotting functions to 'pca' instead of 'auto', as 'auto' is no more useful, so it is deprecated and will be removed in future releases (FutureWarning added).

### Added
- Added public API in `__init__.py` to import classes directly from the package.

### Fixed
- Removed automatic optional dependency installation. Added `umap-learn` to optional dependencies.

## [0.3.6] - 17/07/2026
### Updated
- Improved `plot_dendrogram` function with some modifications, as like as the other radialtree functions (`calculate_positions` and `draw_tree`), added tests for these.

## [0.3.5] - 12/07/2026
### Fixed
- Updated scipy lower bound to >=1.14.1 to ensure prebuilt wheels on Python 3.13

## [0.3.4] - 12/07/2026
### Added
- Added support for Python 3.13 and CI pipeline for dependencies check

## [0.3.3] - 07/07/2026
### Fixed
- `subset()`: replaced `len(self.tokens)` with `self.embeddings.shape[0]` as authoritative size source
- Added `_require_loaded()` guard method with `ValueError` on uninitialized embeddings/tokens; applied to `get_embedding()` and `subset()`
- `_validate_file()`: fixed guard order, removed inconsistent type coercion, added explicit error for compressed files (.gz, .zip)
- `load_from_file()`: GloVe conversion temp file written to cache dir instead of cwd
- removed unnecessary dev dependencies
- substituted `print()` statements with `logging` for better logging practices

## [0.3.2] - 05/07/2026
### Added
- New `classes` property for `EmbeddingLoader` class, allowing access to class labels of embeddings (if available)
- New `color_by_class` parameter for `plot_interactive` method in `Visualizer` class, allowing coloring of points by their class labels (if available)

## [0.3.1] - 04/06/2026
### Fixed
- Restored `plot_dendrogram` function, adapted from radialtree function belonging to https://github.com/koonimaru/omniplot by koonimaru (MIT License)
- Fixed automatic optional dependency installation for `encoding` module
- Fixed other urgent bugs

## [0.3.0] - 19/09/2025
### Added
- Support for contextual embeddings with two modes:
  * `sentences`: visualize entire sentences
  * `word_contexts`: visualize and compare multiple embeddings of the same word in different contexts
- New `encoding` module to embed sentences and words in different contexts, supported by Transformers and PyTorch (optional requirements)
- `load_contextual` method for `EmbeddingLoader` class
- New `type` property for `EmbeddingLoader` class

### Deprecated
- From `/plotting`:
  * `interactive_embeddings` will change name to `plot_interactive` (FutureWarning added)
  * `similarity_heatmap` will change name to `plot_similarity_heatmap` (FutureWarning added)
- Warnings added for imminent property name changes in similarity module and `plot_similarity` (no breaking changes yet)

### Fixed
- Fixed doubled parameter bug in MDS dimensionality reduction
- Fixed support to pairwise distances for all distance types


## [0.2.0] - 30/07/2025
### Added
- new class `Visualizer3D` and parent class `BaseVisualizer`
- new 'reduced' parameter to Visualizer classes and 'auto' value for `red_method` in plot functions to automatically use cached reduced embeddings
- private `_set_embeddings` function to handle embeddings use
- 4 options of 3D plots analogous to 2D versions
- 4 new aesthetic themes: light2, dark2, light3, dark3
- optimized available pretrained json structure

### Disabled
- plot_dendrogram function due to requirements issues, it will be restored in future versions