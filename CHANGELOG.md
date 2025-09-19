## [0.3.0] - /09/2025
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
- new class Visualizer3D and parent class BaseVisualizer
- new 'reduced' parameter to Visualizer classes and 'auto' value for red_method in plot functions to automatically use cached reduced embeddings
- private _set_embeddings function to handle embeddings use
- 4 options of 3D plots analogous to 2D versions
- 4 new aesthetic themes: light2, dark2, light3, dark3
- optimized available pretrained json structure
- temporarily removed radialtree for issues, plot_dendrogram is not available at the moment

### Disabled
- plot_dendrogram function due to requirements issues, it will be restored in future versions