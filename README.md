# WordViz

**WordViz** is a Python visualization library designed for exploring and visualizing word embeddings. Built on top of popular libraries such as `matplotlib`, `plotly`, and `gensim`, WordViz provides intuitive tools for analyzing embeddings through clustering, similarity exploration, and dimensionality reduction, all wrapped in interactive and customizable plots.
With WordViz, users can gain insights into the structure of their word embeddings, making it a valuable tool for researchers and practitioners in natural language processing.

## Main Features

- Load and explore pretrained embeddings (e.g., GloVe, FastText)
- Select from a variety of available embeddings
- Visualize embeddings in 2D with flexible dimensionality reduction options
- Identify and plot the most similar words to a given token
- Visualize clusters of related words
- Interactive plots powered by `plotly`
- Support for both light and dark themes


## Installation

Install the latest version from PyPI:

```bash
pip install wordviz
```


## Usage

You can load and manage embeddings though the `EmbeddingLoader` class, and then visualize them with the `Visualizer` class.

```python
from wordviz.loading import EmbeddingLoader
from wordviz.plotting import Visualizer

loader = EmbeddingLoader()
loader.load_from_file('path/to/your/embedding/file', 'word2vec')

vis = Visualizer(loader)
vis.plot_embeddings()
```

You can explore all functionalities through the example notebook provided in the `docs/` folder:

👉 [View example notebook](docs/example.ipynb)


## Contributing

This project was created as part of my Bachelor's Degree thesis. For now, it remains a personal project and is not yet open to public collaboration.  
However, it will be further developed and eventually opened to contributions.

In the meantime, if you want to suggest features or report bugs, feel free to contact me directly.


## License

This project is licensed under the MIT License.