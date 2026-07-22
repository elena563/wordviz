[![PyPI version](https://img.shields.io/pypi/v/wordviz.svg)](https://pypi.org/project/wordviz/)
[![Python](https://img.shields.io/pypi/pyversions/wordviz.svg)](https://pypi.org/project/wordviz/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![logo](images/logo.png)

**WordViz** is a Python visualization library designed for exploring and visualizing word embeddings. Built on top of popular libraries such as `matplotlib`, `plotly`, and `gensim`, WordViz provides intuitive tools for analyzing embeddings through clustering, similarity exploration, and dimensionality reduction, all wrapped in interactive and customizable plots.
With WordViz, users can gain insights into the structure of their word embeddings, making it a valuable tool for researchers and practitioners in natural language processing.

This project was created as part of my Bachelor's Degree thesis in Statistics and Information Management with title (translated): "Word Embeddings in Practice: Designing a Library for Visualization and Operations"

**version 0.4.0**

Documentation: https://wordviz.readthedocs.io/

Built with:  
  
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-7A76FF?style=flat&logo=plotly&logoColor=white)]()
[![matplotlib](https://img.shields.io/badge/matplotlib-15557C?style=flat)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-f7931e?style=flat&logo=scikitlearn&logoColor=white)]()
[![Gensim](https://img.shields.io/badge/Gensim-8199F7?style=flat)]()
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)]()

## Last Version Updates

### Updated
- Changed Visualizers attribute `reduced` type to dict instead of a single numpy array, to allow for multiple reduced embeddings to be stored and accessed by their reduction method name.
- Changed default value of `red_method` parameter in plotting functions to 'pca' instead of 'auto', as 'auto' is no more useful, so it is deprecated and will be removed in future releases (FutureWarning added).

### Added
- Added public API in `__init__.py` to import classes directly from the package.

### Fixed
- Removed automatic optional dependency installation. Added `umap-learn` to optional dependencies.

See more about previous changes in [CHANGELOG.md](CHANGELOG.md)


## Main Features

- Load and explore pretrained embeddings (e.g., GloVe, FastText)
- Select from a variety of available embeddings
- Visualize embeddings in 2D or 3D with flexible dimensionality reduction options
- Identify and plot the most similar words to a given token
- Visualize clusters of related words
- Interactive plots powered by `plotly`
- Support for many light and dark themes


## Installation

Install the latest version from PyPI:

```bash
pip install wordviz
```

### Notes: Python version compatibility

The only versions that support Python 3.13 are the latest releases starting from version 0.3.5.

If you still have problems with the installation, it is recommended to use Python 3.12.

```bash
uv init --python 3.12
```

This warning will be removed in the next versions of wordviz, as Python 3.13 will be fully supported.


## Usage

You can load and manage embeddings though the `EmbeddingLoader` class, and then visualize them with the `Visualizer` (or `Visualizer3D`) class.

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

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## License

This project is licensed under the MIT License.


## Contacts

Elena Zen - [My Portfolio Website](https://elenazen.it/en) - info.elenazen@gmail.com