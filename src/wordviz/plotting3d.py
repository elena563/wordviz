from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
from .plotting import BaseVisualizer
from .clustering import create_clusters
from .dim_reduction import reduce_dim
from .similarity import n_most_similar


class Visualizer3D(BaseVisualizer):
    def __init__(self, loader):
        super().__init__(loader) 
        self.reduced: dict[str, np.ndarray] = {}
        self.reduced_subset = None

    def _setup_3d(self, reduced_emb, theme, grid, tokens, title, def_title, labels=None):
        '''base private function to config plotly 3d plot'''
        df = pd.DataFrame(reduced_emb, columns=['x', 'y', 'z'])

        style = self.get_theme(theme)
        kwargs = {}
        if labels is not None:
            colors, legend_labels  = self.map_colors(labels, theme)

            df['label'] = labels
            df['name'] = [f'Cluster {label+1}' for label in labels]

            color_discrete_map = {
                legend_labels[label_num][1]: f'rgb({int(color_tuple[0]*255)}, {int(color_tuple[1]*255)}, {int(color_tuple[2]*255)})'
                for label_num, (color_tuple, _) in legend_labels.items()
            }

            kwargs['color'] = 'name'
            kwargs['color_discrete_map'] = color_discrete_map
        else:
            kwargs['color_discrete_sequence'] = [style['points']]

        fig = px.scatter_3d(df, x='x', y='y', z='z',**kwargs)
        fig.update_traces(
            text=tokens,
            hovertemplate='%{text}<extra></extra>',
            hoverlabel=dict(
                bgcolor=style['bg'], 
                font=dict(color=style['text'])),
            marker=dict(size=5, opacity=0.6, line=dict(width=0))
        )
        fig.update_layout(
            height=500,
            title=title if title else def_title,
            title_x=0.5,
            title_xanchor='center',
            scene=dict(bgcolor=style['bg'],
                xaxis=dict(
                    backgroundcolor=style['bg'],
                    showticklabels=False, showgrid=grid,
                    gridcolor=style['grid_color'],
                    zeroline=False, title=None
                ),
                yaxis=dict(
                    backgroundcolor=style['bg'],
                    showticklabels=False, showgrid=grid,
                    gridcolor=style['grid_color'],
                    zeroline=False, title=None
                ),
                zaxis=dict(
                    backgroundcolor=style['bg'],
                    showticklabels=False, showgrid=grid,
                    gridcolor=style['grid_color'],
                    zeroline=False, title=None
                )
            ),
            paper_bgcolor=style['bg'],
            font=dict(color=style['text']),
        )
        return fig


    def plot_static(self, red_method: str = 'auto', grid: bool = True, theme: str = 'light1', title: str = None, nlabels: int = 0, use_subset: bool = False):
        '''
        Creates a simple static 3D scatterplot of the embeddings.

        Parameters
        -----------
        red_method : str, default='auto' -> auto is deprecated, will default to pca in future releases
            Dimensionality reduction method to apply ('pca', 'tsne', 'umap', etc.). If 'auto' searches for cached reduction, if None runs pca.
        grid : bool, default=True
            If True, displays a background grid on the plot.
        theme : str, default='light1'
            Color theme to apply.
        title : str, optional
            Title to display on the plot.
        nlabels : int, default=0
            Number of word labels to display. If 0, no labels are shown.
        use_subset : bool, default=False
            If True, uses the embedding subset instead of the full embeddings.

        Returns
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        '''

        reduced_emb, tokens = self._set_embeddings(use_subset=use_subset, red_method=red_method, dims=3)

        fig, ax, colors = self._setup_plot(theme, grid, title, dims=3)
        ax.scatter(reduced_emb[:, 0], reduced_emb[:, 1], reduced_emb[:, 2], c=colors['points'], alpha=0.5, s=14, marker='o')

        texts = []
        if nlabels > 0:
            sparse_indices = self.select_sparse_labels(reduced_emb, nlabels)
            for i in sparse_indices:
                texts.append(ax.text(reduced_emb[i, 0], reduced_emb[i, 1], reduced_emb[i, 2], tokens[i],
                color=colors['text'], fontsize=7, alpha=1, ha='center', va='bottom'))

        plt.rcParams['figure.dpi'] = 600
        plt.show()
        return fig, ax
    

    def plot_embeddings(self, red_method='auto', grid=True, theme='light1', title=None, use_subset=False):
        '''
        Creates an interactive 3D scatterplot of embeddings using Plotly.

        Parameters:
        -----------
        red_method : str, default='auto' -> auto is deprecated, will default to pca in future releases
            Dimensionality reduction method to apply ('pca', 'tsne', 'umap', etc.). If 'auto' searches for cached reduction, if None runs pca.
        grid : bool, default=True
            Whether to display grid lines.
        theme : str, default='light1'
            Plot color theme.
        title : str, optional
            Title of the plot. If None, no title is shown.
        use_subset : bool, default=False
            If True, uses the embedding subset instead of the full embeddings.

        Returns:
        --------
        fig : plotly.graph_objects.Figure

        Notes:
        ------
        In 3D plotting Plotly.py tends to use GPU to visualize an high number of elements and label, so it is possible that this function does not work properly with a whole embedding set.
        '''
        warnings.warn(
            "Without a suitable GPU, full 3D visualization may be slow or unstable. "
            "It is recommended to use a subset of the data for optimal performance and user experience."
        )
        reduced_emb, tokens = self._set_embeddings(use_subset=use_subset, red_method=red_method, dims=3)

        fig = self._setup_3d(reduced_emb=reduced_emb, theme=theme, grid=grid, tokens=tokens, title=title, def_title="Word Embedding 3D Plot")

        return fig


    def plot_similarity(self, target_word: str, dist: str = 'cosine', n: int = 10, red_method: str = 'pca', grid: bool = True, theme: str = 'light1', title: str = None):
        '''
        Creates a dynamic 3D scatterplot showing the most similar words to a target word.

        Parameters
        -----------
        target_word : str
            The word for which to find and plot the most similar words.
        dist : str, default='cosine'
            Distance metric to use when computing word similarity.
        n : int, default=10
            Number of similar words to display.
        red_method : str, default='pca' -> auto is deprecated, will default to pca in future releases
            Dimensionality reduction method to apply ('pca', 'tsne', 'umap', etc.).
        grid : bool, default=True
            If True, displays a background grid on the plot.
        theme : str, default='light1'
            Color theme to apply to the plot.
        title : str, optional
            Title to display. If None, a default title will be generated.

        Returns
        --------
        fig : plotly.graph_objects.Figure
        '''
        warnings.warn(
            "The parameter names target_word will be renamed to target in a future release. "
            "Please update your code accordingly.",
            FutureWarning
        )
        similar_words, similar_vecs, _ = n_most_similar(self.loader, target_word, dist, n)
        target_vec = self.loader.get_embedding(target_word)
        vectors = np.vstack([target_vec.reshape(1, -1), similar_vecs])
        words = [target_word] + similar_words

        reduced_emb = reduce_dim(vectors, method=red_method, n_dimensions=3)

        fig = self._setup_3d(reduced_emb=reduced_emb, theme=theme, grid=grid, tokens=words, title=title, def_title=f"Top {n} words similar to '{target_word}'")

        style = self.get_theme(theme)
        fig.add_trace(go.Scatter3d(
                x=[target_vec[0]],
                y=[target_vec[1]],
                z=[target_vec[2]],
                mode='markers',
                marker=dict(
                    size=5,
                    color=style['target'],
                    symbol='circle'
                ),
                text=[target_word],  
                hovertemplate='%{text}<extra></extra>',  
                showlegend=False
            ))

        return fig


    def plot_clusters(self, n_clusters=5, method='kmeans', red_method='auto', show_centers=False, grid=True, theme='light1', title=None, nlabels=0, use_subset=False):
        '''
        Creates a 3D scatterplot of clustered embeddings using a clustering algorithm.

        Parameters:
        -----------
        n_clusters : int, default=5
            Number of clusters to generate.
        method : str, default='kmeans'
            Clustering method to use ('kmeans' or others supported by create_clusters).
        red_method : str, default='auto' -> auto is deprecated, will default to pca in future releases
            Dimensionality reduction method to apply ('pca', 'tsne', 'umap', etc.). If 'auto' searches for cached reduction, if None runs pca.
        show_centers : bool, default=False
            If True, displays cluster centers on the plot.
        grid : bool, default=True
            Whether to display grid lines.
        theme : str, default='light1'
            Plot color theme.
        title : str, optional
            Title of the plot. If None, no title is shown.
        nlabels : int, default=0
            Number of token labels to display on the plot.
        use_subset : bool, default=False
            If True, uses the embedding subset instead of the full embeddings.

        Returns:
        --------
        fig : plotly.graph_objects.Figure

        Notes:
        ------
        In 3D plotting Plotly.py tends to use GPU to visualize an high number of elements and label, so it is possible that this function does not work properly with a whole embedding set.
        '''
        warnings.warn(
            "Without a suitable GPU, full 3D visualization may be slow or unstable. "
            "It is recommended to use a subset of the data for optimal performance and user experience."
        )
        reduced_emb, tokens = self._set_embeddings(use_subset=use_subset, red_method=red_method, dims=3)

        clusters, centers, reduced_emb = create_clusters(reduced_emb, n_clusters=n_clusters, method=method)
        clusters_colors, legend_labels = self.map_colors(clusters, theme=theme)

        fig = self._setup_3d(reduced_emb=reduced_emb, theme=theme, grid=grid, tokens=tokens, title=title, def_title=f"3D Clustering Scatterplot", labels=clusters)

        color = 'white' if 'dark' in theme else 'black'
        
        if show_centers and centers is not None:
            fig.add_trace(go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode='markers',
                marker=dict(
                    size=7,
                    color=color,
                    symbol='circle'
                ),
                name='Centers'
            ))

        return fig