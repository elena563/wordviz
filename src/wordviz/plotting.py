from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import gaussian_kde
import warnings
from .clustering import create_clusters
from .dim_reduction import reduce_dim
from .similarity import n_most_similar


class Visualizer:
    def __init__(self, loader):
        self.loader = loader
        self.tokens = loader.tokens
        self.embeddings = loader.embeddings


    def get_theme(self, theme='light1'):
        themes = { 
            'light1': {
                    'bg' : '#f5f5fa',
                    'points' : '#66c2a5',
                    'target' : '#e78ac3',
                    'grid' : '#cccccc',
                    'text' : '#1a1a1a',
                    'scale': 'Viridis'
                },
            'dark1': {
                    'bg' : '#1a1a1a',
                    'points' : '#fc8d62',
                    'target' : '#a6d854',
                    'grid' : '#444444',
                    'text' : '#f0f0f0',
                    'scale': 'Plasma'
                }
        }
        return themes.get(theme, themes['light1'])
    

    def _setup_plot(self, theme, grid, title):
        '''base private function to config matplotlib plot'''
        colors = self.get_theme(theme)

        fig, ax = plt.subplots(figsize=(10, 8))

        fig.patch.set_facecolor(colors['bg'])
        ax.set_facecolor(colors['bg'])

        if grid:  
            ax.grid(True, linestyle='--', color=colors['grid'], alpha=0.6)
            ax.set_axisbelow(True) 
            ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False, color=(0.5, 0.5, 0.5, 0.4))
        else:
            plt.xticks([])
            plt.yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        if title is not None:
            plt.title(title, fontsize=12, fontweight='bold', color=colors['text'])

        return fig, ax, colors
    

    def map_colors(self, labels):
        '''automatizes color and legend label mapping for clustering applied to embeddings'''
        unique_classes = list(set(labels))
        palette = sns.color_palette("Set2", n_colors=len(unique_classes))
        class_to_color = dict(zip(unique_classes, palette))
        colors = [class_to_color[label] for label in labels]
        legend_labels = {label: (class_to_color[label], f'Cluster {label+1}') for label in unique_classes}

        return colors, legend_labels


    def select_sparse_labels(self, embeddings, n):
        '''uses clustering to select n distributed labels to visualize'''
        kmeans = KMeans(n_clusters=n, random_state=0).fit(embeddings)
        centers = kmeans.cluster_centers_
        indices = []

        for center in centers:
            idx = np.argmin(np.linalg.norm(embeddings - center, axis=1))
            indices.append(idx)

        return indices


    def plot_embeddings(self, red_method: str = 'pca', grid: bool = True, theme: str = 'light1', title: str = None, nlabels: int = 0, use_subset: bool = False):   
        '''
        Creates a simple static 2D scatterplot of the embeddings.

        Parameters
        -----------
        red_method : str, default='pca'
            Dimensionality reduction method to apply ('pca', 'tsne', 'umap', etc.).
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

        if use_subset:
            emb, tokens = self.loader.use_subset()
        else:
            emb = self.embeddings
            tokens = self.tokens

        reduced_emb = reduce_dim(emb, method=red_method)

        fig, ax, colors = self._setup_plot(theme, grid, title)
        ax.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=colors['points'], alpha=0.5, s=14, marker='o')

        texts = []
        if nlabels > 0:
            sparse_indices = self.select_sparse_labels(reduced_emb, nlabels)
            for i in sparse_indices:
                texts.append(ax.text(reduced_emb[i, 0], reduced_emb[i, 1], tokens[i],
                color=colors['text'], fontsize=9, alpha=1, ha='center', va='bottom'))

        adjust_text(texts, ax=ax, expand=(1.2, 2), arrowprops=dict(arrowstyle='-', color='k'))
        plt.rcParams['figure.dpi'] = 600
        plt.show()
        return fig, ax
    
    
    def plot_similarity(self, target_word: str, dist: str = 'cosine', n: int = 10, red_method: str = 'umap', grid: bool = True, theme: str = 'light1', title: str = None):
        '''
        Creates a scatterplot showing the most similar words to a target word.

        Parameters
        -----------
        target_word : str
            The word for which to find and plot the most similar words.
        dist : str, default='cosine'
            Distance metric to use when computing word similarity.
        n : int, default=10
            Number of similar words to display.
        red_method : str, default='umap'
            Dimensionality reduction method to apply ('pca', 'tsne', 'umap', etc.).
        grid : bool, default=True
            If True, displays a background grid on the plot.
        theme : str, default='light1'
            Color theme to apply to the plot.
        title : str, optional
            Title to display. If None, a default title will be generated.

        Returns
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        '''
        similar_words, similar_vecs, _ = n_most_similar(self.loader, target_word, dist, n)
        target_vec = self.loader.get_embedding(target_word)
        vectors = np.vstack([target_vec.reshape(1, -1), similar_vecs])
        words = [target_word] + similar_words

        reduced_emb = reduce_dim(vectors, method=red_method)

        if title is None:
            title = f"Top {n} words similar to '{target_word}'"

        fig, ax, colors = self._setup_plot(theme, grid, title)
        
        texts = []
        ax.scatter(reduced_emb[0, 0], reduced_emb[0, 1], c=colors['target'], alpha=0.5, s=20, marker='o')
        texts.append(ax.text(reduced_emb[0, 0], reduced_emb[0, 1], target_word,
                color=colors['text'], fontsize=9, fontweight='bold', alpha=1, ha='center', va='bottom'))
        
        ax.scatter(reduced_emb[1:, 0], reduced_emb[1:, 1], c=colors['points'], alpha=0.5, s=20, marker='o')
        for i, word in enumerate(similar_words):
            texts.append(ax.text(reduced_emb[i+1, 0], reduced_emb[i+1, 1], word,
                    color=colors['text'], fontsize=9, alpha=1, ha='center', va='bottom'))
        
       
        plt.rcParams['figure.dpi'] = 600
        plt.show()  

        return fig, ax


    def plot_topography(self, dist: str = 'cosine', red_method: str = 'isomap', use_subset: bool = True, grid: bool = True, theme: str = 'light1', title: str = None):       
        '''
        Plots word embeddings in a topographical map using dimensionality reduction to maintain word distances in the representation. Allows to visualize word density in the space.

        Parameters
        -----------
        dist : str, default='cosine'
            The distance metric to use for word similarity. Options include 'cosine', 'euclidean', etc.
        red_method : str, default='isomap'
            The dimensionality reduction method to use for visualizing the word embeddings, maintaining word distances. Options include 'umap', 'isomap', and 'mds'.
        use_subset : bool, default=True
            If True, uses a subset of the embeddings for visualization. This is recommended in this plot for larger embeddings.
        grid : bool, default=True
            If True, shows grid lines on the plot.
        theme : str, default='light1'
            The plot theme to use, which controls the colors of the plot.
        title : str, optional
            Title of the plot. If not provided, a default title is used.

        Returns
        --------
        fig : plotly.graph_objs.Figure
        '''

        if use_subset:
            emb, tokens = self.loader.use_subset()
        else:
            emb = self.embeddings
            tokens = self.tokens

        reduced_emb = reduce_dim(emb, red_method, dist=dist)

        x = reduced_emb[:, 0]
        y = reduced_emb[:, 1]
        fig = go.Figure()

        colors = self.get_theme(theme)

        # calculate coordinates for contour plot
        x_grid, y_grid = np.meshgrid(np.linspace(x.min() - 0.5, x.max() + 0.5, 100),
                                np.linspace(y.min() - 0.5, y.max() + 0.5, 100))
    
        kde = gaussian_kde([x, y], bw_method=0.2) 
        z_grid = kde([x_grid.flatten(), y_grid.flatten()]).reshape(x_grid.shape)
        z_grid = np.log1p(z_grid)

        fig.add_trace(go.Contour(
        z=z_grid,
        x=x_grid[0],
        y=y_grid[:, 0],
        colorscale=colors['scale'],
        opacity=0.8,
        contours=dict(
            showlabels=False,
            start=z_grid.min(),
            end=z_grid.max(),
            size=(z_grid.max() - z_grid.min()) / 15),
        colorbar=dict(title="Density"),
        hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=5,
                color='rgba(255, 255, 255, 0.5)',
                line=dict(width=1, color='rgba(0, 0, 0, 0.8)')
            ),
            text=tokens,  
            hovertemplate='%{text}<extra></extra>',  
            showlegend=False
        ))

        fig.update_traces(
            hoverlabel=dict(
                bgcolor=colors['bg'], 
                font=dict(color=colors['text']) 
            )
        )

        fig.update_layout(
            width=900,
            height=700,
            title=title if title else "Word Embedding Topography",
            title_x=0.5,
            title_xanchor='center',
            plot_bgcolor=colors['bg'],
            paper_bgcolor=colors['bg'],
            font=dict(color=colors['text']),
            xaxis=dict(showgrid=grid, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=grid, zeroline=False, showticklabels=False, title="")
        )

        return fig
    

    def similarity_heatmap(self, dist: str = 'cosine', use_subset: bool = True, n: int = 500, theme: str = 'light1', title: bool = None):
        ''' 
        Creates a heatmap showing pairwise distances between word embeddings.

        Parameters
        -----------
        dist : str, default='cosine'
            Distance metric to use for computing similarity between embeddings.
        use_subset : bool, default=True
            If True, uses a subset of the embeddings. Otherwise, uses the full set.
        n : int, optional
            Number of embeddings to subset. Ignored if a subset already exists and use_subset is True.
        theme : str, default='light1'
            Plot color theme to use.
        title : str, optional
            Title for the heatmap. If None, a default title is assigned.

        Returns
        --------
        fig : plotly.graph_objects.Figure
        '''

        if use_subset:
            if n:
                self.loader.subset(n)
                emb = self.loader.embeddings_subset 
                tokens = self.loader.tokens_subset
            else:
                emb, tokens = self.loader.use_subset()
        else:
            emb = self.embeddings
            tokens = self.tokens

        if emb.shape[0] > 500:
            warnings.warn(f"Warning: loading more than 500 embeddings without subsetting will generate more than one heatmap and may result in longer execution times. Consider subsetting before or setting n < 500.")
        
        distances = pairwise_distances(emb, metric=dist)

        colors = self.get_theme(theme)

        fig = px.imshow(distances, x=tokens, y=tokens, text_auto=True, color_continuous_scale=colors['scale'])
        fig.update_layout(
            width=800, height=800,
            title=title if title else "Word Embedding Similarity Heatmap",
            title_x=0.5,
            title_xanchor='center',
            plot_bgcolor=colors['bg'],
            paper_bgcolor=colors['bg'],
            font=dict(color=colors['text']))
        fig.update_coloraxes(colorbar_title='Distance')
        fig.update_traces(
            hovertemplate="Word 1: %{x}<br>Word 2: %{y}<br>Distance: %{z}<extra></extra>",
            hoverlabel=dict(
                bgcolor=colors['bg'], 
                font=dict(color=colors['text']) ))

        return fig
    

    def plot_clusters(self, n_clusters=5, method='kmeans', red_method='pca', show_centers=False, grid=True, theme='light1', title=None, nlabels=0, use_subset=False):
        '''
        Creates a 2D scatterplot of clustered embeddings using a clustering algorithm.

        Parameters:
        -----------
        n_clusters : int, default=5
            Number of clusters to generate.
        method : str, default='kmeans'
            Clustering method to use ('kmeans' or others supported by create_clusters).
        red_method : str, default='pca'
            Dimensionality reduction method to apply before plotting.
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
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        '''

        if use_subset:
            emb, tokens = self.loader.use_subset()
        else:
            emb = self.embeddings
            tokens = self.tokens

        reduced_emb = reduce_dim(emb, method=red_method)

        clusters, centers, reduced_emb = create_clusters(reduced_emb, n_clusters=n_clusters, method=method)
        clusters_colors, legend_labels = self.map_colors(clusters)

        fig, ax, colors = self._setup_plot(theme, grid, title)
        ax.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=clusters_colors, alpha=0.5, s=14, marker='o')

        if show_centers and centers is not None:
            for i in range(n_clusters):
                ax.scatter(centers[i, 0], centers[i, 1], edgecolors="grey", color=colors['text'], s=40, alpha=0.8, marker='o')

        legend_elements = [plt.Line2D([0], [0], marker='o',
                            color=color,                  
                            label=label_text,
                            markerfacecolor=color,       
                            markersize=8,
                            linestyle='None') 
                      for label, (color, label_text) in legend_labels.items()]

        texts=[] 
        if nlabels > 0:
            sparse_indices = self.select_sparse_labels(reduced_emb, nlabels)
            for i in sparse_indices:
                texts.append(ax.text(reduced_emb[i, 0], reduced_emb[i, 1], tokens[i],
                        color=colors['text'], fontsize=9, alpha=1, ha='center', va='bottom'))
                
        ax.legend(handles=legend_elements, facecolor=colors['bg'], labelcolor=colors['text'])
        adjust_text(texts, ax=ax, expand=(1.2, 2), arrowprops=dict(arrowstyle='-', color=colors['text']))
        plt.rcParams['figure.dpi'] = 600
        plt.show()
        return fig, ax


    def interactive_embeddings(self, red_method='pca', grid=True, theme='light1', title=None, use_subset=False):
        '''
        Creates an interactive 2D scatterplot of embeddings using Plotly.

        Parameters:
        -----------
        red_method : str, default='pca'
            Dimensionality reduction method to apply before plotting.
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
        '''

        if use_subset:
            emb, tokens = self.loader.use_subset()
        else:
            emb = self.embeddings
            tokens = self.tokens

        reduced_emb = reduce_dim(emb, method=red_method)

        colors = self.get_theme(theme)

        fig = px.scatter(reduced_emb, reduced_emb[:, 0], reduced_emb[:, 1], color_discrete_sequence=[colors['points']])
        fig.update_traces(
            text=tokens,
            textposition='top center',
            hovertemplate='%{text}<extra></extra>',
            hoverlabel=dict(
                bgcolor=colors['bg'], 
                font=dict(color=colors['text'])),
            marker=dict(size=6, opacity=0.6, line=dict(width=0))
        )
        fig.update_layout(
            height=600,
            title=title if title else "Word Embedding Interactive Plot",
            title_x=0.5,
            title_xanchor='center',
            plot_bgcolor=colors['bg'],
            paper_bgcolor=colors['bg'],
            font=dict(color=colors['text']),
            xaxis=dict(showticklabels=False, showgrid=grid, gridcolor=colors['grid'], zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=grid, gridcolor=colors['grid'], zeroline=False),
            xaxis_title=None,
            yaxis_title=None
        )

        return fig
    

    

