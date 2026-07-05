import os
import json
from adjustText import adjust_text
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like, to_rgb
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, to_tree
import warnings
from .clustering import create_clusters
from .dim_reduction import reduce_dim
from .similarity import n_most_similar, compute_distances
from .utils import compute_positions, draw_tree


class BaseVisualizer:
    def __init__(self, loader):
        self.loader = loader
        self.tokens = loader.tokens
        self.embeddings = loader.embeddings

        with open(os.path.join(os.path.dirname(__file__), 'themes.json')) as f:
            self.themes = json.load(f)

    def list_theme_colors(self):
        '''prints a list of available themes provided by the package'''
        print('background | points  | target  |   grid   | text')
        for theme_name, theme in self.themes.items():
            colors = [v for v in theme.values() if is_color_like(v)]
            sns.palplot(colors)
            plt.title(theme_name)
            plt.show()

    def get_theme(self, theme='light1'):
        return self.themes.get(theme, self.themes['light1'])
    

    def _set_embeddings(self, use_subset=False, n=None, red_method=None, dims=2):
        if use_subset:
            if n:
                self.loader.subset(n)
                emb = self.loader.embeddings_subset
                tokens = self.loader.tokens_subset
            else:
                emb, tokens = self.loader.use_subset()

            cache_attr = 'reduced_subset'
        else:
            emb = self.embeddings
            tokens = self.tokens
            cache_attr = 'reduced'

        if red_method is not None:
            if red_method == 'auto':
                reduced = getattr(self, cache_attr, None)

                if reduced is None:
                    reduced = reduce_dim(emb, method='pca', n_dimensions=dims)
                    setattr(self, cache_attr, reduced)
            else:
                reduced = reduce_dim(emb, method=red_method, n_dimensions=dims)
                setattr(self, cache_attr, reduced)

            return reduced, tokens
        
        return emb, tokens
    

    def _setup_plot(self, theme, grid, title, dims=2):
        '''base private function to config matplotlib plot'''
        style = self.get_theme(theme)

        if dims == 2:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            bg = to_rgb(style['bg'])
            ax.set_facecolor(style['bg'])
            ax.xaxis.set_pane_color((*bg, 1.0))
            ax.yaxis.set_pane_color((*bg, 1.0))
            ax.zaxis.set_pane_color((*bg, 1.0))

            if grid:
                ax.xaxis.gridlines.set_color(style['grid_color'])
                ax.yaxis.gridlines.set_color(style['grid_color'])
                ax.zaxis.gridlines.set_color(style['grid_color'])

        fig.patch.set_facecolor(style['bg'])
        ax.set_facecolor(style['bg'])

        if grid:  
            ax.grid(True, linestyle=style['grid_style'], color=style['grid_color'], alpha=0.6)
            ax.set_axisbelow(True) 
            ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False, color=(0.5, 0.5, 0.5, 0.4))
        else:
            plt.xticks([])
            plt.yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        if title is not None:
            plt.title(title, fontsize=12, fontweight='bold', color=style['text'])

        return fig, ax, style
    

    def map_colors(self, labels, theme):
        '''automatizes color and legend label mapping for clustering applied to embeddings'''
        colors = self.get_theme(theme)

        unique_classes = list(set(labels))
        palette = sns.color_palette(colors['clusters'], n_colors=len(unique_classes))
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

class Visualizer(BaseVisualizer):
    def __init__(self, loader):
        super().__init__(loader) 
        self.reduced = None
        self.reduced_subset = None


    def plot_embeddings(self, red_method: str = 'auto', grid: bool = True, theme: str = 'light1', title: str = None, nlabels: int = 0, use_subset: bool = False):   
        '''
        Creates a simple static 2D scatterplot of the embeddings.

        Parameters
        -----------
        red_method : str, default='auto'
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

        reduced_emb, tokens = self._set_embeddings(use_subset=use_subset, red_method=red_method)

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
    
    
    def plot_similarity(self, target_word: str, dist: str = 'cosine', n: int = 10, red_method: str = 'pca', grid: bool = True, theme: str = 'light1', title: str = None):
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
        red_method : str, default='pca'
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
        warnings.warn(
            "The parameter names target_word will be renamed to target in a future release. "
            "Please update your code accordingly.",
            FutureWarning
        )
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


    def plot_topography(self, red_method: str = 'auto', use_subset: bool = True, grid: bool = True, theme: str = 'light1', title: str = None):       
        '''
        Plots word embeddings in a topographical map using dimensionality reduction to maintain word distances in the representation. Allows to visualize word density in the space.

        Parameters
        -----------
        red_method : str, default='auto'
            Dimensionality reduction method to apply ('pca', 'tsne', 'umap', etc.). If 'auto' searches for cached reduction, if None runs pca.
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

        reduced_emb, tokens = self._set_embeddings(use_subset=use_subset, red_method=red_method)

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

        # add contour
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

        # add points
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
    

    def plot_heatmap(self, use_subset: bool = True, n: int = 500, theme: str = 'light1', title: bool = None):
        ''' 
        Creates a heatmap showing every vectorial value of every word.

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

        emb, tokens = self._set_embeddings(use_subset=use_subset, n=n)

        colors = self.get_theme(theme)

        fig = go.Figure(data=go.Heatmap(
        z=emb,
        x=[f"Dim {i+1}" for i in range(emb.shape[1])],
        y=tokens,
        colorscale=colors['scale'],
        colorbar=dict(title='Value'),
        hovertemplate="Word: %{y}<br>Dimension: %{x}<br>Value: %{z:.3f}<extra></extra>"
        ))

        fig.update_layout(
            title=title if title else "Esempio di heatmap con Word Embedding",
            width=800,
            height=400,
            plot_bgcolor=colors['bg'],
            paper_bgcolor=colors['bg'],
            font=dict(color=colors['text'])
        )

        fig.update_coloraxes(colorbar_title='Value')

        fig.update_traces(
            hovertemplate="Word: %{y}<br>Dimension: %{x}<br>Value: %{z}<extra></extra>",
            hoverlabel=dict(
                bgcolor=colors['bg'],
                font=dict(color=colors['text'])
            )
        )

        return fig
    

    def plot_similarity_heatmap(self, dist: str = 'cosine', use_subset: bool = True, n: int = 500, theme: str = 'light1', title: bool = None):
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

        emb, tokens = self._set_embeddings(use_subset=use_subset, n=n)

        if emb.shape[0] > 500:
            warnings.warn(f"Warning: loading more than 500 embeddings without subsetting will generate more than one heatmap and may result in longer execution times. Consider subsetting before or setting n < 500.")
        
        distances = compute_distances(emb, metric=dist)

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
    
    def similarity_heatmap(self, dist: str = 'cosine', use_subset: bool = True, n: int = 500, theme: str = 'light1', title: bool = None):
        '''
        DEPRECATED: This method will be renamed to `plot_similarity_heatmap` in a future release.

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
        warnings.warn(
            "interactive_embeddings is deprecated and will be renamed to plot_similarity_heatmap in a future release. "
            "Please update your code accordingly.",
            FutureWarning
        )
        return self.plot_similarity_heatmap(dist=dist, use_subset= use_subset, n=n, theme=theme, title=title)
        
    
    

    def plot_clusters(self, n_clusters: int = 5, method: str ='kmeans', red_method: str ='auto', show_centers: bool =False, grid: bool =True, theme: str ='light1', title: str =None, nlabels: int =0, use_subset: bool =False):
        '''
        Creates a 2D scatterplot of clustered embeddings using a clustering algorithm.

        Parameters:
        -----------
        n_clusters : int, default=5
            Number of clusters to generate.
        method : str, default='kmeans'
            Clustering method to use ('kmeans' or others supported by create_clusters).
        red_method : str, default='auto'
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
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        '''

        reduced_emb, tokens = self._set_embeddings(use_subset=use_subset, red_method=red_method)

        clusters, centers, reduced_emb = create_clusters(reduced_emb, n_clusters=n_clusters, method=method)
        clusters_colors, legend_labels = self.map_colors(clusters, theme=theme)

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
    

    def plot_interactive(self, red_method='auto', grid=True, theme='light1', title=None, use_subset=False, color_by_class=False):
        '''
        Creates an interactive 2D scatterplot of embeddings using Plotly.

        Parameters:
        -----------
        red_method : str, default='auto'
            Dimensionality reduction method to apply ('pca', 'tsne', 'umap', etc.). If 'auto' searches for cached reduction, if None runs pca.
        grid : bool, default=True
            Whether to display grid lines.
        theme : str, default='light1'
            Plot color theme.
        title : str, optional
            Title of the plot. If None, no title is shown.
        use_subset : bool, default=False
            If True, uses the embedding subset instead of the full embeddings.
        color_by_class : bool, default=False
            If True, colors the points by their class labels (if available).

        Returns:
        --------
        fig : plotly.graph_objects.Figure
        '''

        reduced_emb, tokens = self._set_embeddings(use_subset=use_subset, red_method=red_method)

        style = self.get_theme(theme)
        classes = self.loader.classes if color_by_class else None

        if classes is not None:
            unique = list(dict.fromkeys(classes))  
            palette = sns.color_palette('tab10', n_colors=len(unique))
            color_map = {label: f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                        for label, (r, g, b) in zip(unique, palette)}
            
            fig = go.Figure()
            for label in unique:
                mask = [i for i, c in enumerate(classes) if c == label]
                fig.add_trace(go.Scatter(
                    x=reduced_emb[mask, 0],
                    y=reduced_emb[mask, 1],
                    mode='markers',
                    name=label,
                    text=[tokens[i] for i in mask],
                    hovertemplate='%{text}<extra></extra>',
                    hoverlabel=dict(bgcolor=style['bg'], font=dict(color=style['text'])),
                    marker=dict(size=6, opacity=0.6, color=color_map[label])
                ))
        else:
            fig = px.scatter(reduced_emb, reduced_emb[:, 0], reduced_emb[:, 1], color_discrete_sequence=[style['points']])
            fig.update_traces(
                text=tokens,
                textposition='top center',
                hovertemplate='%{text}<extra></extra>',
                hoverlabel=dict(
                    bgcolor=style['bg'], 
                    font=dict(color=style['text'])),
                marker=dict(size=6, opacity=0.6, line=dict(width=0))
            )
        fig.update_layout(
            height=500,
            title=title if title else "Word Embedding Interactive Plot",
            title_x=0.5,
            title_xanchor='center',
            plot_bgcolor=style['bg'],
            paper_bgcolor=style['bg'],
            font=dict(color=style['text']),
            xaxis=dict(showticklabels=False, showgrid=grid, gridcolor=style['grid_color'], zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=grid, gridcolor=style['grid_color'], zeroline=False),
            xaxis_title=None,
            yaxis_title=None
        )

        return fig
    
    def interactive_embeddings(self, red_method='auto', grid=True, theme='light1', title=None, use_subset=False):
        '''
        DEPRECATED: This method will be renamed to `plot_interactive` in a future release.

        Parameters:
        -----------
        red_method : str, default='auto'
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
        '''
        warnings.warn(
            "interactive_embeddings is deprecated and will be renamed to plot_interactive in a future release. "
            "Please update your code accordingly.",
            FutureWarning
        )
        return self.plot_interactive(red_method=red_method, grid=grid, theme=theme, title=title, use_subset=use_subset)
        
    
    def plot_dendrogram(self, label_fontsize=10, grid=False, use_subset=False, n: int = 500):
    #(n_clusters=8 ):
        '''
        Creates a 2D circular dendrogram of clustered embeddings using hierarchical clustering.
        This first version of this function does not include title and theme parameters. Adapted from Claude Sonnet 4.5 generation.

        Z = linkage(reduced_emb, method='complete')
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust') 
        clusters_colors, legend_labels = self.map_colors(clusters)

        Z2 = dendrogram(Z, labels=tokens, no_plot=True)
        labels  = [v[1] for v in legend_labels.values()] 

        rt.plot(
            Z2,
            colorlabels={'cluster': clusters_colors},  
            colorlabels_legend={'cluster': {            
                'colors': clusters_colors,
                'labels': labels
            }},
            fontsize=6,            
        )

        Parameters:
        -----------
        n_clusters : int, default=8
            Number of clusters to generate.
        red_method : str, default='auto'
            Dimensionality reduction method for better interpretability ('pca', 'tsne', 'umap', etc.). If 'auto' searches for cached reduction, if None runs pca.
        use_subset : bool, default=False
            If True, uses the embedding subset instead of the full embeddings.

        Returns:
        --------
        fig : matplotlib.figure.Figure
        '''
        emb, tokens = self._set_embeddings(use_subset=use_subset, n=n)
        Z = linkage(emb, method='complete')
        tree = to_tree(Z, rd=False)
        n_leaves = tree.count
        
        if tokens is None:
            tokens = [f'Leaf {i}' for i in range(n_leaves)]
        
        leaf_angles = {}
        leaf_counter = [0]
        max_dist = tree.dist
        node_positions = {}  # {node_id: (angle, radius)}

        compute_positions(tree, leaf_counter, n_leaves, max_dist, leaf_angles, node_positions)

        fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(projection='polar'))

        draw_tree(tree, ax, node_positions, line_color='black', linewidth=1.0)

        label_distance = 1.05 
        for leaf_id, angle in leaf_angles.items():
            label_idx = list(leaf_angles.keys()).index(leaf_id)
            ax.text(angle, label_distance, tokens[label_idx],
                rotation=np.degrees(angle) - 90 if angle < np.pi else np.degrees(angle) + 90,
                rotation_mode='anchor',
                ha='left' if angle < np.pi else 'right',
                va='center',
                fontsize=label_fontsize,
                zorder=2)
        
        ax.set_ylim(0, label_distance + 0.05)
        ax.set_theta_zero_location('N')  # 0° in alto
        ax.set_theta_direction(1)  # senso orario
        
        if not grid:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines['polar'].set_visible(False)
            ax.grid(False)
        else:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.close(fig)
        return fig