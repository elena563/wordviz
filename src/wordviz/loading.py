import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from  gensim.models.fasttext import load_facebook_model
import json
import warnings
import numpy as np

import logging
logger = logging.getLogger(__name__)

from wordviz.helpers.files_helpers import download_file, validate_file, get_cache_dir, extract_archive, export_embedding

class EmbeddingLoader:
    """
        Loads word or sentence embedding.

        Attributes
        ----------
        embeddings_raw : Any
            KeyedVectors format for static embeddings
        embeddings : np.ndarray
            Array of embeddings
        tokens : list of str
            Representative elements for the embeddings in natural language (words, sentences, or other elements to visualize)
        dimension : int
            Dimensionality of the embeddings.
        type: str
            Type of embedding
            - 'word': word embeddings
            - 'sentence': Sentence/document/passage embeddings
            - 'word_context': Word embeddings in different contexts  
            - 'custom': User-defined
        classes : list of str, optional
            Class labels for the embeddings, used for coloring in visualizations.
        """
    def __init__(self):
        self.embeddings_raw = None 
        self.embeddings = None      
        self.tokens = None         
        self.dimension = None
        self.type = None
        self._classes = None
        self.embeddings_subset = None
        self.tokens_subset = None

        with open(os.path.join(os.path.dirname(__file__), 'pretrained_embeddings.json')) as f:
            self.available_pretrained = json.load(f)

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value: list[str]):
        if value is not None:
            if len(value) != len(self.tokens):
                raise ValueError(
                    f"'classes' length ({len(value)}) must match number of tokens ({len(self.tokens)})"
                )
        self._classes = value
    
    def _require_loaded(self, check_tokens=False) -> None:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_from_file() or load_contextual() first.")
        if check_tokens and self.tokens is None:
            raise ValueError("No tokens loaded. Call load_from_file() or load_contextual() first.")


    def load_from_file(self, path: str, format: str) -> np.ndarray:
        '''
        Loads word embeddings from a file in .txt, .vec, or .bin format.

        Parameters
        -----------
        path : str
            Path to the embedding file.
        format : str
            Format of the embedding model: 'word2vec', 'fasttext', or 'glove'.

        Returns
        --------
        np.ndarray
            Loaded embedding matrix.

        Notes:
        ------
        - FastText binary files are supported via Facebook's native loader.
        - Loaded tokens are stored in self.tokens.
        - Embedding matrix is stored in self.embeddings.
        '''

        binary = validate_file(path)

        match format:
            case 'word2vec':
                    self.embeddings_raw = KeyedVectors.load_word2vec_format(path, binary=binary)
            case 'fasttext':
                if binary:
                    self.embeddings_raw = load_facebook_model(path).wv
                else:
                    self.embeddings_raw = KeyedVectors.load_word2vec_format(path, binary=False)
            case 'glove':
                self.embeddings_raw = KeyedVectors.load_word2vec_format(path, binary=False, no_header=True)

        self.tokens = list(self.embeddings_raw.index_to_key)
        self.dimension = self.embeddings_raw.vector_size
        self.type = 'word'

        self.embeddings = self.embeddings_raw.vectors
        logger.info("Embedding loaded from file")

        return self.embeddings

    def load_pretrained(self, model: str, lang: str, source: str, dimension: str, save_file: bool = False, export_dir: str = None) -> np.ndarray:
        '''
        Downloads and loads a pretrained embedding model from an online source.

        Parameters
        -----------
        model : str
            Name of the embedding model ('glove' or 'fasttext').
        lang : str
            Language code of the embedding ('en', 'it').
        source : str
            Data source ('wiki', 'cc').
        dimension : str or int
            Embedding dimensionality (e.g., '300').
        save_file : bool, default=False
            If True, saves the embedding to the specified export directory.
        export_dir : str, optional
            Path to the directory where the file will be exported (used if save_file=True).

        Returns
        --------
        np.ndarray
            Loaded embedding matrix (n_words x dimension).
        '''

        columns = self.available_pretrained["columns"]
        option = next(
            (dict(zip(columns, row)) for row in self.available_pretrained["data"]
            if row[0] == model and row[1] == lang and row[2] == source and row[3] == dimension),
            None
        )
        if option is not None:
            url = option['url']
            filename = option['filename']
        elif model == 'word2vec' and lang == 'en' and source == 'googlenews' and dimension == '300d':
            warnings.warn("The Google News pretrained embeddings are no longer available for download. Please provide a local file path to load them or choose a different pretrained.")
        else:
            raise ValueError(f"Can't find pretrained file with parameters: {model}, {lang}, {source}, {dimension}")
        zip_filename = url.split("/")[-1]

        zip_path = download_file(url, zip_filename)

        dest_dir = get_cache_dir() / model / lang / source / dimension
        dest_dir.mkdir(parents=True, exist_ok=True)
        file_path = dest_dir / filename

        if not file_path.exists():
            extract_archive(zip_path, filename, dest_dir)

        self.embeddings = self.load_from_file(file_path, model)

        if save_file:
            if export_dir is None:
                raise ValueError("Must specify export_dir to save file.")
            export_embedding(file_path, export_dir)

        return self.embeddings
    
    
    def load_contextual(self, embeddings, labels: list, embedding_type: str = 'sentence', classes: list = None) -> np.ndarray:
        """
        Loads embeddings from contextual models.
        
        Parameters
        -----------
        embeddings: various formats
            - numpy.ndarray
            - torch.Tensor
            - List[List[float]] 
        labels: list of str
            labels corresponding to embedding 
        embedding_type: str
            - 'sentence': Sentence/document/passage embeddings
            - 'word_context': Word embeddings in different contexts  
            - 'word': word embeddings
            - 'custom': User-defined
        classes: list of str, optional
            Class labels for the embeddings
        Returns
        --------
        np.ndarray
            Loaded embedding matrix (n_labels x dimension).
        """
        
        embeddings_array = self._normalize_embeddings(embeddings)
        
        self.embeddings = embeddings_array  
        self.tokens = labels
        self.classes = classes
        self.dimension = embeddings_array.shape[1]
        self.type = embedding_type
        logger.info("Contextual embedding loaded")
        
        return self.embeddings

    def _normalize_embeddings(self, embeddings) -> np.ndarray:
        """Converts embeddings to numpy array."""

        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32)
         
        elif hasattr(embeddings, 'detach'):  # torch.Tensor
            return embeddings.detach().cpu().numpy().astype(np.float32)
        
        elif isinstance(embeddings, list):
            return np.array(embeddings, dtype=np.float32)
        
        else:
            try:
                return np.array(embeddings, dtype=np.float32)
            except Exception:
                raise ValueError(f"Cannot convert embeddings of type {type(embeddings)} to numpy array")


    def list_available_pretrained(self) -> None:
        '''prints a list of pretrained embeddings provided by the package'''
        print('model | lang | source | dim')
        for file in self.available_pretrained['data']:
            print(" | ".join(x for x in file[:-2]))

    
    def get_embedding(self, token: str) -> np.ndarray:
        '''returns corresponding embeddings using KeyedVectors object for a string given by the user'''
        self._require_loaded()
        
        if self.type in ("sentence", "word_context"):
            try:
                index = self.tokens.index(token)
            except ValueError:
                raise KeyError(f"Token '{token}' not found")
            return self.embeddings[index]
        elif self.type == "word":
            # prefer keyed vectors if available
            if getattr(self, "embeddings_raw", None) is not None:
                try:
                    return self.embeddings_raw.get_vector(token)
                except KeyError:
                    pass
            try:
                idx = self.tokens.index(token)
            except ValueError:
                raise KeyError(f"Token '{token}' not found")
            return self.embeddings[idx]
        else:
            raise RuntimeError("Unknown embedding type")
            

    def subset(self, n: int = 1000, strategy: str = 'first', random_seed: int = None) -> None:
        '''
        Create a subset of the current embeddings and tokens. Useful for speeding up visualizations or 
        managing memory with large embedding spaces.

        Parameters
        -----------
        n : int, default=1000
            Number of embeddings to retain. If n exceeds the total number of available embeddings, all are retained.
        strategy : str, default='first'
            Selection strategy:
                - 'first': select the first n embeddings in original order.
                - 'random': select n random embeddings.
        random_seed : int, optional
            Seed for reproducible random sampling (only used if strategy is 'random').

        Updates
        --------
        self.tokens_subset : list of str
            List of selected token strings.
        self.embeddings_subset : np.ndarray
            Corresponding selected embedding vectors.
        '''
        self._require_loaded(check_tokens=True)
        
        emb_size = self.embeddings.shape[0]
        
        if n > emb_size:
            logger.info('n is larger than the embedding size, the subset size will be equal to the full size')

        if strategy == 'first':
            indices = list(range(min(n, emb_size)))
        elif strategy == 'random':
            rng = np.random.default_rng(random_seed)
            indices = rng.choice(emb_size, size=min(n, emb_size), replace=False).tolist()
        else:
            raise ValueError("strategy has to be 'first' o 'random'")
        
        self.tokens_subset    = [self.tokens[i] for i in indices]
        self.embeddings_subset = self.embeddings[indices]


    def use_subset(self, n: int = 1000) -> tuple[list[str], np.ndarray]:
        '''returns embedding subset. If None, creates 1000 words subset and returns it.'''

        if self.embeddings_subset is None:
            self.subset(n)
        
        return self.embeddings_subset, self.tokens_subset