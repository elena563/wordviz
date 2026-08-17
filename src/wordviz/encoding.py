# ruff: noqa: E402
from wordviz._optional import require

require("encoding")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoTokenizer


def _validate_model(model: str) -> None:
    """Validates that a model name or path can be loaded."""
    try:
        AutoConfig.from_pretrained(model)
    except Exception as e:
        raise ValueError(
            f"Model '{model}' not found. Check the name or local path."
        ) from e


def encode_sentences(sentences: list[str], model: str = "all-MiniLM-L6-v2", device: str = "auto") -> dict:
    """
    Encodes a list of sentences into embeddings.

    Parameters
    -----------
    sentences : list of str
        Sentences to transform into embeddings.
    model : str, optional, default='all-MiniLM-L6-v2'
        Name of the Hugging Face model to use, or a local model path.
        It is recommended to use a sentence-transformers model.
        Common options:
        - 'all-MiniLM-L6-v2' (fast, English, 384 dimensions)
        - 'paraphrase-MiniLM-L3-v2' (very small, English, 256 dimensions)
        - 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual, 384 dimensions)
        Local paths are supported too: pass the path to a model directory,
        either a sentence-transformers model or a fine-tuned transformers model.
        A plain transformers model is loaded with mean pooling by default.
    device : str, optional, default='auto'
        Device to run the model on. Examples: 'cpu', 'cuda', 'mps', 'auto'.

    Returns
    --------
    dict
        A dictionary with the following keys:
        - 'embeddings' : np.ndarray of shape (n_sentences, dim)
            The sentence embeddings.
        - 'labels' : list of str
            The original input sentences.
        - 'type' : str
            Constant string 'sentence'.
        - 'model' : str
            The model name used to generate embeddings.
        - 'dimensions' : int
            Embedding vector size.
    """

    _validate_model(model)

    st_model = SentenceTransformer(model, device=device)
    embeddings = st_model.encode(sentences, convert_to_numpy=True)

    return {
        "embeddings": embeddings,
        "labels": sentences,
        "type": "sentence",
        "model": model,
        "dimensions": embeddings.shape[1],
    }


def get_model_info(model: str = "distilbert-base-uncased") -> dict:
    """
    Returns metadata about a transformer model.

    Useful to know available layers before calling encode_word_contexts
    with a specific layer_index.

    Parameters
    ----------
    model : str, optional, default='distilbert-base-uncased'
        Name of the Hugging Face model, or a local model path.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'model' : str
            The model name or path.
        - 'num_layers' : int
            Number of hidden layers (excludes embedding layer).
        - 'num_layers_total' : int
            Total layers including the embedding layer (valid range for layer_index).
        - 'hidden_size' : int
            Hidden dimension size.
        - 'num_attention_heads' : int
            Number of attention heads.
        - 'vocab_size' : int
            Vocabulary size.
    """
    config = AutoConfig.from_pretrained(model)
    return {
        "model": model,
        "num_layers": config.num_hidden_layers,
        "num_layers_total": config.num_hidden_layers + 1,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
    }


def _find_word_position(target_word: str, sentence: str, word_ids: list[int | None]) -> int | None:
    """Finds word position in tokenized sentence.

    Parameters
    ----------
    target_word : str
        Word to find.
    sentence : str
        Original sentence (used to determine word index).
    word_ids : list of int or None
        Word IDs from the tokenizer encoding, mapping token indices to word indices.

    Returns
    -------
    int or None
        Token index of the first occurrence of the target word, or None if not found.
    """
    tokens = [t.strip(".,!?;:'\"()[]") for t in sentence.lower().split()]
    target_token = target_word.lower()

    if target_token not in tokens:
        return None

    word_idx = tokens.index(target_token)

    for token_idx, wid in enumerate(word_ids):
        if wid == word_idx:
            return token_idx
    return None


def encode_word_contexts(
    target_word: str,
    sentences: list[str],
    match: str = "exact",
    occurrencies: str = "first",
    model: str = "distilbert-base-uncased",
    device: str = "auto",
    batch_size: int = 32,
    layer_index: int = -1,
) -> dict:
    """
    Encodes a word into contextual embeddings based on the sentence.

    Parameters
    -----------
    target_word: str
        Word to transform into embeddings.
    sentences : list of str
        Sentences to use for word encoding.
    match: str, default='exact' (more options will come)
        choose to include only target word or also derived terms
    occurrencies: str, default='first' (more options will come)
        decide about word occurrence in sentence
    model : str, optional, default='distilbert-base-uncased'
        Name of the Hugging Face model to use, or a local model path.
        Common options:
        - 'bert-base-uncased' (standard, 768 dimensions)
        - 'distilbert-base-uncased' (fast, 768 dimensions)
        - 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual, 384 dimensions)
        Local paths are supported: pass the path to a model directory
        containing config.json and model weights.
    device : str, optional, default='auto'
        Device to run the model on. Examples: 'cpu', 'cuda', 'mps', 'auto'.
    batch_size : int, optional, default=32
        Number of sentences to process in each batch.
    layer_index : int, optional, default=-1
        Which hidden state layer to extract the token embedding from.
        -1 = last layer (default).
        0 = embedding layer (before any transformer block).
        1..N = transformer layer outputs.
        Negative indices count from the end (e.g. -2 = second-to-last layer).

    Returns
    --------
    dict
        A dictionary with the following keys:
        - 'embeddings' : np.ndarray of shape (n_sentences, dim)
            The sentence embeddings.
        - 'labels' : list of str
            The original input sentences.
        - 'type' : str
            Constant string 'sentence'.
        - 'model' : str
            The model name used to generate embeddings.
        - 'dimensions' : int
            Embedding vector size.
    """

    _validate_model(model)

    tokenizer = AutoTokenizer.from_pretrained(model)
    encoding_model = AutoModel.from_pretrained(model)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    encoding_model.to(device)

    num_layers = encoding_model.config.num_hidden_layers + 1  # +1 for embedding layer
    if not (-num_layers <= layer_index < num_layers):
        raise ValueError(
            f"layer_index {layer_index} is out of range for model with {num_layers} layers "
            f"(valid range: {-num_layers} to {num_layers - 1})"
        )

    word_embeddings = []
    valid_sentences = []

    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]

        batch_encoding = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        valid_indices = []
        word_positions = []
        for j, sentence in enumerate(chunk):
            word_pos = _find_word_position(target_word, sentence, batch_encoding.word_ids(batch_index=j))
            if word_pos is not None:
                valid_indices.append(j)
                word_positions.append(word_pos)

        if not valid_indices:
            continue

        inputs = {k: v.to(device) for k, v in batch_encoding.items()}

        with torch.no_grad():
            outputs = encoding_model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[layer_index]

        for j, word_pos in zip(valid_indices, word_positions):
            word_emb = hidden_states[j][word_pos].cpu().numpy()
            word_embeddings.append(word_emb)
            valid_sentences.append(chunk[j])

    embeddings = np.array(word_embeddings)

    return {
        "embeddings": embeddings,
        "labels": valid_sentences,
        "type": "word_context",
        "target_word": target_word,
        "sentences": valid_sentences,
        "model": model,
        "dimensions": embeddings.shape[1],
    }
