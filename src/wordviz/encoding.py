# ruff: noqa: E402
from wordviz._optional import require

require("encoding")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


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

    st_model = SentenceTransformer(model, device=device)
    embeddings = st_model.encode(sentences, convert_to_numpy=True)

    return {
        "embeddings": embeddings,
        "labels": sentences,
        "type": "sentence",
        "model": model,
        "dimensions": embeddings.shape[1],
    }


def _find_word_position(target_word: str, sentence: str, encoding: dict) -> int | None:
    """Finds word position in tokenized sentence"""
    tokens = [t.strip(".,!?;:'\"()[]") for t in sentence.lower().split()]
    target_token = target_word.lower()

    if target_token not in tokens:
        return None

    word_idx = tokens.index(target_token)

    # first occurrence
    word_ids = encoding.word_ids()
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
        Name of the Hugging Face model to use. It is recommended to use a sentence-transformers model.
        Common options:
        - 'bert-base-uncased' (standard, 768 dimensions)
        - 'distilbert-base-uncased' (fast, 768 dimensions)
        - 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual, 384 dimensions)
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

    tokenizer = AutoTokenizer.from_pretrained(model)
    encoding_model = AutoModel.from_pretrained(model)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    encoding_model.to(device)

    word_embeddings = []
    valid_sentences = []

    for sentence in sentences:
        encoding = tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in encoding.items()}

        word_pos = _find_word_position(target_word, sentence, encoding)
        if word_pos is None:
            continue  # skip if word is not found

        with torch.no_grad():
            outputs = encoding_model(**inputs)

        word_emb = outputs.last_hidden_state[0][word_pos].cpu().numpy()
        word_embeddings.append(word_emb)
        valid_sentences.append(sentence)

    embeddings = np.array(word_embeddings)
    labels = [f"{target_word}_ctx_{i}" for i in range(len(valid_sentences))]

    return {
        "embeddings": embeddings,
        "labels": labels,
        "type": "word_context",
        "target_word": target_word,
        "sentences": valid_sentences,
        "model": model,
        "dimensions": embeddings.shape[1],
    }
