import numpy as np
import pytest

encoding = pytest.importorskip("wordviz.encoding")
from transformers import BertConfig, BertModel, BertTokenizer  # noqa: E402

from wordviz import EmbeddingLoader  # noqa: E402
from wordviz.encoding import encode_sentences  # noqa: E402

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "hello",
    "world",
    "test",
    "from",
    "model",
    "this",
    "is",
    "a",
    "sentence",
]

HIDDEN_SIZE = 32

SENTENCES = [
    "hello world",
    "this is a sentence",
    "test from model",
]


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_model")

    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    BertModel(config).save_pretrained(path)

    with open(path / "vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(VOCAB))
    tokenizer = BertTokenizer(vocab_file=str(path / "vocab.txt"))
    tokenizer.save_pretrained(path)

    return path


def test_encode_sentences_from_local_path(tiny_model_path):
    result = encode_sentences(SENTENCES, model=str(tiny_model_path), device="cpu")

    assert isinstance(result["embeddings"], np.ndarray)
    assert result["embeddings"].shape == (len(SENTENCES), HIDDEN_SIZE)
    assert result["dimensions"] == HIDDEN_SIZE
    assert result["labels"] == SENTENCES
    assert result["type"] == "sentence"
    assert result["model"] == str(tiny_model_path)


def test_load_contextual_from_encoding_dict(tiny_model_path):
    result = encode_sentences(SENTENCES, model=str(tiny_model_path), device="cpu")

    loader = EmbeddingLoader()
    loader.load_contextual(result)

    assert loader.embeddings.shape == (len(SENTENCES), HIDDEN_SIZE)
    assert loader.tokens == SENTENCES
    assert loader.type == "sentence"


def test_load_contextual_dict_only_embeddings_required():
    embeddings = np.random.default_rng(42).normal(size=(3, 8)).astype(np.float32)

    loader = EmbeddingLoader()
    loader.load_contextual({"embeddings": embeddings})

    assert loader.embeddings.shape == (3, 8)
    assert loader.type == "sentence"


def test_load_contextual_dict_missing_embeddings_raises():
    loader = EmbeddingLoader()

    with pytest.raises(ValueError, match="embeddings"):
        loader.load_contextual({"labels": ["a", "b"]})
