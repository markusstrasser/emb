import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from emb.embed import EmbeddingEngine, compute_content_hash, KNOWN_MODELS, model_slug, _l2_normalize
from emb.schema import Entry
from emb.cache import EmbeddingCache


def test_model_slug():
    assert model_slug('Alibaba-NLP/gte-modernbert-base') == 'Alibaba-NLP_gte-modernbert-base'
    assert model_slug('qwen3-embedding:8b-q8_0') == 'qwen3-embedding_8b-q8_0'
    assert model_slug('gemini-embedding-2-preview') == 'gemini-embedding-2-preview'
    # gte-768 and gemini-768 share a dim but get distinct slugs (the whole point)
    assert model_slug('Alibaba-NLP/gte-modernbert-base') != model_slug('gemini-embedding-2-preview')


def test_gemini_registered():
    engine = EmbeddingEngine(model='gemini-embedding-2-preview')
    assert engine.dim == 768
    assert engine.backend == 'gemini'
    assert engine.batch_size == 100


def test_l2_normalize_invariant():
    out = _l2_normalize([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
    norms = np.linalg.norm(np.array(out), axis=1)
    assert np.allclose(norms[0], 1.0)
    assert np.allclose(norms[2], 1.0)
    assert np.allclose(out[1], [0.0, 0.0])  # zero vector stays zero (no div-by-zero)


def test_l2_normalize_empty():
    # Regression: np.asarray([]) reshapes to a phantom (1,0) row → must return [] not [[]]
    assert _l2_normalize([]) == []


def test_embed_texts_normalizes_unnormalized_backend_output():
    # Backend returns NON-normalized vectors; embed_texts must enforce unit norm.
    engine = EmbeddingEngine(dim=3)
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32)
    engine._st_model = mock_model
    result = engine.embed_texts(["a", "b"])
    norms = np.linalg.norm(np.array(result), axis=1)
    assert np.allclose(norms, 1.0), f"central normalize invariant violated: {norms}"


def test_compute_content_hash():
    h1 = compute_content_hash("hello world")
    h2 = compute_content_hash("hello world")
    h3 = compute_content_hash("different text")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 32

def test_engine_defaults():
    engine = EmbeddingEngine()
    assert engine.model == 'Alibaba-NLP/gte-modernbert-base'
    assert engine.dim == 768
    assert engine.backend == 'sentence-transformers'

def test_engine_custom_model():
    engine = EmbeddingEngine(model='qwen3-embedding:8b-q8_0')
    assert engine.dim == 4096
    assert engine.backend == 'ollama'

def test_engine_unknown_model():
    engine = EmbeddingEngine(model='custom/model', dim=512, backend='sentence-transformers')
    assert engine.dim == 512
    assert engine.backend == 'sentence-transformers'

def test_embed_texts_with_mock():
    engine = EmbeddingEngine(dim=4)
    # Mock the ST model
    mock_model = MagicMock()
    import numpy as np
    mock_model.encode.return_value = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    engine._st_model = mock_model

    result = engine.embed_texts(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 4
    mock_model.encode.assert_called_once()

def test_embed_entries_uses_cache():
    engine = EmbeddingEngine(dim=4)
    mock_model = MagicMock()
    import numpy as np

    # Return one embedding per input text
    def _encode_side_effect(texts, **kwargs):
        return np.array([[1, 0, 0, 0]] * len(texts), dtype=np.float32)

    mock_model.encode.side_effect = _encode_side_effect
    engine._st_model = mock_model

    cache = EmbeddingCache(dim=4)

    e1 = Entry(id="a", text="hello")
    e2 = Entry(id="b", text="world")

    # First run: both need embedding
    result = engine.embed_entries([e1, e2], cache=cache)
    assert all(e.embedding is not None for e in result)
    assert len(cache) == 2

    # Second run: e3 has same text as e1 (cached), e4 is new
    mock_model.encode.reset_mock()
    mock_model.encode.side_effect = _encode_side_effect
    e3 = Entry(id="c", text="hello")  # same text as e1
    e4 = Entry(id="d", text="new text")

    result2 = engine.embed_entries([e3, e4], cache=cache)
    # e3 reuses cache (same content hash as e1), only e4 needs embedding
    assert e3.embedding is not None
    # The mock should only have been called for the one new entry
    assert mock_model.encode.call_count == 1

def test_embed_entries_checkpoint(tmp_path):
    engine = EmbeddingEngine(dim=4, batch_size=2)
    mock_model = MagicMock()
    import numpy as np
    mock_model.encode.return_value = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    engine._st_model = mock_model

    entries = [Entry(id=f"e{i}", text=f"text {i}") for i in range(10)]
    cache = EmbeddingCache(dim=4)

    checkpoint_dir = tmp_path / "checkpoints"
    engine.embed_entries(entries, cache=cache, checkpoint_dir=checkpoint_dir, checkpoint_interval=4)

    # Cache should have been saved (checkpoint_dir should have files)
    assert (checkpoint_dir / "index.json").exists()
    assert (checkpoint_dir / "vectors.npy").exists()

def test_text_truncation():
    engine = EmbeddingEngine(dim=4, max_chars=10)
    mock_model = MagicMock()
    import numpy as np
    mock_model.encode.return_value = np.array([[1, 0, 0, 0]], dtype=np.float32)
    engine._st_model = mock_model

    engine.embed_texts(["a" * 100])
    # Check that the text passed to encode was truncated
    call_args = mock_model.encode.call_args[0][0]
    assert len(call_args[0]) == 10
