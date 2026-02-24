"""Tests for search engine (dense + BM25 + RRF + reranking + freshness + chunk dedup)."""
import json
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from emb.search import SearchEngine


def _make_index(entries_data, tmp_path, model='test-model', dim=4):
    """Helper: create a minimal index file."""
    entries = []
    for e in entries_data:
        emb = e.get('embedding', np.random.randn(dim).tolist())
        # Normalize
        arr = np.array(emb, dtype=np.float32)
        arr = arr / np.linalg.norm(arr)
        entry = {
            'id': e['id'],
            'text': e.get('text', f"text for {e['id']}"),
            'embedding': arr.tolist(),
        }
        if 'source' in e:
            entry['source'] = e['source']
        if 'title' in e:
            entry['title'] = e['title']
        if 'date' in e:
            entry['date'] = e['date']
        if 'metadata' in e:
            entry['metadata'] = e['metadata']
        entries.append(entry)

    index = {
        'metadata': {'embedding_model': model, 'embedding_dim': dim},
        'entries': entries,
    }
    path = tmp_path / 'test_index.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(index, f)
    return path


def test_load_index(tmp_path):
    path = _make_index([
        {'id': 'a', 'text': 'hello world', 'source': 'test'},
        {'id': 'b', 'text': 'foo bar', 'source': 'test'},
    ], tmp_path)
    engine = SearchEngine(path)
    assert len(engine.entries) == 2
    assert engine.embeddings.shape == (2, 4)


def test_dense_search_returns_sorted(tmp_path):
    """Create entries with known embeddings, verify cosine ranking."""
    query_vec = [1, 0, 0, 0]
    path = _make_index([
        {'id': 'close', 'text': 'close match', 'embedding': [0.9, 0.1, 0, 0]},
        {'id': 'far', 'text': 'far match', 'embedding': [0, 0, 1, 0]},
        {'id': 'medium', 'text': 'medium match', 'embedding': [0.5, 0.5, 0, 0]},
    ], tmp_path)
    engine = SearchEngine(path)
    # Mock query encoding to return our known vector
    engine._encode_query = lambda q: np.array(query_vec, dtype=np.float32) / np.linalg.norm(query_vec)

    results = engine.search("test query", top_k=3)
    assert results[0]['id'] == 'close'
    assert results[-1]['id'] == 'far'
    assert results[0]['similarity'] > results[1]['similarity'] > results[2]['similarity']


def test_source_filtering(tmp_path):
    path = _make_index([
        {'id': 'a', 'text': 'hello', 'source': 'chatgpt'},
        {'id': 'b', 'text': 'world', 'source': 'git'},
        {'id': 'c', 'text': 'foo', 'source': 'chatgpt'},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.random.randn(4).astype(np.float32)

    results = engine.search("test", sources={'git'})
    assert len(results) == 1
    assert results[0]['source'] == 'git'


def test_source_filtering_multiple(tmp_path):
    """Filter to multiple sources."""
    path = _make_index([
        {'id': 'a', 'text': 'hello', 'source': 'chatgpt'},
        {'id': 'b', 'text': 'world', 'source': 'git'},
        {'id': 'c', 'text': 'foo', 'source': 'twitter'},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.random.randn(4).astype(np.float32)

    results = engine.search("test", sources={'chatgpt', 'twitter'})
    assert len(results) == 2
    sources = {r['source'] for r in results}
    assert sources == {'chatgpt', 'twitter'}


def test_min_similarity_filter(tmp_path):
    """Entries below min_similarity are excluded."""
    path = _make_index([
        {'id': 'close', 'text': 'close match', 'embedding': [1, 0, 0, 0]},
        {'id': 'far', 'text': 'far match', 'embedding': [0, 0, 0, 1]},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.array([1, 0, 0, 0], dtype=np.float32)

    results = engine.search("test", min_similarity=0.5)
    assert len(results) == 1
    assert results[0]['id'] == 'close'


def test_since_filter(tmp_path):
    """Only entries after since date are returned."""
    path = _make_index([
        {'id': 'new', 'text': 'recent entry', 'date': '2025-06-01'},
        {'id': 'old', 'text': 'old entry', 'date': '2020-01-01'},
        {'id': 'nodate', 'text': 'no date entry'},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.random.randn(4).astype(np.float32)

    results = engine.search("test", since=datetime(2024, 1, 1))
    assert len(results) == 1
    assert results[0]['id'] == 'new'


def test_bm25_hybrid_search(tmp_path):
    """Hybrid search should work and return results."""
    path = _make_index([
        {'id': 'a', 'text': 'python programming language', 'title': 'Python Guide'},
        {'id': 'b', 'text': 'rust systems programming', 'title': 'Rust Guide'},
        {'id': 'c', 'text': 'cooking recipes pasta', 'title': 'Cooking'},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.random.randn(4).astype(np.float32)

    # Hybrid search should at least not crash and return results
    results = engine.search("python programming", hybrid=True)
    assert len(results) > 0


def test_bm25_boosts_exact_match(tmp_path):
    """Hybrid should rank exact keyword match higher than dense-only."""
    # Give the keyword-match entry a worse dense embedding
    path = _make_index([
        {'id': 'keyword', 'text': 'quantum computing breakthrough', 'embedding': [0, 0, 0, 1]},
        {'id': 'dense', 'text': 'unrelated text about nothing', 'embedding': [1, 0, 0, 0]},
    ], tmp_path)
    engine = SearchEngine(path)
    # Query vec is close to 'dense' but far from 'keyword' in dense space
    engine._encode_query = lambda q: np.array([1, 0, 0, 0], dtype=np.float32)

    # Dense-only: 'dense' wins
    dense_results = engine.search("quantum computing", hybrid=False)
    assert dense_results[0]['id'] == 'dense'

    # Hybrid: BM25 should boost 'keyword' enough to win
    hybrid_results = engine.search("quantum computing", hybrid=True)
    assert hybrid_results[0]['id'] == 'keyword'


def test_chunk_deduplication(tmp_path):
    """Chunks from same parent should be deduped to best one."""
    path = _make_index([
        {'id': 'ep1__chunk_0', 'text': 'first chunk', 'embedding': [1, 0, 0, 0],
         'metadata': {'parent_id': 'ep1', 'chunk_index': 0, 'total_chunks': 2}},
        {'id': 'ep1__chunk_1', 'text': 'second chunk', 'embedding': [0.9, 0.1, 0, 0],
         'metadata': {'parent_id': 'ep1', 'chunk_index': 1, 'total_chunks': 2}},
        {'id': 'ep2__chunk_0', 'text': 'other episode', 'embedding': [0.5, 0.5, 0, 0],
         'metadata': {'parent_id': 'ep2', 'chunk_index': 0, 'total_chunks': 1}},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.array([1, 0, 0, 0], dtype=np.float32)

    results = engine.search("test", top_k=10)
    parent_ids = [r['metadata'].get('parent_id') for r in results]
    # ep1 should appear only once (deduped)
    assert parent_ids.count('ep1') == 1
    assert parent_ids.count('ep2') == 1


def test_chunk_dedup_keeps_best(tmp_path):
    """Dedup should keep the chunk with highest score, not first chunk."""
    # chunk_1 has better embedding for our query
    path = _make_index([
        {'id': 'ep1__chunk_0', 'text': 'first chunk', 'embedding': [0.1, 0.9, 0, 0],
         'metadata': {'parent_id': 'ep1', 'chunk_index': 0}},
        {'id': 'ep1__chunk_1', 'text': 'second chunk', 'embedding': [1, 0, 0, 0],
         'metadata': {'parent_id': 'ep1', 'chunk_index': 1}},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.array([1, 0, 0, 0], dtype=np.float32)

    results = engine.search("test", top_k=10)
    assert len(results) == 1
    assert results[0]['id'] == 'ep1__chunk_1'  # Higher similarity to query


def test_entries_without_parent_id_not_deduped(tmp_path):
    """Entries without parent_id should each appear independently."""
    path = _make_index([
        {'id': 'a', 'text': 'first', 'embedding': [1, 0, 0, 0]},
        {'id': 'b', 'text': 'second', 'embedding': [0.9, 0.1, 0, 0]},
        {'id': 'c', 'text': 'third', 'embedding': [0.5, 0.5, 0, 0]},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.array([1, 0, 0, 0], dtype=np.float32)

    results = engine.search("test", top_k=10)
    assert len(results) == 3


def test_freshness_weighting(tmp_path):
    today = datetime.now().strftime('%Y-%m-%d')
    old = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
    path = _make_index([
        {'id': 'new', 'text': 'recent entry', 'date': today, 'embedding': [0.5, 0.5, 0, 0]},
        {'id': 'old', 'text': 'old entry', 'date': old, 'embedding': [0.5, 0.5, 0, 0]},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.array([0.5, 0.5, 0, 0], dtype=np.float32) / np.linalg.norm([0.5, 0.5, 0, 0])

    # Without freshness: same similarity -> same order (arbitrary)
    results_no_fresh = engine.search("test", freshness_weight=0.0)

    # With freshness: recent should rank higher
    results_fresh = engine.search("test", freshness_weight=0.8)
    assert results_fresh[0]['id'] == 'new'


def test_freshness_no_date_gets_half(tmp_path):
    """Entry with no date should get freshness=0.5 (middle value)."""
    today = datetime.now().strftime('%Y-%m-%d')
    path = _make_index([
        {'id': 'dated', 'text': 'has date', 'date': today, 'embedding': [0.5, 0.5, 0, 0]},
        {'id': 'undated', 'text': 'no date', 'embedding': [0.5, 0.5, 0, 0]},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.array([0.5, 0.5, 0, 0], dtype=np.float32) / np.linalg.norm([0.5, 0.5, 0, 0])

    # With strong freshness, dated (today) should beat undated (0.5)
    results = engine.search("test", freshness_weight=0.9)
    assert results[0]['id'] == 'dated'


def test_sort_by_date(tmp_path):
    path = _make_index([
        {'id': 'old', 'text': 'old entry', 'date': '2020-01-01', 'embedding': [1, 0, 0, 0]},
        {'id': 'mid', 'text': 'mid entry', 'date': '2023-06-15', 'embedding': [0.5, 0.5, 0, 0]},
        {'id': 'new', 'text': 'new entry', 'date': '2025-12-01', 'embedding': [0, 0, 0, 1]},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.array([1, 0, 0, 0], dtype=np.float32)

    results = engine.search("test", sort_by='date')
    assert results[0]['id'] == 'new'
    assert results[1]['id'] == 'mid'
    assert results[2]['id'] == 'old'


def test_add_index_merges(tmp_path):
    path1 = _make_index([
        {'id': 'a', 'text': 'first index', 'source': 'src1'},
    ], tmp_path / 'idx1')
    path2 = _make_index([
        {'id': 'b', 'text': 'second index', 'source': 'src2'},
    ], tmp_path / 'idx2')

    engine = SearchEngine(path1)
    assert len(engine.entries) == 1
    engine.add_index(path2)
    assert len(engine.entries) == 2
    assert engine.embeddings.shape[0] == 2


def test_add_index_searchable(tmp_path):
    """After add_index, new entries should be searchable."""
    path1 = _make_index([
        {'id': 'a', 'text': 'first index', 'source': 'src1', 'embedding': [0, 0, 0, 1]},
    ], tmp_path / 'idx1')
    path2 = _make_index([
        {'id': 'b', 'text': 'second index', 'source': 'src2', 'embedding': [1, 0, 0, 0]},
    ], tmp_path / 'idx2')

    engine = SearchEngine(path1)
    engine.add_index(path2)
    engine._encode_query = lambda q: np.array([1, 0, 0, 0], dtype=np.float32)

    results = engine.search("test")
    assert results[0]['id'] == 'b'  # closer in dense space


def test_add_index_invalidates_fts(tmp_path):
    """After add_index, FTS index should include new entries."""
    path1 = _make_index([
        {'id': 'a', 'text': 'apple banana cherry', 'source': 'src1'},
    ], tmp_path / 'idx1')
    path2 = _make_index([
        {'id': 'b', 'text': 'dragon elephant frog', 'source': 'src2'},
    ], tmp_path / 'idx2')

    engine = SearchEngine(path1)
    # Force FTS build
    engine._ensure_fts_index()
    engine.add_index(path2)
    engine._encode_query = lambda q: np.random.randn(4).astype(np.float32)

    results = engine.search("dragon elephant", hybrid=True)
    ids = {r['id'] for r in results}
    assert 'b' in ids


def test_search_result_format(tmp_path):
    path = _make_index([
        {'id': 'x', 'text': 'hello world', 'source': 'test', 'title': 'Test Entry', 'date': '2024-01-01'},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.random.randn(4).astype(np.float32)

    results = engine.search("hello")
    assert len(results) == 1
    r = results[0]
    assert 'id' in r
    assert 'source' in r
    assert 'title' in r
    assert 'date' in r
    assert 'text' in r
    assert 'similarity' in r
    assert 'metadata' in r


def test_text_truncated_in_output(tmp_path):
    """Output text should be truncated to 300 chars."""
    long_text = 'x' * 1000
    path = _make_index([
        {'id': 'a', 'text': long_text},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.random.randn(4).astype(np.float32)

    results = engine.search("test")
    assert len(results[0]['text']) == 300


def test_top_k_limits_results(tmp_path):
    entries = [{'id': f'e{i}', 'text': f'entry {i}'} for i in range(20)]
    path = _make_index(entries, tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.random.randn(4).astype(np.float32)

    results = engine.search("test", top_k=5)
    assert len(results) == 5


def test_empty_query_bm25(tmp_path):
    """Empty/whitespace query for BM25 should not crash."""
    path = _make_index([
        {'id': 'a', 'text': 'hello world'},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.random.randn(4).astype(np.float32)

    # Should not raise
    results = engine.search("   ", hybrid=True)
    assert isinstance(results, list)


def test_fts5_safe_query():
    """Test FTS5 query quoting for safety."""
    from emb.search import SearchEngine
    assert SearchEngine._fts5_safe_query('hello world') == '"hello" "world"'
    assert SearchEngine._fts5_safe_query('') == ''
    assert SearchEngine._fts5_safe_query('single') == '"single"'


def test_metadata_model_name(tmp_path):
    """Engine should pick up model name from index metadata."""
    path = _make_index([
        {'id': 'a', 'text': 'hello'},
    ], tmp_path, model='custom/model-name')
    engine = SearchEngine(path)
    assert engine._model_name == 'custom/model-name'


def test_rrf_fusion_combines_rankings(tmp_path):
    """RRF should combine dense and BM25 rankings."""
    # Entry 'a' has exact keyword match but bad embedding
    # Entry 'b' has good embedding but no keyword match
    # Entry 'c' has both keyword and decent embedding -> should win with RRF
    path = _make_index([
        {'id': 'keyword_only', 'text': 'machine learning algorithms',
         'embedding': [0, 0, 0, 1]},
        {'id': 'dense_only', 'text': 'unrelated text completely',
         'embedding': [1, 0, 0, 0]},
        {'id': 'both', 'text': 'machine learning is fascinating stuff',
         'embedding': [0.8, 0.2, 0, 0]},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.array([1, 0, 0, 0], dtype=np.float32)

    results = engine.search("machine learning", hybrid=True)
    # 'both' should be in top results since it has both signals
    ids = [r['id'] for r in results]
    # 'both' should rank at least top-2
    assert ids.index('both') <= 1


def test_source_half_lives(tmp_path):
    """Source-specific half-lives should be respected."""
    today = datetime.now().strftime('%Y-%m-%d')
    old_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    path = _make_index([
        {'id': 'git_new', 'text': 'git commit', 'source': 'git', 'date': today,
         'embedding': [0.5, 0.5, 0, 0]},
        {'id': 'git_old', 'text': 'git commit old', 'source': 'git', 'date': old_date,
         'embedding': [0.5, 0.5, 0, 0]},
        {'id': 'book_old', 'text': 'book note', 'source': 'books', 'date': old_date,
         'embedding': [0.5, 0.5, 0, 0]},
    ], tmp_path)
    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.array([0.5, 0.5, 0, 0], dtype=np.float32) / np.linalg.norm([0.5, 0.5, 0, 0])

    # Books: no decay (None half-life -> freshness=1.0 always)
    # Git: 7-day half-life (30 days old -> heavily decayed)
    results = engine.search(
        "test", freshness_weight=0.9,
        source_half_lives={'git': 7, 'books': None}
    )
    # book_old with freshness=1.0 should beat git_old with heavy decay
    ids = [r['id'] for r in results]
    assert ids.index('book_old') < ids.index('git_old')
