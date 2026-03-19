"""Tests for split index format (JSONL + npy)."""
import json
import numpy as np
import pytest
from pathlib import Path
from emb.index import write_index, read_index, convert_json_to_split, convert_split_to_json, check_staleness
from emb.schema import Entry


def _make_entries(n=5, dim=4):
    """Helper: create entries + embeddings."""
    entries = []
    for i in range(n):
        entries.append(Entry(
            id=f'e{i}', text=f'text for entry {i}',
            source='src_a' if i % 2 == 0 else 'src_b',
            title=f'Title {i}', date=f'2025-0{(i % 9) + 1}-01',
            metadata={'idx': i},
        ))
    embeddings = np.random.randn(n, dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return entries, embeddings


def test_write_read_roundtrip(tmp_path):
    entries, embeddings = _make_entries(10)
    idx_dir = tmp_path / 'myindex'

    write_index(entries, embeddings, idx_dir, {'embedding_model': 'test'})

    # Check files exist
    assert (idx_dir / 'metadata.json').exists()
    assert (idx_dir / 'entries.jsonl').exists()
    assert (idx_dir / 'embeddings.npy').exists()

    # Read back
    loaded_entries, loaded_emb, meta = read_index(idx_dir)
    assert len(loaded_entries) == 10
    assert loaded_emb.shape == (10, 4)
    assert meta['embedding_model'] == 'test'

    # Verify content
    for orig, loaded in zip(entries, loaded_entries):
        assert orig.id == loaded.id
        assert orig.text == loaded.text
        assert orig.source == loaded.source

    # Verify embeddings are close (float32 roundtrip)
    np.testing.assert_allclose(loaded_emb, embeddings, atol=1e-6)


def test_source_filtered_loading(tmp_path):
    entries, embeddings = _make_entries(10)
    idx_dir = tmp_path / 'filtered'
    write_index(entries, embeddings, idx_dir)

    # Load only src_a
    loaded, loaded_emb, _ = read_index(idx_dir, sources={'src_a'})
    assert all(e.source == 'src_a' for e in loaded)
    assert len(loaded) == 5  # 0,2,4,6,8
    assert loaded_emb.shape[0] == 5


def test_mmap_mode(tmp_path):
    entries, embeddings = _make_entries(5)
    idx_dir = tmp_path / 'mmap'
    write_index(entries, embeddings, idx_dir)

    # mmap=True (default)
    _, emb_mmap, _ = read_index(idx_dir, mmap=True)
    assert emb_mmap.shape == (5, 4)
    np.testing.assert_allclose(emb_mmap, embeddings, atol=1e-6)

    # mmap=False
    _, emb_copy, _ = read_index(idx_dir, mmap=False)
    np.testing.assert_allclose(emb_copy, embeddings, atol=1e-6)


def test_json_to_split_conversion(tmp_path):
    # Create a legacy JSON index
    entries_data = [
        {'id': 'a', 'text': 'hello', 'source': 'test', 'embedding': [1, 0, 0, 0]},
        {'id': 'b', 'text': 'world', 'source': 'test', 'embedding': [0, 1, 0, 0]},
    ]
    json_path = tmp_path / 'legacy.json'
    with open(json_path, 'w') as f:
        json.dump({
            'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
            'entries': entries_data,
        }, f)

    split_dir = tmp_path / 'split'
    convert_json_to_split(json_path, split_dir)

    loaded, emb, meta = read_index(split_dir)
    assert len(loaded) == 2
    assert loaded[0].id == 'a'
    assert emb.shape == (2, 4)


def test_split_to_json_conversion(tmp_path):
    entries, embeddings = _make_entries(3)
    split_dir = tmp_path / 'split'
    write_index(entries, embeddings, split_dir, {'embedding_model': 'test'})

    json_path = tmp_path / 'output.json'
    convert_split_to_json(split_dir, json_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    assert len(data['entries']) == 3
    assert all('embedding' in e for e in data['entries'])
    assert data['metadata']['embedding_model'] == 'test'


def test_json_split_json_roundtrip(tmp_path):
    """JSON → split → JSON should produce identical entries."""
    entries_data = [
        {'id': f'e{i}', 'text': f'text {i}', 'source': f's{i%2}',
         'title': f'T{i}', 'embedding': np.random.randn(4).tolist()}
        for i in range(5)
    ]
    json1 = tmp_path / 'orig.json'
    with open(json1, 'w') as f:
        json.dump({'metadata': {'embedding_dim': 4}, 'entries': entries_data}, f)

    split_dir = tmp_path / 'split'
    convert_json_to_split(json1, split_dir)

    json2 = tmp_path / 'roundtrip.json'
    convert_split_to_json(split_dir, json2)

    with open(json2, 'r') as f:
        data2 = json.load(f)

    for orig, rt in zip(entries_data, data2['entries']):
        assert orig['id'] == rt['id']
        assert orig['text'] == rt['text']
        np.testing.assert_allclose(orig['embedding'], rt['embedding'], atol=1e-6)


def test_search_engine_with_split_index(tmp_path):
    """SearchEngine should work with split format directory."""
    from emb.search import SearchEngine

    entries, embeddings = _make_entries(5)
    idx_dir = tmp_path / 'searchable'
    write_index(entries, embeddings, idx_dir, {'embedding_model': 'test'})

    engine = SearchEngine(idx_dir)
    assert len(engine.entries) == 5
    # Mock query encoding
    engine._encode_query = lambda q: np.random.randn(4).astype(np.float32)
    results = engine.search("test")
    assert len(results) > 0


def test_check_staleness(tmp_path):
    import time

    # Create source files
    source_dir = tmp_path / 'sources'
    source_dir.mkdir()
    (source_dir / 'data_parsed.json').write_text('{}')

    time.sleep(0.05)  # Ensure different mtime

    # Create index after source files
    idx_dir = tmp_path / 'idx'
    entries, embeddings = _make_entries(2)
    write_index(entries, embeddings, idx_dir)

    # Index is newer → no stale files
    stale = check_staleness(idx_dir, source_dir)
    assert len(stale) == 0

    time.sleep(0.05)
    # Touch a source file to make it newer
    (source_dir / 'new_parsed.json').write_text('{}')

    stale = check_staleness(idx_dir, source_dir)
    assert 'new_parsed.json' in stale


def test_metadata_includes_sources(tmp_path):
    entries, embeddings = _make_entries(10)
    idx_dir = tmp_path / 'meta'
    write_index(entries, embeddings, idx_dir)

    with open(idx_dir / 'metadata.json') as f:
        meta = json.load(f)

    assert 'sources' in meta
    assert meta['sources']['src_a'] == 5
    assert meta['sources']['src_b'] == 5
    assert meta['total_entries'] == 10
