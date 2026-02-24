"""Tests for multi-scale chunking with automatic RRF fusion."""
import json
import numpy as np
import pytest
from emb.schema import Entry
from emb.chunking import multiscale_chunk_entries


def test_multiscale_produces_entries_at_each_scale():
    entry = Entry(
        id="ep1",
        text=" ".join(f"word{i}." for i in range(1000)),
        source="podcast",
        title="Episode 1",
    )
    result = multiscale_chunk_entries([entry], scales=[200, 500])

    scales_found = set()
    for e in result:
        if "scale" in e.metadata:
            scales_found.add(e.metadata["scale"])
    assert scales_found == {200, 500}


def test_multiscale_small_chunks_have_correct_scale():
    entry = Entry(id="ep1", text=" ".join(f"w{i}." for i in range(600)))
    result = multiscale_chunk_entries([entry], scales=[200, 500])

    for e in result:
        if "scale" in e.metadata:
            scale = e.metadata["scale"]
            assert scale in (200, 500)


def test_multiscale_short_entry_appears_once():
    entry = Entry(id="short", text="Very short text.")
    result = multiscale_chunk_entries([entry], scales=[200, 500])
    assert len(result) == 1
    assert result[0].id == "short"


def test_multiscale_parent_id_set():
    entry = Entry(id="long", text=" ".join(f"w{i}." for i in range(1000)))
    result = multiscale_chunk_entries([entry], scales=[200, 500])
    for e in result:
        if "chunk_index" in e.metadata:
            assert e.metadata["parent_id"] == "long"


def test_multiscale_ids_unique():
    entry = Entry(id="ep1", text=" ".join(f"w{i}." for i in range(1000)))
    result = multiscale_chunk_entries([entry], scales=[200, 500])
    ids = [e.id for e in result]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {[x for x in ids if ids.count(x) > 1]}"


def test_multiscale_200_produces_more_chunks_than_500():
    entry = Entry(id="ep1", text=" ".join(f"w{i}." for i in range(1000)))
    result = multiscale_chunk_entries([entry], scales=[200, 500])

    chunks_200 = [e for e in result if e.metadata.get("scale") == 200]
    chunks_500 = [e for e in result if e.metadata.get("scale") == 500]
    assert len(chunks_200) > len(chunks_500)


def test_multiscale_preserves_source_and_date():
    """Source and date should be preserved on all chunks."""
    entry = Entry(
        id="ep1",
        text=" ".join(f"w{i}." for i in range(1000)),
        source="podcast",
        date="2025-01-15",
    )
    result = multiscale_chunk_entries([entry], scales=[200, 500])
    for e in result:
        assert e.source == "podcast"
        assert e.date == "2025-01-15"


def test_multiscale_single_scale_behaves_like_chunk_entries():
    """Single scale should produce same structure as chunk_entries."""
    from emb.chunking import chunk_entries

    entry = Entry(
        id="ep1",
        text=" ".join(f"w{i}." for i in range(1000)),
        source="podcast",
        title="Episode 1",
    )
    single_scale = multiscale_chunk_entries([entry], scales=[200])
    regular = chunk_entries([entry], chunk_tokens=200)

    # Same number of chunks (both use the same underlying chunk_text)
    assert len(single_scale) == len(regular)


def test_multiscale_multiple_entries():
    """Multiple entries should all be chunked at all scales."""
    entries = [
        Entry(id=f"ep{i}", text=" ".join(f"w{j}." for j in range(800)))
        for i in range(3)
    ]
    result = multiscale_chunk_entries(entries, scales=[200, 500])

    # Each entry should have chunks at both scales
    for ep_id in ["ep0", "ep1", "ep2"]:
        ep_chunks = [e for e in result if e.id.startswith(ep_id)]
        scales_for_ep = {e.metadata.get("scale") for e in ep_chunks}
        assert 200 in scales_for_ep
        assert 500 in scales_for_ep


def test_multiscale_fits_in_one_scale_gets_scale_tag():
    """Entry that fits in one chunk at a given scale should get scale tag but no parent_id."""
    # 300 words: fits in 500-token chunk, needs splitting at 200-token
    entry = Entry(id="mid", text=" ".join(f"w{i}." for i in range(300)))
    result = multiscale_chunk_entries([entry], scales=[200, 500])

    # At scale 500: fits in one chunk -> scale tag, id is mid__s500
    s500 = [e for e in result if e.metadata.get("scale") == 500]
    assert len(s500) == 1
    assert s500[0].id == "mid__s500"
    assert "parent_id" not in s500[0].metadata

    # At scale 200: needs splitting -> multiple chunks with parent_id
    s200 = [e for e in result if e.metadata.get("scale") == 200]
    assert len(s200) > 1
    for e in s200:
        assert e.metadata.get("parent_id") == "mid"


def test_multiscale_dedup_works_with_search(tmp_path):
    """Chunk dedup in search should handle multi-scale entries correctly.

    All chunks from the same parent (across scales) share parent_id,
    so dedup picks the best chunk regardless of scale.
    """
    from emb.search import SearchEngine

    # Simulate multi-scale chunks from same parent
    entries = [
        # Scale 200 chunks
        {'id': 'ep1__s200_chunk_0', 'text': 'first chunk small',
         'embedding': [1, 0, 0, 0],
         'metadata': {'parent_id': 'ep1', 'chunk_index': 0, 'scale': 200}},
        {'id': 'ep1__s200_chunk_1', 'text': 'second chunk small',
         'embedding': [0.5, 0.5, 0, 0],
         'metadata': {'parent_id': 'ep1', 'chunk_index': 1, 'scale': 200}},
        # Scale 500 chunk
        {'id': 'ep1__s500_chunk_0', 'text': 'single larger chunk',
         'embedding': [0.9, 0.1, 0, 0],
         'metadata': {'parent_id': 'ep1', 'chunk_index': 0, 'scale': 500}},
        # Different parent
        {'id': 'ep2__s200_chunk_0', 'text': 'other episode',
         'embedding': [0.3, 0.7, 0, 0],
         'metadata': {'parent_id': 'ep2', 'chunk_index': 0, 'scale': 200}},
    ]

    index = {
        'metadata': {'embedding_model': 'test', 'embedding_dim': 4},
        'entries': [],
    }
    for e in entries:
        emb = np.array(e['embedding'], dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        index['entries'].append({
            'id': e['id'], 'text': e['text'],
            'embedding': emb.tolist(), 'metadata': e['metadata'],
        })

    path = tmp_path / 'multiscale_index.json'
    with open(path, 'w') as f:
        json.dump(index, f)

    engine = SearchEngine(path)
    engine._encode_query = lambda q: np.array([1, 0, 0, 0], dtype=np.float32)

    results = engine.search("test", top_k=10)
    # ep1 should appear only once (deduped across scales)
    ep1_results = [r for r in results if r['metadata'].get('parent_id') == 'ep1']
    assert len(ep1_results) == 1
    # Should be the best-matching chunk (ep1__s200_chunk_0 with [1,0,0,0])
    assert ep1_results[0]['id'] == 'ep1__s200_chunk_0'


def test_multiscale_empty_entries():
    """Empty entry list should return empty."""
    result = multiscale_chunk_entries([], scales=[200, 500])
    assert result == []


def test_multiscale_custom_overlap():
    """Custom overlap should be respected."""
    entry = Entry(id="ep1", text=" ".join(f"w{i}." for i in range(1000)))
    result_default = multiscale_chunk_entries([entry], scales=[200])
    result_large = multiscale_chunk_entries([entry], scales=[200], overlap_tokens=100)
    # More overlap = more chunks (more repeated content)
    assert len(result_large) >= len(result_default)
