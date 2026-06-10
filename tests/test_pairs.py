import numpy as np
import pytest

from emb.pairs import find_pairs, iter_pairs


def _normed(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_finds_obvious_pair():
    # rows 0 and 1 nearly identical; row 2 orthogonal
    emb = _normed([[1.0, 0.01, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pairs = find_pairs(emb, threshold=0.9)
    assert (0, 1) in [(i, j) for i, j, _ in pairs]
    assert all(j > i for i, j, _ in pairs)  # upper triangle only


def test_empty_and_singleton():
    assert find_pairs(np.zeros((0, 4), dtype=np.float32)) == []
    assert find_pairs(_normed([[1.0, 0.0, 0.0]])) == []


def test_block_size_independence():
    # Result set must not depend on block_size (determinism trap).
    rng = np.random.default_rng(0)
    emb = _normed(rng.standard_normal((200, 16)))
    base = set((i, j) for i, j, _ in find_pairs(emb, threshold=0.3, block_size=1024))
    for bs in (1, 7, 13, 64, 199):
        got = set((i, j) for i, j, _ in find_pairs(emb, threshold=0.3, block_size=bs))
        assert got == base, f"block_size={bs} changed result set"


def test_max_threshold_band():
    rng = np.random.default_rng(1)
    emb = _normed(rng.standard_normal((100, 8)))
    band = find_pairs(emb, threshold=0.2, max_threshold=0.5)
    assert all(0.2 <= s <= 0.5 for _, _, s in band)


def test_topk_union_deterministic():
    rng = np.random.default_rng(2)
    emb = _normed(rng.standard_normal((80, 12)))
    a = find_pairs(emb, threshold=0.1, top_k=3, block_size=8)
    b = find_pairs(emb, threshold=0.1, top_k=3, block_size=80)
    # Pair set AND order are block-independent; similarity values may differ only by
    # BLAS accumulation-order FP noise (~1e-7), never enough to reorder real pairs.
    assert [(i, j) for i, j, _ in a] == [(i, j) for i, j, _ in b]
    assert np.allclose([s for *_, s in a], [s for *_, s in b], atol=1e-5)


def test_topk_subset_of_full():
    rng = np.random.default_rng(3)
    emb = _normed(rng.standard_normal((60, 10)))
    full = set((i, j) for i, j, _ in find_pairs(emb, threshold=0.2))
    capped = set((i, j) for i, j, _ in find_pairs(emb, threshold=0.2, top_k=2))
    assert capped <= full


def test_sorted_descending():
    rng = np.random.default_rng(4)
    emb = _normed(rng.standard_normal((50, 8)))
    pairs = find_pairs(emb, threshold=0.0)
    sims = [s for _, _, s in pairs]
    assert sims == sorted(sims, reverse=True)


def test_invalid_block_size_raises():
    emb = _normed([[1.0, 0.0], [0.9, 0.1]])
    with pytest.raises(ValueError):
        find_pairs(emb, block_size=0)


def test_nonpositive_top_k_returns_empty():
    emb = _normed([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    assert find_pairs(emb, threshold=0.0, top_k=0) == []


def test_iter_matches_find():
    rng = np.random.default_rng(5)
    emb = _normed(rng.standard_normal((40, 8)))
    streamed = set((i, j) for i, j, _ in iter_pairs(emb, threshold=0.3))
    materialized = set((i, j) for i, j, _ in find_pairs(emb, threshold=0.3))
    assert streamed == materialized
