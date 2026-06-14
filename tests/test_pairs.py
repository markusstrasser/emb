import importlib.util

import numpy as np
import pytest

from emb.pairs import find_pairs, iter_pairs, find_fuzzy_pairs

# fuzzy is an optional backend (emb[fuzzy]); skip only the fuzzy tests if absent.
needs_rapidfuzz = pytest.mark.skipif(
    importlib.util.find_spec("rapidfuzz") is None,
    reason="rapidfuzz not installed (optional dep: emb[fuzzy])",
)


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


# --- fuzzy (rapidfuzz token_set_ratio) pairs -------------------------------------

@needs_rapidfuzz
def test_fuzzy_finds_near_dup_not_unrelated():
    # 0 and 1 differ only by whitespace + punctuation (near-dup); 2 is unrelated.
    texts = ["hello world", "hello  world!", "goodbye moon"]
    pairs = find_fuzzy_pairs(texts, threshold=90.0)
    found = {(i, j) for i, j, _ in pairs}
    assert (0, 1) in found
    assert (0, 2) not in found and (1, 2) not in found
    assert all(j > i for i, j, _ in pairs)  # upper triangle only
    assert all(s >= 90.0 for *_, s in pairs)  # 0-100 scale


@needs_rapidfuzz
def test_fuzzy_token_reorder_is_a_match():
    # token_set_ratio is order-insensitive — reordered tokens should still pair.
    texts = ["quick brown fox", "fox quick brown", "lazy dog sleeps"]
    found = {(i, j) for i, j, _ in find_fuzzy_pairs(texts, threshold=90.0)}
    assert (0, 1) in found
    assert (0, 2) not in found


@needs_rapidfuzz
def test_fuzzy_empty_and_singleton():
    assert find_fuzzy_pairs([]) == []
    assert find_fuzzy_pairs(["only one"]) == []


@needs_rapidfuzz
def test_fuzzy_sorted_descending():
    texts = ["alpha beta", "alpha beta gamma", "alpha bета", "totally different string"]
    pairs = find_fuzzy_pairs(texts, threshold=0.0)
    sims = [s for _, _, s in pairs]
    assert sims == sorted(sims, reverse=True)


@needs_rapidfuzz
def test_fuzzy_max_threshold_band():
    texts = ["acme corp", "acme corporation", "acme corp", "wholly unrelated text here"]
    band = find_fuzzy_pairs(texts, threshold=50.0, max_threshold=99.0)
    # The two identical "acme corp" (score 100) are excluded by the upper band.
    assert all(50.0 <= s <= 99.0 for _, _, s in band)


@needs_rapidfuzz
def test_fuzzy_nonpositive_top_k_returns_empty():
    assert find_fuzzy_pairs(["a a", "a a a", "b b"], threshold=0.0, top_k=0) == []


@needs_rapidfuzz
def test_fuzzy_topk_subset_of_full():
    texts = ["red apple", "red apple pie", "green apple", "blue sky", "blue sky high"]
    full = {(i, j) for i, j, _ in find_fuzzy_pairs(texts, threshold=0.0)}
    capped = {(i, j) for i, j, _ in find_fuzzy_pairs(texts, threshold=0.0, top_k=1)}
    assert capped <= full
