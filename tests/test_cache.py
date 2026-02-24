from emb.cache import EmbeddingCache


def test_put_and_get():
    cache = EmbeddingCache(dim=4)
    cache.put("hash1", [1.0, 2.0, 3.0, 4.0])
    result = cache.get("hash1")
    assert result is not None
    assert len(result) == 4
    assert result[0] == 1.0
    assert cache.get("missing") is None


def test_len_and_contains():
    cache = EmbeddingCache(dim=4)
    assert len(cache) == 0
    assert "x" not in cache
    cache.put("x", [1, 2, 3, 4])
    assert len(cache) == 1
    assert "x" in cache


def test_duplicate_put_ignored():
    cache = EmbeddingCache(dim=2)
    cache.put("x", [1.0, 2.0])
    cache.put("x", [9.0, 9.0])
    assert cache.get("x") == [1.0, 2.0]
    assert len(cache) == 1


def test_save_and_load(tmp_path):
    cache = EmbeddingCache(dim=4)
    cache.put("a", [1.0, 0.0, 0.0, 0.0])
    cache.put("b", [0.0, 1.0, 0.0, 0.0])

    cache_dir = tmp_path / "cache"
    cache.save(cache_dir)

    loaded = EmbeddingCache.load(cache_dir, dim=4)
    assert len(loaded) == 2
    assert loaded.get("a") == [1.0, 0.0, 0.0, 0.0]
    assert loaded.get("b") == [0.0, 1.0, 0.0, 0.0]


def test_save_creates_two_files(tmp_path):
    cache = EmbeddingCache(dim=2)
    cache.put("x", [1.0, 2.0])
    cache_dir = tmp_path / "cache"
    cache.save(cache_dir)
    assert (cache_dir / "index.json").exists()
    assert (cache_dir / "vectors.npy").exists()


def test_load_nonexistent_returns_empty(tmp_path):
    cache = EmbeddingCache.load(tmp_path / "nonexistent", dim=4)
    assert len(cache) == 0


def test_roundtrip_many_entries(tmp_path):
    cache = EmbeddingCache(dim=8)
    for i in range(100):
        cache.put(f"hash_{i}", [float(i)] * 8)

    cache_dir = tmp_path / "big"
    cache.save(cache_dir)

    loaded = EmbeddingCache.load(cache_dir, dim=8)
    assert len(loaded) == 100
    assert loaded.get("hash_50") == [50.0] * 8
