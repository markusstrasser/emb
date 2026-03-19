"""Numpy-backed embedding cache for incremental updates.

Two files on disk:
  cache_dir/index.json  -- {hash: row_index, ...} + metadata
  cache_dir/vectors.npy -- float32 array (N, dim)
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime


class EmbeddingCache:
    def __init__(self, dim: int):
        self.dim = dim
        self.hash_to_idx: Dict[str, int] = {}
        self._vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._overflow: List[np.ndarray] = []  # pending rows not yet in _vectors

    def __len__(self):
        return len(self.hash_to_idx)

    def __contains__(self, content_hash: str):
        return content_hash in self.hash_to_idx

    def _consolidate(self):
        """Merge overflow rows into main array."""
        if not self._overflow:
            return
        parts = [self._vectors] + self._overflow if len(self._vectors) > 0 else self._overflow
        self._vectors = np.concatenate(parts, dtype=np.float32)
        self._overflow.clear()

    def get(self, content_hash: str) -> Optional[List[float]]:
        idx = self.hash_to_idx.get(content_hash)
        if idx is None:
            return None
        total_main = len(self._vectors)
        if idx < total_main:
            return self._vectors[idx].tolist()
        # In overflow
        overflow_idx = idx - total_main
        offset = 0
        for arr in self._overflow:
            if overflow_idx < offset + len(arr):
                return arr[overflow_idx - offset].tolist()
            offset += len(arr)
        return None

    def put(self, content_hash: str, embedding):
        if content_hash in self.hash_to_idx:
            return
        idx = len(self._vectors) + sum(len(a) for a in self._overflow)
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        self._overflow.append(vec)
        self.hash_to_idx[content_hash] = idx

    def save(self, cache_dir: Path):
        """Write cache to disk."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._consolidate()
        arr = self._vectors if len(self._vectors) > 0 else np.empty((0, self.dim), dtype=np.float32)
        np.save(cache_dir / "vectors.npy", arr)

        with open(cache_dir / "index.json", "w") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "count": len(self.hash_to_idx),
                    "updated_at": datetime.now().isoformat(),
                    "hashes": self.hash_to_idx,
                },
                f,
            )

    @classmethod
    def load(cls, cache_dir: Path, dim: int) -> "EmbeddingCache":
        """Load cache from disk. Returns empty cache if files don't exist or dim mismatches."""
        cache = cls(dim=dim)
        cache_dir = Path(cache_dir)

        idx_path = cache_dir / "index.json"
        vec_path = cache_dir / "vectors.npy"

        if not idx_path.exists() or not vec_path.exists():
            return cache

        try:
            with open(idx_path, "r") as f:
                idx_data = json.load(f)

            cached_dim = idx_data.get("dim")
            if cached_dim is not None and cached_dim != dim:
                import sys
                print(
                    f"  Warning: cache dim={cached_dim} != model dim={dim}. "
                    f"Ignoring stale cache (re-embedding all).",
                    file=sys.stderr,
                )
                return cache

            vectors = np.load(vec_path)
            cache._vectors = vectors.astype(np.float32)
            cache.hash_to_idx = {
                h: int(i) for h, i in idx_data["hashes"].items()
            }
        except Exception:
            pass

        return cache
