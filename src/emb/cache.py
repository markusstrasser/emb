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
        self.vectors: List[List[float]] = []

    def __len__(self):
        return len(self.hash_to_idx)

    def __contains__(self, content_hash: str):
        return content_hash in self.hash_to_idx

    def get(self, content_hash: str) -> Optional[List[float]]:
        idx = self.hash_to_idx.get(content_hash)
        if idx is None:
            return None
        v = self.vectors[idx]
        return v.tolist() if hasattr(v, "tolist") else list(v)

    def put(self, content_hash: str, embedding: List[float]):
        if content_hash in self.hash_to_idx:
            return
        idx = len(self.vectors)
        self.vectors.append(embedding)
        self.hash_to_idx[content_hash] = idx

    def save(self, cache_dir: Path):
        """Write cache to disk."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        arr = (
            np.array(self.vectors, dtype=np.float32)
            if self.vectors
            else np.empty((0, self.dim), dtype=np.float32)
        )
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
            cache.vectors = list(vectors)
            cache.hash_to_idx = {
                h: int(i) for h, i in idx_data["hashes"].items()
            }
        except Exception:
            pass

        return cache
