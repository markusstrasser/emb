"""All-pairs similarity over an embedding matrix — for duplicate/interference detection.

Unlike SearchEngine (top-k against a query), this finds high-similarity PAIRS within a
single corpus. Used by anki (duplicate detection ≥0.85, interference 0.50–0.80).

Design:
  - Row-block streaming over the UPPER TRIANGLE only — never materialize the full N×N
    matrix (72K² × 4B ≈ 20 GB; even 1,649² is wasteful to hold when only pairs above a
    threshold matter).
  - Determinism independent of block size: similarity is symmetric and each unordered
    pair (i, j) with i<j is emitted exactly once, so block boundaries never change the
    result set. An optional per-item top-K cap is applied as a DIRECTED count over the
    full row (computed block-by-block but reduced deterministically), then the surviving
    directed pairs are symmetrized into an unordered union.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np


def find_pairs(
    embeddings: np.ndarray,
    threshold: float = 0.85,
    max_threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    block_size: int = 1024,
) -> List[Tuple[int, int, float]]:
    """Find all (i, j, similarity) pairs with i < j above `threshold`.

    Args:
        embeddings: (N, dim) float32, assumed L2-normalized (cosine == dot product).
        threshold: minimum similarity to emit (inclusive).
        max_threshold: if set, also require similarity <= max_threshold (e.g. for
            interference detection in a band like 0.50–0.80, excluding near-duplicates).
        top_k: if set, keep only each item's top-K most-similar partners (DIRECTED,
            stable tie-break by partner index), then emit the symmetrized union. Caps
            output explosion on dense corpora without depending on block boundaries.
        block_size: rows per block; tunes memory, never the result.

    Returns pairs sorted by similarity descending, then (i, j) ascending for stable
    ordering. For streaming very large corpora use `iter_pairs` instead.
    """
    n = embeddings.shape[0]
    if n < 2:
        return []
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if top_k is not None and top_k <= 0:
        return []  # asking for the top-0 partners of each item → no pairs
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)

    if top_k is None:
        pairs = list(_iter_upper_triangle(emb, threshold, max_threshold, block_size))
    else:
        pairs = _topk_union(emb, threshold, max_threshold, top_k, block_size)

    pairs.sort(key=lambda p: (-p[2], p[0], p[1]))
    return pairs


def iter_pairs(
    embeddings: np.ndarray,
    threshold: float = 0.85,
    max_threshold: Optional[float] = None,
    block_size: int = 1024,
) -> Iterator[Tuple[int, int, float]]:
    """Stream (i, j, sim) pairs (i < j) above threshold without sorting or buffering
    the full list. Order is block-then-column; use find_pairs for sorted output."""
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    yield from _iter_upper_triangle(emb, threshold, max_threshold, block_size)


def _iter_upper_triangle(
    emb: np.ndarray, threshold: float, max_threshold: Optional[float], block_size: int
) -> Iterator[Tuple[int, int, float]]:
    n = emb.shape[0]
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        # Block rows [start:end] against all columns >= start (upper triangle).
        sims = emb[start:end] @ emb[start:].T  # (block, n-start)
        for local_i in range(end - start):
            i = start + local_i
            row = sims[local_i]
            # Only columns strictly right of i (global), i.e. local col > (i - start).
            lo = i - start + 1
            mask = row[lo:] >= threshold
            if max_threshold is not None:
                mask &= row[lo:] <= max_threshold
            cols = np.nonzero(mask)[0]
            for c in cols:
                j = start + lo + int(c)
                yield (i, j, float(row[lo + int(c)]))


def _topk_union(
    emb: np.ndarray, threshold: float, max_threshold: Optional[float],
    top_k: int, block_size: int,
) -> List[Tuple[int, int, float]]:
    """Directed top-K per item over the FULL similarity row, then symmetrized union.

    Computed block-by-block for memory, but each item's neighbor set is reduced
    deterministically (stable sort by -sim, then partner index), so the result does
    not depend on block_size.
    """
    n = emb.shape[0]
    directed: dict = {}  # i -> list of (sim, j)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sims = emb[start:end] @ emb.T  # (block, n) — full rows for directed top-K
        for local_i in range(end - start):
            i = start + local_i
            row = sims[local_i].copy()
            row[i] = -np.inf  # exclude self
            mask = row >= threshold
            if max_threshold is not None:
                mask &= row <= max_threshold
            cand = np.nonzero(mask)[0]
            if cand.size == 0:
                continue
            # Stable: sort by (-sim, j). argsort is stable on ties; sort j ascending
            # first, then by -sim, so equal sims keep ascending-j order.
            cand = cand[np.argsort(cand, kind='stable')]
            order = np.argsort(-row[cand], kind='stable')
            keep = cand[order][:top_k]
            directed[i] = [(float(row[j]), int(j)) for j in keep]

    seen = set()
    out: List[Tuple[int, int, float]] = []
    for i, neighbors in directed.items():
        for sim, j in neighbors:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b, sim))
    return out


def write_pairs_jsonl(
    pairs: List[Tuple[int, int, float]],
    out_path: Path,
    ids: Optional[List] = None,
    dropped: int = 0,
) -> int:
    """Write pairs as JSONL. If `ids` given, emit id_1/id_2 alongside index_1/index_2.
    Reports `dropped` (e.g. by a top-K cap) so truncation is never silent."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for i, j, sim in pairs:
            rec = {'index_1': i, 'index_2': j, 'similarity': round(sim, 6)}
            if ids is not None:
                rec['id_1'] = ids[i]
                rec['id_2'] = ids[j]
            f.write(json.dumps(rec, default=str) + '\n')
    if dropped:
        import sys
        print(f"  Note: {dropped} pair(s) dropped by top-K cap (not written)", file=sys.stderr)
    return len(pairs)
