"""Search engine with dense + BM25 + RRF + reranking + freshness."""
import json
import math
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Callable
from datetime import datetime
from dateutil import parser as date_parser
from emb.schema import Entry

RRF_K = 60  # Reciprocal Rank Fusion constant
DEFAULT_RERANKER = 'tomaarsen/Qwen3-Reranker-0.6B-seq-cls'


def expand_sources(sources: Set[str], groups: Dict[str, Set[str]]) -> Set[str]:
    """Expand source group aliases. e.g. {'health'} + groups → {'research','docs','healthkit'}"""
    expanded = set()
    for s in sources:
        if s in groups:
            expanded.update(groups[s])
        else:
            expanded.add(s)
    return expanded


class NeighborIndex:
    """Generic spreading activation for structural neighbors.

    The mechanism is generic — callers provide a key_extractor function that maps
    entries to relation keys (e.g. 'chatgpt_conv:abc123', 'git_repo:myrepo').
    """

    def __init__(self, entries: List[Entry], key_extractor: Callable):
        """Build index. key_extractor(entry) -> List[str] of relation keys."""
        self._index: Dict[str, List[int]] = {}
        for i, entry in enumerate(entries):
            for key in key_extractor(entry):
                self._index.setdefault(key, []).append(i)
        self._key_extractor = key_extractor

    def expand(self, anchor_indices: List[int], entries: List[Entry],
               valid_indices: Set[int], max_neighbors: int = 30) -> Set[int]:
        """Find structural neighbors of anchors within valid_indices."""
        neighbor_set: Set[int] = set()
        anchor_set = set(anchor_indices)

        for idx in anchor_indices:
            entry = entries[idx]
            for key in self._key_extractor(entry):
                for n_idx in self._index.get(key, []):
                    if n_idx not in anchor_set and n_idx in valid_indices:
                        neighbor_set.add(n_idx)
            if len(neighbor_set) >= max_neighbors:
                break

        return neighbor_set

    def as_post_processor(self, entries: List[Entry], boost_weight: float = 0.3):
        """Return a post_processor callable for SearchEngine.search()."""
        def _spread(query: str, results: List[dict], valid_indices: Set[int]) -> List[dict]:
            anchor_count = min(10, len(results))
            anchor_indices = [r['idx'] for r in results[:anchor_count]]
            neighbor_indices = self.expand(anchor_indices, entries, valid_indices)

            if not neighbor_indices:
                return results

            anchor_avg_score = sum(
                r['score'] for r in results[:anchor_count]
            ) / max(anchor_count, 1)
            anchor_set = set(anchor_indices)

            for r in results:
                if r['idx'] in neighbor_indices and r['idx'] not in anchor_set:
                    old_score = r['score']
                    r['score'] = old_score * (1 - boost_weight) + anchor_avg_score * boost_weight
                    r['_expanded'] = True

            results.sort(key=lambda x: x['score'], reverse=True)
            return results

        return _spread


class SearchEngine:
    """Semantic search over an emb index."""

    def __init__(self, index_path: Path, source_half_lives: Optional[Dict[str, Optional[float]]] = None):
        index_path = Path(index_path)
        with open(index_path, 'r') as f:
            data = json.load(f)

        entries = [Entry.from_dict(e) for e in data['entries']]
        embeddings = np.array(
            [e.embedding for e in entries], dtype=np.float32
        )
        metadata = data.get('metadata', {})
        self._init_common(entries, embeddings, metadata, source_half_lives)

    @classmethod
    def from_data(cls, entries: List[Entry], embeddings: np.ndarray,
                  metadata: Optional[Dict] = None,
                  source_half_lives: Optional[Dict[str, Optional[float]]] = None) -> 'SearchEngine':
        """Create from pre-loaded data. entries: List[Entry], embeddings: np.ndarray."""
        engine = object.__new__(cls)
        engine._init_common(entries, embeddings, metadata or {}, source_half_lives)
        return engine

    def _init_common(self, entries: List[Entry], embeddings: np.ndarray,
                     metadata: Dict, source_half_lives: Optional[Dict[str, Optional[float]]] = None):
        self.entries = entries
        self.embeddings = embeddings
        self.metadata = metadata
        self._source_half_lives = source_half_lives

        # Pre-parse dates
        self._parsed_dates = []
        for entry in self.entries:
            parsed = None
            if entry.date:
                try:
                    parsed = date_parser.parse(str(entry.date), fuzzy=True)
                    if parsed.tzinfo is not None:
                        parsed = parsed.replace(tzinfo=None)
                except Exception:
                    pass
            self._parsed_dates.append(parsed)

        # Model config from index metadata
        self._model_name = self.metadata.get('embedding_model', 'Alibaba-NLP/gte-modernbert-base')
        self._embedding_model = None  # lazy
        self._fts_db = None  # lazy
        self._reranker = None  # lazy

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query string to embedding vector."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                self._model_name, trust_remote_code=True
            )
        emb = self._embedding_model.encode(
            [query], normalize_embeddings=True
        )[0]
        return np.array(emb, dtype=np.float32)

    def _ensure_fts_index(self):
        """Build FTS5 index lazily."""
        if self._fts_db is not None:
            return
        self._fts_db = sqlite3.connect(':memory:')
        cur = self._fts_db.cursor()
        cur.execute('''
            CREATE VIRTUAL TABLE fts_entries USING fts5(
                title, text, source,
                content='',
                tokenize='porter unicode61'
            )
        ''')
        for i, entry in enumerate(self.entries):
            cur.execute(
                'INSERT INTO fts_entries(rowid, title, text, source) VALUES (?, ?, ?, ?)',
                (i, entry.title or '', entry.text or '', entry.source or '')
            )
        self._fts_db.commit()

    @staticmethod
    def _fts5_safe_query(query: str) -> str:
        """Make a query safe for FTS5 MATCH, preserving AND/OR/NOT operators."""
        tokens = query.split()
        if not tokens:
            return ''
        FTS5_OPERATORS = {'AND', 'OR', 'NOT'}
        FTS5_SPECIAL = set('"*^:{}()[]')
        safe_tokens = []
        for t in tokens:
            if not t.strip():
                continue
            if t in FTS5_OPERATORS:
                safe_tokens.append(t)
            else:
                cleaned = ''.join(c for c in t if c not in FTS5_SPECIAL)
                if cleaned:
                    safe_tokens.append(f'"{cleaned}"')
        return ' '.join(safe_tokens)

    def _bm25_search(self, query: str, limit: int = 200) -> List[Tuple[int, float]]:
        """BM25 search via FTS5. Returns (index, score) tuples."""
        self._ensure_fts_index()
        safe = self._fts5_safe_query(query)
        if not safe:
            return []
        cur = self._fts_db.cursor()
        try:
            rows = cur.execute('''
                SELECT rowid, bm25(fts_entries, 10.0, 1.0, 0.0) as score
                FROM fts_entries WHERE fts_entries MATCH ?
                ORDER BY score LIMIT ?
            ''', (safe, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(r[0]), -r[1]) for r in rows]

    def _get_reranker(self, model: str = DEFAULT_RERANKER):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(model)
        return self._reranker

    def _rerank(self, query: str, candidates: List[dict], top_k: int,
                provenance: bool = False) -> List[dict]:
        """Rerank candidates with cross-encoder."""
        reranker = self._get_reranker()
        instruction = 'Given a web search query, retrieve relevant passages that answer the query.'
        prefixed = f'Instruct: {instruction}\nQuery: {query}'
        pairs = []
        for c in candidates:
            e = c['entry']
            doc = f"{e.title or ''} {e.text[:500]}"
            pairs.append((prefixed, doc))
        scores = reranker.predict(pairs)
        for i, c in enumerate(candidates):
            c['rerank_score'] = float(scores[i])
            if provenance and 'provenance' in c:
                c['provenance']['pre_rerank_rank'] = i
                c['provenance']['rerank_score'] = round(float(scores[i]), 4)
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        if provenance:
            for rank, c in enumerate(candidates[:top_k]):
                if 'provenance' in c:
                    pre = c['provenance'].get('pre_rerank_rank', rank)
                    c['provenance']['post_rerank_rank'] = rank
                    c['provenance']['rerank_delta'] = pre - rank
        return candidates[:top_k]

    def _compute_freshness(
        self, entry_date: Optional[datetime], source: str = None,
        half_life_days: float = 365, source_half_lives: Dict[str, Optional[float]] = None,
    ) -> float:
        """Exponential decay freshness weight."""
        if source_half_lives and source in source_half_lives:
            hl = source_half_lives[source]
            if hl is None:
                return 1.0
            half_life_days = hl
        if not entry_date:
            return 0.5
        age = (datetime.now() - entry_date).days
        if age <= 0:
            return 1.0
        return math.pow(0.5, age / half_life_days)

    def _deduplicate(self, results: List[dict], top_k: int,
                     dedup_key: Optional[Callable] = None) -> List[dict]:
        """Deduplicate results. Default: by parent_id. Custom: by dedup_key callable."""
        if dedup_key is None:
            dedup_key = lambda r: r['entry'].metadata.get('parent_id')

        deduped = []
        seen_keys = {}
        for r in results:
            key = dedup_key(r)
            if key is None:
                deduped.append(r)
            elif key not in seen_keys:
                seen_keys[key] = r
                deduped.append(r)
        return deduped[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 20,
        sources: Optional[Set[str]] = None,
        min_similarity: float = 0.0,
        since: Optional[datetime] = None,
        sort_by: str = 'relevance',
        freshness_weight: float = 0.0,
        source_half_lives: Optional[Dict[str, Optional[float]]] = None,
        hybrid: bool = False,
        rerank: bool = False,
        # Extension points
        entry_filter: Optional[Callable] = None,
        dedup_key: Optional[Callable] = None,
        post_processors: Optional[List[Callable]] = None,
        provenance: bool = False,
    ) -> List[dict]:
        """Search the index.

        Args:
            entry_filter: Optional callable(entry) -> bool for custom filtering.
            dedup_key: Optional callable(result_dict) -> Optional[str] for dedup grouping.
            post_processors: List of callable(query, results, valid_indices) -> results.
            provenance: If True, attach per-result provenance dict with scoring details.

        Returns list of dicts with: id, source, title, date, text, similarity, metadata
        """
        # Merge constructor-level and call-level half-lives
        effective_half_lives = self._source_half_lives
        if source_half_lives is not None:
            if effective_half_lives:
                effective_half_lives = {**effective_half_lives, **source_half_lives}
            else:
                effective_half_lives = source_half_lives

        query_emb = self._encode_query(query)
        sims = self.embeddings @ query_emb

        # Filter
        valid = set()
        for i, entry in enumerate(self.entries):
            if sources and entry.source not in sources:
                continue
            if entry_filter and not entry_filter(entry):
                continue
            if min_similarity > 0 and float(sims[i]) < min_similarity:
                continue
            if since and (not self._parsed_dates[i] or self._parsed_dates[i] < since):
                continue
            valid.add(i)

        # BM25
        bm25_ranks = {}
        if hybrid:
            bm25_results = self._bm25_search(query, limit=200)
            rank = 0
            for idx, _ in bm25_results:
                if idx in valid:
                    bm25_ranks[idx] = rank
                    rank += 1

        # Dense ranking
        dense_ranked = sorted(valid, key=lambda i: -sims[i])
        dense_ranks = {idx: rank for rank, idx in enumerate(dense_ranked)}

        # Score
        results = []
        for i in valid:
            entry = self.entries[i]
            sim = float(sims[i])

            if hybrid:
                base_score = 1.0 / (RRF_K + dense_ranks[i])
                if i in bm25_ranks:
                    base_score += 1.0 / (RRF_K + bm25_ranks[i])
            else:
                base_score = sim

            freshness = None
            if freshness_weight > 0:
                freshness = self._compute_freshness(
                    self._parsed_dates[i], entry.source, source_half_lives=effective_half_lives
                )
                score = base_score * (1 - freshness_weight + freshness_weight * freshness)
            else:
                score = base_score

            r = {'idx': i, 'entry': entry, 'similarity': sim, 'score': score}

            if provenance:
                prov = {
                    'dense_sim': round(sim, 4),
                    'dense_rank': dense_ranks[i],
                }
                if hybrid:
                    prov['bm25_rank'] = bm25_ranks.get(i)
                    prov['rrf_dense'] = round(1.0 / (RRF_K + dense_ranks[i]), 6)
                    prov['rrf_bm25'] = round(
                        1.0 / (RRF_K + bm25_ranks[i]) if i in bm25_ranks else 0.0, 6
                    )
                    prov['rrf_total'] = round(base_score, 6)
                if freshness is not None:
                    prov['freshness_factor'] = round(freshness, 4)
                r['provenance'] = prov

            results.append(r)

        # Sort
        if sort_by == 'date':
            results.sort(key=lambda x: (self._parsed_dates[x['idx']] or datetime.min, x['score']), reverse=True)
        else:
            results.sort(key=lambda x: x['score'], reverse=True)

        # Post-processors (e.g. spreading activation)
        if post_processors:
            for proc in post_processors:
                results = proc(query, results, valid)

        # Dedup + rerank pool
        pool_size = top_k * 3 if rerank else top_k
        results = self._deduplicate(results, pool_size, dedup_key=dedup_key)

        if rerank and results:
            results = self._rerank(query, results, top_k, provenance=provenance)

        results = results[:top_k]

        # Format output
        formatted = []
        for r in results:
            e = r['entry']
            out = {
                'id': e.id, 'source': e.source, 'title': e.title,
                'date': e.date, 'text': e.text[:300],
                'similarity': r['similarity'], 'metadata': e.metadata,
            }
            if provenance and 'provenance' in r:
                out['provenance'] = r['provenance']
            if 'rerank_score' in r:
                out['rerank_score'] = r['rerank_score']
            formatted.append(out)
        return formatted

    def add_index(self, index_path: Path):
        """Merge another index into this engine (for cross-corpus search)."""
        index_path = Path(index_path)
        with open(index_path, 'r') as f:
            data = json.load(f)
        new_entries = [Entry.from_dict(e) for e in data['entries']]
        new_embeddings = np.array([e.embedding for e in new_entries], dtype=np.float32)

        self.entries.extend(new_entries)
        self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Re-parse dates for new entries
        for entry in new_entries:
            parsed = None
            if entry.date:
                try:
                    parsed = date_parser.parse(str(entry.date), fuzzy=True)
                    if parsed.tzinfo:
                        parsed = parsed.replace(tzinfo=None)
                except Exception:
                    pass
            self._parsed_dates.append(parsed)

        # Invalidate FTS cache
        self._fts_db = None
