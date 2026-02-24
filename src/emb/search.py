"""Search engine with dense + BM25 + RRF + reranking + freshness."""
import json
import math
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from dateutil import parser as date_parser
from emb.schema import Entry

RRF_K = 60  # Reciprocal Rank Fusion constant
DEFAULT_RERANKER = 'tomaarsen/Qwen3-Reranker-0.6B-seq-cls'


class SearchEngine:
    """Semantic search over an emb index."""

    def __init__(self, index_path: Path):
        index_path = Path(index_path)
        with open(index_path, 'r') as f:
            data = json.load(f)

        self.metadata = data.get('metadata', {})
        self.entries = [Entry.from_dict(e) for e in data['entries']]
        self.embeddings = np.array(
            [e.embedding for e in self.entries], dtype=np.float32
        )

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
        """Quote tokens for FTS5 MATCH safety."""
        tokens = query.split()
        if not tokens:
            return ''
        return ' '.join(f'"{t}"' for t in tokens if t.strip())

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

    def _rerank(self, query: str, candidates: List[dict], top_k: int) -> List[dict]:
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
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
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

    def _deduplicate_chunks(self, results: List[dict], top_k: int) -> List[dict]:
        """Group by parent_id, keep best score per parent."""
        deduped = []
        seen_parents = {}
        for r in results:
            meta = r['entry'].metadata
            parent_id = meta.get('parent_id')
            if parent_id is None:
                deduped.append(r)
            elif parent_id not in seen_parents:
                seen_parents[parent_id] = r
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
    ) -> List[dict]:
        """Search the index.

        Returns list of dicts with: id, source, title, date, text, similarity, metadata
        """
        query_emb = self._encode_query(query)
        sims = self.embeddings @ query_emb

        # Filter
        valid = set()
        for i, entry in enumerate(self.entries):
            if sources and entry.source not in sources:
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
                score = 1.0 / (RRF_K + dense_ranks[i])
                if i in bm25_ranks:
                    score += 1.0 / (RRF_K + bm25_ranks[i])
            else:
                score = sim

            if freshness_weight > 0:
                freshness = self._compute_freshness(
                    self._parsed_dates[i], entry.source, source_half_lives=source_half_lives
                )
                score = score * (1 - freshness_weight + freshness_weight * freshness)

            results.append({
                'idx': i, 'entry': entry, 'similarity': sim, 'score': score,
            })

        # Sort
        if sort_by == 'date':
            results.sort(key=lambda x: (self._parsed_dates[x['idx']] or datetime.min, x['score']), reverse=True)
        else:
            results.sort(key=lambda x: x['score'], reverse=True)

        # Chunk dedup + rerank pool
        pool_size = top_k * 3 if rerank else top_k
        results = self._deduplicate_chunks(results, pool_size)

        if rerank and results:
            results = self._rerank(query, results, top_k)

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
