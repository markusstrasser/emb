"""Embedding engine with sentence-transformers and Ollama backends."""
import hashlib
import json
import signal
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from emb.schema import Entry
from emb.cache import EmbeddingCache


# Known models: name -> (dim, backend, default_batch_size)
KNOWN_MODELS = {
    'Alibaba-NLP/gte-modernbert-base': (768, 'sentence-transformers', 64),
    'qwen3-embedding:8b-q8_0': (4096, 'ollama', 16),
}

DEFAULT_MODEL = 'Alibaba-NLP/gte-modernbert-base'


def _model_is_cached(model_name: str) -> bool:
    """Check if a sentence-transformers model is already downloaded."""
    from pathlib import Path
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    if not cache_dir.exists():
        return False
    # HF cache uses models--org--name format
    slug = 'models--' + model_name.replace('/', '--')
    return (cache_dir / slug).exists()


def _confirm_model_download(model_name: str) -> bool:
    """Ask user to confirm model download. Returns True if confirmed or non-interactive."""
    import sys
    if not sys.stdin.isatty():
        return True  # non-interactive: proceed silently
    try:
        resp = input(f"Model '{model_name}' not found locally. Download it? [Y/n] ").strip().lower()
        return resp in ('', 'y', 'yes')
    except (EOFError, KeyboardInterrupt):
        return False


def compute_content_hash(text: str) -> str:
    """Compute MD5 hash of text for cache keys."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


class EmbeddingEngine:
    """Generate embeddings with caching and incremental updates."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dim: Optional[int] = None,
        batch_size: Optional[int] = None,
        backend: Optional[str] = None,
        ollama_url: str = 'http://localhost:11434/api/embed',
        max_chars: int = 16000,
    ):
        info = KNOWN_MODELS.get(model, (768, 'sentence-transformers', 64))
        self.model = model
        self.dim = dim or info[0]
        self.backend = backend or info[1]
        self.batch_size = batch_size or info[2]
        self.ollama_url = ollama_url
        self.max_chars = max_chars
        self._st_model = None  # lazy

    def _get_st_model(self):
        """Lazy-load sentence-transformers model, confirming download if needed."""
        if self._st_model is None:
            if not _model_is_cached(self.model):
                if not _confirm_model_download(self.model):
                    raise SystemExit("Model download declined.")
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'true'
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model, trust_remote_code=True)
            self._st_model.max_seq_length = 512
        return self._st_model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns normalized embeddings."""
        truncated = []
        n_truncated = 0
        for t in texts:
            if len(t) > self.max_chars:
                truncated.append(t[:self.max_chars])
                n_truncated += 1
            else:
                truncated.append(t)
        if n_truncated:
            import sys
            print(f"  Warning: {n_truncated} text(s) truncated to {self.max_chars} chars", file=sys.stderr)
        if self.backend == 'ollama':
            return self._embed_ollama(truncated)
        return self._embed_st(truncated)

    def _embed_st(self, texts: List[str]) -> List[List[float]]:
        model = self._get_st_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        from urllib import request as urllib_request
        data = json.dumps({'model': self.model, 'input': texts}).encode()
        req = urllib_request.Request(
            self.ollama_url, data=data,
            headers={'Content-Type': 'application/json'}
        )
        resp = urllib_request.urlopen(req, timeout=300)
        result = json.loads(resp.read())
        normed = []
        for emb in result['embeddings']:
            arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            normed.append(arr.tolist())
        return normed

    def _embed_adaptive(self, texts: List[str]) -> List[List[float]]:
        """Embed with adaptive batch halving on OOM."""
        size = self.batch_size
        while size > 1:
            try:
                results = []
                for i in range(0, len(texts), size):
                    results.extend(self.embed_texts(texts[i:i + size]))
                return results
            except Exception:
                size = max(1, size // 2)
        # Last resort: single-entry with zero fallback
        results = []
        for t in texts:
            try:
                results.append(self.embed_texts([t])[0])
            except Exception:
                results.append([0.0] * self.dim)
        return results

    def embed_entries(
        self,
        entries: List[Entry],
        cache: Optional[EmbeddingCache] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 256,
    ) -> List[Entry]:
        """Embed entries with caching and checkpointing.

        - Computes content_hash for each entry
        - Reuses cached embeddings where available
        - Embeds remaining in batches
        - Checkpoints cache periodically
        - Handles Ctrl+C (saves cache before exit)

        Returns entries with embedding field populated.
        """
        if cache is None:
            cache = EmbeddingCache(dim=self.dim)

        # Separate cached vs needs-embedding
        needs_embedding = []
        for entry in entries:
            entry.content_hash = compute_content_hash(entry.text)
            cached = cache.get(entry.content_hash)
            if cached is not None:
                entry.embedding = cached
            else:
                needs_embedding.append(entry)

        reused = len(entries) - len(needs_embedding)
        if reused > 0:
            print(f"  Reusing {reused} cached embeddings")

        if not needs_embedding:
            print("  All embeddings cached, nothing to generate")
            return entries

        total = len(needs_embedding)
        print(f"  Generating {total} new embeddings (batch_size={self.batch_size})...")

        start_time = time.time()
        last_checkpoint = 0
        interrupted = False

        def _handle_interrupt(signum, frame):
            nonlocal interrupted
            if interrupted:
                raise KeyboardInterrupt
            interrupted = True
            print(f"\n  Ctrl+C -- saving cache before exit...")

        old_handler = signal.signal(signal.SIGINT, _handle_interrupt)

        done = 0
        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch = needs_embedding[batch_start:batch_end]
            texts = [e.text for e in batch]

            try:
                embeddings = self.embed_texts(texts)
            except Exception:
                embeddings = self._embed_adaptive(texts)

            for i, entry in enumerate(batch):
                entry.embedding = embeddings[i]
                cache.put(entry.content_hash, embeddings[i])

            done = batch_end
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total}] {rate:.1f}/s, ETA {eta/60:.1f}m", flush=True)

            if checkpoint_dir and (done - last_checkpoint) >= checkpoint_interval:
                cache.save(checkpoint_dir)
                last_checkpoint = done

            if interrupted:
                break

        signal.signal(signal.SIGINT, old_handler)
        if checkpoint_dir:
            cache.save(checkpoint_dir)

        elapsed = time.time() - start_time
        if interrupted:
            print(f"\n  Interrupted after {done}/{total} ({elapsed/60:.1f}m)")
            print(f"  Cache saved: {len(cache)} vectors")
            raise SystemExit(0)

        print(f"  Done: {total} embeddings in {elapsed:.1f}s ({total/max(elapsed,0.01):.1f}/s)")
        return entries
