"""Split index format: entries.jsonl + embeddings.npy + metadata.json.

Replaces monolithic JSON with a split directory format:
  myindex/
    metadata.json      # embedding_model, dim, total_entries, sources, generated_at
    entries.jsonl       # one JSON per line: {id, text, source, title, date, metadata, content_hash}
    embeddings.npy      # float32 (N × dim), mmap-able

Benefits: ~75% size reduction, mmap for zero-copy embeddings, source-filtered loading,
streaming-friendly JSONL.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from emb.schema import Entry


def write_index(entries: List[Entry], embeddings: np.ndarray,
                path: Path, metadata: Optional[Dict] = None):
    """Write split index directory.

    Args:
        entries: List of Entry objects (embedding field ignored, uses embeddings array).
        embeddings: float32 ndarray of shape (N, dim).
        path: Directory to write to (created if needed).
        metadata: Optional metadata dict (embedding_model, etc.).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Compute source counts
    source_counts: Dict[str, int] = {}
    for e in entries:
        s = e.source or 'unknown'
        source_counts[s] = source_counts.get(s, 0) + 1

    # Metadata
    meta = {
        'total_entries': len(entries),
        'embedding_dim': int(embeddings.shape[1]) if len(embeddings) > 0 else 0,
        'sources': source_counts,
        'generated_at': datetime.now().isoformat(),
    }
    if metadata:
        meta.update(metadata)

    with open(path / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Entries as JSONL (no embedding field)
    with open(path / 'entries.jsonl', 'w') as f:
        for entry in entries:
            d = entry.to_dict()
            d.pop('embedding', None)
            f.write(json.dumps(d, ensure_ascii=False))
            f.write('\n')

    # Embeddings as numpy
    np.save(path / 'embeddings.npy', embeddings.astype(np.float32))


def read_index(path: Path, sources: Optional[Set[str]] = None,
               mmap: bool = True) -> Tuple[List[Entry], np.ndarray, Dict]:
    """Read split index. Optional source filtering + mmap for zero-copy embeddings.

    Args:
        path: Directory containing metadata.json, entries.jsonl, embeddings.npy.
        sources: If provided, only load entries from these sources.
        mmap: If True, memory-map the embeddings (zero-copy, OS pages on demand).

    Returns:
        (entries, embeddings, metadata) tuple.
    """
    path = Path(path)

    with open(path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Load embeddings
    mmap_mode = 'r' if mmap else None
    all_embeddings = np.load(path / 'embeddings.npy', mmap_mode=mmap_mode)

    if sources is None:
        # Fast path: load all entries
        entries = []
        with open(path / 'entries.jsonl', 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(Entry.from_dict(json.loads(line)))
        return entries, np.asarray(all_embeddings, dtype=np.float32), metadata

    # Filtered path: only load matching entries
    entries = []
    indices = []
    with open(path / 'entries.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get('source') in sources:
                entries.append(Entry.from_dict(d))
                indices.append(i)

    if indices:
        embeddings = np.array(all_embeddings[indices], dtype=np.float32)
    else:
        dim = metadata.get('embedding_dim', 0)
        embeddings = np.empty((0, dim), dtype=np.float32)

    return entries, embeddings, metadata


def convert_json_to_split(json_path: Path, output_dir: Path):
    """One-shot migration from monolithic JSON index to split format."""
    json_path = Path(json_path)
    output_dir = Path(output_dir)

    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    raw_entries = data.get('entries', [])

    entries = []
    embedding_list = []
    for e in raw_entries:
        emb = e.pop('embedding', None)
        entry = Entry.from_dict(e)
        entries.append(entry)
        if emb is not None:
            embedding_list.append(emb)
        else:
            dim = metadata.get('embedding_dim', 768)
            embedding_list.append([0.0] * dim)

    embeddings = np.array(embedding_list, dtype=np.float32)
    write_index(entries, embeddings, output_dir, metadata)


def convert_split_to_json(split_dir: Path, json_path: Path):
    """Backwards compat conversion from split format to monolithic JSON."""
    entries, embeddings, metadata = read_index(split_dir, mmap=False)

    output_entries = []
    for i, entry in enumerate(entries):
        d = entry.to_dict()
        d['embedding'] = embeddings[i].tolist()
        output_entries.append(d)

    with open(json_path, 'w') as f:
        json.dump({'metadata': metadata, 'entries': output_entries}, f)


def check_staleness(index_path: Path, source_dir: Path,
                    pattern: str = "*_parsed.json") -> List[str]:
    """Return list of source files newer than the index.

    Args:
        index_path: Path to index directory or JSON file.
        source_dir: Directory containing source data files.
        pattern: Glob pattern for source files.

    Returns:
        List of filenames that are newer than the index.
    """
    index_path = Path(index_path)

    if index_path.is_dir():
        # Split format — use metadata.json mtime
        meta_path = index_path / 'metadata.json'
        if not meta_path.exists():
            return []
        index_mtime = os.path.getmtime(meta_path)
    else:
        if not index_path.exists():
            return []
        index_mtime = os.path.getmtime(index_path)

    newer = []
    for p in Path(source_dir).glob(pattern):
        if os.path.getmtime(p) > index_mtime:
            newer.append(p.name)
    return newer
