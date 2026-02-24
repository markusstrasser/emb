"""Text chunking with sentence-aware boundaries.

Splits long text at sentence boundaries (. ! ?) into chunks of ~chunk_tokens words.
Falls back to word-level splitting if no sentence boundaries found.
"""
import re
from typing import List
from emb.schema import Entry

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def chunk_text(
    text: str,
    chunk_tokens: int = 500,
    overlap_tokens: int = 50,
) -> List[str]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    words = text.split()
    if len(words) <= chunk_tokens:
        return [text]

    sentences = _SENTENCE_RE.split(text)
    if len(sentences) <= 1:
        return _chunk_by_words(words, chunk_tokens, overlap_tokens)

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_word_count + sent_words > chunk_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            # Overlap: keep trailing sentences within overlap budget
            overlap_sents = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_len = len(s.split())
                if overlap_count + s_len > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_count += s_len
            current_sentences = overlap_sents
            current_word_count = overlap_count

        current_sentences.append(sent)
        current_word_count += sent_words

    if current_sentences:
        final = " ".join(current_sentences)
        if chunks and len(final.split()) < chunk_tokens * 0.25:
            chunks[-1] = chunks[-1] + " " + final
        else:
            chunks.append(final)

    return chunks


def _chunk_by_words(words: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Fallback: fixed-size word-level chunking with overlap."""
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
        if start + overlap >= len(words):
            break
    return chunks


def chunk_entries(
    entries: List[Entry],
    chunk_tokens: int = 500,
    overlap_tokens: int = 50,
) -> List[Entry]:
    """Chunk entries with long text, preserving metadata.

    Short entries pass through unchanged. Long entries produce
    child entries with parent_id in metadata.
    """
    result = []
    for entry in entries:
        if not entry.text or len(entry.text.split()) <= chunk_tokens:
            result.append(entry)
            continue

        chunks = chunk_text(entry.text, chunk_tokens, overlap_tokens)
        if len(chunks) == 1:
            result.append(entry)
        else:
            for i, chunk_str in enumerate(chunks):
                child_meta = dict(entry.metadata)
                child_meta["parent_id"] = entry.id
                child_meta["chunk_index"] = i
                child_meta["total_chunks"] = len(chunks)
                result.append(Entry(
                    id=f"{entry.id}__chunk_{i}",
                    text=chunk_str,
                    source=entry.source,
                    title=f"{entry.title} (part {i+1}/{len(chunks)})" if entry.title else None,
                    date=entry.date,
                    metadata=child_meta,
                ))
    return result
