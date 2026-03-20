"""CAG (Context-Augmented Generation) — bypass embeddings, span cheap LLM over raw text.

Single batch: stuff all entries into one Gemini call.
Multi-batch: parallel map (extract per shard) + reduce (synthesize across shards).
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Set

from emb.schema import Entry

logger = logging.getLogger(__name__)

# Token budget — Gemini 3.1 Flash Lite: 1M input, 65K output
CHARS_PER_TOKEN = 4
MAX_INPUT_TOKENS = 1_048_576
RESERVED_TOKENS = 70_000  # system prompt + query + output headroom
MAX_CORPUS_TOKENS = MAX_INPUT_TOKENS - RESERVED_TOKENS
MAX_CORPUS_CHARS = MAX_CORPUS_TOKENS * CHARS_PER_TOKEN

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
MAX_WORKERS = 6  # parallel shard calls


@dataclass
class Batch:
    entries: list[Entry]
    text: str
    est_tokens: int


def pack_batches(
    entries: list[Entry],
    max_chars: int = MAX_CORPUS_CHARS,
) -> list[Batch]:
    """Pack entries into batches that fit within token budget.

    Greedy bin-packing: fill each batch until adding the next entry would overflow,
    then start a new batch. Preserves input order.
    """
    batches: list[Batch] = []
    cur_parts: list[str] = []
    cur_entries: list[Entry] = []
    cur_chars = 0

    for entry in entries:
        block = _format_entry(entry)
        block_len = len(block)

        # Single entry exceeds budget — truncate it into its own batch
        if block_len > max_chars:
            if cur_parts:
                _flush_batch(batches, cur_parts, cur_entries)
                cur_parts, cur_entries, cur_chars = [], [], 0
            truncated = block[:max_chars]
            batches.append(Batch(
                entries=[entry],
                text=truncated,
                est_tokens=len(truncated) // CHARS_PER_TOKEN,
            ))
            continue

        if cur_chars + block_len > max_chars:
            _flush_batch(batches, cur_parts, cur_entries)
            cur_parts, cur_entries, cur_chars = [], [], 0

        cur_parts.append(block)
        cur_entries.append(entry)
        cur_chars += block_len

    if cur_parts:
        _flush_batch(batches, cur_parts, cur_entries)

    return batches


def cag_search(
    query: str,
    entries: list[Entry],
    model: str | None = None,
    sources: Set[str] | None = None,
    max_chars: int = MAX_CORPUS_CHARS,
) -> dict:
    """Search entries via CAG — stuff into Gemini context, get prose answer.

    Auto-selects single-batch or map-reduce based on corpus size.
    Returns dict with 'answer', 'model', 'batches', 'entries', 'tokens_used'.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("CAG requires google-genai. Install with: uv pip install 'emb[cag]'")

    if model is None:
        model = os.environ.get("EMB_CAG_MODEL", DEFAULT_MODEL)

    # Source filter
    if sources:
        entries = [e for e in entries if e.source in sources]

    if not entries:
        return {"answer": "No entries match filters.", "model": model, "batches": 0, "entries": 0, "tokens_used": 0}

    batches = pack_batches(entries, max_chars=max_chars)
    client = genai.Client()

    logger.info(f"CAG: {len(entries)} entries -> {len(batches)} batch(es), model={model}")

    if len(batches) == 1:
        return _single_batch(client, query, batches[0], model)
    else:
        return _map_reduce(client, query, batches, model)


# --- Prompts ---

SEARCH_SYSTEM = """\
You are a search assistant. The user's corpus is loaded into context.
Answer the query using ONLY the provided entries. Cite entries by their title or ID.
If the entries don't contain relevant information, say so — do not hallucinate."""

MAP_SYSTEM = """\
You are searching one shard of a larger corpus. Extract ALL information relevant to the query.
Include entry titles/IDs for attribution. If nothing is relevant, say "No relevant entries in this shard."
Be thorough — your output feeds into a synthesis step."""

REDUCE_SYSTEM = """\
You are synthesizing search results from multiple corpus shards.
Combine shard results into a coherent answer. Cite by entry title/ID.
Deduplicate overlapping information. If all shards found nothing relevant, say so."""


# --- Internal ---

def _format_entry(entry: Entry) -> str:
    """Format a single entry for context stuffing."""
    parts = [f"--- [{entry.source or 'unknown'}] {entry.title or entry.id} ---\n"]
    if entry.date:
        parts.append(f"Date: {entry.date}\n")
    parts.append(entry.text)
    parts.append("\n\n")
    return "".join(parts)


def _flush_batch(
    batches: list[Batch],
    parts: list[str],
    entries: list[Entry],
) -> None:
    text = "".join(parts)
    batches.append(Batch(
        entries=list(entries),
        text=text,
        est_tokens=len(text) // CHARS_PER_TOKEN,
    ))


def _single_batch(client, query: str, batch: Batch, model: str) -> dict:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=f"Corpus ({len(batch.entries)} entries):\n\n{batch.text}\n\nQuery: {query}",
        config=types.GenerateContentConfig(
            system_instruction=SEARCH_SYSTEM,
            temperature=0.2,
            max_output_tokens=8192,
        ),
    )

    usage = response.usage_metadata
    tokens = usage.total_token_count if usage else batch.est_tokens

    return {
        "answer": response.text or "(empty response)",
        "model": model,
        "batches": 1,
        "entries": len(batch.entries),
        "tokens_used": tokens,
    }


def _map_reduce(client, query: str, batches: list[Batch], model: str) -> dict:
    from google.genai import types

    # Map phase: parallel per-shard extraction
    def _map_one(batch: Batch) -> str:
        resp = client.models.generate_content(
            model=model,
            contents=f"Corpus shard ({len(batch.entries)} entries):\n\n{batch.text}\n\nQuery: {query}",
            config=types.GenerateContentConfig(
                system_instruction=MAP_SYSTEM,
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
        return resp.text or ""

    shard_results = []
    workers = min(len(batches), MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_map_one, b): i for i, b in enumerate(batches)}
        # Collect in order
        ordered = [None] * len(batches)
        for future in as_completed(futures):
            idx = futures[future]
            ordered[idx] = future.result()
        shard_results = ordered

    # Reduce phase: synthesize across shards
    combined = "\n\n".join(
        f"=== Shard {i+1}/{len(batches)} ({len(b.entries)} entries) ===\n{result}"
        for i, (b, result) in enumerate(zip(batches, shard_results))
    )

    response = client.models.generate_content(
        model=model,
        contents=f"Shard results:\n\n{combined}\n\nOriginal query: {query}",
        config=types.GenerateContentConfig(
            system_instruction=REDUCE_SYSTEM,
            temperature=0.2,
            max_output_tokens=8192,
        ),
    )

    total_entries = sum(len(b.entries) for b in batches)
    total_tokens = sum(b.est_tokens for b in batches)

    return {
        "answer": response.text or "(empty response)",
        "model": model,
        "batches": len(batches),
        "entries": total_entries,
        "tokens_used": total_tokens,
    }
