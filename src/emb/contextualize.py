"""Contextual retrieval: LLM-generated context prepending.

Calls a cheap LLM to generate a short (1-2 sentence) context for each entry,
then prepends it to the text. This dramatically improves retrieval quality
(Anthropic research: 35-49% fewer failed retrievals).

Uses google-genai SDK directly (Gemini models).
"""
import sys
import asyncio
from typing import List, Optional
from emb.schema import Entry


CONTEXT_PROMPT = """You are helping improve search retrieval. Given an entry from a knowledge base, write a SHORT (1-2 sentence) context that situates this entry. Focus on: what topic is discussed, key entities, and how it relates to the source.

Source type: {source}
Title: {title}
Date: {date}

Entry text:
{text}

Write ONLY the contextualizing sentences, nothing else. Be concise."""


def build_context_prompt(entry: Entry) -> str:
    """Build the LLM prompt for an entry."""
    return CONTEXT_PROMPT.format(
        source=entry.source or "unknown",
        title=entry.title or "untitled",
        date=entry.date or "unknown",
        text=entry.text[:2000],  # Cap input to avoid huge prompts
    )


async def contextualize_entry(entry: Entry, model: str, client) -> Entry:
    """Add LLM-generated context to a single entry."""
    if entry.metadata.get("contextualized"):
        return entry  # Already done

    prompt = build_context_prompt(entry)
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config={"max_output_tokens": 150, "temperature": 0.0},
        )
        context = response.text.strip()
    except Exception as e:
        print(f"  Warning: LLM call failed for {entry.id}: {e}", file=sys.stderr)
        return entry  # Return unchanged on error

    # Prepend context
    new_meta = dict(entry.metadata)
    new_meta["contextualized"] = True
    new_meta["context_model"] = model

    return Entry(
        id=entry.id,
        text=f"{context}\n\n{entry.text}",
        source=entry.source,
        title=entry.title,
        date=entry.date,
        metadata=new_meta,
    )


async def contextualize_batch(
    entries: List[Entry],
    model: str = "gemini-2.0-flash",
    concurrency: int = 10,
    client=None,
) -> List[Entry]:
    """Contextualize entries with bounded concurrency.

    Args:
        entries: Entries to contextualize.
        model: Gemini model name (e.g. "gemini-2.0-flash")
        concurrency: Max parallel LLM calls.
        client: google.genai.Client instance (created if not provided)
    """
    if client is None:
        from google import genai
        client = genai.Client()

    # Split into already-done and needs-work
    needs_work = [e for e in entries if not e.metadata.get("contextualized")]
    already_done = [e for e in entries if e.metadata.get("contextualized")]

    if not needs_work:
        print("  All entries already contextualized, nothing to do")
        return entries

    print(f"  Contextualizing {len(needs_work)} entries ({len(already_done)} already done)")
    print(f"  Model: {model}, concurrency: {concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    done_count = 0
    total = len(needs_work)

    async def _process(entry):
        nonlocal done_count
        async with semaphore:
            result = await contextualize_entry(entry, model, client)
            done_count += 1
            if done_count % 50 == 0 or done_count == total:
                print(f"  [{done_count}/{total}]", flush=True)
            return result

    tasks = [_process(e) for e in needs_work]
    contextualized = await asyncio.gather(*tasks)

    # Rebuild in original order
    result_map = {e.id: e for e in contextualized}
    result_map.update({e.id: e for e in already_done})
    return [result_map.get(e.id, e) for e in entries]


def contextualize_sync(
    entries: List[Entry],
    model: str = "gemini-2.0-flash",
    concurrency: int = 10,
    client=None,
) -> List[Entry]:
    """Synchronous wrapper for contextualize_batch."""
    return asyncio.run(contextualize_batch(entries, model, concurrency, client))
