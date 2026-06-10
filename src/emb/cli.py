"""emb CLI — embed, index, and search text corpora."""
import json
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

app = typer.Typer(help="emb: Embed, index, and search text corpora")
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())


@app.command()
def version():
    """Show version."""
    from emb import __version__
    print(f"emb {__version__}")


@app.command()
def embed(
    input_file: str = typer.Argument(..., help="JSONL input file (or '-' for stdin)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output path (directory for split, file for json)"),
    model: str = typer.Option("Alibaba-NLP/gte-modernbert-base", "--model", "-m", help="Embedding model"),
    chunk: bool = typer.Option(False, "--chunk", help="Enable sentence-aware chunking"),
    chunk_tokens: int = typer.Option(500, "--chunk-tokens", help="Words per chunk"),
    overlap_tokens: int = typer.Option(50, "--overlap-tokens", help="Overlap words between chunks"),
    force: bool = typer.Option(False, "--force", help="Force re-embed all (ignore cache)"),
    scales: str = typer.Option(None, "--scales", help="Multi-scale chunk sizes, comma-separated (e.g. '200,500')"),
):
    """Embed entries from JSONL into a searchable split index.

    Always writes the split format (entries.jsonl + mmap embeddings.npy + metadata.json).
    To read or migrate a legacy monolithic-JSON index, use `emb convert`.
    """
    import numpy as np
    from emb.io import read_jsonl
    from emb.embed import EmbeddingEngine
    from emb.cache import EmbeddingCache
    from emb.chunking import chunk_entries, multiscale_chunk_entries

    # Read input
    console.print(f"Reading entries from {input_file}...")
    entries = read_jsonl(input_file)
    if not entries:
        console.print("[red]No valid entries found in input.[/red]")
        raise typer.Exit(1)
    console.print(f"  Loaded {len(entries)} entries")

    # Chunk if requested
    if scales:
        scale_list = [int(s.strip()) for s in scales.split(',')]
        before = len(entries)
        entries = multiscale_chunk_entries(entries, scales=scale_list, overlap_tokens=overlap_tokens)
        console.print(f"  Multi-scale chunked ({scale_list}): {before} -> {len(entries)} entries")
    elif chunk:
        before = len(entries)
        entries = chunk_entries(entries, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        console.print(f"  Chunked: {before} -> {len(entries)} entries")

    # Set up engine + cache
    from emb.embed import model_slug
    engine = EmbeddingEngine(model=model)
    # Namespace the cache by model slug: two models can share a dimension (gte-768 vs
    # gemini-768) but produce vectors in incompatible spaces. A dim check alone can't
    # tell them apart, so a shared cache dir would yield silent wrong-vector hits.
    cache_dir = output.parent / f".emb_cache_{output.stem}" / model_slug(model)

    if force:
        cache = EmbeddingCache(dim=engine.dim)
    else:
        cache = EmbeddingCache.load(cache_dir, dim=engine.dim)
        if len(cache):
            console.print(f"  Cache: {len(cache)} existing embeddings")

    # Embed
    entries = engine.embed_entries(entries, cache=cache, checkpoint_dir=cache_dir)

    meta = {
        'embedding_model': engine.model,
        'embedding_dim': engine.dim,
    }

    from emb.index import write_index
    embeddings = np.array([e.embedding for e in entries], dtype=np.float32)
    write_index(entries, embeddings, output, meta)
    total_bytes = sum(f.stat().st_size for f in output.iterdir())
    console.print(f"  Split index written: {output}/ ({total_bytes / 1024 / 1024:.1f} MB, {len(entries)} entries)")


@app.command()
def search(
    index_file: Path = typer.Argument(..., help="Index file (JSON) or directory (split format)"),
    query: Optional[str] = typer.Argument(None, help="Search query"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Comma-separated source filter"),
    hybrid: bool = typer.Option(False, "--hybrid", help="Enable BM25+dense hybrid (RRF)"),
    fusion: str = typer.Option("rrf", "--fusion", help="Hybrid fusion: rrf or convex"),
    convex_alpha: float = typer.Option(0.5, "--convex-alpha", help="Dense weight for convex fusion (0-1)"),
    rerank: bool = typer.Option(False, "--rerank", help="Cross-encoder reranking"),
    fresh: float = typer.Option(0.0, "--fresh", "-f", help="Freshness weight (0.0-1.0)"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show full text preview"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    cag: bool = typer.Option(False, "--cag", help="CAG mode: skip embeddings, span Gemini over raw text"),
    cag_model: Optional[str] = typer.Option(None, "--cag-model", help="Gemini model for CAG (default: gemini-3.1-flash-lite-preview)"),
):
    """Search an index."""
    if cag:
        _cag_search(index_file, query, sources, json_output, cag_model)
        return

    from emb.search import SearchEngine

    engine = SearchEngine(index_file)

    if interactive:
        _interactive_loop(engine)
        return

    if not query:
        console.print("[red]Provide a query or use --interactive[/red]")
        raise typer.Exit(1)

    source_set = set(s.strip() for s in sources.split(',')) if sources else None

    results = engine.search(
        query, top_k=top_k, sources=source_set,
        hybrid=hybrid, rerank=rerank, freshness_weight=fresh,
        fusion=fusion, convex_alpha=convex_alpha,
    )

    if json_output:
        print(json.dumps(results, indent=2, default=str))
    else:
        _display_results(results, detailed=detailed)


def _cag_search(index_file: Path, query, sources, json_output, cag_model):
    """CAG mode: load entries, stuff into Gemini context, get prose answer."""
    from emb.io import read_jsonl
    from emb.cag import cag_search

    if not query:
        console.print("[red]CAG mode requires a query[/red]")
        raise typer.Exit(1)

    # Load entries — from split index (entries.jsonl) or raw JSONL
    index_file = Path(index_file)
    if index_file.is_dir():
        entries_path = index_file / "entries.jsonl"
        if not entries_path.exists():
            console.print(f"[red]No entries.jsonl in {index_file}/[/red]")
            raise typer.Exit(1)
        entries = read_jsonl(str(entries_path))
    elif str(index_file).endswith('.jsonl'):
        entries = read_jsonl(str(index_file))
    else:
        # Legacy JSON format — load entries without embeddings
        with open(index_file, 'r') as f:
            from emb.schema import Entry
            data = json.load(f)
        entries = [Entry.from_dict(e) for e in data.get('entries', [])]

    source_set = set(s.strip() for s in sources.split(',')) if sources else None

    # Show post-filter stats
    filtered = [e for e in entries if e.source in source_set] if source_set else entries
    total_chars = sum(len(e.text) for e in filtered)
    est_tokens = total_chars // 4
    console.print(f"  {len(filtered):,} entries ({len(entries):,} total), ~{est_tokens:,} tokens")

    result = cag_search(query, entries, model=cag_model, sources=source_set)

    if json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        console.print(f"\n[dim]{result['model']} | {result['batches']} batch(es) | {result['entries']} entries | ~{result['tokens_used']:,} tokens[/dim]\n")
        console.print(result['answer'])


def _interactive_loop(engine):
    """Interactive search REPL."""
    console.print("\n[bold]emb search -- Interactive Mode[/bold]")
    console.print("Commands: !sources <s>, !top <n>, !hybrid, !rerank, !fresh <n>, !detailed, !quit\n")

    sources = None
    top_k = 10
    hybrid = False
    rerank = False
    fresh = 0.0
    detailed = False

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query == '!quit':
            break
        if query.startswith('!sources'):
            parts = query.split(maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                sources = set(s.strip() for s in parts[1].split(','))
                console.print(f"  Sources: {sources}")
            else:
                sources = None
                console.print("  Sources: all")
            continue
        if query.startswith('!top'):
            try:
                top_k = int(query.split()[1])
                console.print(f"  Top-k: {top_k}")
            except (IndexError, ValueError):
                console.print("  Usage: !top N")
            continue
        if query == '!hybrid':
            hybrid = not hybrid
            console.print(f"  Hybrid: {hybrid}")
            continue
        if query == '!rerank':
            rerank = not rerank
            console.print(f"  Rerank: {rerank}")
            continue
        if query.startswith('!fresh'):
            try:
                fresh = float(query.split()[1])
                console.print(f"  Freshness: {fresh}")
            except (IndexError, ValueError):
                console.print("  Usage: !fresh 0.3")
            continue
        if query == '!detailed':
            detailed = not detailed
            console.print(f"  Detailed: {detailed}")
            continue

        results = engine.search(
            query, top_k=top_k, sources=sources,
            hybrid=hybrid, rerank=rerank, freshness_weight=fresh,
        )
        _display_results(results, detailed=detailed)


def _display_results(results, detailed=False):
    """Display search results with color coding."""
    if not results:
        console.print("  No results found.")
        return

    source_colors = {
        'chatgpt': 'blue', 'twitter': 'cyan', 'logseq': 'green',
        'git': 'yellow', 'raycast': 'magenta', 'podcast': 'bright_black',
        'research': 'red', 'docs': 'white', 'claude': 'blue',
    }
    for i, r in enumerate(results, 1):
        color = source_colors.get(r.get('source', ''), 'white')
        src = (r.get('source') or 'unknown').upper()
        title = r.get('title') or r.get('id', '')
        console.print(f"\n[bold {color}][{i}] [{src}] {title}[/bold {color}]")

        parts = [f"Sim: {r['similarity']:.3f}"]
        if 'rerank_score' in r:
            parts.append(f"Rerank: {r['rerank_score']:.3f}")
        date_str = r.get('date') or ''
        console.print(f"  Date: {date_str} | {' | '.join(parts)}")

        if detailed:
            console.print(f"  {r.get('text', '')}")
        console.print(f"  ID: {r.get('id', '')}")


@app.command()
def read(
    index_file: Path = typer.Argument(..., help="Index directory (split) or JSON file"),
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(200, "--top-k", "-k", help="How many hits to read over"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Comma-separated source filter"),
    since: Optional[str] = typer.Option(None, "--since", help="ISO date filter (e.g. 2026-01-01)"),
    hybrid: bool = typer.Option(True, "--hybrid/--dense", help="Hybrid BM25+dense retrieval (default on)"),
    rerank: bool = typer.Option(False, "--rerank", help="Cross-encoder rerank the hits before reading"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Gemini model (default: gemini-3.1-flash-lite-preview)"),
    max_tokens: int = typer.Option(0, "--max-tokens", help="Cap corpus tokens sent to the model (0 = no cap)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the cost confirmation prompt"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Locate with the local index, then READ the hits with a long-context model.

    The June-2026 pattern: cheap exact retrieval narrows the corpus, a long-context
    model synthesizes over only the hits. Distinct from `search --cag` (which spans the
    model over the WHOLE corpus for exhaustive recall) — `read` is bounded and cheap.
    """
    from datetime import datetime
    from dateutil import parser as date_parser
    from emb.search import SearchEngine
    from emb.cag import cag_search, _format_entry, CHARS_PER_TOKEN, DEFAULT_MODEL

    model = model or DEFAULT_MODEL
    engine = SearchEngine(index_file)
    source_set = set(s.strip() for s in sources.split(',')) if sources else None
    since_dt = date_parser.parse(since) if since else None

    results = engine.search(
        query, top_k=top_k, sources=source_set, since=since_dt,
        hybrid=hybrid, rerank=rerank,
    )
    if not results:
        console.print("  No hits to read.")
        raise typer.Exit(0)

    # Pull the FULL entries for the hits (search output truncates text to 300 chars).
    by_id = {e.id: e for e in engine.entries}
    hits = [by_id[r['id']] for r in results if r['id'] in by_id]

    # Cost preflight — gemini-3.1-flash-lite: $0.25/MTok in, $1.50/MTok out (verified).
    est_in_tokens = sum(len(_format_entry(e)) for e in hits) // CHARS_PER_TOKEN
    est_out_tokens = 8192
    in_rate, out_rate = 0.25, 1.50
    if 'flash-lite' not in model:
        in_rate, out_rate = (0.50, 3.0) if '3-flash' in model else (1.50, 9.0)
    est_cost = est_in_tokens / 1e6 * in_rate + est_out_tokens / 1e6 * out_rate

    console.print(f"  {len(hits)} hits → ~{est_in_tokens:,} input tokens, model={model}")
    console.print(f"  [dim]Estimated cost: ${est_cost:.4f} (in ${est_in_tokens/1e6*in_rate:.4f} + out ~${est_out_tokens/1e6*out_rate:.4f})[/dim]")
    if est_cost > 1.0 and not yes:
        import sys
        if sys.stdin.isatty():
            resp = input(f"  This will cost ~${est_cost:.2f}. Proceed? [y/N] ").strip().lower()
            if resp not in ('y', 'yes'):
                console.print("  Aborted.")
                raise typer.Exit(0)
        else:
            console.print(f"[yellow]  Cost ~${est_cost:.2f} exceeds $1 and no TTY to confirm. Pass --yes to proceed.[/yellow]")
            raise typer.Exit(1)

    from emb.cag import MAX_CORPUS_CHARS
    max_chars = max_tokens * CHARS_PER_TOKEN if max_tokens > 0 else MAX_CORPUS_CHARS
    result = cag_search(query, hits, model=model, max_chars=max_chars)

    if json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        console.print(f"\n[dim]{result['model']} | {result['batches']} batch(es) | {result['entries']} entries | ~{result['tokens_used']:,} tokens[/dim]\n")
        console.print(result['answer'])


@app.command()
def pairs(
    index_file: Path = typer.Argument(..., help="Index directory (split) or JSON file"),
    threshold: float = typer.Option(0.85, "--threshold", "-t", help="Min similarity"),
    max_threshold: Optional[float] = typer.Option(None, "--max-threshold", help="Max similarity (for bands, e.g. interference 0.50-0.80)"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Keep only each item's top-K partners (caps output)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write pairs as JSONL (default: print summary)"),
):
    """Find all high-similarity PAIRS within a corpus (duplicates, interference)."""
    import numpy as np
    from emb.pairs import find_pairs, write_pairs_jsonl

    index_file = Path(index_file)
    if index_file.is_dir():
        from emb.index import read_index
        entries, embeddings, _ = read_index(index_file)
    else:
        with open(index_file) as f:
            from emb.schema import Entry
            data = json.load(f)
        entries = [Entry.from_dict(e) for e in data['entries']]
        embeddings = np.array([e.embedding for e in entries], dtype=np.float32)

    console.print(f"  {len(entries)} entries, computing pairs (threshold={threshold})...")
    found = find_pairs(embeddings, threshold=threshold, max_threshold=max_threshold, top_k=top_k)
    console.print(f"  Found {len(found)} pairs")

    if output:
        ids = [e.id for e in entries]
        write_pairs_jsonl(found, output, ids=ids)
        console.print(f"  Written to {output}")
    else:
        for i, j, sim in found[:50]:
            console.print(f"  {sim:.4f}  [{entries[i].id}]  ~  [{entries[j].id}]")
        if len(found) > 50:
            console.print(f"  ... and {len(found) - 50} more (use --output to write all)")


@app.command()
def contextualize(
    input_file: str = typer.Argument(..., help="JSONL input file"),
    output_file: str = typer.Argument(..., help="JSONL output file"),
    model: str = typer.Option("gemini-2.0-flash", "--model", "-m", help="Gemini model name"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Max parallel LLM calls"),
):
    """Add LLM-generated context to entries (Anthropic's Contextual Retrieval)."""
    from emb.io import read_jsonl, write_jsonl
    from emb.contextualize import contextualize_sync

    entries = read_jsonl(input_file)
    console.print(f"  Loaded {len(entries)} entries")

    results = contextualize_sync(entries, model=model, concurrency=concurrency)

    count = write_jsonl(results, output_file)
    console.print(f"  Written {count} entries to {output_file}")


@app.command()
def info(
    index_file: Path = typer.Argument(..., help="Index file (JSON) or directory (split format)"),
):
    """Show index statistics."""
    index_file = Path(index_file)

    if index_file.is_dir():
        meta_path = index_file / 'metadata.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        n_entries = meta.get('total_entries', '?')

        # Show disk size breakdown
        sizes = {}
        for child in index_file.iterdir():
            sizes[child.name] = child.stat().st_size
        total = sum(sizes.values())

        console.print(f"\n[bold]Index: {index_file}/ (split format)[/bold]")
        console.print(f"  Entries: {n_entries}")
        console.print(f"  Total size: {total / 1024 / 1024:.1f} MB")
        for name, size in sorted(sizes.items()):
            console.print(f"    {name}: {size / 1024 / 1024:.1f} MB")
    else:
        with open(index_file, 'r') as f:
            data = json.load(f)
        meta = data.get('metadata', {})
        n_entries = len(data.get('entries', []))
        size_mb = index_file.stat().st_size / 1024 / 1024
        console.print(f"\n[bold]Index: {index_file} (JSON format, {size_mb:.1f} MB)[/bold]")
        console.print(f"  Entries: {n_entries}")

    console.print(f"  Model: {meta.get('embedding_model', 'unknown')}")
    console.print(f"  Dim: {meta.get('embedding_dim', 'unknown')}")
    console.print(f"  Generated: {meta.get('generated_at', 'unknown')}")

    sources = meta.get('sources', {})
    if sources:
        console.print(f"\n  Sources:")
        for s, count in sorted(sources.items(), key=lambda x: -x[1]):
            console.print(f"    {s}: {count}")


@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="Input index (JSON file or split directory)"),
    output_path: Path = typer.Argument(..., help="Output path (directory for split, file for JSON)"),
):
    """Convert between JSON and split index formats."""
    from emb.index import convert_json_to_split, convert_split_to_json

    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.is_dir():
        # Split -> JSON
        console.print(f"Converting split format → JSON: {input_path}/ → {output_path}")
        convert_split_to_json(input_path, output_path)
        size_mb = output_path.stat().st_size / 1024 / 1024
        console.print(f"  Written: {output_path} ({size_mb:.1f} MB)")
    else:
        # JSON -> Split
        console.print(f"Converting JSON → split format: {input_path} → {output_path}/")
        convert_json_to_split(input_path, output_path)
        total = sum(f.stat().st_size for f in output_path.iterdir())
        console.print(f"  Written: {output_path}/ ({total / 1024 / 1024:.1f} MB)")


@app.command()
def merge(
    inputs: list[Path] = typer.Argument(..., help="Index JSON files to merge"),
    output: Path = typer.Option(..., "--output", "-o", help="Output merged index"),
):
    """Merge multiple indices into one."""
    from datetime import datetime

    all_entries = []
    all_meta_sources = {}
    model = None
    dim = None

    for idx_path in inputs:
        console.print(f"  Reading {idx_path}...")
        with open(idx_path, 'r') as f:
            data = json.load(f)

        meta = data.get('metadata', {})
        if model is None:
            model = meta.get('embedding_model')
            dim = meta.get('embedding_dim')
        elif meta.get('embedding_model') != model:
            console.print(f"[yellow]Warning: model mismatch in {idx_path} ({meta.get('embedding_model')} vs {model})[/yellow]")

        for e in data.get('entries', []):
            all_entries.append(e)
            s = e.get('source', 'unknown')
            all_meta_sources[s] = all_meta_sources.get(s, 0) + 1

    # Deduplicate by id
    seen_ids = set()
    deduped = []
    for e in all_entries:
        if e['id'] not in seen_ids:
            seen_ids.add(e['id'])
            deduped.append(e)

    merged = {
        'metadata': {
            'total_entries': len(deduped),
            'sources': all_meta_sources,
            'embedding_model': model,
            'embedding_dim': dim,
            'generated_at': datetime.now().isoformat(),
            'merged_from': [str(p) for p in inputs],
        },
        'entries': deduped,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(merged, f)

    console.print(f"\n  Merged: {len(all_entries)} -> {len(deduped)} entries (deduped)")
    console.print(f"  Written to: {output}")


if __name__ == "__main__":
    app()
