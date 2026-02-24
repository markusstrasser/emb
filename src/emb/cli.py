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
    output: Path = typer.Option(..., "--output", "-o", help="Output index JSON path"),
    model: str = typer.Option("Alibaba-NLP/gte-modernbert-base", "--model", "-m", help="Embedding model"),
    chunk: bool = typer.Option(False, "--chunk", help="Enable sentence-aware chunking"),
    chunk_tokens: int = typer.Option(500, "--chunk-tokens", help="Words per chunk"),
    overlap_tokens: int = typer.Option(50, "--overlap-tokens", help="Overlap words between chunks"),
    force: bool = typer.Option(False, "--force", help="Force re-embed all (ignore cache)"),
    scales: str = typer.Option(None, "--scales", help="Multi-scale chunk sizes, comma-separated (e.g. '200,500')"),
):
    """Embed entries from JSONL into a searchable index."""
    from emb.io import read_jsonl
    from emb.embed import EmbeddingEngine
    from emb.cache import EmbeddingCache
    from emb.chunking import chunk_entries, multiscale_chunk_entries
    from datetime import datetime

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
    engine = EmbeddingEngine(model=model)
    cache_dir = output.parent / f".emb_cache_{output.stem}"

    if force:
        cache = EmbeddingCache(dim=engine.dim)
    else:
        cache = EmbeddingCache.load(cache_dir, dim=engine.dim)
        if len(cache):
            console.print(f"  Cache: {len(cache)} existing embeddings")

    # Embed
    entries = engine.embed_entries(entries, cache=cache, checkpoint_dir=cache_dir)

    # Write index
    index_data = {
        'metadata': {
            'total_entries': len(entries),
            'sources': {},
            'embedding_model': engine.model,
            'embedding_dim': engine.dim,
            'generated_at': datetime.now().isoformat(),
        },
        'entries': [e.to_dict() for e in entries],
    }
    # Count sources
    for e in entries:
        s = e.source or 'unknown'
        index_data['metadata']['sources'][s] = index_data['metadata']['sources'].get(s, 0) + 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(index_data, f)

    size_mb = output.stat().st_size / 1024 / 1024
    console.print(f"  Index written: {output} ({size_mb:.1f} MB, {len(entries)} entries)")


@app.command()
def search(
    index_file: Path = typer.Argument(..., help="Index JSON file to search"),
    query: Optional[str] = typer.Argument(None, help="Search query"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Comma-separated source filter"),
    hybrid: bool = typer.Option(False, "--hybrid", help="Enable BM25+dense hybrid (RRF)"),
    rerank: bool = typer.Option(False, "--rerank", help="Cross-encoder reranking"),
    fresh: float = typer.Option(0.0, "--fresh", "-f", help="Freshness weight (0.0-1.0)"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show full text preview"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Search an index."""
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
    )

    if json_output:
        print(json.dumps(results, indent=2, default=str))
    else:
        _display_results(results, detailed=detailed)


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
def contextualize(
    input_file: str = typer.Argument(..., help="JSONL input file"),
    output_file: str = typer.Argument(..., help="JSONL output file"),
    model: str = typer.Option("gemini/gemini-2.0-flash", "--model", "-m", help="LLM model (litellm format)"),
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
    index_file: Path = typer.Argument(..., help="Index JSON file"),
):
    """Show index statistics."""
    with open(index_file, 'r') as f:
        data = json.load(f)

    meta = data.get('metadata', {})
    entries = data.get('entries', [])

    console.print(f"\n[bold]Index: {index_file}[/bold]")
    console.print(f"  Entries: {len(entries)}")
    console.print(f"  Model: {meta.get('embedding_model', 'unknown')}")
    console.print(f"  Dim: {meta.get('embedding_dim', 'unknown')}")
    console.print(f"  Generated: {meta.get('generated_at', 'unknown')}")

    sources = meta.get('sources', {})
    if sources:
        console.print(f"\n  Sources:")
        for s, count in sorted(sources.items(), key=lambda x: -x[1]):
            console.print(f"    {s}: {count}")


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
