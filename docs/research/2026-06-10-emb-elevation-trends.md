# Elevating emb — Consumer-Driven Improvements + 2025-26 Retrieval Trends

**Question:** Given emb's consumers, how to elevate/improve it; what are the newest trends and solutions in embedding/retrieval for local-first search?
**Tier:** Standard | **Date:** 2026-06-10
**Ground truth:** Consumer audit (this session) + `phenome/docs/research/embedding-stack-upgrade-2026-05-29.md` (12 days old, covers models + rerankers with local evals — reused, not re-researched).

## Consumer reality (drives everything below)

**phenome is the only real consumer** [DATA: Explore sweep of ~/Projects, 2026-06-10]. ~2.3M entries across 25+ personal-data sources. It uses nearly the full API surface: `SearchEngine` (hybrid+rerank+freshness), `NeighborIndex`, `EmbeddingEngine`, `EmbeddingCache`, `chunk_text`, `read_index`/`write_index`, `expand_sources`. Anki uses its own Ollama path; nothing else consumes emb.

Observed consumer pain points (from phenome code, not speculation):
1. **Overfetch hack**: `fetch_k = max(top_k * 5, 200)` when `entry_filter` is active — post-scoring filtering forces overfetch to avoid K-truncation.
2. **Cache migration complexity**: 4-tier fallback (model slugs → legacy flat files → old JSON embeddings).
3. **Dual index formats**: split + legacy JSON both supported; split saves ~17s load.
4. **Rerank eval cost**: 45-75 min, so reranking is gated, not default.
5. **Late chunking lives in phenome** (`LateChunker` wrapper), not in emb — a consumer carrying library-shaped code.

## Claims Table

| # | Claim | Evidence | Confidence | Source | Status |
|---|-------|----------|------------|--------|--------|
| 1 | Brute-force dense at 2.3M×768 fp32 = ~7 GB resident; FAISS-style Flat is ~2-4 ms p95 only up to ~1M, then ANN wins decisively | FAISS production benchmark: IVF ~0.83 ms p95 @10M (recall@10>0.95, 4.2 GB); HNSW 0.42 ms (12.8 GB); Flat ~4 ms near 1M | MED (B-grade source, numbers plausible) | [markaicode.com/benchmarks/faiss-production-benchmark-latency](https://markaicode.com/benchmarks/faiss-production-benchmark-latency/) | VERIFIED-SINGLE-SOURCE |
| 2 | int8 quantization cuts memory ~75% at ~95-99% recall; 1-bit/binary cuts 32× at 70-90% recall unless rescored | usearch docs (int8, ~95% recall on 1536d); Weaviate RQ (8-bit, 97-99.9% recall); pgvector int8vec PR (no recall loss observed, smaller+cache-friendlier) | HIGH (3 independent sources agree) | usearch overview; [weaviate.io/blog/8-bit-rotational-quantization](https://weaviate.io/blog/8-bit-rotational-quantization); [pgvector PR #977](https://github.com/pgvector/pgvector/pull/977) | VERIFIED |
| 3 | Two-phase quantized search (fast traversal on quantized vectors → rescore top candidates at full precision) keeps recall@10 ≥99% — the standard pattern now (Elastic BBQ default since 9.2) | Elastic recall benchmark, DBPedia + Jina v5 | HIGH | [elastic.co/search-labs/blog/recall-vector-search-quantization](https://www.elastic.co/search-labs/blog/recall-vector-search-quantization) | VERIFIED |
| 4 | For local single-machine use, usearch (HNSW, SIMD, native f16/int8, single-dependency) and LanceDB (IVF, disk-based) are the practitioner picks over raw FAISS; sqlite-vec is brute-force and fine only <~1M | node-vector-bench (sqlite-vec/USearch/LanceDB/DuckDB-VSS at 1k-2M); Vector-Arena | MED | [github.com/photostructure/node-vector-bench](https://github.com/photostructure/node-vector-bench) | VERIFIED |
| 5 | Hybrid BM25+dense with RRF + cross-encoder rerank on a short list is now THE standard production architecture (the exact thing emb already implements) | 4+ independent 2026 guides converge: 3-stage (retrieve 100-1000 → RRF → cross-encode); hybrid beats dense-only ~26-31% NDCG | HIGH | digitalapplied.com, atlan.com/know/hybrid-rag, prompt20, cadence.withremote.ai | VERIFIED |
| 6 | Contextual embeddings (prepend 50-100 token chunk context before embedding) cut retrieval failures ~49% alone, ~67% with reranker | Anthropic contextual-retrieval numbers, echoed in production guides | HIGH | [cadence.withremote.ai/blog/production-rag-architecture](https://cadence.withremote.ai/blog/production-rag-architecture) | VERIFIED |
| 7 | The 2026 frontier shift: retrieval as agent-operated toolkit (iterative grep/glob/read + semantic + BM25), not one-shot top-k — "Direct Corpus Interaction"; Claude Code/Cursor/Devin moved away from pre-indexed vector corpora | Multiple independent 2026 posts; LightOn ColGrep ships late-interaction "stronger grep" for agents (17M/130M models, 70% win rate vs grep with Claude Code, −56% search ops) | HIGH (trend), MED (vendor numbers) | shaped.ai, sesamedisk.com DCI, [lighton.ai LateOn-Code/ColGrep](https://lighton.ai/lighton-blogs/lateon-code-colgrep-lighton) | VERIFIED |
| 8 | MUVERA (FDE) makes ColBERT-style multi-vector retrieval run at single-vector MIPS speed (~90% latency reduction on BEIR); usable Python/Rust implementations exist | Google Research algo (arXiv:2405.19504); muvera-py, muvera-rs, Weaviate/Milvus integrations | HIGH (algo), MED (production maturity outside vendors) | [github.com/sionic-ai/muvera-py](https://github.com/sionic-ai/muvera-py), lib.rs/crates/muvera-rs | VERIFIED |
| 9 | Model delta since 2026-05-29 memo: nothing changes its conclusions. Granite R2-311M (Apache-2.0, **768d, ModernBERT-based, MRL, 32K ctx**) confirmed 65.2 MTEB multilingual retrieval — architecturally the closest drop-in upgrade to gte-modernbert-base | HF blog via keepingupwith.ai; prior memo claim #7 | HIGH | [keepingupwith.ai granite R2](https://keepingupwith.ai/articles/ibm-granite-embedding-multilingual-r2-97m-and-311m-parameter-models-top-mteb-mul/) | VERIFIED |
| 10 | Qwen3-Embedding-0.6B license: prior memo says Apache-2.0; one 2026-05 SEO source says "Tongyi Qianwen custom commercial" | Conflict between sources | LOW | presenc.ai vs prior memo | CONFLICT — check HF model card before adopting |

## Key Findings

**1. emb's architecture is validated, not obsolete.** Hybrid BM25+dense+RRF+cross-encoder is exactly what 2026 production guides converged on (claim 5). The CAG and contextualize modules put emb *ahead* of most local libraries on the architecture-trend axis. The gaps are in the **dense execution layer** and the **API shape for its one real consumer**, not the design.

**2. The dense path is the one place emb is behind state of practice.** Brute-force fp32 numpy over 2.3M×768 (~7 GB) is the 2023 answer. The 2026 answer at this scale is HNSW/IVF + int8 with full-precision rescoring: ~4× less memory, ~99% recall, ms-level queries (claims 1-4). usearch fits emb's minimal-dependency character best (single package, native int8/f16, mmap views).

**3. The biggest trend shift is *who* searches.** Retrieval is becoming an agent-operated toolkit — iterative, mixed-modality (grep + semantic + structured), not one-shot top-k (claim 7). emb's consumer *is* agents (phenome search serves Claude sessions). emb already has the pieces (FTS5, dense, filters); what's missing is exposing them as composable primitives rather than one blended `search()`.

**4. Filter pushdown is a free, consumer-driven win.** phenome's `fetch_k = top_k*5` overfetch exists because emb filters *after* scoring. With brute-force (or usearch's filtered search), pre-masking candidate rows by `sources`/`entry_filter` before scoring eliminates the hack and improves quality (no K-truncation).

**5. Models/rerankers: already settled 12 days ago.** Granite R2-311M or Qwen3-Embedding-0.6B as commercial-safe text upgrades (probe first); reranker change is lowest-ROI; Jina v5 Omni lost the abstract-media eval to Gemini. Nothing in the last 12 days changes this (claim 9). New since then: LightOn LateOn-Code/ColGrep — relevant as a *pattern* (local late-interaction for agents), not as a dependency for personal-corpus search.

**6. MUVERA/late-interaction: defer.** Real and now practical (claim 8), but for a personal corpus where hybrid+rerank already hits recall targets, it adds a second index representation and model for marginal gain. Re-evaluate only if rerank-level precision is needed at zero rerank latency.

## Recommendations (priority order)

1. **ANN + int8 index backend (usearch) with exact-rescore** — `emb index` builds `.usearch` sidecar next to the npy; `SearchEngine` uses it when present, falls back to exact. Two-phase: int8 traversal → fp32 rescore of top ~200. Probe first on phenome's real index: measure current query latency + RSS vs usearch (verify-before discipline).
2. **Filter pushdown** — apply `sources`/`entry_filter` as a candidate mask *before* dense scoring (and as FTS5 WHERE for BM25). Deletes phenome's overfetch hack. No new deps.
3. **Absorb late chunking from phenome** — `LateChunker` is library code living in the consumer. Move into `emb.chunking`.
4. **Model upgrade probe** (already recommended in 2026-05-29 memo): Granite R2-311M (768d — same dim, index format unchanged) vs Qwen3-Embedding-0.6B vs current, on phenome's eval set. Add MRL-truncation support while touching this.
5. **Agent-facing search primitives** — expose `dense_search`, `bm25_search`, `grep` (FTS5 phrase/prefix), `read_entry` as separate composable calls (CLI subcommands and/or MCP server), alongside the blended `search()`. Aligns with the DCI/agentic-retrieval shift; phenome's agent callers can iterate instead of one-shot.
6. **Kill legacy JSON format + flatten cache slugs** — default-to-breaking: one format (split), one cache layout, delete the migration paths phenome compensates for.
7. **Defer:** MUVERA/multi-vector, reranker swap, learned-sparse (SPLADE).

## What's Uncertain
- Actual query latency/RSS pain at phenome's 2.3M scale — no measurement exists; rec 1's value rides on the probe.
- Qwen3-Embedding license (claim 10 conflict) — verify on HF model card.
- FAISS p95 numbers (claim 1) are single-B-grade-source; directionally consistent with claims 2-4 but don't quote them as gospel.

## Addendum 2026-06-10 — anki as second consumer; backend scope (user direction)

User: emb **will also serve anki**. Scope decisions from the follow-up:

- **Anki gets embedding, not search.** Its jobs (duplicate detection >0.85, interference pairs 0.5-0.8) are all-pairs band similarity, not query→top-k — `SearchEngine` is irrelevant. Reuse: `EmbeddingEngine` (anki's `qwen3-embedding:8b-q8_0` is already in `KNOWN_MODELS`/ollama backend — deletes its hand-rolled HTTP + manual normalization), `EmbeddingCache`, plus one new `emb pairs` utility (all-pairs similarity within a band; two concrete callers in anki). No card-search caller exists → no SearchEngine wiring.
- **Gemini backend: promoted to do.** Third `EmbeddingEngine` backend (`gemini`), absorbing phenome's `generate_gemini_embeddings.py` (real existing caller; May-29 eval keeps Gemini for abstract media; `google-genai` already an optional dep).
- **Scientific (SPECTER2-style) embeddings: skip** — per-domain spaces break cross-source score comparability in phenome's one-shared-space design; no felt pain on paper retrieval. **Gene embeddings: no** — no caller, different modality (genomics outputs indexed as text narratives).
- **agent-infra: not a consumer.** Verified no vector embeddings in corpus-core/corpus_mcp/git history; "selve" export path is legacy (selve gone). Its corpus reaches embeddings only via phenome's unified index. [DATA: grep sweep 2026-06-10]

## Correction + CAG economics 2026-06-10 (supersedes recommendation 1)

**Scale correction:** phenome's unified index is **72,045 entries** (768d, 211MB npy, 155MB jsonl ≈ 54M tokens) [DATA: `indexed/unified/metadata.json`, 2026-06-10]. The Explore agent's "2.3M entries" was wrong by ~30×. At 72K vectors, brute-force numpy is ~5ms and 211MB — **there is no dense-speed or memory problem. Recommendation 1 (usearch ANN+int8) is DEFERRED until the index approaches ~500K-1M entries.** Filter pushdown (rec 2) stands — it's a quality fix (K-truncation), not a speed fix. Claim 1's relevance was overstated by the bad scale number; the benchmark numbers themselves stand.

**"Why not just span Gemini Flash over everything?"** (user question) — verified pricing 2026-06: gemini-3.1-flash-lite $0.25/$1.50 per MTok (1M ctx), gemini-3-flash $0.50/$3, gemini-3.5-flash $1.50/$9; batch/flex −50%, context caching up to −90% on cached reads; >200K-token prompts can trigger long-context rates on Vertex [SOURCE: costgoat.com/pricing/gemini-api, openrouter.ai, metacto.com — 3 sources agree].

At 54M corpus tokens:
- **Full-corpus CAG** = ~57 shards × 1M → **~$13.5/query** list ($7 flex) and ~4-7 min at 6 workers (emb's cag.py map-reduce). vs $0 / <10ms local search. Not a default; an exhaustive-recall mode.
- **Slice CAG** (one source / time window, 1-3M tokens) = $0.25-0.75, 30-90s. This is what `--cag` post-filter already does — the right shape.
- **Hybrid read stage** (search → top-200 → ~100-300K tokens → flash-lite) = **$0.03-0.08/query**. The sweet spot.

**What people actually do, June 2026** [SOURCE: 10-source trend sweep above]: nobody spans multi-10M-token corpora per query. The consensus is **locate with a cheap index, read with a long-context model** — indexes retrieve coarser units (whole docs/sections, not chunks) and the LLM reads a fat slice; agentic loops (grep+BM25+dense+read as tools) handle the iterative cases; pure CAG is the standard only for corpora that FIT the window (<1M tokens, stable, repeat-queried → context caching makes re-reads ~90% cheaper). Long-context degradation over lexically-mismatched content keeps effective recall below nominal window size [TRAINING-DATA: NoLiMa/RULER-class results, pre-frontier but uncontradicted by the 2026 practitioner sources].

**Implication for emb:** the architecture is already the June-2026 shape. The upgrade is making CAG the *read stage* of search (search/filter → slice → span) rather than an either/or bypass — i.e., a `--read`/synthesis mode that feeds top-k or a filtered slice to flash-lite, with full-corpus spanning reserved for explicit exhaustive queries. Anki's corpus (~0.3-1.5M tokens of cards) is the one consumer that genuinely fits in-window, but its jobs are all-pairs similarity, not generation — CAG doesn't apply there either.

## Addendum 2026-06-10 (2) — Packages/SaaS landscape: is there a clear winner?

**Question:** clear best-of-breed replacing a self-owned library, or duct-tape? Verdict: **the component level has matured (no longer duct-tape), but no migration is warranted; one package and one SaaS deserve honest mention.**

1. **txtai is the closest thing to a "clear winner" in emb's exact niche** — batteries-included embedded Python search: native hybrid BM25+dense in one query, built-in RRF/convex fusion, cross-encoder reranking, SPLADE, semantic graphs, multi-provider embeddings (incl. Gemini), `pip install`, Apache-2.0, 10K+ stars. Real-world signal: nexi-lab/nexus replaced a 3,000+ LOC custom search stack with it after evaluating 15 candidates (txtai scored highest, 12/15 feature coverage) [SOURCE: github.com/nexi-lab/nexus/issues/2663]. **If starting from zero today, txtai would be the right base.** emb exists, is 2,200 LOC, has live consumers and features txtai lacks in this exact shape (freshness half-lives, spreading activation post-processors, CAG read-stage, split-index mmap format) — migration buys nothing now. Re-evaluate if emb's maintenance burden grows (dependency-evaluation discipline).
2. **Engines:** vector DBs are commodity infra; hybrid is table stakes (Weaviate best hybrid quality, Qdrant best filtering, LanceDB best embedded disk-based with FTS+hybrid+rerankers, Chroma prototyping) [SOURCE: youngju.dev 2026-05 deep dive; aifoss.dev]. All server-class options are ops overkill for one operator at 72K entries. LanceDB is the storage-layer candidate if scale ever demands real indexing (pairs with the deferred-ANN trigger).
3. **Frameworks (LlamaIndex/Haystack/LangChain): consensus says skip at this scale.** Practitioner reports converge: frameworks pay off for teams needing governance/observability; for 1-2 devs, direct pipelines are faster and safer (LangChain silently passing empty context on retrieval failure is the canonical horror story) [SOURCE: dev.to synsun 2026-03; markaicode 2026-05 "skip the framework if your team is 1-2 developers"]. Roll-your-own validated for this stack.
4. **SaaS — one genuinely notable entry: Gemini File Search API** (Nov 2025): fully managed RAG inside the Gemini API — **storage free, query-time embedding free, pay only first-index at $0.15/M tokens** + normal model tokens; supports gemini-embedding-2 incl. images; built-in citations; no audio/video [SOURCE: ai.google.dev/gemini-api/docs/file-search; blog.google 2025-11-06]. phenome's 54M tokens ≈ **$8 one-time**. Deferred alternative, surfaced explicitly: NOT for phenome (mail/messages/health to Google cloud; no freshness/spreading-activation/filter policy layer; lock-in), but **plausible for intel's corpus** (public-data analyses, ~zero sensitivity) if Phase 4's local index ever feels like overhead. OpenAI vector stores lose at personal scale ($0.10/GB/day recurring + $2.50/1K search calls).
5. **Personal-AI products: the category collapsed rather than matured.** Rewind sunset Dec 2025 (Meta acquired Limitless); Bee→Amazon; pendants dying; Screenpipe ($400, local, screen+audio only) is the lone local-first survivor; an 18-month category review concludes "not one of them does the whole job" — nothing searches mail+chats+notes+media cross-source [SOURCE: 9to5mac 2025-12-05; memx.app 2026-05-30; screenpipe.com 2026-03]. **phenome remains ahead of the commercial market; nothing obviates building.**

## Search Log
- Phase 0 dedup: found + reused `embedding-stack-upgrade-2026-05-29.md` (models + rerankers axes — skipped re-research).
- Explore agent: consumer sweep of ~/Projects (phenome sole consumer).
- Exa advanced ×3: local ANN benchmarks (10 results), retrieval-architecture trends 2026 (10 results), model recency delta since 2026-05-15 (6 results).
- Brave ×1: MUVERA implementations.
- Disconfirmation: "grep beats vector DB"/"agents don't need vector search" pieces read as the adversarial axis against embedding-centric design — conclusion: they argue for *toolkit* exposure, not for deleting the dense index; hybrid corpora (non-code, no filenames) still need semantic recall.
