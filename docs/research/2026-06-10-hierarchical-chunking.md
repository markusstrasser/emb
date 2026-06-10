# Hierarchical & Structure-Aware Chunking — 2025-2026 Frontier

Research memo for `emb`. Date: 2026-06-10. Tier: Standard-Deep.
Feeds chunking decisions for three corpora: (a) **phenome** — 72K-entry personal corpus
(chat logs, git, notes, docs, papers, social); (b) **intel** — ~2,650 markdown analysis files
organized by folder/entity; (c) **Anki** flashcards.

Source grades follow `source-grading` conventions: A = peer-reviewed / vendor-primary /
controlled benchmark; B = strong industry write-up with numbers or named eval; C = blog/marketing.
Tags: [TRAINING-DATA] from model memory; [ESTIMATED] my arithmetic; [UNVERIFIED] single weak source.

---

## TL;DR verdict

1. **The single most important 2026 result** (arXiv:2602.16974, "Beyond Chunk-Then-Embed",
   Feb 2026, Grade A — the first paper to unify all these methods on one benchmark):
   **optimal chunking is task-dependent, and the two tasks split cleanly.**
   - **In-corpus retrieval** (find the right *document/chunk* across a collection — this is `emb`'s job):
     **simple structure-based methods (paragraph, fixed-size) beat LLM-guided methods.**
   - **In-document retrieval** (needle within one long doc): LumberChunker (LLM-guided) wins.
   - **Contextualized chunking** (late chunking / contextual retrieval) helps in-corpus but
     *degrades* in-document — it encodes document-wide themes that blur intra-doc discriminability.

2. **Semantic chunking (embedding-distance breakpoints) is still net-negative or break-even**
   under realistic cost. Chroma 2024 and NAACL 2025 ("Is Semantic Chunking Worth the
   Computational Cost?", verify_claim confidence 1.0) both hold up in 2026. Multiple 2025-2026
   replications (chemistry RAG 2506.17277; hyperparameter study 2505.08445) independently find
   **recursive/fixed-size beats naive semantic** at a fraction of the cost. The exception is
   *structure-informed* chunking (Markdown headers, AST) and *LLM-informed* clustering (Chroma's
   `ClusterSemanticChunker` ~91.9% recall), not vanilla sentence-similarity breakpoints.

3. **For `emb`'s corpora the highest-leverage move is structure-derived boundaries +
   small-to-big (parent-document) retrieval, not any clever embedding trick.** All three corpora
   are already structured (markdown headers, git/chat record boundaries, one Anki card = one unit).
   Exploit that. Add **late chunking** as a cheap, training-free contextual upgrade if the embedding
   model supports it. Reserve **contextual retrieval** (LLM-per-chunk) for the intel corpus *only if*
   measured retrieval misses justify it.

4. **The LLM-decided-boundary idea: defer for in-corpus retrieval.** The 2026 benchmark is explicit
   that LLM-guided segmentation *loses* to structure-based methods for the cross-document retrieval
   `emb` does. It wins only for needle-in-one-long-document. Your corpora are already segmented by
   structure for free. Spending GPT-5.3/Flash-lite per document buys you nothing measurable on the
   task you actually have — and costs [ESTIMATED] ~$3-15 to re-segment intel, ~$30-130 for phenome.
   Use the cheap model for *enrichment* (titles/summaries for contextual retrieval) instead, where
   the benchmark evidence is positive.

---

## Q1 — Hierarchical / structure-aware chunking: state of the art

### Claims table

| Method | What it does | Index cost | Retrieval result (sourced) | Grade |
|---|---|---|---|---|
| **Fixed-size / recursive** (512 tok, 10-20% overlap) | Char/token split on separators | 1x (baseline) | Chroma: 85-90% recall @400 tok; strong default. 69% vs semantic's 54% on one set (CallSphere). "Recursive R100-0 consistently best" (chemistry RAG, 2506.17277) | A/B |
| **Semantic (embedding breakpoint)** | Embed sentences, split where cosine drops | ~3-14x | NAACL 2025: **not consistently better than fixed-size**, cost unjustified (conf 1.0). Up to +9% recall on clean prose but inconsistent | A |
| **Structure/layout-aware** (Markdown headers, AST, page) | Split on document's own boundaries | ~1x (O(n) regex) | Page-level won NVIDIA 2024 (0.648 acc, lowest variance) for paginated docs. "Single biggest, easiest improvement" when structure exists (langcopilot, B). 2602.16974: **best for in-corpus** | A/B |
| **Late chunking** (Jina, 2409.04701) | Embed whole doc with long-ctx model, pool token-vecs into chunks *after* | ~near-zero extra | +10-12% on anaphora-heavy text; BEIR gains grow with doc length; "doesn't consistently beat naive across all models/datasets" (2504.19754) | A |
| **Contextual retrieval** (Anthropic, late 2024) | LLM prepends a doc-context blurb to each chunk before embedding + BM25 | High (1 LLM call/chunk) | 35% fewer top-20 failures (embeddings); 49% +BM25; **67% +reranker** (stacked, Anthropic) | A |
| **RAPTOR** (recursive cluster + summary tree) | Build tree of abstractive summaries; retrieve across levels | High | +20% absolute on QuALITY vs prior SOTA (GPT-4). Enhanced variant +AGC: 65.5% QuALITY, -76% summary nodes (Frontiers, Jan 2026) | A |
| **Parent-document / small-to-big** | Retrieve precise small chunk, return its parent section/doc | ~1x + storage | Widely recommended 2025-2026 default for structured docs (AWS layout-aware, langcopilot, Sarthak) | B |

### What actually wins, when

- **`emb` does in-corpus retrieval** (find the right item across 72K/2,650/N entries). Per the
  decisive 2026 benchmark (2602.16974, A): **structure-based segmentation wins this regime.** The
  clever LLM-guided methods (LumberChunker) win the *other* regime (in-document needle search),
  which `emb` does not do.
- **Semantic chunking remains the over-hyped loser.** Chroma 2024 (B, the eval that started the
  "fixed-size beats semantic" meme) + NAACL 2025 (A) + two independent 2025-2026 replications all
  agree. Only *cluster-based / LLM-informed* semantic variants (Chroma's ClusterSemanticChunker
  91.9% recall) and *structure-informed* variants beat the simple baseline. Embedding-distance
  breakpoints are not worth the ~3-14x cost.
- **Late chunking** is the best *cheap contextual* upgrade: near-zero extra index cost, +10-12% on
  reference-heavy ("it/this/the aforementioned") text, but requires a **mean-pooling long-context
  embedding model** (jina-embeddings-v3, BGE-M3-long). It does NOT consistently beat naive across
  all model/dataset combos (2504.19754, A) — measure before committing.
- **Contextual retrieval** is the accuracy champion (67% fewer failures stacked) but pays an LLM
  call per chunk. Worth it only for **static, high-value** corpora where misses are expensive.
- **RAPTOR** is for **long-document cross-section synthesis** (a single book/report where the answer
  spans sections). `emb`'s units are mostly short and atomic; RAPTOR is overkill except possibly as
  a folder/entity-level *summary layer* on intel (see Q2).
- **Consensus decision rule** (AICraftGuide B, CallSphere B, firecrawl B, alexcloudstar B): build
  the boring pipeline first — **recursive ~512-token + structure-aware splitting + hybrid (dense+BM25)
  + reranker** — instrument it, and only add late/contextual when your own evals prove the boundary
  is costing answers. "Chunking in 2026 rewards restraint."

---

## Q2 — Exploiting existing file / folder / entity structure

This is `emb`'s strongest lever and the literature is unambiguous.

- **Structure-derived boundaries beat computed boundaries when structure exists.** Markdown headers
  (`MarkdownHeaderTextSplitter`), then recursive-split any over-long section, is "often the single
  biggest and easiest improvement you can make" (langcopilot, B). Azure AI Search ships a
  Document-Layout skill doing exactly this (Microsoft Learn, A-vendor). Tools: `chunkweaver`
  (stdlib, regex header/table detectors, beats naive 600-char 15-4 wins p=0.019, B), `semchunk` +
  Kanon-2 AI mode (+12% over fixed-size on Legal RAG, B-vendor), `chonky` (neural paragraph splitter,
  strips markup first). For code: AST-aware splitting (split on function/class boundaries).
- **Parent-document / small-to-big retrieval is the canonical pattern for hierarchical corpora.**
  Index the small chunk for precision; at retrieval, return the parent section/document/file for
  context. AWS's reference build stores a **chunk → section → chapter** (grandparent) hierarchy as
  metadata and expands on demand (B-vendor). LangChain `ParentDocumentRetriever` is the off-the-shelf
  impl. Pinecone calls it "chunk expansion" and recommends it to keep retrieval low-latency while
  preserving context (B-vendor).
- **Multi-granularity indexes are now a named research direction.** FreeChunker (arXiv:2510.20356,
  Oct 2025 → v2 Feb 2026, Grade A) treats sentences as atomic and lets the retriever assemble
  *arbitrary sentence combinations* on demand — no fixed boundary at all — and reports best avg
  retrieval + best time-efficiency on LongBench V2. MultiDocFusion (EMNLP 2025, A) does hierarchical
  + multimodal section-header trees. This is the "chunk + section + doc" idea formalized.
- **Metadata-filtered hierarchy is free and high-value.** Attach `{folder, entity, doc_type, header_path,
  section}` to every chunk and filter at query time. Sarthak (B) recommends type-routing: classify
  doc by type → route to the right chunker → store consistent metadata → type-aware retrieval. For
  intel's entity-keyed docs this directly enables entity-scoped search.

**Implication for `emb`:** all three corpora come pre-segmented. Markdown headers (intel),
record/message boundaries (phenome chat/git), and one-card-one-chunk (Anki) ARE your chunk
boundaries — don't recompute them. Layer parent/metadata retrieval on top.

---

## Q3 — LLM-decided chunk boundaries (the user's proposal): explicit verdict

**Verdict: do NOT use a cheap LLM to decide chunk boundaries for `emb`. Use it for chunk
*enrichment* instead.** The evidence is specifically against the proposed use:

- **The decisive benchmark says LLM-guided segmentation loses on `emb`'s task.** 2602.16974 (A):
  "simple structure-based methods outperform LLM-guided alternatives for **in-corpus retrieval**."
  LumberChunker (the canonical LLM-boundary method, EMNLP 2024, +7.37% DCG@20 on GutenQA) only wins
  **in-document** needle search. `emb` does in-corpus retrieval. The user's intuition ("cheap
  intelligence picks better boundaries") is correct *for the wrong task*.
- **It is done in practice, with named tools** — so the idea isn't crazy, just mis-targeted:
  - **LumberChunker** (2024) — iteratively prompts an LLM to find content-shift points.
  - **LGMGC** (2025) — uses LLM EOS-token logits for split points (cheaper, no generation).
  - **TopoChunker** (arXiv:2603.18409, 2026, A) — agentic, routes only complex docs through a VLM;
    R@3 83.26%, beats LumberChunker, **23.5% cheaper** by not LLM-processing simple structure.
  - **semchunk AI mode / Kanon-2** (2026, B) — LLM-derived structural spans, +6% over vanilla semchunk.
  - **AutoChunker** (ACL 2025 Industry, A) — LLM structural chunking + noise elimination.
  - The recurring 2026 finding (TopoChunker, AutoChunker): **don't apply LLM reasoning uniformly** —
    it's wasteful on structurally-simple text. `emb`'s corpora are structurally simple/pre-segmented.
- **Cost math** ([ESTIMATED], Gemini Flash-lite ~$0.10/1M in, GPT-5.3-class cheap tier similar order):
  - LumberChunker-style needs to feed each doc through the model (~1.65x the doc's tokens, per
    TopoChunker's measurement of LumberChunker overhead).
  - **intel** (~2,650 files, est. ~1.5K tokens avg = ~4M tokens × 1.65 ≈ 6.6M tokens) ≈ **$0.7-7**
    one-time depending on tier — *cheap in absolute terms*, but buys ~0 on in-corpus retrieval.
  - **phenome** (72K entries, est. ~300 tokens avg = ~22M tokens × 1.65 ≈ 36M) ≈ **$3.6-36** one-time.
    Most phenome entries are *already* atomic (one message, one commit, one card) — boundary
    detection is moot.
  - Verdict: cost is not the blocker; **the retrieval-quality gain is the blocker (≈ zero on this task).**
- **Where the cheap LLM IS worth paying:** **contextual retrieval enrichment** — one cheap call per
  chunk/section to generate a 1-2 sentence "where this fits in the document" blurb, prepended before
  embedding. Same per-chunk cost, but here the benchmark evidence is *positive* (35-67% fewer failures,
  Anthropic A). Anthropic prompt-caching the document makes this materially cheaper. For intel's
  ~2,650 high-value analysis files this is the defensible LLM spend, not boundary-picking.

---

## Q4 — Long-context & late-interaction: does chunking become obsolete?

**No. Chunking is still load-bearing for `emb`, and arguably more so at 72K entries.**

- **Long-context "just read the whole doc" does not scale to a corpus.** alexcloudstar (B, Apr 2026):
  long-context cost scales linearly with data; at ~20M tokens (real KB size) it's "architecturally
  impossible." Needle-in-haystack accuracy still drops 10-20% between 50k-500k tokens even on 2026
  models. The honest 2026 production pattern is **RAG to narrow → long-context to reason** (retrieve
  top-k full items at natural boundaries, hand 100K tokens to the model). Even Jina's own late-chunking
  paper notes long-context embedders still perform *better on short texts* → "chunking generally
  improves retrieval even with models that support long contexts" (A). Lost-in-the-middle persists
  (Pinecone, B).
- **Late interaction / ColBERT (multi-vector) is complementary, not a chunking-killer.** ColBERTv2 /
  Jina-ColBERT-v2 / PLAID store per-token embeddings and score via max-sim. They **still operate on
  chunks** — they change the *embedding granularity*, not the need to segment. Wins specifically on:
  **code corpora** (identifier matches inside semantic paragraphs), **mixed code+prose**, **multilingual**,
  and **very long chunks** where pooled vectors lose discriminative power (+8-14 recall points,
  futureagi B). Cost: 4-10x index storage (PLAID → 2-3x at recall parity). **Directly relevant to
  phenome's git/code content** — keep as an option for the code-heavy slice.
- **When chunking is still load-bearing:** large corpora (cost/scale), precision retrieval, exact-term
  matching, anything where you can't afford to read everything per query. That is exactly `emb`.
- **When chunking is genuinely obsolete:** single short document fully in-window, one-off Q&A, no corpus.
  Not `emb`'s situation.

---

## Concrete recommendation for `emb`'s three corpora

**Shared substrate (build once):** structure-derived boundaries + parent/metadata retrieval +
hybrid (dense + BM25) + a reranker. This is the "boring pipeline" every 2026 source converges on,
and the 2026 benchmark says it's the *right* choice for in-corpus retrieval. Defer everything clever
until an eval (use Chroma's `chunking_evaluation` harness, MIT, off-the-shelf) proves a specific
boundary is costing answers.

| Corpus | Chunk unit (structure-derived) | Granularity strategy | LLM use | Notes |
|---|---|---|---|---|
| **phenome** (72K) | Native record: one message / commit / note / card; papers split by section header then recursive-512 | Small-to-big: index the record, return thread/file as parent. Consider **late chunking** for long notes/papers (needs long-ctx embedder). **ColBERT** for the git/code slice. | None for boundaries. Optional enrichment only on long papers. | Most entries already atomic → boundary detection moot. Heterogeneous → **type-route** (chat/git/note/paper/social) to per-type chunker. |
| **intel** (~2,650 md) | `MarkdownHeaderTextSplitter` (header_path metadata) → recursive-split over-long sections | Multi-granularity: chunk + section + file. Entity/folder as filterable metadata for entity-scoped search. Optional RAPTOR-style folder/entity summary layer for cross-doc synthesis. | **This is where LLM spend is justified**: contextual-retrieval enrichment (1-2 sentence doc-context blurb per section, prompt-cache the file). Highest-value, static, well-structured. | Markdown headers are free, reliable boundaries — the literature's ideal case. |
| **Anki** | One card = one chunk (front+back). Never split. | Flat index + deck/tag metadata. | None — cards are pre-authored atomic units. | Boundary/semantic/LLM chunking all irrelevant here. |

**Sequencing:** (1) ship structure + parent + hybrid + reranker for all three; (2) add late chunking
to phenome long-form if the embedder supports mean-pooling long context; (3) add contextual-retrieval
enrichment to intel *only* if eval shows misses; (4) add ColBERT to phenome's code slice if pooled
embeddings underperform on identifier queries. **Skip:** vanilla semantic (embedding-breakpoint)
chunking and LLM boundary-picking entirely.

---

## Sources (grades)

- arXiv:2602.16974 "Beyond Chunk-Then-Embed" (Zhou et al., Feb 2026) — **A**, the decisive unifying benchmark.
- arXiv:2410.13070 / NAACL-Findings 2025 "Is Semantic Chunking Worth the Computational Cost?" (Qu/Tu/Bao) — **A** (verify_claim conf 1.0).
- Chroma "Evaluating Chunking Strategies" (Smith & Troynikov 2024) + `chunking_evaluation` repo — **B** (verify_claim conf 0.95).
- arXiv:2409.04701 Late Chunking (Jina) — **A**. arXiv:2504.19754 late-vs-contextual comparison — **A**.
- Anthropic Contextual Retrieval (late 2024) numbers via Atlan/AICraftGuide — **A** (primary)/**B** (relay).
- RAPTOR +20% QuALITY [TRAINING-DATA, corroborated by Atlan B]; Frontiers RAPTOR+AGC (Jan 2026) — **A**.
- arXiv:2510.20356 FreeChunker (cross-granularity) — **A**. arXiv:2603.18409 TopoChunker — **A**. EMNLP 2025 MultiDocFusion — **A**. ACL-Industry 2025 AutoChunker — **A**.
- LumberChunker EMNLP-Findings 2024 — **A**. arXiv:2506.17277 chemistry RAG, arXiv:2505.08445 hyperparameter study — **A** (replications).
- RAG-vs-long-context (alexcloudstar), ColBERT/late-interaction (futureagi), decision rules (CallSphere, firecrawl, langcopilot, Weaviate, Pinecone, AWS layout-aware, Azure Doc-Layout, chunkweaver, semchunk/Kanon-2) — **B/C** industry.
