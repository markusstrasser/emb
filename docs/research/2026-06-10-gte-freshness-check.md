# gte-modernbert Freshness Check — Local Text Embeddings for intel markdown corpus

Date: 2026-06-10. Tier: Quick-to-Standard freshness check (NOT a full survey).
Builds on (do not re-read unless needed):
- `phenome/docs/research/embedding-stack-upgrade-2026-05-29.md` (model landscape, 2026-05-29)
- `emb/docs/research/2026-06-10-hierarchical-chunking.md` (chunking, today)

Usecase: ~2,650-file / ~32.5K-chunk **English** markdown analysis corpus (intel: financial/entity
dossiers, memory notes). Text-only queries. Hybrid dense+BM25+RRF+rerank. markdown-header chunking +
parent/metadata. Current model: `Alibaba-NLP/gte-modernbert-base` (768d, ~150M, local, ST).
Constraints: local, ≤500M params preferred, Apache/MIT license.

Source grades: A = vendor-primary card / arXiv / controlled benchmark; B = strong write-up w/ numbers; C = blog.

---

## TL;DR

1. **Nothing in the last ~2 weeks beats the sub-500M English leaders.** The only genuinely-new
   release in-window is **Granite Embedding *Multilingual* R2** (311M/97m, blog 2026-05-14, Grade A) —
   but that's a *multilingual* model and is the wrong tool for an English-only corpus. The *English*
   Granite R2 (`granite-embedding-english-r2`, 149M) has actually been out since **Aug 2025** — the
   prior memo's framing ("new") was slightly off. No new sub-500M English-retrieval SOTA appeared.

2. **Prior memo correction (load-bearing):** it recommended Granite **R2-311M (65.2)** — that 65.2 is
   the **multilingual** retrieval score. For English-only data, IBM's own card says use the **English**
   model (`granite-embedding-english-r2`, 149M, 768d) instead, "since it doesn't need to allocate
   capacity across 200+ languages." Right pick for this corpus = the 149M English model, not the 311M ML one.

3. **Single best model for THIS usecase = a near-tie; re-index is NOT clearly worth it.** On the
   standard **BEIR retrieval** suite, the *current* `gte-modernbert-base` scores **55.33**, which
   actually **edges** `granite-embedding-english-r2`'s **53.1 BEIR(15)** (both from vendor cards). Granite
   wins on the broader **MTEB-v2 aggregate (62.8)** and on long-doc/RAG sub-suites (MLDR 40.7, MTRAG
   56.7), but those gains don't cleanly map to short-chunk in-corpus dense retrieval. **gte-modernbert
   is not "mid-pack" for pure retrieval — it's competitive with the best small English models.**

4. **Verdict: do NOT re-embed for a model swap alone.** The retrieval-quality delta over gte-modernbert
   is marginal-to-ambiguous (BEIR slightly favors gte; MTEB-aggregate slightly favors Granite — both
   well within "depends on your corpus" noise). For a personal 32.5K-chunk corpus already running a
   rerank stage (which absorbs most small-model dense-retrieval gaps), the re-index cost isn't justified
   by leaderboard points. Spend the effort on **contextual enrichment** instead (positive evidence; see Q4).

---

## Q1 — Anything NEWER than 2026-05-29?

| Model | Released | Params | Dim | License | English retrieval | New? | Grade |
|---|---|---|---|---|---|---|---|
| **Granite Embedding Multilingual R2** (311m / 97m) | blog 2026-05-14 | 311M / 97M | 768 / 384 | Apache-2.0 | 65.2 **multilingual** retrieval (#2); NOT English-tuned | YES (in-window) but multilingual | A |
| `granite-embedding-english-r2` | **Aug 15 2025** | 149M | 768 | Apache-2.0 | BEIR 53.1, MTEB-v2(41) 62.8, MLDR 40.7 | No (older than memo implied) | A |
| Qwen3-Embedding-0.6B | Jun 2025 | 0.6B (>500M) | up to 1024 | Apache-2.0 | MTEB-v2 ~64.3 agg, retrieval 64.7 | No | A |
| EmbeddingGemma-300m | 2025 | 300M | 768 (MRL) | Gemma terms (not Apache/MIT) | ~60-63, "doesn't rank high for its size" | No | B |
| GeeVec-Embeddings-1.0-Lite | ~Apr 2026 | 366M (PseudoMoE) | 256-4096 | check card; `trust_remote_code` | 74.66 **MMTEB multilingual** nDCG@10 (SOTA <1B) | seen, but multilingual + remote-code | B |
| pplx-embed(-context)-v1-0.6B | paper Feb 2026 | 0.6B (>500M) | 1024 | check card | int8-native, contextual variant; web-scale tuned | seen | A(paper) |

**Conclusion:** No new model in the last ~2 weeks displaces the established sub-500M English leaders.
The headline "new" item (Granite **Multilingual** R2) is multilingual and irrelevant to an English corpus.
For sub-500M **English** retrieval, the field as of 2026-06-10 is unchanged from 2026-05-29:
`granite-embedding-english-r2` (149M) and the current `gte-modernbert-base` (150M) are the contenders;
Qwen3-Embedding-0.6B is the stronger model but **exceeds the ≤500M preference** (it's 0.6B).

---

## Q2 — Single best model for THIS usecase (English md, local, ≤500M, Apache/MIT)

**Recommendation: stay on `gte-modernbert-base`. If you ever rebuild from scratch, pick
`ibm-granite/granite-embedding-english-r2` (149M, 768d, Apache-2.0) — but it is a lateral move, not an upgrade.**

Concrete MTEB-v2 / BEIR English-retrieval numbers (vendor cards, Grade A):

| Model | Params | Dim | BEIR retrieval (15) | MTEB-v2 (41 agg) | License | Ctx |
|---|---|---|---|---|---|---|
| `gte-modernbert-base` (current) | ~150M | 768 | **55.33** | — (mid-60s agg) | Apache-2.0 | 8192 |
| `granite-embedding-english-r2` | 149M | 768 | 53.1 | **62.8** | Apache-2.0 | 8192 |
| `granite-embedding-small-english-r2` | 47M | 384 | 50.9 | 61.1 | Apache-2.0 | 8192 |
| Qwen3-Embedding-0.6B (>500M) | 0.6B | ≤1024 (MRL) | ~64.7 | ~64.3 | Apache-2.0 | 32K |

Why not the others, for this specific corpus:
- **Qwen3-Embedding-0.6B** is the genuine quality step-up (~+9 BEIR over both 150M models) but it
  **breaks the ≤500M constraint** (0.6B, last-token pooling, heavier on M3). It's the right pick *only*
  if you relax the size budget and decide retrieval quality is a felt pain. It is NOT a "drop-in" (1024d
  vs 768d → full re-index + index-dim change).
- **EmbeddingGemma-300m** — non-Apache/MIT (Gemma license) and "doesn't rank high for its size"; skip.
- **GeeVec-Lite / pplx-embed** — multilingual / `trust_remote_code` / >500M / contextual-RAG-tuned;
  added maintenance surface, no clean English-retrieval win at ≤500M. Skip for this corpus.

---

## Q3 — Is re-embedding worth it?

**No — not for a model swap.** Honest magnitude:

- gte-modernbert → granite-english-r2: **BEIR -2.2 (gte wins), MTEB-v2-agg +~? (Granite wins)** — i.e.
  **directionally ambiguous and small**. This is a lateral move, not an upgrade. Granite's real edges
  are CoIR (code), MLDR (long-doc), MTRAG (RAG-multiturn) — none central to short-chunk dense retrieval
  over markdown dossiers.
- gte-modernbert → Qwen3-0.6B: **~+9 BEIR**, a real upgrade — but costs the ≤500M budget, a 768d→1024d
  index rebuild, and slower local inference. Only worth it if retrieval misses are an *observed* pain.
- **Your rerank stage masks most of the gap.** A cross-encoder reranker recovers most of the
  small-model dense-retrieval shortfall (the Granite/gte BEIR delta is ~2 points pre-rerank). With a
  reranker already in the pipeline, swapping a 150M dense model for another 150M dense model is
  near-invisible end-to-end.
- 32.5K chunks re-embeds in minutes on M3, so cost isn't the blocker — **the absence of a quality delta is.**

**Decision rule:** keep gte-modernbert. Re-index ONLY if (a) you relax to Qwen3-0.6B for a real ~+9 BEIR
jump AND have measured retrieval misses, or (b) you're rebuilding the index for another reason anyway
(then granite-english-r2 is a fine, equally-licensed, equally-dimensioned choice — flip a coin).

---

## Q4 — Chunking: anything new since today's memo?

Nothing that overturns it. Two in-window data points seen, both *confirming* the memo's direction:
- **pplx-embed-context-v1** (Perplexity, paper Feb 2026): a *contextual* embedding variant "for document
  chunks in RAG where surrounding context matters" — this is the **contextual-enrichment** lever the
  chunking memo already flagged as the positive LLM use, now shipped as a model rather than a pipeline
  step. Validates "enrich, don't re-segment."
- **vstash / bge-small-rrf-v3** (arXiv:2604.15484): local-first hybrid retrieval with adaptive RRF
  fusion — same dense+BM25+RRF shape this corpus already runs; no new chunking idea, just fusion tuning.

**Net:** the highest-leverage chunking move for this corpus remains what the 2026-06-10 memo said —
markdown-header boundaries + parent/metadata retrieval, plus optional **contextual-retrieval enrichment**
(1-2 sentence "where this fits" blurb per section, prompt-cached per file) on these high-value, static
dossiers. No re-chunking needed; no new method beats structure-derived boundaries for in-corpus retrieval.

---

## Sources
- IBM Granite R2 cards (Grade A): huggingface.co/ibm-granite/granite-embedding-english-r2 (BEIR 53.1,
  MTEB-v2 62.8, release Aug 15 2025); huggingface.co/blog/ibm-granite/granite-embedding-multilingual-r2
  (2026-05-14, 311m=65.2 multilingual retrieval).
- gte-modernbert-base BEIR 55.33 — huggingface.co/Alibaba-NLP/gte-modernbert-base (via verify_claim conf 0.9, Grade A vendor card).
- CodeSOTA MTEB table updated 2026-05-17 (B); Ailog MTEB 2026 (Jan/May, B); Modal MTEB blog (B).
- Qwen3-Embedding-0.6B Apache-2.0, MTEB-v2 ~64.3 — CodeSOTA / Qwen card (A).
- pplx-embed-context-v1 (arXiv:2602.11151, A); vstash bge-small-rrf-v3 (arXiv:2604.15484, B); GeeVec-Lite card (B).
- verify_claim ×3 (Granite/Qwen size+rank; gte BEIR vs Granite). Note: a verify_claim "contradicted"
  verdict flagged that Qwen3-0.6B exceeds 500M (correct) and that vendor MTEB numbers vary by task
  selection (correct caveat — treat all single-number rankings as ±1-2 noise).
