# emb — Maintainer Handoff & Architecture

**Last updated:** 2026-06-11 eve (rerank program DONE — rerank is opt-in-after-eval, default reranker gte @96bc8e8; bakeoff audited to /eval standard, invariant claims in EXPERIMENT.md §6b; external positioning memo added; phenome eval is the open item) · **Audience:** the next agent/maintainer of `emb` and its consumers.
**Stance:** breaking changes welcome; no backward-compat shims, no legacy wrappers. Keep the
representation deep and the surface small. This doc is living — update it when the design moves.

---

## 0. What emb is (one paragraph)

`emb` is ~2,500 LOC of *glue* over big-team engines (sentence-transformers/HuggingFace, SQLite
FTS5, numpy, google-genai). It is **policy over mechanism**: the engines do the ML; emb owns the
*retrieval policy* — exact pre-filtering, content-hash incremental caching, model-space safety,
hybrid fusion, freshness decay, spreading activation, passage-windowed reranking, a long-context
read stage, and all-pairs similarity. The whole point is that "harness ongoing work" already
happens at the engine layer; emb is the thin, correct, owned policy on top. (This framing is why
txtai was evaluated and rejected — see §5.)

---

## 1. Architecture (layers, data flow, invariants)

```
Entry(schema.py)                # id, text, source, title, date, metadata, embedding, content_hash
   │
EmbeddingEngine(embed.py)       # backends: sentence-transformers (gte, local) · ollama · gemini (multimodal)
   │   ├─ embed_texts()         # one Content per text for gemini (a list[str] FUSES to one vector — Phase-0 trap)
   │   ├─ embed_media()         # (description, [(data,mime),…]) → native multi-part fusion (gemini only)
   │   ├─ _l2_normalize()       # CENTRAL INVARIANT: every vector emb returns is unit-norm, all backends
   │   └─ EmbeddingCache        # content-hash → vector; incremental; namespaced by model_slug (space safety)
   │
Split index (index.py)          # dir/ = entries.jsonl + embeddings.npy (mmap) + metadata.json
   │                            # legacy monolithic JSON is READ-only via `emb convert`; embed never writes it
SearchEngine(search.py)
   │   ├─ dense                 # brute-force  sims = E @ q  (exact; ~5ms at 72K; NO ANN until >500K)
   │   ├─ filter BEFORE rank    # sources/entry_filter/min_sim/since build `valid` pre-ranking (exact)
   │   ├─ BM25 (FTS5)           # pushdown: valid_rowids via json_each → no starvation on narrow filters
   │   ├─ fusion                # 'rrf' (default) | 'convex' (min-max normalized α·dense+(1-α)·bm25)
   │   ├─ freshness             # per-source exponential half-life
   │   ├─ NeighborIndex         # spreading-activation post-processor (generic key_extractor)
   │   └─ rerank                # passage-windowed MAX-POOL cross-encoder (BERT-MaxP) — see §3
   │
CAG read-stage (cag.py)         # `emb read`: hybrid-retrieve top-k → span gemini-flash-lite over hits
pairs (pairs.py)                # `emb pairs`: all-pairs similarity, upper-triangle streaming, det. top-K
```

**Load-bearing invariants — do not break without reading why:**
1. **L2-normalization is central** (`embed._l2_normalize`). Cosine == dot product everywhere. Tests assert ‖v‖≈1 per backend.
2. **Caches are namespaced by `model_slug`** (`cli.py`, `EmbeddingCache` dir layout). gte-768 and gemini-768 share a *dimension* but are different *spaces*; a dim check can't tell them apart, so the slug dir prevents silent wrong-vector cache hits.
3. **Filters run BEFORE ranking** (`search.py` builds `valid` first). Dense is exact at this scale. The ONLY truncation surface is the BM25 leg — fixed by the json_each rowid pushdown. (phenome's old `fetch_k=top_k*5` "fix" was deleted; it rested on a wrong "filter runs post-retrieval" comment.)
4. **Query encoding routes through the index's backend** (`_encode_query` reads `metadata['embedding_model']`). Unknown model → hard error, never a silent default into a foreign space.
5. **FTS rowid == entry enumerate index** — the pushdown's `rowid IN (valid)` relies on this; documented at `_bm25_search`.

---

## 2. Consumers (who uses what)

| Consumer | Uses | Index / model | Notes |
|---|---|---|---|
| **phenome** (full API) | search + rerank + read + freshness + spreading | text: gte-768; media: gemini-768 (**separate** spaces, separate indexes) | Heaviest reranker user → got the §3 windowing fix, **but ⚠ rerank measured NET-NEGATIVE on intel (9/10→3/10 topical)** — phenome needs an own-domain rerank-on/off eval before trusting `rerank=True`; biomedical chunks may behave differently, don't assume either way. Got BM25 pushdown; `fetch_k` hack + LateChunker deleted. |
| **anki** (embed + pairs only) | EmbeddingEngine(gemini) + EmbeddingCache + `find_pairs` | gemini-768 multimodal | Type-routed cards (text-only / multipart / occlusion→base-image). dedup ≥0.85, interference 0.74–0.85 (gemini-calibrated). **Uses pairs, not rerank → unaffected by §3.** |
| **intel** (search via MCP) | SearchEngine hybrid (NO rerank) via 7th MCP tool `semantic_search` | gte-768, 32,084 chunks | Header-aware chunking (code-fence-safe), 52 scaffolding files excluded. **hybrid-only → §3 doesn't touch it in production.** Needle lookups should route to the DuckDB FTS *entity* tools, not semantic_search. |

agent-infra is **not** a consumer (the old `selve` integration was retired).

---

## 3. The reranker (just rebuilt — understand this)

`_rerank` was scoring `f"{title} {text[:500]}"` — blind to anything past char 500. Measured cost
(2026-06-10 bake-off): needle recall@5 was **1/10 with rerank vs 5/10 without** — rerank actively
*demoted* docs whose match lay deeper than 500 chars.

**Now: passage-windowed max-pool (BERT-MaxP).** Each candidate's full chunk → overlapping windows
(`passage_windows`, 1200-char / 800-stride, title prepended to each, ≤12 windows/doc); score the
query against every window; **the doc takes its MAX window score**. Relevant content anywhere
surfaces. Max-pool is ≥ the old single-window approach on the topical case too (it considers
strictly more of the doc). `CrossEncoder(max_length=512)` so a window is never internally truncated.

**Trade-off:** more (query, window) pairs → higher rerank latency on long docs (bounded by the
12-window cap). For a personal lib the quality win dominates; if latency bites, lower
`RERANK_MAX_WINDOWS` or pre-cap candidate text. Tunable, no code change needed.

**✓ Empirically validated 2026-06-11** (`evals/retrieval_backend_bakeoff`, both lanes):
needle recall@5 **1/10 → 5/10** — the truncation harm is gone, and the windowing mechanism
demonstrably rescues buried content (3 rescues across lanes that no-rerank missed).

**⚠ But the rerank STAGE is now measurably net-negative on intel-style corpora** — this is the
model+pool, not the windowing: topical lane **hybrid 9/10 → rerank 3/10** (7 demotions, 1 rescue).
Across all 20 queries: 3 rescues, 9 demotions. Recall below retrieval-alone is exactly the
"Drowning in Documents" (arXiv:2411.11767) regime — a 0.6B cross-encoder re-sorting a 100-doc
pool of confusable siblings discards the RRF dual-leg rank evidence in favor of one confident
window score. **The fix program RAN 2026-06-11** (sweep: both lanes × {qwen3, gte-windowed,
gte-native-8192} × pools {25,50,100} over frozen drift-guarded pools; hermetic candidate
snapshot in `evals/.../runs/2026-06-11/pool_snapshot.json.gz`): **no config beat hybrid on both
lanes** → `rerank=True` is **opt-in-after-eval**, never free insurance. But gte beat qwen3 in
every cell → `DEFAULT_RERANKER` is now `gte-reranker-modernbert-base` (96bc8e8; ¼ params,
ONNX-able, Qwen prefix conditional). Windowed MaxP retained — gte-native loses the deep-needle
rescues (6 vs 7 @100; long-doc dilution, frontier-memo Q1 confirmed). Known-good opt-in for
needle-heavy use: gte windowed @100 → needle 7/10, best local measured (FS: 8/10); gte@25 is
net-even with hybrid at near-free latency. Full grid: `evals/retrieval_backend_bakeoff/EXPERIMENT.md` §4c.

---

## 4. Decisions register (settled — don't re-litigate; cite these)

| Decision | Verdict | Recorded in |
|---|---|---|
| Reranker model + default | **`gte-reranker-modernbert-base`, opt-in only** — beat Qwen3-0.6B in every sweep cell; NO rerank config beat hybrid-alone on intel (bar: both lanes) → `rerank=True` is opt-in-after-eval. Windowed MaxP kept (native-8192 loses deep needles). | evals `DECISIONS.md` → `emb-rerank-default`; `retrieval_backend_bakeoff/EXPERIMENT.md` §4c |
| Text embedding model | **Keep `gte-modernbert-base`** — competitive-with-best on English retrieval (BEIR 55.3 edges granite-en-r2 53.1); re-embed not worth it. Qwen3-Embedding-0.6B is the only real upgrade but breaks ≤500M + forces 768→1024d. | `docs/research/2026-06-10-gte-freshness-check.md` |
| Multimodal model | **`gemini-embedding-2-preview`** — won the abstract-media two-judge eval over Jina v5 Omni. | `phenome/docs/research/embedding-stack-upgrade-2026-05-29.md` |
| Adopt txtai? | **No.** Glue-inversion (we already harness the engines); bus-factor-1; its SQL+vector is post-filter (ships phenome's worst historical bug as design); litellm route (evicted). Cherry-picked convex fusion only. | `docs/research/2026-06-10-txtai-deepdive.md`; agent-infra `decisions/2026-06-10-dependency-glue-inversion.md` |
| File Search for intel? | **No (provisional).** FS wins *needle* recall (8/10 vs emb's 5/10) but loses on $0/offline/16×-faster + topical parity; not enough to flip a free incumbent on one query class. | `~/Projects/evals/DECISIONS.md` → `intel-search-backend`; `~/Projects/evals/retrieval_backend_bakeoff/EXPERIMENT.md` |
| Chunking strategy | **Structure-derived** (markdown headers) + parent/metadata. REJECT LLM-decided boundaries for in-corpus retrieval (arXiv:2602.16974). Cheap-LLM spend goes to *contextual enrichment*, not boundary-picking. | `docs/research/2026-06-10-hierarchical-chunking.md` |
| ANN index? | **No** at 72K (brute-force ~5ms exact). Trigger: >500K entries OR measured dense >100ms. | `.claude/plans/88ec9920-emb-elevation.md` (Deferred table) |

---

## 5. The eval that drove the rerank fix (so the next model trusts the number)

Bake-off: `~/Projects/evals/retrieval_backend_bakeoff/` (canonical; not in emb).
- **v1 A/B** used title-matching queries (query terms = filename) → both 12/12 → *false parity*. A softball.
- **v2 hard eval** (paraphrased, keyword-stripped, distractor slice): **FS 8/10 · emb-hybrid 5/10 · emb+rerank 1/10**.
- The 5→1 was the 500-char rerank truncation → §3 fix.
- Caveat at the time: v2 queries skewed *tangential-needle* (FS's home turf). **Resolved 2026-06-11**:
  the topical lane ran (emb-hybrid 9/10 · FS 6/10 · kw-baseline 0/10) — "parity" was wrong in emb's
  FAVOR; and the rerank conditions were re-measured post-fix (EXPERIMENT.md §4b, §4c).
- Method lesson now enforced: `evals/CLAUDE.md` requires a discrimination check (trivial-pass baseline + a should-fail case) before any comparison eval.
- **Audited to /eval standard 2026-06-11** (review mode): SCREENING power declared (N=10/lane,
  Wilson ±0.227 — verdicts rest on ranks/cell-sweeps/structure, never one margin), prereg
  provenance mapped to pre-scoring commits, 5 invariant claims extracted with transfer status
  (EXPERIMENT.md §6b), deviations consolidated (§7). External positioning: we run the private
  instantiation of what RTEB/FreshStack/EnterpriseRAG-Bench simulate publicly — see
  `docs/research/2026-06-11-benchmark-positioning.md`.

---

## 6. Open items (next steps, with exact commands)

*(2026-06-11: the bake-off is DONE — needle 5/5/8, topical 9/3/6 for hybrid/rerank/FS; verdict
"hold local emb, unconditional" landed in `evals/DECISIONS.md`. What remains is the rerank
program the results demand.)*

*(2026-06-11 later: items 1–2 — the pool sweep and the swap eval — RAN; outcome in §3 and
`evals/.../EXPERIMENT.md` §4c. Default reranker swapped to gte (96bc8e8); rerank stays opt-in.)*

1. **phenome rerank-on/off eval** (own domain, own queries) — phenome reranks by default and the
   intel sweep says that's unvalidated insurance (no config beat hybrid there). The method
   transfers directly: freeze phenome pools via a `sweep_rerank.py`-style pools stage, snapshot
   candidate texts, score gte-windowed @{25,100} vs hybrid. Until run, phenome's `rerank=True`
   is a belief, not a measurement.
2. **Opt-in latency extras, only if a consumer actually opts into rerank**: window pruning
   (top-3 windows by dense score — quality-positive per EviRerank) and ONNX INT8 on the gte
   encoder (~2× CPU). Don't build ahead of a consumer. Evidence:
   `docs/research/2026-06-10-reranker-frontier-check.md`.
3. **anki interference band** (0.74–0.85) is gemini-calibrated provisionally — validate against real
   confusion pairs over time; re-tune if the model changes.
4. **(Optional, deeper)** File Search's only durable edge is *chunk granularity*. If needle retrieval
   becomes important for a consumer, the right emb answer is finer/multi-granularity chunking
   (chunk + section + doc), not adopting FS. Don't build until a consumer needs it.

---

## 7. Operational safety (LEARNED THE HARD WAY — read before running anything heavy)

On 2026-06-10 three parallel torch eval jobs (each: embedding model + cross-encoder over a full
index, with `PYTORCH_ENABLE_MPS_FALLBACK=1` spilling oversized tensors to CPU RAM) hit ~44 GB of
demand on an **18 GB** Mac (`sysctl hw.memsize`; an earlier version of this doc wrongly said 36 GB)
→ OOM freeze → forced reboot. Same evening, a SECOND incident refined the lesson: a single
well-behaved torch eval (no MPS fallback, ~7 GB) was evicted to swap mid-run by **aggregate agent
load** — 8 concurrent claude sessions + node tooling — and stalled unrecoverably in uninterruptible
swap-thrash (RSS 10 MB, state U; SIGKILL queued for minutes). "Quiet machine" means quiet of agent
sessions too, not just torch jobs. Guards now in place:

- **Never run >1 local model/rerank job at once.** `pretool-heavy-load-guard.sh` BLOCKS a new heavy
  job when any python is already resident >8 GB (name-independent RAM signal).
- **Never set `PYTORCH_ENABLE_MPS_FALLBACK=1`** on these jobs — it silently spills to RAM. The hook
  warns; drop it so an oversized tensor errors cleanly. (gte/Qwen on CPU is slow but safe.)
- **`emb.embed.require_ram()`** refuses to load a model when RAM is critically low (gte 1.5 GB,
  reranker 2.5 GB) — a clear error beats a silent SIGKILL.
- **MPS is the DAYTIME failure mode** (2026-06-11): the Apple-GPU pool is unified memory shared
  with everything the user runs — a cross-encoder auto-placed on MPS OOM'd at "other allocations:
  14 GB" while the user worked. It fails CLEAN (RuntimeError) — good — but for background/eval
  jobs on a busy machine, **pin the model to CPU** (`CrossEncoder(..., device="cpu")`) and split
  embedder/reranker into separate processes (`topical_emb_lowram.py` pattern, ≤4 GB each).
  Night failure = swap eviction; day failure = MPS pool. Same root: 18 GB unified.
- **Route LLM *generation* through llmx**, not raw SDKs: `llmx ... --lite bare` = $0 GPT (ChatGPT
  sub); `--flex` = 50%-off gemini; or `~/Projects/skills/scripts/llm-dispatch.py`. `pretool-raw-openai-guard.sh`
  (wired) blocks raw `import openai` in `.py`. **SDK exceptions:** embeddings (`embed_content`) and
  Gemini **File Search** (`file_search_stores`) — llmx has no wrapper for those.

---

## 8. Map (where everything lives)

- Library: `~/Projects/emb` · elevation plan: `~/Projects/emb/.claude/plans/88ec9920-emb-elevation.md`
- Research memos: `~/Projects/emb/docs/research/2026-06-10-*.md` (chunking, screenshot-vs-text,
  txtai, gte-freshness, reranker-frontier-check) · `2026-06-11-benchmark-positioning.md`
  (where this bakeoff sits vs RTEB / FreshStack / EnterpriseRAG-Bench / TREC ToT)
- Bake-off + verdicts: `~/Projects/evals/retrieval_backend_bakeoff/` + `~/Projects/evals/DECISIONS.md`
- Consumers: `~/Projects/phenome` (full), `~/Projects/anki` (embed+pairs), `~/Projects/intel` (`tools/intelligence_mcp.py` + `tools/semantic_index.py`)
- Hooks: `~/Projects/skills/hooks/pretool-{heavy-load-guard,raw-openai-guard}.sh`
- Memories: `~/.claude/projects/-Users-alien-Projects-emb/memory/` (route-model-calls-through-llmx, research-subagent-model-default)
