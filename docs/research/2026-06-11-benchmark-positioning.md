# Benchmark positioning — retrieval_backend_bakeoff vs the external frontier

**Question:** Are there academic/industry benchmarks relevant to our bakeoff? Are we pushing
the frontier, replicating it, or orthogonal — and why is our use case different?
**Tier:** Standard · **Date:** 2026-06-11
**Ground truth going in:** frontier memo `2026-06-10-reranker-frontier-check.md` (reranker
benchmarks: BEIR/TREC-DL/LoCo, Drowning-in-Documents, depth studies); our own
`evals/retrieval_backend_bakeoff/EXPERIMENT.md` (needle 5/5/8, topical 9/3/6, sweep §4c).

## Claims table

| # | Claim | Evidence | Confidence | Source | Status |
|---|---|---|---|---|---|
| 1 | The field's flagship answer to "public benchmarks overfit" is RTEB (Oct 2025): hybrid open + **private held-out** retrieval datasets, explicitly anti-"teaching to the test" | HF blog + InfoQ | HIGH | [SOURCE: huggingface.co/blog/rteb] | VERIFIED |
| 2 | RTEB's private column was **temporarily removed Jan 2026** because one co-developing vendor (Voyage/MongoDB) had structural access to the private data — community-leaderboard trust failure | MTEB issue #3934, locked Decision | HIGH | [SOURCE: github.com/embeddings-benchmark/mteb/issues/3934] | VERIFIED |
| 3 | EnterpriseRAG-Bench (Onyx, 2026) is the closest public analog to our managed-vs-local question: 500K synthetic enterprise docs, 500 questions, 10 question categories, leaderboard scoring managed services (OpenAI File Search 61.0, Amazon Q 49.0, Azure AI Search 48.4, Vertex AI Search 41.9 vs tuned self-hosted 72.4) | onyx.app + GitHub (395★, MIT) | HIGH | [SOURCE: onyx.app/enterpriserag-bench] | VERIFIED |
| 4 | Known-item retrieval is institutionalized as a distinct query class: TREC Tip-of-the-Tongue track 2023–2025 (2025: open-domain, 6.4M-doc corpus); success@k/RR metrics for one-relevant-doc tasks — the academic lineage of our needle lane | track overviews + guidelines | HIGH | [SOURCE: arxiv.org/pdf/2601.20671; trec-tot.github.io] | VERIFIED |
| 5 | LLM-elicited synthetic queries are a validated eval method: ToT 2024/2025 test queries include LLM-generated ones, shown to rank systems with high correlation to human CQA-derived queries | arXiv:2502.17776 (CMU/UNC/MSR) | HIGH | [SOURCE: arxiv.org/pdf/2502.17776] | VERIFIED |
| 6 | Contamination-aware fresh-query benchmarks are an active frontier: FreshStack (NeurIPS 2025 D&B; Stack Overflow 2023–24 questions + live GitHub corpora; included into RTEB), LiveRAG (SIGIR 2025, 500 unseen questions live) | project sites + challenge report | HIGH | [SOURCE: fresh-stack.github.io; arxiv.org/html/2507.04942] | VERIFIED |
| 7 | Reranker-depth/harm literature is active 2025–26: beyond Drowning (2411.11767), "Rerank Before You Reason" (2601.14224) studies depth-vs-cost tradeoffs; R²R (2511.19987) documents generalist rerankers missing domain nuance + surface-form overfitting — consistent with our per-corpus-validation stance | scite results | MED-HIGH | [SOURCE: scite, DOIs above] | VERIFIED |
| 8 | No published benchmark evaluates managed RAG services on **real** private/personal corpora — enterprise benchmarks are synthetic (EnterpriseRAG-Bench: "Redwood Inference"; AeroVelo etc.); vendor comparisons on real data are blog-grade (Tonic.ai 2025) | axis-2 sweep | MED (absence claim) | — | INFERENCE |

## Positioning verdict

**Mostly orthogonal-confirming: we run the private instantiation of exactly the experiment
the public frontier is trying to simulate.** Specifics:

- **Replication (valuable, not novel):** our rerank-net-negative result independently confirms
  Drowning-in-Documents on terrain academia structurally cannot reach (a real private corpus
  with a strong dense first stage). MaxP windowing is settled lineage. Replications on private
  corpora are rare *because the data is private* — that's the value, not the mechanism.
- **Aligned with the frontier's direction, arrived independently:** private held-out eval
  (RTEB), fresh-authored contamination-free queries (FreshStack), query-class stratification
  (TREC ToT; EnterpriseRAG-Bench's 10 categories), LLM-elicited queries (ToT 2024-25 validated
  the method we used). Our needle/topical lanes map cleanly onto known-item vs ad-hoc.
- **Possibly novel micro-claim:** I4 — *pool depth is a needle↔topical trade whenever gold
  pre-rerank rank distributions differ by query class* (needle golds deep at 56/60, topical at
  0–2, so no single rerank depth serves both). Implicit in the depth literature; we haven't
  seen it stated with per-query rank evidence. SCREENING-grade N — a paper would need N≥100+
  and multiple corpora. [INFERENCE]
- **The durable transferable part is the method kit, not the numbers:** discrimination probe
  with a mechanized degenerate baseline, drift guard for living corpora, hermetic candidate
  snapshots, frozen-pool model swaps, phase-split low-RAM scoring, prereg + decision register.
  EnterpriseRAG-Bench ships a leaderboard; we ship a *decision procedure* an operator can run
  on their own data in an afternoon.

## Why our use case is structurally different

1. **Population-of-one economics.** Benchmark science optimizes external validity (will this
   number transfer to YOUR data?). We have no transfer problem — corpus, queries, and consumer
   are the same entity. Internal validity + cheapness dominate; that's why SCREENING-N is
   rational for us and would be inadequate for a leaderboard.
2. **The trust problem dissolves.** RTEB's private column failed over who-can-see-the-data
   (claim 2). Self-evaluation on self-owned data has no leaderboard incentive and no vendor
   asymmetry — the failure mode that broke the field's flagship private benchmark cannot occur.
3. **Living corpus.** Public benchmarks freeze; our corpus rebuilds nightly. Hence drift
   guards + hermetic snapshots as first-class artifacts (FreshStack versions datasets for the
   same reason).
4. **Decision-grade, not leaderboard-grade.** Verdicts land in a decisions register wired to
   production defaults, with pre-registered bars. Academia ranks systems; we close decisions.

## What's uncertain

- Claim 8 is an absence claim from one sweep — a real-private-corpus managed-service eval may
  exist unpublished (enterprises run them internally; that's the point).
- No large-scale independent replication of Drowning's harm finding surfaced beyond
  corroborating depth studies; our result adds one small private-corpus data point.
- Whether I4 is stated anywhere in the depth literature — not found in this pass, not proven
  absent.

## Search log

Exa advanced ×5 (private-corpus benchmarks; managed-vs-self-hosted; RTEB; ToT/known-item;
FreshStack/LiveRAG), scite ×1 (Drowning follow-ups). 6 calls, all axes productive.
