# Resume Checkpoint — bio_embedding_bakeoff

> Rewritten 2026-06-11 (the canceled /compact's PreCompact hook had overwritten this with a
> STALE auto-checkpoint listing all tasks as pending — they are DONE). Trust this file.

## Status: v1 AND v2 COMPLETE + committed. Verdict CONCLUSIVE: HOLD gte.

v1 done (queries → probe → embed → score → neutral arm → EXPERIMENT §5-9 → DECISIONS/BENCHMARKS
→ /critique close + window arm + construct audit §10).

**v2 BUILT + RUN (2026-06-11, this session)** — the §10c graded-relevance rebuild that resolves
v1's inconclusive. In `evals` repo (`bio_embedding_bakeoff/`): `PREREGISTRATION_v2.md` (locked +
cross-model-hardened pre-data), `PREDICTION_v2.md` (TIE predicted — confirmed), `run_v2.py`
(phrasings→embed_q→pool→judge→score→audit→hardness), `consumer_lane.py`, `decide.py`,
`spotcheck_{prep,score}.py`; `evalcore/irmetrics.py` (graded nDCG/MAP) + `stats.paired_bootstrap_mean_diff`
/`prob_superiority_paired`. **Verdict: generator-robust TIE → HOLD gte** (pgx+clinical isolation
graded nDCG@10 Δ=-0.002 CI[-0.023,+0.020]; consumer lane 89% top-10 overlap; κ=0.669; label-noise
flaw fixed 80/80). EXPERIMENT.md §11, DECISIONS row resolved. **One open curator task:** fill
`bio_embedding_bakeoff/spotcheck_v2.csv` (120 blind pairs) → run `spotcheck_score.py` for the
judge↔human κ gate (≥0.65 human-validates the LLM-judge lane; currently validated by cross-family
agreement only). Switch independently gated on the gemini-embedding-2 PREVIEW endpoint regardless.

## Verdict
**HOLD `gte-modernbert-base`** (default-to-incumbent). Do NOT switch phenome's emb text index to
`gemini-embedding-2`. Crucially: the eval is **INCONCLUSIVE on embedder quality**, NOT a parity
finding. gemini2's flash-query lead (prereg pgx+clinical P=0.974) collapsed to a tie (0.504) under
a neutral GPT generator, and 3× document content didn't help it. The deeper reason (construct audit
§10): single-gold recall@5 scores **co-relevant sibling memos as misses** on 23/48 queries → the
design can't rank the embedders. Don't do an irreversible re-embed on this.

## The ONE open thread → handoff plan
`~/Projects/evals/.claude/plans/3cab7495-bio-embed-v2-graded-relevance.md`
A v2 graded-relevance eval that fixes v1's flaws. **It opens with a §0 go/no-go gate** — only build
if retrieval quality is a felt pain (gemini2 is a preview endpoint; the switch is irreversible).
Default action: none. If building, `/critique model` the plan first (plan-review-gate).

## Key locations
- Eval: `~/Projects/evals/bio_embedding_bakeoff/` — `EXPERIMENT.md` (§10 = the audit + v2 design),
  `run.py`, `bio_embed_modal.py` (Modal gte, L40S), `robustness_{neutral,window}.py`,
  `audit_label_noise.py` (the miss-inspection evidence), `runs/scored*.json`.
- Verdict register: `evals/DECISIONS.md` (`phenome-bio-text-embedder`) + `evals/BENCHMARKS.md`.
- Skill lesson folded: `~/Projects/skills/eval/SKILL.md` (Phase 3 necessary-not-sufficient + eyeball
  misses + single-gold validity), commit skills@4964033.
- RAM: 18 GB Mac, heavily loaded → keep ML embedding on Modal, never local (memory
  `local-ml-route-to-modal`).

## Commits (evals)
b6e0a5e v2 handoff plan · 41d2502 contamination-framing fix · 4bad2d7 construct audit (§10) ·
8c8ded4 window arm · 81f6db6 /critique-close hardening · dc2d047 DECISIONS row · e58c0ff verdict ·
51dcf76 Modal gte backend · 7299428 frozen queries · a6c8a8b harness · 5568d4e prereg.
