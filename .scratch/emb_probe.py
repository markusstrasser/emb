#!/usr/bin/env python3
"""Same separation probe, but using the in-house emb engine (gte-modernbert-base)."""
import numpy as np
from emb.embed import EmbeddingEngine

eng = EmbeddingEngine()  # default Alibaba-NLP/gte-modernbert-base

seeds = [
    "why didn't you find this bug",
    "did you check the git log first",
    "you should have looked at the prior decisions",
    "we already discussed this earlier",
    "isn't there a better package for this",
    "why are you hand-rolling this instead of using a library",
    "you didn't check what already exists",
    "look at the ideas file before proposing",
]
pos = [
    "SOOO why didn't you find this bug?",
    "why don't you check the system",
    "why did you not find thid",
    "isn't there better packages ... linguistics ... fuzzy matching",
    "how come you missed that in the logs",
    "shouldn't you have read the decision doc",
    "Is model2vec better than what emb offers?",  # the user's current message (approach-correction)
]
neg = [
    "please add a test for the parser",
    "let's ship the corpus fix and commit",
    "run the smoke test",
    "what reasoning level are you using for opus",
    "go on",
    "ok do all 3",
    "find the duckdb import and fix it",
    "can you check the tests pass before committing",
]

def emb(xs):
    e = np.array(eng.embed_texts(xs), dtype=np.float32)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

S = emb(seeds)
def maxsim(xs):
    E = emb(xs); return (E @ S.T).max(axis=1)

ps, ns = maxsim(pos), maxsim(neg)
print("POSITIVES (should be HIGH):")
for t, s in zip(pos, ps): print(f"  {s:.3f}  {t[:60]}")
print("\nNEGATIVES (should be LOW):")
for t, s in zip(neg, ns): print(f"  {s:.3f}  {t[:60]}")
print(f"\npos min={ps.min():.3f} mean={ps.mean():.3f} | neg max={ns.max():.3f} mean={ns.mean():.3f}")
gap = ps.min() - ns.max()
print(f"margin = {gap:+.3f}  {'CLEAN' if gap>0 else 'OVERLAP'}   thr≈{(ps.min()+ns.max())/2:.3f}")
