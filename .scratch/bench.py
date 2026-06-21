#!/usr/bin/env python3
"""Head-to-head: is emb-contrastive actually better than regex / fuzzy / abs-semsearch
for detecting blindspot flags? Labeled set with HARD negatives (lexical traps)."""
import re, time
import numpy as np
from rapidfuzz import process, fuzz
from emb.embed import EmbeddingEngine

# --- labeled data (pos = blindspot/correction flags; neg = normal, incl. lexical traps)
pos = [
 "SOOO why didn't you find this bug?", "why don't you check the system",
 "why did you not find thid", "isn't there better packages ... fuzzy matching",
 "how come you missed that in the logs", "shouldn't you have read the decision doc",
 "you keep rediscovering stuff we already built", "did you even look at the git history",
 "we literally talked about this yesterday", "that's already in ideas.md",
 "you should've grepped before proposing", "this was decided in the ADR, why redo it",
 "i shouldn't have to tell you to check first", "go check what exists before building",
]
neg = [
 "please add a test for the parser", "let's ship the corpus fix and commit",
 "run the smoke test", "what reasoning level are you using", "go on", "ok do all 3",
 "find the duckdb import and fix it", "can you check the tests pass before committing",
 "look at the new PR when you get a chance", "did you want me to deploy this",
 "why is the build slow", "check the logs for the error", "we should add caching here",
 "read the file and summarize it", "implement the feature in foo", "commit and push",
]
labels = [1]*len(pos) + [0]*len(neg); msgs = pos + neg

# --- method 1: regex (the shipped Tier-0)
RGX = re.compile(
 r"why (?:did|didn'?t|don'?t|aren'?t|haven'?t|wouldn'?t|didnt|dont) (?:you|the loop|it|we) (?:not |never |fail(?:ed)? to )?(?:find|catch|check|look|notice|see|spot|read|consult)"
 r"|(?:you|it) (?:should|could) have (?:found|caught|checked|looked|noticed|seen|read)"
 r"|did (?:you|the loop) (?:check|look at|read|see|notice|find|consider)"
 r"|we (?:already|just) (?:discussed|decided|did|tried|talked|covered|said)"
 r"|already (?:discussed|decided|exists|covered) "
 r"|(?:check|look at|read|consult) the (?:git|commit|log|history|ideas|docs|decision|prior|reasoning)"
 r"|you (?:didn'?t|never|forgot to|should've|shouldve) (?:check|look|read|consult|find|catch|grep)", re.I)
def regex_scores(): return [1.0 if RGX.search(m) else 0.0 for m in msgs]

# --- seeds
blind = ["why didn't you find this bug","did you check the git log first","you should have looked at prior decisions","we already discussed this","isn't there a better package for this","you didn't check what already exists","you missed something obvious","this was already decided"]
normal = ["add a test for the parser","fix the bug in the resolver","run the tests","commit this","implement the feature","what model are you using","summarize the file","check the tests pass","look at the PR","deploy this"]

# --- method 2: fuzzy (rapidfuzz max token_set_ratio over blind seeds), 0..100
def fuzzy_scores():
    return [max(fuzz.token_set_ratio(m, s) for s in blind) for m in msgs]

# --- emb (abs + contrastive)
eng = EmbeddingEngine()
def E(xs):
    e=np.array(eng.embed_texts(xs),dtype=np.float32); return e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
B,N,M = E(blind),E(normal),E(msgs)
abs_scores  = (M@B.T).max(1)
cont_scores = (M@B.T).max(1) - (M@N.T).max(1)

def best(scores, name, binary=False):
    scores=np.asarray(scores,dtype=float); y=np.asarray(labels)
    if binary:
        pred=scores>=0.5
        rec=pred[y==1].mean(); fp=pred[y==0].sum(); prec=(pred&(y==1)).sum()/max(pred.sum(),1)
        print(f"  {name:16s} recall={rec*100:4.0f}%  FP={fp}/{ (y==0).sum() }  prec={prec*100:3.0f}%  (binary)")
        return
    # sweep thresholds: report best F1, and recall at zero-FP
    ths=sorted(set(scores)); bestf1=0; bf=None; rec0=0
    for t in ths:
        pred=scores>=t
        tp=(pred&(y==1)).sum(); fp=(pred&(y==0)).sum(); fn=((~pred)&(y==1)).sum()
        prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
        if f1>bestf1: bestf1,bf=f1,(t,prec,rec,fp)
        if fp==0 and rec>rec0: rec0=rec
    t,prec,rec,fp=bf
    print(f"  {name:16s} bestF1={bestf1:.2f} (thr={t:.3g} prec={prec*100:3.0f}% rec={rec*100:3.0f}% FP={fp})  recall@0FP={rec0*100:3.0f}%")

print("n_pos=%d n_neg=%d  (hard negatives: 'check the logs','did you want me to deploy','why is the build slow')\n"%(len(pos),len(neg)))
t0=time.time(); rs=regex_scores(); best(rs,"regex(Tier0)",binary=True)
t1=time.time(); best(fuzzy_scores(),"fuzzy(rapidfuzz)")
t2=time.time(); best(abs_scores,"emb abs-sim")
best(cont_scores,"emb contrastive")
t3=time.time()
print(f"\nspeed: regex {(t1-t0)*1e3:.1f}ms  fuzzy {(t2-t1)*1e3:.1f}ms  emb(embed 50 texts, incl load) {(t3-t2)*1e3:.0f}ms")
# show which the cheap methods MISS that contrastive catches
print("\nparaphrases caught by contrastive but missed by regex:")
for m,r,c in zip(msgs,rs,cont_scores):
    if labels[msgs.index(m)]==1 and r==0 and c>0.05: print(f"  +{c:.2f} regex=MISS  {m[:50]}")
