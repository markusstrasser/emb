import numpy as np
from emb.embed import EmbeddingEngine
eng = EmbeddingEngine()
blind = ["why didn't you find this bug","did you check the git log first","you should have looked at the prior decisions","we already discussed this","isn't there a better package for this","you didn't check what already exists","you missed something obvious"]
normal = ["add a test for the parser","fix the bug in the resolver","run the tests","commit this and move on","implement the feature","what model are you using","summarize the file","check the tests pass before committing"]
pos = ["SOOO why didn't you find this bug?","why don't you check the system","why did you not find thid","isn't there better packages ... fuzzy matching","how come you missed that in the logs","shouldn't you have read the decision doc","Is model2vec better than what emb offers?"]
neg = ["please add a test for the parser","let's ship the corpus fix and commit","run the smoke test","what reasoning level are you using","go on","ok do all 3","find the duckdb import and fix it","can you check the tests pass before committing"]
def emb(xs):
    e=np.array(eng.embed_texts(xs),dtype=np.float32); return e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
B,N=emb(blind),emb(normal)
def score(xs):
    E=emb(xs); return (E@B.T).max(1)-(E@N.T).max(1)   # contrastive: blind-sim minus normal-sim
ps,ns=score(pos),score(neg)
print("POS (contrastive, should be >0):"); [print(f"  {s:+.3f}  {t[:52]}") for s,t in zip(ps,pos)]
print("NEG (should be <0):"); [print(f"  {s:+.3f}  {t[:52]}") for s,t in zip(ns,neg)]
g=ps.min()-ns.max(); print(f"\nmargin={g:+.3f} {'CLEAN' if g>0 else 'OVERLAP'}  pos.min={ps.min():+.3f} neg.max={ns.max():+.3f}")
