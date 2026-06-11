"""Determine how gemini embed_content batches. Does contents=[str,str] give N or 1 embedding?"""
import time
import numpy as np
from google import genai
from google.genai import types

MODEL = "gemini-embedding-2-preview"
DIM = 768
client = genai.Client()

texts5 = [f"distinct sentence about topic {w}" for w in
          ["mitochondria", "Paris", "python", "quantum", "BM25"]]

print("A) contents = list[str] (5 strings):")
r = client.models.embed_content(model=MODEL, contents=texts5,
                                config=types.EmbedContentConfig(output_dimensionality=DIM))
print(f"   -> {len(r.embeddings)} embeddings")

print("B) contents = list[Content], one per text:")
r = client.models.embed_content(
    model=MODEL,
    contents=[types.Content(parts=[types.Part(text=t)]) for t in texts5],
    config=types.EmbedContentConfig(output_dimensionality=DIM))
print(f"   -> {len(r.embeddings)} embeddings")

print("C) loop single-call x5, timing:")
t0 = time.time()
embs = []
for t in texts5:
    r = client.models.embed_content(model=MODEL, contents=t,
                                    config=types.EmbedContentConfig(output_dimensionality=DIM))
    embs.append(r.embeddings[0].values)
print(f"   -> {len(embs)} embeddings, {time.time()-t0:.2f}s ({(time.time()-t0)/5*1000:.0f}ms/item)")

# If B works, find the per-request content limit
if True:
    print("\nD) batch limit via list[Content]:")
    for n in (50, 100, 250):
        try:
            t0 = time.time()
            r = client.models.embed_content(
                model=MODEL,
                contents=[types.Content(parts=[types.Part(text=f"text {i}")]) for i in range(n)],
                config=types.EmbedContentConfig(output_dimensionality=DIM))
            print(f"   batch={n}: OK ({len(r.embeddings)} back, {time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"   batch={n}: FAIL — {str(e)[:140]}")
            break
