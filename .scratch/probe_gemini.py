"""Phase-0 probe: gemini-embedding-2-preview normalization, batch limit, multi-part fusion, cost."""
import io
import time
import numpy as np
from google import genai
from google.genai import types

MODEL = "gemini-embedding-2-preview"
DIM = 768

client = genai.Client()


def _make_image(color, size=(64, 64)):
    from PIL import Image
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def embed_texts(texts):
    # CRITICAL: one Content per text. A bare list[str] fuses into ONE embedding.
    r = client.models.embed_content(
        model=MODEL,
        contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
        config=types.EmbedContentConfig(output_dimensionality=DIM),
    )
    return np.array([e.values for e in r.embeddings], dtype=np.float32)


def embed_multipart(parts_list):
    """Each item is a list of types.Part."""
    contents = [types.Content(parts=parts) for parts in parts_list]
    r = client.models.embed_content(
        model=MODEL, contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=DIM),
    )
    return np.array([e.values for e in r.embeddings], dtype=np.float32)


print("=== 1. TEXT: 10 texts, normalization check ===")
texts = [
    "The mitochondria is the powerhouse of the cell.",
    "Cellular respiration produces ATP in mitochondria.",  # near-dup of above
    "The capital of France is Paris.",
    "Photosynthesis converts light to chemical energy.",
    "Python is a programming language.",
    "Snakes are reptiles found on every continent except Antarctica.",
    "Reciprocal rank fusion combines ranked lists.",
    "BM25 is a bag-of-words ranking function.",
    "The French Revolution began in 1789.",
    "Quantum entanglement links particle states.",
]
t0 = time.time()
E = embed_texts(texts)
dt = time.time() - t0
norms = np.linalg.norm(E, axis=1)
print(f"  shape={E.shape} dtype={E.dtype} latency={dt:.2f}s")
print(f"  L2 norms: min={norms.min():.5f} max={norms.max():.5f} mean={norms.mean():.5f}")
print(f"  NORMALIZED AT 768d? {'YES' if np.allclose(norms, 1.0, atol=1e-3) else 'NO — must L2-normalize in emb'}")
En = E / norms[:, None]
print(f"  cos(mitochondria pair 0,1) = {En[0] @ En[1]:.4f} (expect high)")
print(f"  cos(0, Paris 2)            = {En[0] @ En[2]:.4f} (expect low)")
print(f"  cos(BM25 7, RRF 6)         = {En[6] @ En[7]:.4f} (expect moderate)")

print("\n=== 2. BATCH LIMIT probe ===")
for n in (50, 100, 200):
    try:
        t0 = time.time()
        r = client.models.embed_content(
            model=MODEL, contents=[f"text number {i}" for i in range(n)],
            config=types.EmbedContentConfig(output_dimensionality=DIM),
        )
        print(f"  batch={n}: OK ({len(r.embeddings)} returned, {time.time()-t0:.2f}s)")
    except Exception as e:
        print(f"  batch={n}: FAIL — {str(e)[:120]}")
        break

print("\n=== 3. MULTI-PART text+image fusion (anki card shape) ===")
# Card A: text front/back + red image; Card B: same text + blue image; Card C: different text + red image
red, blue = _make_image((220, 30, 30)), _make_image((30, 30, 220))
try:
    M = embed_multipart([
        [types.Part(text="Front: What color is a stop sign? Back: Red."),
         types.Part.from_bytes(data=red, mime_type="image/png")],
        [types.Part(text="Front: What color is a stop sign? Back: Red."),
         types.Part.from_bytes(data=blue, mime_type="image/png")],
        [types.Part(text="Front: What is the boiling point of water? Back: 100C."),
         types.Part.from_bytes(data=red, mime_type="image/png")],
    ])
    Mn = M / np.linalg.norm(M, axis=1)[:, None]
    print(f"  multi-part shape={M.shape}, norms={np.linalg.norm(M, axis=1).round(4)}")
    print(f"  cos(sameText+diffImg A,B) = {Mn[0] @ Mn[1]:.4f}")
    print(f"  cos(diffText+sameImg A,C) = {Mn[0] @ Mn[2]:.4f}")
    print("  MULTI-PART FUSION: OK")
except Exception as e:
    print(f"  MULTI-PART FAIL — {str(e)[:200]}")

print("\n=== 4. GIF + audio mime probe (anki has 105 gif, 6 mp3) ===")
for mime, data in [("image/gif", _make_image((10, 200, 10))), ("audio/mp3", b"\xff\xfb\x90\x00" + b"\x00" * 200)]:
    try:
        r = client.models.embed_content(
            model=MODEL,
            contents=[types.Content(parts=[
                types.Part(text="test"),
                types.Part.from_bytes(data=data, mime_type=mime)])],
            config=types.EmbedContentConfig(output_dimensionality=DIM),
        )
        print(f"  {mime}: OK")
    except Exception as e:
        print(f"  {mime}: FAIL — {str(e)[:120]}")
