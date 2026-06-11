"""Probe real mp3 audio support + token/cost metadata for gemini-embedding-2."""
import glob
import os
import numpy as np
from google import genai
from google.genai import types

MODEL = "gemini-embedding-2-preview"
DIM = 768
client = genai.Client()
MEDIA = os.path.expanduser("~/Library/Application Support/Anki2/alien/collection.media")

mp3 = sorted(glob.glob(f"{MEDIA}/*.mp3"))[0]
gif = sorted(glob.glob(f"{MEDIA}/*.gif"))[0]
png = sorted(glob.glob(f"{MEDIA}/*.png"))[0]

print(f"Testing real media files:")
for label, path, mime in [("mp3", mp3, "audio/mpeg"), ("gif", gif, "image/gif"), ("png", png, "image/png")]:
    data = open(path, "rb").read()
    try:
        r = client.models.embed_content(
            model=MODEL,
            contents=[types.Content(parts=[
                types.Part(text="anki card front/back"),
                types.Part.from_bytes(data=data, mime_type=mime)])],
            config=types.EmbedContentConfig(output_dimensionality=DIM),
        )
        norm = np.linalg.norm(r.embeddings[0].values)
        meta = getattr(r, "metadata", None) or getattr(r, "usage_metadata", None)
        print(f"  {label} ({len(data)} bytes, {mime}): OK norm={norm:.4f} meta={meta}")
    except Exception as e:
        print(f"  {label}: FAIL — {str(e)[:140]}")

# also try audio/mp3 vs audio/mpeg mime
print("\nMime variants for mp3:")
data = open(mp3, "rb").read()
for mime in ("audio/mpeg", "audio/mp3"):
    try:
        r = client.models.embed_content(
            model=MODEL,
            contents=[types.Content(parts=[types.Part.from_bytes(data=data, mime_type=mime)])],
            config=types.EmbedContentConfig(output_dimensionality=DIM))
        print(f"  {mime}: OK")
    except Exception as e:
        print(f"  {mime}: FAIL — {str(e)[:100]}")

# Token metadata for a plain text call (for cost estimation)
print("\nToken metadata (text call):")
r = client.models.embed_content(model=MODEL, contents="a moderately sized anki card with some text content here",
                                config=types.EmbedContentConfig(output_dimensionality=DIM))
print("  full response attrs:", [a for a in dir(r) if not a.startswith("_")])
for emb in r.embeddings:
    print("  embedding attrs:", [a for a in dir(emb) if not a.startswith("_")])
    st = getattr(emb, "statistics", None)
    print("  statistics:", st)
