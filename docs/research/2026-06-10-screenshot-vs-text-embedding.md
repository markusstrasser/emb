# Native multi-part (text+image) vs rendered-screenshot embedding for text-heavy items

Date: 2026-06-10
Question: For a text-heavy item (Anki card: front/back text + a supporting image), is it better to embed
(A) NATIVE multi-part input (text tokens + image) fused by a multimodal embedder, or
(B) a rendered SCREENSHOT/composite (text-as-pixels + image) embedded as one image?
Plus: if these embeddings are later searched JOINTLY with a separate corpus of native-text docs
(text query -> text doc), does the screenshot representation hurt text-query <-> item alignment?

Source grades: DSE (EMNLP 2024 main, A), ColPali (ICLR 2025, A), "Doc-as-Image Falls Short for
Scientific Retrieval" (arXiv 2604.18508, 2026, B-preprint), "Towards Text-Image Interleaved Retrieval"
(ACL 2025 long, A), MLLM text-bias (arXiv 2512.19115, B-preprint), modality-gap follow-ups
(I0T arXiv 2412.14384, contrastive-gap 2405.18570, spectral/OT 2505.03703, ReAlign 2602.07026 — B/C preprints),
practitioner blogs (Vespa/Mixpeek/Intixel — C, vendor/marketing but technically specific).

---

## TL;DR DECISION RULE

- **Use NATIVE multi-part (text tokens + image) when** the item is text-heavy and the *text is the
  question/answer* — i.e. prose Q&A flashcards, definitions, cloze-on-text. This is the default for
  flashcards. It preserves machine-readable text, aligns with the LM pretraining distribution, and
  keeps the item in the same region of space as a native-text corpus (best cross-corpus alignment).
- **Use SCREENSHOT/composite-image when** the *visual artifact IS the question* and rendering it as
  pixels is the only faithful representation: image-occlusion cards (masked region is the answer),
  diagram/figure/table/equation-layout cards, handwriting, math notation where LaTeX extraction is
  fragile, or cards whose layout/spatial relationships carry meaning. Here OCR/text-extraction loses
  the signal and the screenshot path wins (this is exactly the DSE/ColPali regime).
- **Cross-corpus implication:** YES, screenshot representation hurts text-query <-> item alignment when
  the same model space also holds native-text docs. A residual modality gap exists even in modern
  multimodal embedders; text-text pairs sit closer than text->image-of-text pairs. So if your retrieval
  workload is overwhelmingly `text query -> text doc`, embedding a fraction of items as screenshots
  drops them into the "image cluster," systematically disadvantaging them against the native-text
  competitors for the SAME text query. Keep text-heavy items native; reserve screenshots for items
  where the pixels are irreplaceable, and accept they'll compete on a slightly different footing.

---

## CLAIMS TABLE

| # | Claim | Evidence | Grade |
|---|-------|----------|-------|
| 1 | DSE (screenshot embedding) BEATS BM25 by +17 pts top-1 on text-intensive Wiki-SS/NQ, and is *competitive with* (not superior to) neural text retrieval. | DSE EMNLP'24: DSE top-1 46.2 vs BM25 ~29; on par with E5; Phi-3 (same LM, text input) ~4 pts HIGHER than DSE. | A |
| 2 | On text-image MIXED docs (slides), screenshot embedding BEATS OCR-text retrieval by >15 pts nDCG@10 — because OCR drops visual/layout content. | DSE EMNLP'24 SlideVQA; DPR (neural text) fails to beat even BM25 here due to noisy OCR text. | A |
| 3 | Current VLMs "still cannot fully capture the text content in a screenshot" — text-as-pixels is lossier than native text for the textual channel. | DSE authors' own analysis (Phi-3 text > DSE; "room to fully capture textual nuances"). | A |
| 4 | ColPali (late-interaction page-image) outperforms text pipelines on ViDoRe across ALL domains, but the gap is "particularly stark" on visually complex tasks (infographics, figures, tables); margin shrinks on text-centric docs. | ColPali ICLR'25; HF/Vespa write-ups. | A |
| 5 | ColPali/ColQwen's text-from-pixels works *despite* aggressive downscaling — base PaliGemma resizes to 448x448, text becomes "barely legible"; newer ColQwen variants raise resolution / keep aspect ratio precisely to fix text-heavy/non-square docs. | Ceshine Lee preprocessing notes (2026-01); ColQwen2 release. | C (technical blog, consistent w/ papers) |
| 6 | Page-as-image carries a heavy COST: multi-vector storage. ColPali/ColQwen use ~1024 patch vectors/page; vs single-vector DSE, ColQwen is ~36-64x the memory for +6.5-7.3 pts nDCG@5. Token-merging (Light-ColPali) recovers most of that. | ACL'25 findings (token reduction study). | A |
| 7 | For TEXT-HEAVY / prose content specifically, document-as-image UNDERPERFORMS text and interleaved representations. Even for figure queries, best model was ColQwen with text+VLM captions (a text representation). Interleaved text+image > flat document-image. | "Doc-as-Image Falls Short for Scientific Retrieval" arXiv 2604.18508 (2026). | B |
| 8 | Pixel-based inputs degrade MORE on text-centric reasoning than on visual tasks; VLM doc performance is sensitive to layout/rendering template (surface-presentation sensitivity). | 2604.18508 citing PixelWorld, Lyu/Cheng 2025. | B |
| 9 | "Interleaved context is the key" — native interleaved text+image retrievers beat non-interleaved (incl. single-image / flattened) baselines; but naive interleaving causes "disproportionate visual dominance in the embedding space." | "Towards Text-Image Interleaved Retrieval" ACL'25 (wikiHow-TIIR, MME). | A |
| 10 | MLLM embedding space is "overwhelmingly dominated by textual semantics" yet *homogenizes* embeddings when bridging modalities for generation — reducing discriminability. Implication: text query is in the model's comfort zone; image-of-text embeddings get projected toward text but lose distinctiveness. | arXiv 2512.19115 (2025). | B |
| 11 | A modality gap persists in CLIP-family and is "detrimental for multimodal retrieval"; it is a contrastive-loss / low-uniformity artifact, not fundamental, and is reducible post-hoc (mean-shift, standardization, spectral/OT). But absent that correction it is real and present. | Mind-the-Gap (2022); contrastive-gap 2405.18570; I0T 2412.14384; spectral/OT 2505.03703; ReAlign 2602.07026. | B (cluster of preprints + one seminal A) |
| 12 | 2026 best practice for mixed corpora: two-stage hybrid — text/OCR search for fast cheap filtering, vision (ColPali) rerank for visual precision. Pure-vision is "overkill" for born-digital clean text (storage + latency). | Mixpeek 2026 comparison; Vespa. | C |
| 13 | Vision-based retrieval is far more ROBUST to document degradation than OCR pipelines (-9% vs -45% under high distortion); hybrid best. Relevant only if YOUR card images are degraded/scanned. | lanl/lost-ocr "Lost in OCR Translation?" 2025. | B |

---

## DETAILED FINDINGS BY SUB-QUESTION

### 1. DSE (Ma et al., EMNLP 2024)
DSE embeds a full document SCREENSHOT into a single dense vector via a VLM bi-encoder; query is encoded
by a text tower. Two regimes:
- **Text-intensive (Wiki-SS, 1.3M Wikipedia screenshots, NQ questions):** DSE top-1 **46.2** vs BM25 ~29
  (+17 pts), top-20 +10 pts. On par with E5 (neural text). BUT **Phi-3 with native text input beats DSE
  by ~4 pts top-1 using the same backbone LM** — direct evidence that text-as-pixels is a *lossier
  channel for pure text* than native text. Authors: VLMs "still cannot fully capture the text content in
  a screenshot."
- **Mixed text+image (SlideVQA, 50k slides):** DSE beats ALL OCR-text baselines by **>15 pts nDCG@10**.
  Here screenshot wins decisively because OCR extraction loses visual/layout content and produces noisy
  text (DPR can't even beat BM25). Notably 7/50 DSE "false negatives" actually had the answer in
  image captions/tables — screenshot captured info text extraction missed.
- Net: screenshot is **competitive (slightly behind) on pure text, clearly ahead on visual/layout**.

### 2. ColPali / ColQwen (Faysse et al., ICLR 2025) + cost
Late-interaction (ColBERT-style MaxSim) over page-image patch grids. On ViDoRe it beats OCR+BGE-M3 and
even Claude-Sonnet-captioning pipelines. **Win is "particularly stark" on infographics, figures, tables;
margin narrows on text-centric docs** (it still wins there too, but text pipelines are closest). Wins
where text extraction is brittle: equations, charts, multi-column layout, scans.
Cost: multi-vector (~1024 vectors/page). ACL'25 token-reduction study: ColQwen is +6.5-7.3 nDCG@5 over
single-vector DSE but at **36.7-64.4x memory**; Light-ColPali/ColQwen recovers ~98-99% effectiveness at
~3-25% memory. Resolution matters: original ColPali downscales to 448x448 (text "barely legible");
ColQwen2 fixes this with higher res + aspect-ratio preservation, which is *why* it scores higher on
text-heavy docs.

### 3. Text-query -> image-of-text is a harder/lossier direction
Confirmed. Multiple lines:
- DSE itself: native-text model (Phi-3) > screenshot DSE on the same LM for pure-text retrieval (claim 3).
- arXiv 2604.18508 (2026): "pixel-based inputs degrade MORE on text-centric reasoning tasks"; for
  scientific (prose/equation-heavy) docs, document-as-image UNDERPERFORMS text and interleaved reps —
  even figure queries are best served by text+caption, not flat image. Document-as-image forces the
  model to infer content boundaries/cross-references implicitly from appearance.
- Surface-sensitivity: VLM retrieval quality varies with layout/rendering template — a brittleness
  native text doesn't have.
So matching a text query against rendered-text pixels is reliably harder on PROSE than text->text, while
the direction *flips* in favor of pixels when the content is genuinely visual/structured.

### 4. Cross-modal space alignment / modality gap
- Seminal: Liang et al. "Mind the Gap" (2022) — image and text embeddings occupy separate cones; the gap
  is detrimental for multimodal retrieval/clustering/zero-shot.
- 2024-2026 follow-ups confirm the gap PERSISTS and is reducible but not auto-zero: contrastive-gap
  (2405.18570: it's a low-uniformity artifact of two-encoder contrastive loss); I0T (2412.14384:
  post-hoc standardization drives gap ~0 but you must DO it); spectral/OT methods (2505.03703); ReAlign
  (2602.07026: 3-component gap — anchor displacement, anisotropic residual, spherical-norm centroid
  drift — correctable with closed-form linear ops on unpaired data).
- MLLM-specific (2512.19115): MLLM embedding space is text-dominated and *homogenizes* embeddings while
  bridging modalities, reducing discriminability of image-derived embeddings.
- Consequence for joint corpora: text-text pairs align more tightly than text->image-of-text pairs
  unless the model was contrastively trained to put text-query and image-of-text in the same place AND
  you apply gap-correction. A flashcard embedded as a screenshot lands in the image region; a text query
  scored against a mixed corpus will favor native-text docs over the screenshot card, all else equal.
  ColQwen markets "maintains strong text-only retrieval" — but that's relative; the residual gap is the
  thing that bites a SMALL screenshot minority inside a LARGE native-text corpus.

### 5. Practitioner verdict 2026 (flashcards / text+supporting-media)
- "Towards Text-Image Interleaved Retrieval" (ACL'25): native INTERLEAVED text+image beats both
  text-only and flattened-single-image baselines on genuinely multimodal items — BUT naive interleaving
  causes "disproportionate visual dominance," so a good multimodal embedder (e.g. Matryoshka token
  compression) matters. This is the strongest support for the NATIVE multi-part default on cards that
  have BOTH a real text Q/A AND a supporting image.
- 2026 hybrid consensus (Mixpeek/Vespa/Intixel): vision-as-image is "overkill" for born-digital clean
  text (storage + latency); reserve it for visually complex content; production pattern is text-filter +
  vision-rerank.
- For Anki specifically:
  - **Prose Q&A card (text front/back + decorative or supporting image):** NATIVE multi-part. The answer
    lives in the text; native text keeps alignment with a text corpus and avoids the lossy pixel channel.
    Feed the supporting image as the image part only if it carries retrieval signal; otherwise text-only.
  - **Image-occlusion card (masked region IS the question):** SCREENSHOT/composite — the pixels are the
    item; there is no faithful text representation of "what's under the mask."
  - **Diagram / labeled-figure / table / equation-layout card:** SCREENSHOT or vision path — text
    extraction is lossy/fragile exactly here (DSE +15 on slides; ColPali stark on figures/tables/equations).
  - **Handwriting / scanned card:** vision path (robustness: -9% vs OCR's -45% under degradation).

---

## CROSS-CORPUS RECOMMENDATION (the specific ask)
If the end state is one shared index where text queries hit BOTH these items AND a native-text doc corpus:
1. Default flashcards to NATIVE multi-part (text + optional image). This maximizes text-query alignment
   and keeps cards in the same region as native-text docs.
2. Only render-to-screenshot the cards whose meaning is visual (occlusion, diagram, table, equation,
   handwriting). Accept they sit slightly off the text manifold.
3. If you do mix screenshot items into a text-dominated index, apply a modality-gap correction
   (mean-centering / I0T standardization / ReAlign closed-form) before joint search, OR keep screenshot
   items in a separate sub-index and fuse scores (RRF / two-stage rerank) rather than competing them
   head-to-head in raw cosine. The literature says the gap is real but cheaply correctable post-hoc.
4. Don't pay ColPali multi-vector storage (36-64x) unless visual precision is the bottleneck; single-
   vector multimodal embedding (DSE-style) or native text+image is the right cost tier for a personal
   flashcard corpus, and Light-ColPali shows multi-vector is mostly compressible if you ever need it.

Findings: COMPLETE.
