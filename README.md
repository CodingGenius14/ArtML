````markdown
# Lyric-to-Album-Cover: Modular NLP + Diffusion Pipeline

Code and evaluation artifacts for the paper:

> *When NLP Meets Lyrics: Characterizing Domain-Shift Failures in Affective Text Analysis and a Modular Framework for Lyric-Conditioned Visual Synthesis*
> Anonymous Authors — Under Review, NeurIPS 2026

---

## Overview

An end-to-end pipeline that takes song lyrics as input and generates album cover artwork via five sequential stages:

1. **Lyric preprocessing** — custom emotion-aware stopword filtering + WordNet lemmatization
2. **Keyword extraction** — hybrid TF-IDF + emotion-boosted frequency scoring
3. **Emotion classification** — j-hartmann/emotion-english-distilroberta-base
4. **Prompt engineering** — V1–V5 narrative templates with palette mapping and genre-aware override
5. **Image generation** — Stable Diffusion v1.5 (V1–V4) and SDXL (V5)

---

## Repo Structure

    ├── main.ipynb                  # Program 1: SD v1.5 pipeline (produces V1–V4)
    ├── improved_main.ipynb         # Program 2: SDXL pipeline with genre-aware override (produces V5)
    ├── final_implementation.ipynb  # CLIP evaluation and scoring
    ├── generated_outputs/          # All 15 generated images (V1–V5 × 3 genres)
    ├── album_cover.png             # Sample output
    └── requirements.txt

---

## Requirements

```bash
pip install -r requirements.txt
```

NLTK resources needed:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## Reproducing the Main Results

**Program 1 — SD v1.5, produces V1–V4:**

Open and run `main.ipynb`. Runs on CPU (no GPU required, ~15 min/image). All cells use fixed `seed=42`.

**Program 2 — SDXL with genre-aware override, produces V5:**

Open and run `improved_main.ipynb`. Requires a GPU with ≥16GB VRAM.

**CLIP evaluation:**

Open and run `final_implementation.ipynb`. Upload the 15 generated images from `generated_outputs/` when prompted. Outputs CLIP-prompt similarity scores matching Table 6 in the paper.

---

## Generated Outputs

`generated_outputs/` contains all 15 images organized by genre and pipeline version:

| File | Genre | Version |
|---|---|---|
| `sad_introspective_v1.png` – `sad_v5.png` | Sad / Introspective | V1–V5 |
| `upbeat_pop_v1.png` – `pop_v5.png` | Upbeat Pop | V1–V5 |
| `rap_v1.png` – `rap_v5.png` | Rap / Ambition | V1–V5 |

---

## Fixed Seeds and Reproducibility

All generation and evaluation use `seed=42` throughout. Model version strings:

| Component | Version string |
|---|---|
| SD v1.5 | `runwayml/stable-diffusion-v1-5` |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` |
| Emotion classifier | `j-hartmann/emotion-english-distilroberta-base` |
| CLIP | `openai/clip-vit-base-patch32` |

---

## Citation

Anonymous submission — citation information will be added upon publication.
````
