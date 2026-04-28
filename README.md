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
├── program1/               # SD v1.5 pipeline (produces V1–V4)
├── program2/               # SDXL pipeline with genre-aware override (produces V5)
├── evaluation/
│   ├── clip_eval.ipynb     # CLIP-prompt and CLIP-lyric scoring
│   └── human_study/        # Preference data (CSV) and analysis scripts
├── outputs/
│   ├── rap/                # Generated images V1–V5, rap genre
│   ├── sad/                # Generated images V1–V5, sad/introspective genre
│   └── pop/                # Generated images V1–V5, upbeat pop genre
└── figures/                # All paper figures (PDF + PNG)

---

## Requirements
torch
transformers
diffusers
accelerate
Pillow
nltk
scikit-learn
numpy
pandas
matplotlib

Install with:

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

**Generate images (Program 1 — SD v1.5, V1–V4):**

```bash
cd program1
python generate.py --genre sad --version v4 --seed 42
```

**Generate images (Program 2 — SDXL, V5):**

```bash
cd program2
python generate_sdxl.py --genre rap --seed 42
```

All experiments use `--seed 42` throughout. V1–V4 run on CPU (no GPU required, ~15 min/image). V5 requires a GPU with ≥16GB VRAM.

**CLIP evaluation:**

```bash
cd evaluation
jupyter nbconvert --to notebook --execute clip_eval.ipynb
```

Scores are written to `evaluation/clip_scores.csv` and match Table 6 in the paper.

---

## Human Preference Data

`evaluation/human_study/preferences.csv` contains the raw forced-choice responses from all 56 participants (genre, selected version, no PII collected). Analysis scripts reproducing Tables 4–5 and all statistical tests are in `evaluation/human_study/analysis.py`.

---

## Fixed Seeds and Reproducibility

All generation, evaluation, and statistical analysis use `seed=42` unless otherwise noted. Model version strings:

| Component | Version string |
|---|---|
| SD v1.5 | `runwayml/stable-diffusion-v1-5` |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` |
| Emotion classifier | `j-hartmann/emotion-english-distilroberta-base` |
| CLIP | `openai/clip-vit-base-patch32` |

---

## Citation

Anonymous submission — citation information will be added upon publication.
