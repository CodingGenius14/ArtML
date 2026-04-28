# Translating Musical Emotion into Visual Form
### An Iterative NLP and Generative AI Pipeline for Lyric-to-Album-Cover Synthesis


---

## Overview

This repository contains the full implementation of an end-to-end pipeline that automatically generates album cover artwork from song lyrics using NLP and text-to-image diffusion models.

The pipeline runs in five sequential stages:

```
Raw Lyrics → Preprocessing → Keyword Extraction → Emotion Classification → Prompt Engineering → Image Generation
```

Two program iterations are included:
- **Program 1** (`main.ipynb`, `improved_main.ipynb`): Stable Diffusion v1.5, produces V1–V4
- **Program 2** (`final_implementation.ipynb`): Stable Diffusion XL, produces V5 with genre-aware prompt construction

---

## Key Findings

| Finding | Detail |
|---|---|
| Standard NLP stopwords destroy lyric signal | NLTK removes emotionally critical tokens like `dark`, `lost`, `pain`, `cry` |
| RAKE fails on lyrics | Phrase-level scoring misses single-word emotional anchors; TF-IDF + boosting adopted |
| Hip-hop misclassification | `j-hartmann` DistilRoBERTa predicts `fear` (0.331 confidence) for rap ambition lyrics |
| Prompt engineering dominates | 65-word V4 narrative template outperforms 20-word baseline on identical model weights |
| SDXL eliminates most text hallucinations | SD v1.5 hallucinated garbled text in nearly all outputs; SDXL substantially reduces this |

**Human evaluation (n=21):**

| Genre | Mean Preference (1–5) | % Preferring V5 |
|---|---|---|
| Rap / Ambition | 4.00 ± 1.58 | **66.7%** |
| Upbeat Pop | 3.81 ± 1.25 | **38.1%** |
| Sad / Introspective | 2.71 ± 1.65 | 28.6% |

---

## Repository Structure

```
ArtML/
├── main.ipynb                  # Program 1 baseline (V1 outputs)
├── improved_main.ipynb         # Program 1 improved (V2–V4 outputs)
├── final_implementation.ipynb  # Program 2 — SDXL + genre-aware prompts (V5 outputs)
├── requirements.txt            # Full dependency list
├── generated_outputs/          # All 15 generated album covers (V1–V5, 3 genres)
└── album_cover.png             # Sample output
```

---

## Setup

### Prerequisites

- Python 3.9+
- A HuggingFace account and access token (for SDXL in `final_implementation.ipynb`)
- **GPU strongly recommended for Program 2 (SDXL).** CPU-only generation takes 5–25 minutes per image on SD v1.5 and is impractical for SDXL.

### Install dependencies

```bash
pip install -r requirements.txt
```

You will also need NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### HuggingFace token (required for SDXL)

```bash
huggingface-cli login
```

Or set the environment variable:

```bash
export HF_TOKEN=your_token_here
```

---

## Usage

### Program 1 — SD v1.5 (V1–V4)

Open and run `improved_main.ipynb` end-to-end. This notebook:
1. Accepts raw lyric text input
2. Applies custom emotion-aware preprocessing and lemmatization
3. Runs TF-IDF + emotion-boosted keyword extraction
4. Classifies emotion via `j-hartmann/emotion-english-distilroberta-base`
5. Constructs a V4 narrative prompt
6. Generates a 512×512 album cover via Stable Diffusion v1.5

```python
# Set your lyrics here (top of notebook)
lyrics = """
[Verse 1]
Your lyrics here...
"""
```

### Program 2 — SDXL + Genre-Aware Prompts (V5)

Open and run `final_implementation.ipynb`. This notebook adds:
- Genre detection heuristic (keyword-based)
- Genre-aware prompt override for hip-hop/rap content
- Negative prompting to suppress text hallucinations
- SDXL for 1024×1024 output

---

## Generated Outputs

All 15 generated images are stored in `generated_outputs/`.

| Version | Sad | Pop | Rap |
|---|---|---|---|
| V1 | Lone figure, umbrella, foggy park | Colorful figure mid-leap | Plain brown texture (RAKE failure) |
| V2 | Blurry silhouette in fog | Black-and-white rainy alleyway ⚠️ | Horror-style face ⚠️ |
| V3 | Teal/magenta face close-up | Warm orange/teal illustration | Graffiti silhouette |
| V4 | Figure in illuminated corridor ✓ | Vibrant woman, light explosions ✓ | Dramatic figure between rock formations |
| V5 | Solitary figure, violet fog, streetlamp ✓ | Woman in confetti, golden-hour sunset ✓ | Figure overlooking city skyline at sunset ✓ |

⚠️ = documented failure mode · ✓ = preferred by plurality of evaluators

---

## Pipeline Details

### Custom Emotion-Aware Stopword List

Standard NLTK stopwords remove 179 tokens including emotionally critical words. Our custom list retains 47 of these, including:

`hurt · fear · love · lost · pain · dark · light · fall · shadow · cry · dream · night · hope · down · through · away · still · ever · never · always`

### Keyword Extraction

```
TF-IDF score + (2× multiplier if token in emotion anchor list)
→ merge and deduplicate
→ top 7 terms passed to prompt builder
```

Emotion anchor list: `hurt, fear, love, hate, sad, happy, cry, pain, joy, lost, dark, light, night, dream`

### Emotion-to-Palette Mapping (V3+)

| Emotion | Visual Palette |
|---|---|
| Sadness | Deep blues and purples |
| Joy | Warm brightness, golden tones |
| Anger | Reds and oranges |
| Fear | Desaturated greys, cold blues |
| Neutral | Muted earth tones |

### SD v1.5 Generation Config (Program 1)

| Parameter | Value | Notes |
|---|---|---|
| Inference steps | 30 | 15 = artifacts; 50 = marginal gain |
| Guidance scale | 7.5 | >10 = oversaturation; <6 = off-prompt |
| Seed | 42 | Fixed across all runs for reproducibility |
| Resolution | 512×512 | |
| Attention slicing | Enabled | ~40% memory reduction for CPU |

---

## Failure Documentation

This project deliberately documents all failures. Notable failure modes:

**Stopword failure:** Standard NLTK on sad lyrics returned `figure somehow` and `rise`, missing `darkness`, `shadow`, `memories`, `tears`.

**RAKE failure:** Phrase-level scoring on rap lyrics returned `ever seem`.

**Emotion misclassification:** `j-hartmann` model predicts `fear` (0.331) for hip-hop lyrics about urban ambition. Root cause: model trained on Twitter/Reddit/news captions; hip-hop bravado is out-of-distribution.

**V2 style conflict:** Adding `"cinematic, dramatic, high contrast"` style descriptors to pop lyrics overrode the 0.980-confidence joy signal, producing a black-and-white rainy alleyway.

**SD v1.5 text hallucinations:** Garbled text appeared in nearly all SD v1.5 outputs. Addressed in V5 via SDXL + negative prompting.

---

## License

MIT License. See `LICENSE` for details.
