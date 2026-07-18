# ArtML — Lyric-to-Album-Cover Synthesis

Code, generated outputs, and study data for:

> **Confidence Misread as Fear: Register Shift and Genre Conditioning in Lyric-to-Image Synthesis**
> Aarush Kandukoori, Karthik Murugan, Jai Nukala, Eunsu Kang
> School of Computer Science, Carnegie Mellon University
> NeurIPS 2026 Creative AI Track

A five-stage pipeline that generates album cover artwork from song lyrics using
NLP and text-to-image diffusion, and two failure modes it exposes.

## The two findings

**1. Register shift: confidence misread as fear.**
`j-hartmann/emotion-english-distilroberta-base`, fine-tuned on Twitter and
Reddit, collapses hip-hop's assertive register onto the *fear* class. It reads
sad lyrics at 0.699 sadness and pop at 0.980 joy — confident and correct — but
on rap it returns a flat distribution across all seven classes topping out at
0.331, leaving no decisive signal. Bravado ("I run this city"), hyperbole
("Crown on my head"), and first-person assertion ("Came up from nothing") are
statistically associated with aggression in standard English corpora. Because
the mechanism is register shift rather than a model-specific quirk, the same
failure should be expected from other standard-corpus classifiers applied to
non-standard registers. Mitigated here by a genre-conditioned prompt override.

**2. CLIP similarity and human preference diverge.**
Across 15 images rated by 56 participants, the proxy metric and the humans
disagree in both directions. For pop, V3 scores 35.32 and V5 scores 35.17 while
raters prefer V5 by 5:1. For sad, the V4 image records the lowest CLIP score in
the entire study (20.31) while winning 57.1% of votes. Optimizing against CLIP
would have selected against the human evaluators in both cases.

## Repository layout

```
├── main.ipynb                      Program 1 — SD v1.5 pipeline (V1–V4)
├── improved_main.ipynb             Program 2 — SDXL + genre override (V5)
├── final_implementation.ipynb      Final adopted configuration
├── generated_outputs/              All 15 generated covers (5 versions × 3 genres)
├── appendix/
│   ├── A_lyric_corpus.md           Full synthetic lyric text for all 3 genres
│   ├── B_prompt_templates.md       Prompt templates V1–V5, instantiated
│   └── C_core_implementation.py    Core algorithmic components, importable
├── data/
│   ├── preference_counts.csv       Human study vote counts (n = 56)
│   └── README.md                   Study design and data schema
├── analysis/
│   ├── reproduce_stats.py          Regenerates Table 2 of the paper
│   └── requirements.txt
└── requirements.txt                Pipeline dependencies
```

### Generated output file map

The files in `generated_outputs/` map to the pipeline versions referenced in
the paper and figures as follows:

| Genre (paper label) | V1 | V2 | V3 | V4 | V5 |
| --- | --- | --- | --- | --- | --- |
| Sad / Introspective | `sad_introspective_v1.png` | `sad_introspective_v2.png` | `sad_introspective_v3.png` | `sad_introspective_v4.png` | `sad_v5.png` |
| Upbeat Pop | `upbeat_pop_v1.png` | `upbeat_pop_v2.png` | `upbeat_pop_v3.png` | `upbeat_pop_v4.png` | `pop_v5.png` |
| Rap / Ambition | `rap_v1.png` | `rap_v2.png` | `rap_v3.png` | `rap_v4.png` | `rap_v5.png` |

All 15 covers are present. V1–V4 (SD v1.5, from `main.ipynb`) are 512×512. The
three V5 covers (SDXL + genre override, from `improved_main.ipynb`) use the
shortened `{sad,pop,rap}_v5.png` filenames expected by `analysis/clip_eval.py`.
The committed V5 PNGs are 548×548 exports; regenerate from `improved_main.ipynb`
for the full 1024×1024 resolution described under "Generation configuration".

## Reproducing the statistics

```bash
pip install -r analysis/requirements.txt
python analysis/reproduce_stats.py
```

This regenerates the chi-squared goodness-of-fit tests, Cramér's V, one-sided
binomial tests against the 20% chance baseline, Cohen's h, mean preference
scores, and bootstrap 95% confidence intervals (10,000 resamples, seed 42) —
all values in Table 2. The Friedman test requires the per-participant response
matrix; see `data/README.md`.

## Pipeline stages

| Stage | Method | Failure addressed |
| --- | --- | --- |
| 1. Preprocessing | Lemmatization + custom stopwords | NLTK's default list strips 47 emotionally load-bearing tokens |
| 2. Keyword extraction | TF-IDF + emotion-boosted frequency | RAKE penalizes the single-word anchors that dominate lyrics |
| 3. Emotion classification | DistilRoBERTa, 7-way | Hip-hop register → fear (0.331), no decisive signal |
| 4. Prompt engineering | Narrative template + palette + genre override | Style descriptors override the emotion signal |
| 5. Image generation | SDXL + negative prompting | Text hallucination artifacts in SD v1.5 |

## Generation configuration

Fixed seed 42 throughout, so differences between outputs reflect prompt and
configuration changes rather than sampling noise. V1–V4: SD v1.5, 512×512, 30
denoising steps, guidance scale 7.5, attention slicing, CPU-only (~15 min per
image). V5: SDXL, 1024×1024, with negative prompting to suppress text
artifacts.

15 steps showed visible denoising artifacts; 50 steps gave marginal gains at
2× cost. Guidance above 10 produced oversaturation and anatomical distortion,
below 6 produced off-prompt drift.

## Limitations

Evaluation used three synthetic lyric samples and 56 raters. Real lyrics carry
greater diversity, including mixed emotional content, ironic phrasing, and
genre-blending, and the pipeline has not been tested on songs where lyrical
sentiment contradicts musical tonality. All generation ran on CPU. The
forced-choice design limits cross-genre comparison. The confidence-as-fear
finding is characterized on a single classifier; the argument that it reflects
register shift rather than a model-specific artifact is mechanistic, and
confirming it across other standard-corpus classifiers remains open.

## Citation

```bibtex
@inproceedings{kandukoori2026confidence,
  title     = {Confidence Misread as Fear: Register Shift and Genre
               Conditioning in Lyric-to-Image Synthesis},
  author    = {Kandukoori, Aarush and Murugan, Karthik and
               Nukala, Jai and Kang, Eunsu},
  booktitle = {NeurIPS 2026 Creative AI Track},
  year      = {2026}
}
```

## Note on the corpus

All lyric samples in `appendix/A_lyric_corpus.md` are synthetic, written for
this project to avoid copyright complications. Commercial tracks referenced in
the paper are cited to characterize genre register only; none of their lyrics
are reproduced here.
