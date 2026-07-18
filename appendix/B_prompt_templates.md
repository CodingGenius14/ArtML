# Appendix B — Prompt Template Examples

Five template versions were evaluated. V1–V4 ran on Stable Diffusion v1.5
(512x512, seed 42, 30 steps, guidance 7.5, CPU). V5 ran on SDXL (1024x1024)
and adds the genre-conditioned override.

## V1 — Baseline (~20 words)

> An album cover art that conveys {emotion} emotions featuring {rake_keywords}.

Instantiated on sad lyrics:

> An album cover art that conveys sadness emotions featuring figure somehow, rise.

No artistic direction. Combined with RAKE's degraded keyword output, produces
incoherent or texturally empty results.

## V2 — Style descriptors (~30 words)

> A dark, moody album cover art conveying {emotions}. Imagery: {keywords}.
> Style: dramatic, cinematic, high contrast.

Introduces the critical interaction failure: for pop lyrics, "cinematic, high
contrast" semantically overrides a 0.980-confidence joy signal, producing a
black-and-white rainy alleyway for an upbeat song. Received zero votes.

## V3 — Emotion-to-palette mapping (~45 words)

> Album cover: {PRIMARY_EMOTION} and {secondary}. Visual palette:
> {mapped_palette}. Imagery: {keywords}. Mood: {mood}. Style: cinematic
> photography, dramatic lighting.

Resolves the V2 style/emotion conflict by encoding the palette explicitly.

## V4 — Narrative art direction (~65 words)

> A cinematic album cover depicting the emotional journey of {emotion}.
> Scene: {keywords} bathed in {palette}. Mood: {mood}. Art direction:
> dramatic lighting, high contrast, professional photography.

Instantiated on sad lyrics:

> A cinematic album cover depicting the emotional journey of sadness and fear.
> Scene: shadows, cry, night, fall, darkness bathed in deep blues and purples.
> Mood: melancholic and brooding. Art direction: dramatic lighting, high
> contrast, professional photography.

## V5 — Genre-conditioned override + SDXL (~70 words)

Instantiated on rap lyrics:

> A cinematic hip-hop album cover depicting confidence, ambition, and urban
> power. Scene: city skyline at golden hour, solitary figure on a rooftop,
> chain, crown, neon lights below. Mood: triumphant and determined.
> Art direction: dramatic low-angle shot, warm golden tones, photorealistic.

Negative prompt used for all V5 generation:

> text, watermark, words, letters, typography, caption, subtitle
