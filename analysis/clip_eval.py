"""
CLIP evaluation for the lyric-to-album-cover pipeline.

Computes CLIP similarity scores between each generation prompt and its
corresponding generated image, for every pipeline version (V1..V5) across the
three lyric genres, and reports the results table used in the paper:

    Confidence Misread as Fear: Register Shift and Genre Conditioning
    in Lyric-to-Image Synthesis
    (NeurIPS 2026 Creative AI Track)

This is a standalone extraction of ``CLIP_Scores_Generation.ipynb``. The
notebook uploaded the images interactively via ``google.colab.files``; this
script instead reads them from a directory on disk (``generated_outputs/`` by
default) so it can be run from a normal shell:

    pip install torch torchvision transformers Pillow matplotlib numpy
    python analysis/clip_eval.py --images-dir generated_outputs

Model: ``openai/clip-vit-base-patch32``. Higher scores indicate stronger
semantic alignment between the image and its conditioning prompt.

Note: the V5 covers (``sad_v5.png``, ``pop_v5.png``, ``rap_v5.png``) are the
SDXL + genre-override outputs from ``improved_main.ipynb``. Any image the script
cannot find on disk is skipped with a warning, so it still runs on whatever
subset of the covers is present.
"""

import argparse
import os

import numpy as np


# ---------------------------------------------------------------------------
# Generation prompts, verbatim per pipeline version and genre.
# ---------------------------------------------------------------------------
PROMPTS = {
    "sad": {
        "V1": "An album cover art that conveys sadness emotions featuring figure somehow, rise",
        "V2": "A dark, moody album cover art conveying sadness, fear emotions. Imagery: darkness, memories, shadows, pain, light. Style: dramatic, cinematic, high contrast.",
        "V3": "Album cover: SADNESS and fear. Visual palette: deep blues and purples. Imagery: shadows, cry, night, fall, darkness. Mood: melancholic, brooding. Style: cinematic photography, dramatic lighting.",
        "V4": "A cinematic album cover depicting the emotional journey of sadness and fear. Scene: shadows, cry, night, fall, darkness bathed in deep blues and purples. Mood: melancholic and brooding. Art direction: dramatic lighting, high contrast, professional photography.",
        "V5": "A cinematic album cover depicting grief and introspection. Scene: solitary figure in deep violet fog beneath a streetlamp, bare winter trees, fallen leaves, deep blues and purples. Mood: melancholic, isolated, and quietly hopeful. Art direction: atmospheric lighting, soft focus, professional photography.",
    },
    "pop": {
        "V1": "An album cover art that conveys joy emotions featuring bright dancing alive",
        "V2": "A vibrant, energetic album cover art conveying joy emotions. Imagery: bright, dancing, alive, joy, light. Style: dramatic, cinematic, high contrast.",
        "V3": "Album cover: JOY. Visual palette: warm brightness and golden tones. Imagery: bright, dancing, alive, joy, light. Mood: euphoric, energetic. Style: cinematic photography, dramatic lighting.",
        "V4": "A cinematic album cover depicting the emotional journey of joy and energy. Scene: bright, dancing, alive, soar, free bathed in warm brightness and golden tones. Mood: euphoric and energetic. Art direction: dramatic lighting, high contrast, professional photography.",
        "V5": "A cinematic album cover depicting pure joy and celebration. Scene: woman dancing in golden-hour confetti shower, arms outstretched, warm pinks and oranges, glitter and light bokeh. Mood: euphoric, free, alive. Art direction: sunset backlight, shallow depth of field, photorealistic.",
    },
    "rap": {
        "V1": "An album cover art that conveys fear emotions featuring ever seem",
        "V2": "A dark, dramatic album cover art conveying fear emotions. Imagery: city, rise, hustle, top, run. Style: dramatic, cinematic, high contrast.",
        "V3": "Album cover: FEAR. Visual palette: desaturated greys and cold blues. Imagery: rise, city, hustle, grind, watch. Mood: tense, determined. Style: cinematic photography, dramatic lighting.",
        "V4": "A cinematic album cover depicting determination and ambition. Scene: figure between large rock formations, dramatic sky, conviction and isolation. Mood: determined and powerful. Art direction: dramatic lighting, high contrast, professional photography.",
        "V5": "A cinematic hip-hop album cover depicting confidence, ambition, and urban power. Scene: silhouetted figure in hat and gold chains crouching on rooftop overlooking city skyline at crimson sunset, Empire State Building visible. Mood: triumphant and determined. Art direction: dramatic low-angle shot, warm golden-red tones, graphic novel illustration style.",
    },
}

# Image filename per genre and version, relative to --images-dir.
IMAGE_FILES = {
    "sad": {
        "V1": "sad_introspective_v1.png",
        "V2": "sad_introspective_v2.png",
        "V3": "sad_introspective_v3.png",
        "V4": "sad_introspective_v4.png",
        "V5": "sad_v5.png",
    },
    "pop": {
        "V1": "upbeat_pop_v1.png",
        "V2": "upbeat_pop_v2.png",
        "V3": "upbeat_pop_v3.png",
        "V4": "upbeat_pop_v4.png",
        "V5": "pop_v5.png",
    },
    "rap": {
        "V1": "rap_v1.png",
        "V2": "rap_v2.png",
        "V3": "rap_v3.png",
        "V4": "rap_v4.png",
        "V5": "rap_v5.png",
    },
}

GENRES = ["sad", "pop", "rap"]
VERSIONS = ["V1", "V2", "V3", "V4", "V5"]

GENRE_LABELS = {"sad": "Sad / Introspective", "pop": "Upbeat Pop", "rap": "Rap / Ambition"}
GENRE_COLORS = {"sad": "#5b6dcd", "pop": "#e09f3e", "rap": "#9e2a2b"}


def load_clip():
    """Load the CLIP model and processor once."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, torch


def clip_score(model, processor, torch, image_path, text):
    """CLIP score = cosine similarity scaled by 100 (logits_per_image)."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[text], images=image,
        return_tensors="pt", padding=True, truncation=True, max_length=77,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits_per_image.item()


def compute_scores(images_dir):
    """Return results[genre][version] = score for every image found on disk."""
    model, processor, torch = load_clip()
    results = {}
    missing = []
    print("Computing CLIP scores...")
    print("-" * 50)
    for genre in GENRES:
        results[genre] = {}
        for v in VERSIONS:
            fname = IMAGE_FILES[genre][v]
            path = os.path.join(images_dir, fname)
            if not os.path.exists(path):
                missing.append(fname)
                print(f"{genre.upper():4s} {v}: SKIPPED (missing {fname})")
                continue
            score = clip_score(model, processor, torch, path, PROMPTS[genre][v])
            results[genre][v] = score
            print(f"{genre.upper():4s} {v}: {score:.4f}")
    if missing:
        print(f"\n{len(missing)} image(s) not found and skipped: {', '.join(missing)}")
    return results


def print_table(results):
    print("\n" + "=" * 60)
    print("CLIP SCORE RESULTS")
    print("=" * 60)
    print(f"{'Version':<8} {'Sad':>10} {'Pop':>10} {'Rap':>10}")
    print("-" * 40)
    for v in VERSIONS:
        cells = []
        for genre in GENRES:
            s = results[genre].get(v)
            cells.append(f"{s:>10.4f}" if s is not None else f"{'--':>10}")
        print(f"{v:<8} " + " ".join(cells))
    print("-" * 40)
    for genre in GENRES:
        scores = list(results[genre].values())
        if not scores:
            continue
        best_v = max(results[genre], key=results[genre].get)
        print(f"{genre.upper()} — mean: {np.mean(scores):.4f}, "
              f"best: {best_v} ({results[genre][best_v]:.4f})")


def print_latex(results):
    """Emit the LaTeX table body used in the paper."""
    print("\n% -- PASTE THIS INTO paper.tex " + "-" * 30)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{CLIP similarity scores between generation prompts and "
          "corresponding generated images, computed using "
          "\\texttt{openai/clip-vit-base-patch32}. Higher scores indicate "
          "stronger semantic alignment between the image and its conditioning prompt.}")
    print("\\label{tab:clip}")
    print("\\begin{tabular}{lrrr}")
    print("\\toprule")
    print("Version & Sad / Introspective & Upbeat Pop & Rap / Ambition \\\\")
    print("\\midrule")
    for v in VERSIONS:
        b_o = "\\textbf{" if v == "V5" else ""
        b_c = "}" if v == "V5" else ""
        cells = []
        for genre in GENRES:
            s = results[genre].get(v)
            cells.append(f"{b_o}{s:.2f}{b_c}" if s is not None else "--")
        print(f"{b_o}{v}{b_c} & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def make_figure(results, out_path):
    """Render the two-panel CLIP-score figure. Requires all five versions."""
    if any(len(results[g]) < len(VERSIONS) for g in GENRES):
        print(f"\nFigure skipped: needs all {len(VERSIONS)} versions for every genre.")
        return

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 150, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for genre in GENRES:
        scores = [results[genre][v] for v in VERSIONS]
        ax.plot(VERSIONS, scores, marker="o", markersize=8, linewidth=2,
                label=GENRE_LABELS[genre], color=GENRE_COLORS[genre])
        ax.scatter(["V5"], [results[genre]["V5"]], s=150, color=GENRE_COLORS[genre],
                   edgecolor="black", linewidth=1.5, zorder=5)
    ax.set_xlabel("Pipeline Version")
    ax.set_ylabel("CLIP Score (cosine similarity x 100)")
    ax.set_title("CLIP Score Progression: V1 -> V5", pad=10)
    ax.legend(loc="best", frameon=True, framealpha=0.95)

    ax2 = axes[1]
    x = np.arange(len(GENRES))
    width = 0.35
    v1_scores = [results[g]["V1"] for g in GENRES]
    v5_scores = [results[g]["V5"] for g in GENRES]
    ax2.bar(x - width / 2, v1_scores, width, label="V1 (Baseline)",
            color="#cccccc", edgecolor="black", linewidth=0.8)
    ax2.bar(x + width / 2, v5_scores, width, label="V5 (Final)",
            color=[GENRE_COLORS[g] for g in GENRES],
            edgecolor="black", linewidth=0.8)
    for i, (v1, v5) in enumerate(zip(v1_scores, v5_scores)):
        delta = v5 - v1
        ax2.annotate(f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}",
                     xy=(i + width / 2, v5), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=9, color="darkgreen" if delta >= 0 else "red",
                     fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([GENRE_LABELS[g].split("/")[0].strip() for g in GENRES])
    ax2.set_ylabel("CLIP Score")
    ax2.set_title("V1 vs V5 CLIP Score Comparison", pad=10)
    ax2.legend(frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"\nFigure saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", default="generated_outputs",
                        help="Directory holding the generated cover PNGs "
                             "(default: generated_outputs)")
    parser.add_argument("--figure", default=None,
                        help="If set, path to save the two-panel CLIP-score "
                             "figure (e.g. fig_clip_scores.png)")
    parser.add_argument("--latex", action="store_true",
                        help="Also print the LaTeX table body for the paper")
    args = parser.parse_args()

    results = compute_scores(args.images_dir)
    print_table(results)
    if args.latex:
        print_latex(results)
    if args.figure:
        make_figure(results, args.figure)


if __name__ == "__main__":
    main()
