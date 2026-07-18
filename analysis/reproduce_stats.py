"""
Reproduces the statistical results reported in Table 2 of

    Confidence Misread as Fear: Register Shift and Genre Conditioning
    in Lyric-to-Image Synthesis
    (NeurIPS 2026 Creative AI Track)

Run:
    python analysis/reproduce_stats.py

Inputs
------
data/preference_counts.csv
    Aggregate vote counts (n = 56 participants, forced choice, one vote per
    genre). These counts are sufficient for the chi-squared, binomial, and
    bootstrap results.

Note on the Friedman test
-------------------------
The Friedman test reported in the paper is a within-subjects test and requires
the per-participant response matrix, not aggregate counts. Place that file at
data/preference_responses.csv (schema in data/README.md) and this script will
run it; otherwise that test is skipped with a notice.
"""

import csv
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COUNTS = os.path.join(ROOT, "data", "preference_counts.csv")
RESPONSES = os.path.join(ROOT, "data", "preference_responses.csv")

N_VERSIONS = 5
CHANCE = 1.0 / N_VERSIONS
SEED = 42


def load_counts(path):
    """genre -> [V1..V5] vote counts"""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["genre"]] = [int(row[f"V{i}"]) for i in range(1, N_VERSIONS + 1)]
    return out


def cramers_v(chi2, n, k):
    """Cramér's V for a 1 x k goodness-of-fit table."""
    return float(np.sqrt(chi2 / (n * (k - 1))))


def cohens_h(p1, p2):
    """Effect size for a difference of two proportions."""
    return float(abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))))


def bootstrap_ci(counts, n_boot=10_000, seed=SEED):
    """Percentile CI on mean preference, treating version index 1..5 as ordinal."""
    rng = np.random.default_rng(seed)
    votes = np.repeat(np.arange(1, N_VERSIONS + 1), counts)
    means = rng.choice(votes, size=(n_boot, votes.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    counts = load_counts(COUNTS)

    print(f"{'Genre':<20}{'chi2':>8}{'p_unif':>12}{'V':>8}"
          f"{'p_binom':>12}{'h':>8}{'mean':>8}{'  95% CI':>18}")
    print("-" * 94)

    for genre, c in counts.items():
        n = sum(c)
        expected = [n / N_VERSIONS] * N_VERSIONS

        chi2, p_unif = stats.chisquare(c, f_exp=expected)
        v = cramers_v(chi2, n, N_VERSIONS)

        v5 = c[-1]
        p_binom = stats.binomtest(v5, n, CHANCE, alternative="greater").pvalue
        h = cohens_h(v5 / n, CHANCE)

        votes = np.repeat(np.arange(1, N_VERSIONS + 1), c)
        mean, sd = votes.mean(), votes.std(ddof=1)
        lo, hi = bootstrap_ci(c)

        print(f"{genre:<20}{chi2:>8.2f}{p_unif:>12.2e}{v:>8.3f}"
              f"{p_binom:>12.2e}{h:>8.3f}{mean:>8.2f}   [{lo:.2f}, {hi:.2f}]"
              f"   sd={sd:.2f}")

    if os.path.exists(RESPONSES):
        with open(RESPONSES) as f:
            rows = list(csv.DictReader(f))
        matrix = np.array([[int(r["sad"]), int(r["pop"]), int(r["rap"])] for r in rows])
        chi2, p = stats.friedmanchisquare(*matrix.T)
        k, n = matrix.shape[1], matrix.shape[0]
        w = chi2 / (n * (k - 1))
        print(f"\nFriedman (within-subjects): chi2={chi2:.2f}, df={k-1}, "
              f"p={p:.3f}, Kendall's W={w:.3f}")
    else:
        print(f"\nFriedman test skipped: {RESPONSES} not found.")
        print("Add the per-participant matrix to reproduce that row of Table 2.")


if __name__ == "__main__":
    main()
