# Human Preference Study Data (n = 56)

## Design

Forced choice. Each participant viewed all five images (V1–V5) for each of the
three lyric genres as three separate tasks, and selected the one that best
visually represented the emotional content of the lyrics. Version labels were
hidden. No time limit.

## `preference_counts.csv`

Aggregate vote counts per genre. Sufficient to reproduce the chi-squared,
binomial, effect-size, and bootstrap results in Table 2 of the paper.

| column | meaning |
| --- | --- |
| `genre` | Sad / Introspective, Upbeat Pop, or Rap / Ambition |
| `V1`–`V5` | number of participants selecting that pipeline version |
| `n` | total respondents for that genre (56) |

Verify with:

```bash
python analysis/reproduce_stats.py
```

## `preference_responses.csv` — TO BE ADDED

The Friedman test reported in the paper is a within-subjects test and cannot be
computed from aggregate counts. It needs one row per participant:

| column | meaning |
| --- | --- |
| `participant_id` | anonymous integer, 1–56 |
| `sad` | version chosen for the sad lyrics, 1–5 |
| `pop` | version chosen for the pop lyrics, 1–5 |
| `rap` | version chosen for the rap lyrics, 1–5 |

Example:

```csv
participant_id,sad,pop,rap
1,5,5,5
2,2,4,5
3,5,3,4
```

Once present, `reproduce_stats.py` picks it up automatically and runs the
Friedman test and Kendall's W. Do not include any identifying information about
respondents.
