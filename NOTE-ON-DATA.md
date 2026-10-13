# Note on data, provenance and usage terms

## Source

Review texts were collected from publicly available app-store pages
(**Google Play** and **Apple App Store**) of **20 digital-health mobile apps**,
using the official/公开 interfaces (`google-play-scraper` for Google Play and the
iTunes RSS customer-reviews feed for the App Store). No private or personally
identifiable information is included: the released `data/absa_long.csv` contains only
the review text (already truncated), the app name, the platform and the labels derived
in this study.

## What is released

| File | Content |
|---|---|
| `data/absa_long.csv` | Aspect-level long table: one row per (review, aspect) pair, with the aspect-sentiment label and its confidence flag. |
| `lexicon/*.json` | The aspects trigger-word lexicon, the sentiment lexicon (positive/negative words, negators, intensifiers) and the annotation configuration (window size, scoring rule). |

Reviewer-identifiable fields (author name, review id hashes, timestamps) are **not**
released beyond what is necessary for reproducibility.

## Usage terms

- The released data and lexicons are provided **for academic reproducibility and
  non-commercial research only**, in connection with the associated paper.
- Copyright of the original review texts remains with their authors and the
  respective platform; the texts are redistributed here solely for the purpose of
  verifying the published results, under fair-use / research-exemption principles.
- If you are a rights holder and wish a specific item to be removed, please contact
  the authors and it will be removed promptly.
- Please cite the paper (see `CITATION.cff`) when using these files.

## Reproducibility note

`data/absa_long.csv` holds **35,717** rows. Running `code/data_prep.py` filters it
(valid aspect ∧ valid label ∧ non-empty text) down to **31,349** rows, which are then
split 8:1:1 by `review_id` into **25,066 / 3,182 / 3,101** — the exact numbers reported
in the paper.
