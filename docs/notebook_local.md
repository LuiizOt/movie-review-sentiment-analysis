# `notebook_local.ipynb` — Local Re-run of the Sentiment Analysis

[← Back to the project README](../README.md)

## Purpose

`notebook_local.ipynb` is a re-run of the main notebook, [`notebook.ipynb`](../notebook.ipynb), in an updated local environment. It implements **the same pipeline**: EDA, text normalization, the shared evaluation routine, and Models 0, 1, 3 and 4. The main notebook is the version that was executed on the TripleTen platform and approved in code review.

The local re-run is useful for two reasons:

- it shows that the results reproduce in a different environment;
- it runs with current library versions.

## Execution environment

| Item | `notebook.ipynb` (main) | `notebook_local.ipynb` |
|---|---|---|
| Where it was run | TripleTen platform | Local machine |
| Python | 3.9 | 3.13 |
| Matplotlib style | `plt.style.use('seaborn')` | `plt.style.use('seaborn-v0_8')` |
| Dataset path | `datasets/imdb_reviews.tsv` (originally `/datasets/imdb_reviews.tsv`) | `./datasets/imdb_reviews.tsv` |
| Library versions | Not recorded | pandas 3.0.2, NumPy 2.4.4, scikit-learn 1.8.0, LightGBM 4.6.0, spaCy 3.8.14, NLTK 3.9.4, Matplotlib 3.10.9, seaborn 0.13.2 |

The `'seaborn'` style name used in the main notebook was deprecated in Matplotlib 3.6 and is not available in recent versions. In a current environment, run `notebook_local.ipynb`, which uses the renamed style `'seaborn-v0_8'`.

## Results

The evaluation metrics are **identical** in both notebooks:

| Model | Train F1 | Test F1 | Test accuracy | Test APS | Test ROC-AUC |
|---|---|---|---|---|---|
| Model 0 — Dummy (most frequent) | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 |
| Model 1 — NLTK + TF-IDF + Logistic Regression | 0.94 | 0.88 | 0.88 | 0.95 | 0.95 |
| Model 3 — spaCy + TF-IDF + Logistic Regression | 0.93 | 0.88 | 0.88 | 0.95 | 0.95 |
| Model 4 — spaCy + TF-IDF + LightGBM | 0.91 | 0.86 | 0.86 | 0.93 | 0.94 |

## Differences in the hand-written review predictions

Probabilities that each hand-written review is positive (main notebook / local re-run):

| Review (excerpt) | Model 1 | Model 3 | Model 4 |
|---|---|---|---|
| "I simply did not like this movie…" | 0.17 / 0.18 | 0.24 / 0.24 | 0.58 / 0.62 |
| "I was bored and fell asleep…" | 0.07 / 0.07 | 0.06 / 0.07 | 0.37 / 0.39 |
| "I was truly fascinated by this film." | 0.73 / 0.73 | 0.75 / 0.76 | 0.65 / 0.68 |
| "The actors looked tired and uninterested…" | 0.02 / 0.02 | 0.01 / 0.00 | 0.05 / 0.05 |
| "I did not expect the new version to be this good…" | 0.22 / 0.22 | 0.19 / 0.19 | 0.60 / 0.60 |
| "The movie has its pros and cons…" | 0.41 / 0.42 | 0.39 / 0.40 | 0.47 / 0.51 |
| "What a terrible attempt at comedy…" | 0.01 / 0.01 | 0.02 / 0.02 | 0.09 / 0.05 |
| "Releasing it on Netflix was a bold move…" | 0.74 / 0.73 | 0.92 / 0.92 | 0.85 / 0.81 |

The differences are small for the Logistic Regression models and somewhat larger for LightGBM. They most likely come from the different library versions used in each environment, in particular scikit-learn, LightGBM, and spaCy with its `en_core_web_sm` model. The conclusions are the same in both runs.

## Other differences

- The local notebook's saved outputs include library warnings that contain local file paths.
- The local notebook explains in a Markdown cell why BERT was not used. The main notebook only marks the BERT section as not implemented.
