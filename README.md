# Movie Review Sentiment Analysis

## Project Overview

The Film Junky Union, a community for classic movie enthusiasts, is developing a system to filter and categorize movie reviews, starting with the automatic detection of negative reviews. This project builds an NLP pipeline on about 47,000 IMDB reviews labeled by polarity, comparing text normalization, NLTK stop-word removal, spaCy lemmatization and TF-IDF features combined with Logistic Regression and LightGBM classifiers.

**Key outcome:** TF-IDF + Logistic Regression reached an **F1-score of 0.88 and ROC-AUC of 0.95 on the test set**, above the required F1 of 0.85, with the simplest and cheapest pipeline among the models tested. The same test metrics were reproduced in two independent runs, in different environments (see [Notebooks](#notebooks)).

## Business Problem

Manually moderating thousands of user reviews does not scale. An automatic classifier lets the community flag negative reviews and organize content by sentiment. The model must be accurate on both classes, which is why the F1-score, the balance between precision and recall, is the success criterion.

## Objectives

- Train a model that classifies movie reviews as positive or negative.
- Reach an **F1-score of at least 0.85** on the test set.
- Compare different text preprocessing approaches and classifiers.

## Dataset Description

`imdb_reviews.tsv` (tab-separated), provided for educational purposes as part of the TripleTen Data Science program, contains 47,331 reviews and 17 columns.

| Column | Description |
|---|---|
| `review` | Review text (model input) |
| `pos` | **Target:** 1 = positive, 0 = negative |
| `ds_part` | Predefined split: `train` (23,796 reviews) or `test` (23,535 reviews) |
| `rating` | Review rating on a 1–10 scale; only the values 1–4 and 7–10 occur (no neutral ratings) |
| `tconst`, `title_type`, `primary_title`, `original_title`, `start_year`, `end_year`, `runtime_minutes`, `is_adult`, `genres`, `average_rating`, `votes` | Movie metadata |
| `sp` | Text version of the label (`neg` / `pos`) |
| `idx` | Record index |

**Class balance:** 23,715 negative and 23,616 positive reviews; the two classes have similar sizes in both splits.

## Technologies and Libraries

- Python
- pandas, NumPy
- Matplotlib, seaborn
- scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `DummyClassifier`, `sklearn.metrics`)
- NLTK (English stop words)
- spaCy (`en_core_web_sm`, lemmatization)
- LightGBM (`LGBMClassifier`)
- tqdm
- Jupyter Notebook

## Methodology

1. **Exploratory analysis:** number of movies and reviews per year, number of reviews per movie (exact counts and KDE), class balance, rating distributions, and polarity distribution per year and per movie for each split. The classes are balanced, so no resampling is needed, and the predefined `ds_part` split is used as is.
2. **Text normalization:** lowercasing, accent removal (Unicode NFKD), replacement of every non-letter character with a space, and whitespace collapsing.
3. **Evaluation routine:** a single function reports accuracy, F1, average precision (APS) and ROC-AUC for the training and test sets, and plots F1 vs. threshold, the ROC curve and the precision-recall curve. This makes all models directly comparable.
4. **Models:**
   - **Model 0:** `DummyClassifier` (most frequent class) as a baseline;
   - **Model 1:** TF-IDF with NLTK English stop words + Logistic Regression;
   - **Model 3:** spaCy lemmatization + TF-IDF + Logistic Regression;
   - **Model 4:** spaCy lemmatization + TF-IDF + LightGBM.

   The numbering follows the variable names in the code. TF-IDF is always fitted on the training corpus only and then applied to the test corpus.
5. **BERT:** evaluated as an option but not implemented, because computing BERT embeddings for tens of thousands of reviews on a CPU can take several hours, and the lighter models already met the requirement.
6. **Sanity check:** the trained models score eight hand-written reviews with clear, negative and mixed sentiment.

## Results and Model Performance

| Model | Train F1 | **Test F1** | Test accuracy | Test APS | Test ROC-AUC |
|---|---|---|---|---|---|
| Model 0 — Dummy (most frequent) | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 |
| **Model 1 — NLTK + TF-IDF + Logistic Regression** | 0.94 | **0.88** | 0.88 | 0.95 | 0.95 |
| Model 3 — spaCy + TF-IDF + Logistic Regression | 0.93 | **0.88** | 0.88 | 0.95 | 0.95 |
| Model 4 — spaCy + TF-IDF + LightGBM | 0.91 | 0.86 | 0.86 | 0.93 | 0.94 |

Requirement: test F1 ≥ 0.85. Metrics are rounded to two decimals by the evaluation routine.

**Hand-written reviews (Model 1, probability of a positive review)**

| Review (excerpt) | P(positive) |
|---|---|
| "I simply did not like this movie…" | 0.17 |
| "I was bored and fell asleep halfway through the movie." | 0.07 |
| "I was truly fascinated by this film." | 0.73 |
| "The actors looked tired and uninterested…" | 0.02 |
| "I did not expect the new version to be this good…" | 0.22 |
| "The movie has its pros and cons, but overall it is a decent film…" | 0.41 |
| "What a terrible attempt at comedy…" | 0.01 |
| "Releasing it on Netflix was a bold move, and I really enjoyed…" | 0.74 |

Values from the main notebook. The local re-run produces probabilities within ±0.01 for Model 1 (see [docs/notebook_local.md](docs/notebook_local.md)).

## Key Findings

- **Three of the four models meet the requirement**, and TF-IDF + Logistic Regression is the strongest (test F1 0.88, ROC-AUC 0.95).
- **Lemmatization added cost without benefit:** spaCy lemmatization (Model 3) produced the same test metrics as simple normalization (Model 1) while adding a slow preprocessing step.
- **Gradient boosting did not help on sparse TF-IDF features:** LightGBM (Model 4) was slightly worse than Logistic Regression.
- **The model handles clear sentiment well but struggles with nuance:** clearly negative reviews receive very low probabilities, and the mixed review receives an intermediate one (0.41). However, the positive review "I did not expect the new version to be this good…" received only 0.22, which shows a weakness with negations and contrastive phrasing that a bag-of-words model cannot capture.

## Conclusions

**Achievement of objectives:** the objective was achieved. Model 1 (NLTK + TF-IDF + Logistic Regression) is the recommended solution, as it combines the best test performance with simplicity and low computational cost.

**Limitations**

- There is a gap between training and test F1 (0.94 vs. 0.88), which indicates some overfitting; regularization was not tuned.
- No hyperparameters were tuned (for example, the `TfidfVectorizer` n-gram range or the Logistic Regression `C`).
- BERT, which could handle negation and context better, was not implemented.
- The `kernel='epa'` argument passed to `sns.kdeplot` is no longer supported by seaborn, which fell back to a Gaussian kernel, as the warnings in the saved outputs show.
- The main notebook uses the Matplotlib style name `'seaborn'`, which is no longer available in recent Matplotlib versions (see [How to Run](#how-to-run)).
- The saved outputs of `notebook_local.ipynb` contain library warnings with local file paths.

**Potential improvements:** add word n-grams (for example, bigrams to capture "not good"), tune regularization with cross-validation on the training set, and evaluate transformer embeddings (BERT or DistilBERT) on a GPU.

## Notebooks

| Notebook | Description |
|---|---|
| [`notebook.ipynb`](notebook.ipynb) | **Main notebook.** Executed on the TripleTen platform (Python 3.9) and approved in code review; the results above come from this file. |
| [`notebook_local.ipynb`](notebook_local.ipynb) | Local re-run of the same analysis with Python 3.13 and current library versions. The test metrics are identical; only the hand-written review probabilities differ slightly. See [docs/notebook_local.md](docs/notebook_local.md). |

## Repository Structure

```text
movie-review-sentiment-analysis/
├── README.md
├── notebook.ipynb
├── notebook_local.ipynb
├── docs/
│   └── notebook_local.md
└── datasets/
    └── imdb_reviews.tsv
```

## How to Run

The two notebooks have different environment requirements:

| Notebook | Python | Matplotlib | Notes |
|---|---|---|---|
| `notebook.ipynb` (main) | 3.9 | **below 3.6** | Uses `plt.style.use('seaborn')`, which is not available in recent Matplotlib versions. The exact library versions of the platform run were not recorded. |
| `notebook_local.ipynb` | 3.13 | 3.10.9 | Tested with pandas 3.0.2, NumPy 2.4.4, scikit-learn 1.8.0, LightGBM 4.6.0, spaCy 3.8.14, NLTK 3.9.4, seaborn 0.13.2 and tqdm 4.67.3. |

1. Create and activate a Python environment that matches the notebook you want to run:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install the dependencies and the language resources:

   ```bash
   # for notebook.ipynb (Python 3.9)
   pip install pandas numpy scikit-learn lightgbm spacy nltk "matplotlib<3.6" seaborn tqdm jupyter ipywidgets
   # for notebook_local.ipynb (Python 3.13)
   pip install pandas numpy scikit-learn lightgbm spacy nltk matplotlib seaborn tqdm jupyter ipywidgets

   # language resources (both notebooks)
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('stopwords')"
   ```

3. Start Jupyter **from the project root**, so that the relative path `datasets/imdb_reviews.tsv` resolves correctly, and open the notebook:

   ```bash
   jupyter notebook notebook.ipynb
   ```

**Execution notes**

- spaCy lemmatization of about 47,000 reviews is run twice (for Models 3 and 4) and can take a long time on a CPU.
- `imdb_reviews.tsv` is about 63 MB. It is below GitHub's 100 MB hard limit but above the 50 MB warning threshold, so Git LFS is recommended if the dataset is versioned.
- Chart titles are partly in Portuguese, as in the original run.
- In a current Python environment, run `notebook_local.ipynb`. The main notebook fails at `plt.style.use('seaborn')` with recent Matplotlib versions.

## Author

**Luiz Otávio Carneiro Filho**
Geospatial Engineer | GIS & Data Science
[LinkedIn](https://www.linkedin.com/in/luizot/)
