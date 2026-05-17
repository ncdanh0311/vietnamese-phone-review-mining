# Vietnamese Phone Review Topic Clustering and Sentiment Analysis

## 1. Overview

This project mines Vietnamese phone review data from the UIT-ViSFD dataset. The pipeline combines two main approaches:

- **K-Means clustering** to group reviews into topic clusters based on TF-IDF features.
- **LinearSVC sentiment classification** to classify overall sentiment into `Positive`, `Negative`, and `Neutral`.

The notebooks are used to present and analyze each step. The `.py` files in `src/` keep the core logic reusable, testable, and runnable through the full pipeline with `python main.py`.

## 2. Dataset

Dataset source: [UIT-ViSFD](https://github.com/LuongPhan/UIT-ViSFD)

| Dataset | Path | Samples |
|---|---|---:|
| Train | `datasets/raw/Train.csv` | 7,786 |
| Test | `datasets/raw/Test.csv` | 2,224 |

Each row contains the review text, star rating, timestamp, and aspect-based labels in this format:

```text
{CAMERA#Positive};{BATTERY#Negative};{PRICE#Neutral};
```

Main aspects include `CAMERA`, `BATTERY`, `FEATURES`, `PERFORMANCE`, `SCREEN`, `PRICE`, `GENERAL`, `DESIGN`, `SER&ACC`, and `OTHERS`.

## 3. Cleaned Project Structure

```text
vietnamese-phone-sentiment-mining/
├── configs/
│   └── config.yaml
├── datasets/
│   ├── raw/
│   │   ├── Train.csv
│   │   └── Test.csv
│   ├── cleaned/
│   │   ├── train_clean.csv
│   │   └── test_clean.csv
│   └── stopwords/
│       └── vietnamese-stopwords.txt
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_preprocessing_text.ipynb
│   ├── 03_tfidf_kmeans_clustering.ipynb
│   ├── 04_svm_classification.ipynb
│   └── 05_evaluation_visualization.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── classification.py
│   └── visualization.py
├── results/
│   ├── csv/
│   ├── figures/
│   └── models/
├── main.py
├── requirements.txt
├── README.md
└── README_VI.md
```

Removed unnecessary files and folders: `src/__pycache__/`, `datasets/processed/`, `results/logs/`, `datasets/cleaned/trainprocessed.csv`, and `datasets/cleaned/testprocessed.csv`.

## 4. Processing Pipeline

```text
Raw CSV
  -> EDA
  -> Text cleaning
  -> Overall sentiment and main_aspect assignment
  -> TF-IDF vectorization
  -> K-Means clustering
  -> Cluster vs. true aspect comparison
  -> LinearSVC classification
  -> Evaluation and visualization
```

The preprocessing pipeline in `src/preprocessing.py` includes:

1. Convert text to lowercase.
2. Remove punctuation and emoji.
3. Normalize numbers into the `number` token.
4. Remove Vietnamese stopwords.
5. Tokenize text with `pyvi.ViTokenizer`.
6. Remove repeated consecutive words.
7. Parse `label` into `sentiment`, `main_aspect`, and `aspects_list`.

## 5. EDA Visuals

### Sentiment Distribution

![Sentiment distribution](results/figures/eda_sentiment_distribution.png)

### Main Aspect Distribution

![Aspect distribution](results/figures/eda_aspect_distribution.png)

### Star Rating vs. Sentiment

![Star vs sentiment](results/figures/eda_star_vs_sentiment.png)

### Top Words in Positive and Negative Reviews

![Top words](results/figures/eda_top_words_positive_negative.png)

## 6. K-Means Clustering

TF-IDF configuration:

```python
TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

K-Means is tested with `K=2..10`. The best K by Silhouette Score in the current run is:

| Metric | Value |
|---|---:|
| Best K | 10 |
| Silhouette | 0.0086 |

The Silhouette Score is very low, which suggests that TF-IDF + K-Means does not separate topics clearly on this review dataset. Some clusters still have clear dominant aspects, such as clusters `0` and `6` leaning strongly toward `CAMERA`, and cluster `7` leaning toward `BATTERY`.

### Elbow and Silhouette

![Elbow](results/figures/elbow.png)

![Silhouette](results/figures/silhouette.png)

### Cluster Size and PCA

![Cluster size](results/figures/kmeans_cluster_size.png)

![PCA K-Means](results/figures/pca_kmeans.png)

### Sentiment by Topic Cluster

![Sentiment by cluster](results/figures/sentiment_by_cluster.png)

Word clouds for individual clusters are saved from `results/figures/wordcloud_cluster_0.png` to `wordcloud_cluster_9.png`.

## 7. SVM Sentiment Classification

The classifier uses `LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)`. The baseline model is `DummyClassifier(strategy="most_frequent")`.

Results after rerunning the notebooks and `main.py`:

| Metric | Value |
|---|---:|
| Baseline F1 macro | 0.2218 |
| SVM F1 macro | 0.6631 |
| SVM F1 weighted | 0.8000 |
| Accuracy | 0.8008 |
| Precision macro | 0.6635 |
| Recall macro | 0.6629 |
| Cross-validation F1 macro mean | 0.6419 |
| Cross-validation F1 macro std | 0.0131 |

SVM clearly improves over the baseline, but macro F1 is still limited because the `Neutral` class has fewer samples and is harder to distinguish from mildly positive or mildly negative reviews.

### Confusion Matrix

![Confusion matrix](results/figures/confusion_matrix.png)

## 8. Notebook Workflow

All notebooks were executed successfully in this order:

| Notebook | Role | Main Outputs |
|---|---|---|
| `01_explore_dataset.ipynb` | Explore raw data | EDA figures |
| `02_preprocessing_text.ipynb` | Clean text and create sentiment/aspect fields | `train_clean.csv`, `test_clean.csv` |
| `03_tfidf_kmeans_clustering.ipynb` | TF-IDF, K selection, K-Means | K-Means model, vectorizer, clustered CSV, PCA/word clouds |
| `04_svm_classification.ipynb` | Train and evaluate SVM | SVM model, metrics, confusion matrix |
| `05_evaluation_visualization.ipynb` | Summarize results | Insights and summary charts |

## 9. How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py
```

Run notebooks through the Jupyter UI:

```bash
python -m jupyter notebook
```

Or execute notebooks from the terminal:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_explore_dataset.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing_text.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_tfidf_kmeans_clustering.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_svm_classification.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_evaluation_visualization.ipynb
```

## 10. Important Output Files

| File | Meaning |
|---|---|
| `datasets/cleaned/train_clean.csv` | Preprocessed training data |
| `datasets/cleaned/test_clean.csv` | Preprocessed test data |
| `results/csv/kmeans_k_scores.csv` | Inertia and Silhouette for each K |
| `results/csv/kmeans_cluster_summary.csv` | Dominant aspect and match ratio for each cluster |
| `results/csv/kmeans_clustered.csv` | Training data with cluster labels |
| `results/csv/svm_metrics_summary.csv` | Summary metrics for SVM |
| `results/csv/svm_results.csv` | SVM predictions on the test set |
| `results/models/tfidf_vectorizer.pkl` | Trained TF-IDF vectorizer |
| `results/models/kmeans_model.pkl` | K-Means model |
| `results/models/svm_model.pkl` | LinearSVC model |

## 11. Conclusion

SVM sentiment classification performs well compared with the baseline, especially in accuracy and weighted F1. However, macro F1 is around `0.6631`, showing that the model still needs improvement on smaller or more ambiguous classes.

K-Means with TF-IDF produces a few clusters with dominant aspects, but the Silhouette Score of `0.0086` is very low. This indicates that sparse TF-IDF vectors and multi-aspect reviews make clear topic separation difficult for K-Means.

## 12. Future Work

- Try PhoBERT or Vietnamese sentence embeddings instead of TF-IDF.
- Use BERTopic, NMF, or HDBSCAN for topic modeling.
- Move from overall sentiment classification to aspect-level sentiment analysis.
- Balance the dataset or add more samples for the `Neutral` class.
- Refine stopwords and add a phone-domain vocabulary.
