# Topic Clustering and Sentiment Analysis for Vietnamese Phone Reviews

## 1. Student Information

| Student ID | Full Name | Class | Email |
|---|---|---|---|
|  |  |  |  |

## 2. Project Overview

This project applies text mining and machine learning techniques to Vietnamese phone customer reviews. The pipeline includes text preprocessing, TF-IDF feature extraction, topic clustering with K-Means, and sentiment classification with LinearSVC.

The project focuses on three main goals:

- Discover review topics automatically using unsupervised clustering.
- Classify overall sentiment into Positive, Negative, and Neutral.
- Identify which product aspects receive the most negative feedback.

## 3. Research Questions

1. Can K-Means group Vietnamese phone reviews into topics that are similar to the real aspect labels?
2. How well does SVM perform in classifying sentiment into Positive, Negative, and Neutral?
3. Which review topic receives the highest proportion of negative customer feedback?

## 4. Data Information:
Data Information:
Train Data: 7,786 samples
Dev: 1,112 samples
Test Data: 2,224 samples
Link: [Dataset](https://github.com/LuongPhan/UIT-ViSFD?tab=readme-ov-file)

Input files:

```text
datasets/raw/Train.csv
datasets/raw/Test.csv
datasets/stopwords/vietnamese-stopwords.txt
```

Each raw CSV file contains the following columns:

| Column | Description |
|---|---|
| `index` | Original row index |
| `comment` | Vietnamese customer review text |
| `n_star` | Star rating from 1 to 5 |
| `date_time` | Review date/time |
| `label` | Aspect-based sentiment labels |

The `label` column has an aspect-based format such as:

```text
{CAMERA#Positive};{BATTERY#Negative};{PRICE#Neutral};
```

Common aspects include:

```text
CAMERA, BATTERY, PRICE, FEATURES, PERFORMANCE, GENERAL, SER&ACC, OTHERS
```

## 5. Project Structure

```text
vietnamese-phone-sentiment-mining/
├── configs/
│   └── config.yaml
├── datasets/
│   ├── raw/
│   │   ├── Train.csv
│   │   └── Test.csv
│   ├── cleaned/
│   │   ├── trainprocessed.csv
│   │   └── testprocessed.csv
│   ├── processed/
│   └── stopwords/
│       └── vietnamese-stopwords.txt
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_preprocessing_text.ipynb
│   ├── 03_tfidf_kmeans_clustering.ipynb
│   ├── 04_svm_classification.ipynb
│   └── 05_evaluation_visualization.ipynb
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── classification.py
│   └── visualization.py
├── results/
│   ├── models/
│   ├── csv/
│   ├── figures/
│   └── logs/
├── main.py
├── requirements.txt
├── README.md
└── README_VI.md
```

## 6. Overall Pipeline

```text
Raw CSV files
  -> Exploratory Data Analysis
  -> Text preprocessing
  -> Overall sentiment assignment
  -> Main aspect extraction
  -> TF-IDF feature extraction
  -> K-Means topic clustering
  -> Cluster-aspect comparison
  -> LinearSVC sentiment classification
  -> Evaluation and business insights
```

## 7. Notebook Workflow

| Notebook | Purpose | Main Outputs |
|---|---|---|
| `01_explore_dataset.ipynb` | Explore raw data, label distribution, star rating, review length, aspects, and top words | EDA figures in `results/figures/` |
| `02_preprocessing_text.ipynb` | Clean text, assign overall sentiment, extract main aspect | `train_clean.csv`, `test_clean.csv` |
| `03_tfidf_kmeans_clustering.ipynb` | Build TF-IDF features, choose K, run K-Means, analyze clusters | K-Means model, TF-IDF vectorizer, clustered CSV |
| `04_svm_classification.ipynb` | Train and evaluate LinearSVC sentiment classifier | SVM model, prediction CSV, confusion matrix |
| `05_evaluation_visualization.ipynb` | Summarize results and generate final insights | Final charts and project conclusions |

## 8. Text Preprocessing

The preprocessing pipeline is implemented in `src/preprocessing.py`.

Steps:

1. Convert text to lowercase.
2. Remove punctuation and emoji.
3. Normalize all numbers to the token `number`.
4. Remove Vietnamese stopwords.
5. Tokenize Vietnamese text with `pyvi.ViTokenizer`.
6. Remove repeated consecutive words.

The project also converts aspect-based labels into one overall sentiment label:

```text
Positive / Negative / Neutral
```

## 9. TF-IDF Feature Extraction

TF-IDF is used to convert cleaned review text into numerical vectors. The configuration is:

```python
TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

`sublinear_tf=True` reduces the effect of words that appear many times in the same review.

## 10. K-Means Topic Clustering

K-Means is used to group reviews into topics based on TF-IDF vectors.

The project tests K from 2 to 10 and uses:

- Elbow Method
- Silhouette Score

After clustering, each cluster is compared with the real `main_aspect` label using a crosstab table. A cluster is considered more interpretable if one aspect dominates the cluster.

## 11. SVM Sentiment Classification

The sentiment classifier uses:

```python
LinearSVC(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
```

The evaluation includes:

- Majority-class baseline with `DummyClassifier`
- 5-fold cross-validation
- F1 Macro
- F1 Weighted
- Accuracy
- Precision Macro
- Recall Macro
- Confusion Matrix
- Error analysis

`class_weight="balanced"` is used because sentiment classes may be imbalanced.

## 12. Expected Results

| Metric | Target |
|---|---:|
| K-Means Silhouette Score | >= 0.08 |
| SVM F1 Macro | >= 0.75 |
| SVM improvement over baseline | >= 10% |
| Good cluster-aspect match | Match ratio >= 50% |

Actual results should be filled in after running all notebooks.

## 13. How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in this order:

```text
01 -> 02 -> 03 -> 04 -> 05
```

Or run the main pipeline:

```bash
python main.py
```

## 14. Running on Google Colab

Upload the project zip file to Colab, then run:

```python
!unzip vietnamese-phone-sentiment-mining.zip
%cd vietnamese-phone-sentiment-mining
!pip install -r requirements.txt
```

Then open and run the notebooks in order.

## 15. Running on Kaggle

Upload the project zip as a Kaggle dataset/input, then run:

```python
!unzip /kaggle/input/<your-dataset-name>/vietnamese-phone-sentiment-mining.zip -d /kaggle/working/
%cd /kaggle/working/vietnamese-phone-sentiment-mining
!pip install -r requirements.txt
```

Then run the notebooks or execute:

```python
!python main.py
```

## 16. Limitations

- TF-IDF does not fully capture semantic meaning or negation.
- K-Means can be sensitive to initialization and sparse text vectors.
- The stopword list may miss domain-specific phone review terms.
- Neutral sentiment is often difficult to distinguish from mild positive or mild negative reviews.

## 17. Future Work

- Replace TF-IDF with Vietnamese contextual embeddings such as PhoBERT.
- Try topic modeling methods such as LDA, NMF, or BERTopic.
- Perform aspect-level sentiment analysis instead of assigning one sentiment to the whole review.
- Collect more data to reduce class imbalance.

## 18. Conclusion

This project provides a complete data mining workflow for Vietnamese phone reviews. It combines unsupervised topic discovery with supervised sentiment classification, then connects the results to business insights about product aspects such as battery, camera, price, performance, and customer service.
