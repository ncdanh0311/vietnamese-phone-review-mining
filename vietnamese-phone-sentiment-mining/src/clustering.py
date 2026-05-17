from pathlib import Path

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def build_tfidf(texts, max_features=3000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=tuple(ngram_range),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )
    X = vectorizer.fit_transform(texts.fillna('') if hasattr(texts, 'fillna') else texts)
    return vectorizer, X


def _resolve_plot_dir(save_path):
    if save_path is None:
        return None
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_optimal_k(X, k_range=range(2, 11), save_path=None):
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        rows.append({
            'k': k,
            'inertia': model.inertia_,
            'silhouette': silhouette_score(X, labels),
        })

    scores = pd.DataFrame(rows)
    plot_dir = _resolve_plot_dir(save_path)
    if plot_dir is not None:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=scores, x='k', y='inertia', marker='o')
        plt.title('Elbow Method cho K-Means')
        plt.xlabel('Số cụm K')
        plt.ylabel('Inertia')
        plt.tight_layout()
        plt.savefig(plot_dir / 'elbow.png', dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=scores, x='k', y='silhouette', marker='o', color='darkorange')
        plt.title('Silhouette Score theo số cụm K')
        plt.xlabel('Số cụm K')
        plt.ylabel('Silhouette Score')
        plt.tight_layout()
        plt.savefig(plot_dir / 'silhouette.png', dpi=150)
        plt.close()
    return scores


def run_kmeans(X, n_clusters, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return model, labels, score


def get_top_keywords(model, vectorizer, n_top=10):
    terms = np.array(vectorizer.get_feature_names_out())
    top_keywords = {}
    order_centroids = model.cluster_centers_.argsort(axis=1)[:, ::-1]
    for cluster_id in range(model.n_clusters):
        top_keywords[cluster_id] = terms[order_centroids[cluster_id, :n_top]].tolist()
    return top_keywords


def compare_cluster_vs_aspect(df, cluster_col, aspect_col):
    crosstab = pd.crosstab(df[cluster_col], df[aspect_col])
    summary_rows = []
    for cluster_id, row in crosstab.iterrows():
        total = row.sum()
        top_aspect = row.idxmax() if total else 'UNKNOWN'
        match_ratio = row.max() / total if total else 0
        summary_rows.append({
            'cluster': cluster_id,
            'cluster_name': top_aspect,
            'match_ratio': match_ratio,
            'total': total,
        })
    return crosstab, pd.DataFrame(summary_rows)


def plot_wordclouds(df, cluster_col, text_col, save_dir):
    from wordcloud import WordCloud

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id in sorted(df[cluster_col].unique()):
        text = ' '.join(df.loc[df[cluster_col] == cluster_id, text_col].dropna().astype(str))
        if not text.strip():
            continue
        wc = WordCloud(width=1000, height=600, background_color='white', collocations=False).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'WordCloud cụm {cluster_id}')
        plt.tight_layout()
        plt.savefig(save_dir / f'wordcloud_cum_{cluster_id}.png', dpi=150)
        plt.close()


def plot_pca_clusters(X, labels, cluster_names, save_path):
    dense_X = X.toarray() if hasattr(X, 'toarray') else X
    points = PCA(n_components=2, random_state=42).fit_transform(dense_X)
    plot_df = pd.DataFrame({
        'PC1': points[:, 0],
        'PC2': points[:, 1],
        'cluster': labels,
        'cluster_name': [cluster_names.get(label, f'Cụm {label}') for label in labels],
    })
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        hue='cluster_name',
        palette='tab10',
        s=18,
        alpha=0.75,
    )
    plt.title('PCA 2D các cụm K-Means')
    plt.xlabel('Thành phần chính 1')
    plt.ylabel('Thành phần chính 2')
    plt.legend(title='Chủ đề cụm', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
