from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.classification import (
    cross_validate_svm,
    get_baseline_f1,
    plot_confusion_matrix,
    prediction_results_dataframe,
    train_svm,
)
from src.clustering import (
    build_tfidf,
    compare_cluster_vs_aspect,
    find_optimal_k,
    plot_pca_clusters,
    run_kmeans,
)
from src.preprocessing import preprocess_dataframe, read_filestopwords
from src.visualization import (
    plot_label_distribution,
    plot_sentiment_by_cluster,
    plot_star_vs_sentiment,
    plot_top_words,
)


def load_config(path='configs/config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dirs(config):
    for path in config['results'].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    Path('datasets/cleaned').mkdir(parents=True, exist_ok=True)


def main():
    config = load_config()
    ensure_dirs(config)
    figures_dir = Path(config['results']['figures_dir'])
    csv_dir = Path(config['results']['csv_dir'])
    models_dir = Path(config['results']['models_dir'])

    stop_words = read_filestopwords(config['data']['stopwords'])
    train = pd.read_csv(config['data']['train_raw'], encoding='utf-8')
    test = pd.read_csv(config['data']['test_raw'], encoding='utf-8')

    train_clean = preprocess_dataframe(train, stop_words)
    test_clean = preprocess_dataframe(test, stop_words)
    keep_cols = ['comment', 'sentiment', 'main_aspect', 'n_star', 'aspects_list']
    train_clean[keep_cols].to_csv(config['data']['train_clean'], index=False, encoding='utf-8-sig')
    test_clean[keep_cols].to_csv(config['data']['test_clean'], index=False, encoding='utf-8-sig')

    plot_label_distribution(
        train_clean,
        'sentiment',
        'Sentiment distribution in training data',
        figures_dir / 'eda_sentiment_distribution.png',
    )
    plot_label_distribution(
        train_clean,
        'main_aspect',
        'Main aspect distribution in training data',
        figures_dir / 'eda_aspect_distribution.png',
    )
    plot_label_distribution(
        train_clean,
        'n_star',
        'Star rating distribution in training data',
        figures_dir / 'eda_star_distribution.png',
    )
    plot_star_vs_sentiment(train_clean, figures_dir / 'eda_star_vs_sentiment.png')
    plot_top_words(train_clean, 'comment', 'sentiment', figures_dir / 'eda_top_words_positive_negative.png')

    vectorizer, X_train = build_tfidf(
        train_clean['comment'],
        max_features=config['tfidf']['max_features'],
        ngram_range=config['tfidf']['ngram_range'],
        min_df=config['tfidf']['min_df'],
        max_df=config['tfidf']['max_df'],
        sublinear_tf=config['tfidf']['sublinear_tf'],
    )
    scores = find_optimal_k(
        X_train,
        range(config['kmeans']['k_range'][0], config['kmeans']['k_range'][1] + 1),
        save_path=figures_dir,
    )
    scores.to_csv(csv_dir / 'kmeans_k_scores.csv', index=False, encoding='utf-8-sig')
    best_k = int(scores.sort_values('silhouette', ascending=False).iloc[0]['k'])

    kmeans, clusters, silhouette = run_kmeans(X_train, best_k, config['kmeans']['random_state'])
    train_clean['cluster'] = clusters
    _, cluster_summary = compare_cluster_vs_aspect(train_clean, 'cluster', 'main_aspect')
    cluster_names = dict(zip(cluster_summary['cluster'], cluster_summary['cluster_name']))
    train_clean['cluster_name'] = train_clean['cluster'].map(cluster_names)
    train_clean.to_csv(csv_dir / 'kmeans_clustered.csv', index=False, encoding='utf-8-sig')
    cluster_summary.to_csv(csv_dir / 'kmeans_cluster_summary.csv', index=False, encoding='utf-8-sig')

    plot_label_distribution(
        train_clean,
        'cluster',
        'K-Means cluster size',
        figures_dir / 'kmeans_cluster_size.png',
    )
    plot_sentiment_by_cluster(
        train_clean,
        'cluster',
        'sentiment',
        figures_dir / 'sentiment_by_cluster.png',
    )
    plot_pca_clusters(X_train, clusters, cluster_names, figures_dir / 'pca_kmeans.png')
    joblib.dump(kmeans, models_dir / 'kmeans_model.pkl')
    joblib.dump(vectorizer, models_dir / 'tfidf_vectorizer.pkl')

    X_test = vectorizer.transform(test_clean['comment'])
    svm = train_svm(X_train, train_clean['sentiment'], C=config['svm']['C'])
    baseline = get_baseline_f1(X_train, train_clean['sentiment'], X_test, test_clean['sentiment'])
    cv_scores = cross_validate_svm(X_train, train_clean['sentiment'], cv=config['svm']['cv_folds'])
    y_pred = svm.predict(X_test)
    metrics_summary = pd.DataFrame([{
        'baseline_f1_macro': baseline,
        'f1_macro': f1_score(test_clean['sentiment'], y_pred, average='macro'),
        'f1_weighted': f1_score(test_clean['sentiment'], y_pred, average='weighted'),
        'accuracy': accuracy_score(test_clean['sentiment'], y_pred),
        'precision_macro': precision_score(test_clean['sentiment'], y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(test_clean['sentiment'], y_pred, average='macro', zero_division=0),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
    }])
    svm_results = prediction_results_dataframe(svm, X_test, test_clean['sentiment'], test_clean)
    svm_results.to_csv(csv_dir / 'svm_results.csv', index=False, encoding='utf-8-sig')
    metrics_summary.to_csv(csv_dir / 'svm_metrics_summary.csv', index=False, encoding='utf-8-sig')
    plot_confusion_matrix(
        svm,
        X_test,
        test_clean['sentiment'],
        ['Negative', 'Neutral', 'Positive'],
        figures_dir / 'confusion_matrix.png',
    )
    joblib.dump(svm, models_dir / 'svm_model.pkl')

    print(f'Best K: {best_k}')
    print(f'K-Means Silhouette: {silhouette:.4f}')
    print(f'Baseline F1 Macro: {baseline:.4f}')
    print(f'SVM F1 Macro: {metrics_summary.loc[0, "f1_macro"]:.4f}')
    print('Saved cleaned data, models, metrics, and figures to results/.')


if __name__ == '__main__':
    main()
