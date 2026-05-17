from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.classification import cross_validate_svm, get_baseline_f1, prediction_results_dataframe, train_svm
from src.clustering import build_tfidf, compare_cluster_vs_aspect, find_optimal_k, run_kmeans
from src.preprocessing import preprocess_dataframe, read_filestopwords


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

    stop_words = read_filestopwords(config['data']['stopwords'])
    train = pd.read_csv(config['data']['train_raw'], encoding='utf-8')
    test = pd.read_csv(config['data']['test_raw'], encoding='utf-8')

    train_clean = preprocess_dataframe(train, stop_words)
    test_clean = preprocess_dataframe(test, stop_words)
    keep_cols = ['comment', 'sentiment', 'main_aspect', 'n_star', 'aspects_list']
    train_clean[keep_cols].to_csv(config['data']['train_clean'], index=False, encoding='utf-8-sig')
    test_clean[keep_cols].to_csv(config['data']['test_clean'], index=False, encoding='utf-8-sig')

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
        save_path=config['results']['figures_dir'],
    )
    scores.to_csv('results/csv/kmeans_k_scores.csv', index=False, encoding='utf-8-sig')
    best_k = int(scores.sort_values('silhouette', ascending=False).iloc[0]['k'])
    kmeans, clusters, silhouette = run_kmeans(X_train, best_k, config['kmeans']['random_state'])
    train_clean['cluster'] = clusters
    _, cluster_summary = compare_cluster_vs_aspect(train_clean, 'cluster', 'main_aspect')
    cluster_names = dict(zip(cluster_summary['cluster'], cluster_summary['cluster_name']))
    train_clean['cluster_name'] = train_clean['cluster'].map(cluster_names)
    train_clean.to_csv('results/csv/kmeans_clustered.csv', index=False, encoding='utf-8-sig')
    cluster_summary.to_csv('results/csv/kmeans_cluster_summary.csv', index=False, encoding='utf-8-sig')
    joblib.dump(kmeans, 'results/models/kmeans_model.pkl')
    joblib.dump(vectorizer, 'results/models/tfidf_vectorizer.pkl')

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
    svm_results.to_csv('results/csv/svm_results.csv', index=False, encoding='utf-8-sig')
    metrics_summary.to_csv('results/csv/svm_metrics_summary.csv', index=False, encoding='utf-8-sig')
    joblib.dump(svm, 'results/models/svm_model.pkl')

    print(f'Best K: {best_k}')
    print(f'K-Means Silhouette: {silhouette:.4f}')
    print(f'Baseline F1 Macro: {baseline:.4f}')
    print(f'SVM F1 Macro: {metrics_summary.loc[0, "f1_macro"]:.4f}')
    print('Đã lưu dữ liệu, mô hình và kết quả vào thư mục results/.')


if __name__ == '__main__':
    main()
