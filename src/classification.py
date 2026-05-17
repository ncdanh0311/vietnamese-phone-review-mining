from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC


def train_svm(X_train, y_train, C=1.0):
    model = LinearSVC(C=C, max_iter=2000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_names):
    y_pred = model.predict(X_test)
    metrics = {
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
    }
    print(classification_report(y_test, y_pred, labels=label_names, zero_division=0))
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')
    return metrics


def cross_validate_svm(X, y, cv=5):
    model = LinearSVC(C=1.0, max_iter=2000, random_state=42, class_weight='balanced')
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=splitter, scoring='f1_macro')
    for idx, score in enumerate(scores, start=1):
        print(f'Fold {idx}: F1 Macro = {score:.4f}')
    print(f'Mean +/- Std: {scores.mean():.4f} +/- {scores.std():.4f}')
    return scores


def get_baseline_f1(X_train, y_train, X_test, y_test):
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')


def analyze_errors(model, X_test, y_test, df_test, n=5):
    y_pred = model.predict(X_test)
    result = df_test.copy()
    result['y_true'] = list(y_test)
    result['y_pred'] = y_pred
    errors = result[result['y_true'] != result['y_pred']].head(n)
    return errors[['comment', 'y_true', 'y_pred']]


def plot_confusion_matrix(model, X_test, y_test, labels, save_path):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - LinearSVC')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return cm


def prediction_results_dataframe(model, X_test, y_test, df_test):
    y_pred = model.predict(X_test)
    return pd.DataFrame({
        'comment': df_test['comment'].values,
        'y_true': np.array(y_test),
        'y_pred': y_pred,
    })
