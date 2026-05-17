from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _prepare_save_path(save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_label_distribution(df, col, title, save_path):
    counts = df[col].value_counts()
    percentages = (counts / counts.sum() * 100).round(2)
    plot_df = pd.DataFrame({'label': counts.index, 'count': counts.values, 'percentage': percentages.values})
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=plot_df, x='label', y='count', hue='label', palette='Set2', legend=False)
    for idx, row in plot_df.iterrows():
        ax.text(idx, row['count'], f"{row['percentage']}%", ha='center', va='bottom')
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(_prepare_save_path(save_path), dpi=150)
    plt.close()
    return plot_df


def plot_star_vs_sentiment(df, save_path):
    table = pd.crosstab(df['n_star'], df['sentiment'], normalize='index') * 100
    plt.figure(figsize=(9, 5))
    sns.heatmap(table, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('Sentiment ratio by star rating (%)')
    plt.xlabel('Sentiment')
    plt.ylabel('Star rating')
    plt.tight_layout()
    plt.savefig(_prepare_save_path(save_path), dpi=150)
    plt.close()
    return table


def plot_sentiment_by_cluster(df, cluster_col, sentiment_col, save_path):
    table = pd.crosstab(df[cluster_col], df[sentiment_col], normalize='index') * 100
    plot_df = table.reset_index().melt(id_vars=cluster_col, var_name='sentiment', value_name='percentage')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x=cluster_col, y='percentage', hue='sentiment', palette='Set2')
    plt.title('Sentiment ratio by topic cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Ratio (%)')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig(_prepare_save_path(save_path), dpi=150)
    plt.close()
    return table


def _top_words(texts, n=20):
    counter = Counter()
    for text in texts.dropna().astype(str):
        counter.update(text.split())
    return pd.DataFrame(counter.most_common(n), columns=['word', 'count'])


def plot_top_words(df, text_col, sentiment_col, save_path):
    sentiments = ['Positive', 'Negative']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    results = {}
    for ax, sentiment in zip(axes, sentiments):
        words = _top_words(df.loc[df[sentiment_col] == sentiment, text_col], n=20)
        results[sentiment] = words
        sns.barplot(data=words, y='word', x='count', hue='word', ax=ax, palette='viridis', legend=False)
        ax.set_title(f'Top 20 words - {sentiment}')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(_prepare_save_path(save_path), dpi=150)
    plt.close(fig)
    return results
