import re
import string
from typing import Iterable, List, Tuple

import pandas as pd
from pyvi import ViTokenizer


def remove_punctuation(comment):
    translator = str.maketrans('', '', string.punctuation)
    new_string = comment.translate(translator)
    new_string = re.sub('[\n ]+', ' ', new_string)
    emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
    return re.sub(emoji_pattern, '', new_string)


def read_filestopwords(path='datasets/stopwords/vietnamese-stopwords.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def remove_stopword(comment, stop_words):
    return ' '.join([w for w in comment.split() if w not in stop_words])


def normalize_numbers(text):
    return re.sub(r'\d+', 'number', text)


def remove_repeated_words(text):
    words = text.split()
    return ' '.join([words[i] for i in range(len(words))
                     if i == 0 or words[i] != words[i-1]])


def full_pipeline(text, stop_words):
    text = str(text).lower()
    text = remove_punctuation(text)
    text = normalize_numbers(text)
    text = remove_stopword(text, stop_words)
    text = ViTokenizer.tokenize(text)
    text = remove_repeated_words(text)
    return text.strip()


def assign_overall_label(row):
    p, n, neu = row['positive_count'], row['negative_count'], row['neutral_count']
    if p > neu and p > n:   return 'Positive'
    elif n > neu and n > p: return 'Negative'
    elif n == neu:          return 'Negative'
    elif neu == p:          return 'Positive'
    else:                   return 'Neutral'


def extract_main_aspect(label_str):
    matches = re.findall(r'\{(\w+)#(\w+)\}', str(label_str))
    if not matches:
        return 'OTHERS'
    for aspect, _ in matches:
        if aspect != 'OTHERS':
            return aspect
    return 'OTHERS'


def extract_aspects_list(label_str: str) -> List[Tuple[str, str]]:
    """Return all aspect-sentiment pairs, including names such as SER&ACC."""
    matches = re.findall(r'\{([^#{};]+)#(Positive|Negative|Neutral)\}', str(label_str))
    return [(aspect.strip(), sentiment.strip()) for aspect, sentiment in matches]


def count_sentiments(label_str: str) -> Tuple[int, int, int]:
    aspects = extract_aspects_list(label_str)
    positive_count = sum(sentiment == 'Positive' for _, sentiment in aspects)
    neutral_count = sum(sentiment == 'Neutral' for _, sentiment in aspects)
    negative_count = sum(sentiment == 'Negative' for _, sentiment in aspects)
    return positive_count, neutral_count, negative_count


def add_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    counts = result['label'].apply(count_sentiments)
    result[['positive_count', 'neutral_count', 'negative_count']] = pd.DataFrame(
        counts.tolist(), index=result.index
    )
    result['sentiment'] = result.apply(assign_overall_label, axis=1)
    return result


def add_aspect_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result['main_aspect'] = result['label'].apply(extract_main_aspect)
    result['aspects_list'] = result['label'].apply(extract_aspects_list)
    return result


def preprocess_dataframe(
    df: pd.DataFrame,
    stop_words: Iterable[str],
    text_col: str = 'comment',
) -> pd.DataFrame:
    result = df.copy()
    result['raw_comment'] = result[text_col].astype(str)
    result[text_col] = result[text_col].fillna('').astype(str).apply(
        lambda text: full_pipeline(text, stop_words)
    )
    result = add_sentiment_columns(result)
    result = add_aspect_columns(result)
    result['word_count'] = result[text_col].str.split().str.len()
    return result


def preprocessing_steps_example(text: str, stop_words: Iterable[str]) -> pd.DataFrame:
    rows = []
    current = str(text)
    rows.append(('Trước', current))
    current = current.lower()
    rows.append(('Sau bước 1 (lowercase)', current))
    current = remove_punctuation(current)
    rows.append(('Sau bước 2 (remove_punctuation)', current))
    current = normalize_numbers(current)
    rows.append(('Sau bước 3 (normalize_numbers)', current))
    current = remove_stopword(current, stop_words)
    rows.append(('Sau bước 4 (remove_stopword)', current))
    current = ViTokenizer.tokenize(current)
    rows.append(('Sau bước 5 (tokenize)', current))
    current = remove_repeated_words(current)
    rows.append(('Sau bước 6 (remove_repeated)', current))
    return pd.DataFrame(rows, columns=['Bước', 'Kết quả'])
