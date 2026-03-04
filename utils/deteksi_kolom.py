import pandas as pd


def is_datetime_column(series: pd.Series) -> bool:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.notna().mean() > 0.8


def is_valid_text_column(series: pd.Series) -> bool:
    texts = series.dropna().astype(str)

    if len(texts) == 0:
        return False

    if is_datetime_column(texts):
        return False

    # rata-rata panjang kata
    if texts.str.len().mean() < 20:
        return False

    # rata-rata jumlah kata
    if texts.str.split().apply(len).mean() < 5:
        return False

    vocab_ratio = (
        texts.str.split()
        .explode()
        .nunique() /
        max(texts.str.split().explode().count(), 1)
    )

    return vocab_ratio > 0.03


def detect_text_column(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == object:
            if is_valid_text_column(df[col]):
                return col
    return None
