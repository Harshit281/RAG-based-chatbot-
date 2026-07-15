import os
from typing import Tuple

import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    expected = {"ID", "Topic", "Content"}
    if not expected.issubset(df.columns):
        raise ValueError(f"CSV is missing required columns: {expected - set(df.columns)}")

    df = df[['ID', 'Topic', 'Content']].copy()
    df['Topic'] = df['Topic'].astype(str)
    df['Content'] = df['Content'].astype(str)
    return df
