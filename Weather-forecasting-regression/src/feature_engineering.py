
from __future__ import annotations
import pandas as pd

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    return df


def add_lag_features(df: pd.DataFrame, target_col: str='temp', lags: list[int]|None=None) -> pd.DataFrame:
    if lags is None:
        lags = [1, 2, 3]
    df = df.copy()
    for l in lags:
        df[f'{target_col}_lag_{l}'] = df[target_col].shift(l)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str='temp', windows: list[int]|None=None) -> pd.DataFrame:
    if windows is None:
        windows = [3, 7]
    df = df.copy()
    for w in windows:
        df[f'{target_col}_rollmean_{w}'] = df[target_col].rolling(w).mean()
    return df


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with newly created NaNs from lags/rollings
    return df.dropna().reset_index(drop=True)
