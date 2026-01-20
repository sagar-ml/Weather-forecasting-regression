
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from .utils import ensure_dir

def read_raw_csv(path: str|Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Sort by date if present
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    # Replace obvious invalids
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Interpolate numeric columns
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].interpolate(limit_direction='both')
    # Forward/backward fill as backup
    df[num_cols] = df[num_cols].ffill().bfill()
    # Cap outliers (winsorize at 1st/99th percentiles)
    for c in num_cols:
        lo, hi = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df

def save_clean(df: pd.DataFrame, out_path: str|Path) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
