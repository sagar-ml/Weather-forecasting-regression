
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib

from .data_cleaning import read_raw_csv, basic_clean, save_clean
from .feature_engineering import add_calendar_features, add_lag_features, add_rolling_features, finalize_features
from .utils import ensure_dir

DATA_RAW = Path('data/raw/weather_raw.csv')
DATA_CLEAN = Path('data/cleaned/weather_cleaned.csv')
MODEL_PATH = Path('data/cleaned/model.pkl')
PLOT_PATH = Path('data/cleaned/pred_vs_actual.png')


def train_and_evaluate():
    # Read & clean
    df = read_raw_csv(DATA_RAW)
    df = basic_clean(df)
    save_clean(df, DATA_CLEAN)

    # Features
    df = add_calendar_features(df)
    df = add_lag_features(df, 'temp', lags=[1,2,3])
    df = add_rolling_features(df, 'temp', windows=[3,7])
    df = finalize_features(df)

    feature_cols = [
        'humidity','pressure','windspeed',
        'temp_lag_1','temp_lag_2','temp_lag_3',
        'temp_rollmean_3','temp_rollmean_7',
        'month','day','dayofweek'
    ]
    X = df[feature_cols]
    y = df['temp']

    # time-aware split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Baseline: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    def metrics(y_true, y_hat):
        mae = mean_absolute_error(y_true, y_hat)
        rmse = mean_squared_error(y_true, y_hat, squared=False)
        r2 = r2_score(y_true, y_hat)
        return mae, rmse, r2

    lr_mae, lr_rmse, lr_r2 = metrics(y_test, lr_pred)
    rf_mae, rf_rmse, rf_r2 = metrics(y_test, rf_pred)

    print('
=== Linear Regression ===')
    print(f'MAE: {lr_mae:.3f} | RMSE: {lr_rmse:.3f} | R2: {lr_r2:.3f}')
    print('=== Random Forest ===')
    print(f'MAE: {rf_mae:.3f} | RMSE: {rf_rmse:.3f} | R2: {rf_r2:.3f}')

    # Choose best model
    best_model, best_pred, name = (rf, rf_pred, 'RandomForest') if rf_rmse < lr_rmse else (lr, lr_pred, 'LinearRegression')
    ensure_dir(MODEL_PATH.parent)
    joblib.dump(best_model, MODEL_PATH)
    print(f"
Saved best model: {name} → {MODEL_PATH}")

    # Plot predicted vs actual
    plt.figure(figsize=(10,4))
    plt.plot(y_test.values, label='Actual', marker='o', linewidth=1)
    plt.plot(best_pred, label='Predicted', marker='o', linewidth=1)
    plt.title('Predicted vs Actual Temperature')
    plt.xlabel('Time index (test set)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f'Plot saved to {PLOT_PATH}')

if __name__ == '__main__':
    train_and_evaluate()
