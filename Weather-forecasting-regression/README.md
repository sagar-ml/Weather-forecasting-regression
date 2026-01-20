
# 🌦️ Weather Forecasting using Regression

Predict next-day **temperature (°C)** from historical weather data using Python, Pandas, and scikit‑learn.

## ✨ Features
- Data cleaning (NaN handling, interpolation, outlier caps)
- Feature engineering (lags, rolling means, calendar features)
- Models: Linear Regression & Random Forest
- Evaluation: MAE, RMSE, R² + plots (Actual vs Predicted)
- Optional **Streamlit** app for quick predictions

## 🗂️ Project Structure
```
weather-forecasting-regression/
├─ data/
│  ├─ raw/          # input CSVs
│  └─ cleaned/      # cleaned CSVs
├─ notebooks/       # EDA / modelling notebooks
├─ src/
│  ├─ data_cleaning.py
│  ├─ feature_engineering.py
│  ├─ model.py
│  └─ utils.py
├─ app/
│  └─ streamlit_app.py
├─ requirements.txt
└─ README.md
```

## 🚀 Quickstart
```bash
# 1) create & activate a venv (optional)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) run training (from project root)
python -m src.model

# 4) (optional) run the Streamlit app
streamlit run app/streamlit_app.py
```

## 📊 Output
- Trained model saved to `data/cleaned/model.pkl`
- Metrics printed in console
- Plot saved to `data/cleaned/pred_vs_actual.png`

## 📝 Dataset
A small synthetic dataset is provided at `data/raw/weather_raw.csv` for quick testing. You can replace it with any city’s historical weather CSV (columns: `date, temp, humidity, pressure, windspeed`).

## 📄 License
MIT
