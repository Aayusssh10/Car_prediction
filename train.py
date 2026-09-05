"""
Train the used-car resale-price model.

Data source (first that exists wins):
  1. Car Dataset Processed.csv   - original processed dataset (categoricals as text)
  2. recovered_training_data.csv - 5 features + target recovered from the old .pkl
                                   (already integer-encoded, no registration_year)

Output: car_price_model.pkl  - a full Pipeline(StandardScaler -> estimator),
so the app can feed raw encoded numbers straight in without scaling them itself.

Run:  python train.py
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
FEATURES = ["insurance_validity", "fuel_type", "kms_driven", "ownsership", "transmission"]
TARGET = "price(in lakhs)"

# encodings - must stay identical to app.py
D_INSURANCE = {"Comprehensive": 0, "Third Party insurance": 1, "Third Party": 1,
               "Zero Dep": 2, "Not Available": 3}
D_FUEL = {"Petrol": 0, "Diesel": 1, "CNG": 2}
D_TRANSMISSION = {"Manual": 0, "Automatic": 1}
D_OWNERSHIP = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3,
               "Fourth Owner": 4, "Fifth Owner": 5}


def load_data() -> pd.DataFrame:
    if os.path.exists("Car Dataset Processed.csv"):
        print("Loading Car Dataset Processed.csv")
        df = pd.read_csv("Car Dataset Processed.csv")
        df["insurance_validity"] = df["insurance_validity"].map(D_INSURANCE)
        df["fuel_type"] = df["fuel_type"].map(D_FUEL)
        df["transmission"] = df["transmission"].map(D_TRANSMISSION)
        df["ownsership"] = df["ownsership"].map(D_OWNERSHIP)
    elif os.path.exists("recovered_training_data.csv"):
        print("Loading recovered_training_data.csv (no registration_year available)")
        df = pd.read_csv("recovered_training_data.csv")
    else:
        raise FileNotFoundError(
            "Need 'Car Dataset Processed.csv' or 'recovered_training_data.csv' in this folder."
        )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[FEATURES + [TARGET]].dropna()
    # drop data-entry errors: a used car is not 70,000 lakhs, and not 800,000 km
    df = df[(df[TARGET] > 0) & (df[TARGET] <= 150)]
    df = df[(df["kms_driven"] > 0) & (df["kms_driven"] <= 400_000)]
    print(f"Cleaning: {before} -> {len(df)} rows ({before - len(df)} dropped)")
    return df.reset_index(drop=True)


def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    rmse = mean_squared_error(y_te, pred) ** 0.5
    print(f"  {name:<16}  test R2={r2:6.3f}   MAE={mae:5.2f} L   RMSE={rmse:5.2f} L")
    return r2, model


def main():
    df = clean(load_data())
    X = df[FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train {len(X_tr)}  |  Test {len(X_te)}\n")

    knn = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("knn", KNeighborsRegressor())]),
        {"knn__n_neighbors": [3, 5, 7, 9, 11, 15], "knn__weights": ["uniform", "distance"]},
        cv=5, scoring="r2",
    )
    candidates = {
        "LinearRegression": Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())]),
        "KNN (tuned)": knn,
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
    }

    print("Model comparison:")
    results = {name: evaluate(name, m, X_tr, X_te, y_tr, y_te) for name, m in candidates.items()}

    best_name = max(results, key=lambda n: results[n][0])
    best_model = results[best_name][1]
    if isinstance(best_model, GridSearchCV):
        print(f"\nBest KNN params: {best_model.best_params_}")
        best_model = best_model.best_estimator_

    # refit best on ALL data before shipping
    best_model.fit(X, y)
    joblib.dump(best_model, "car_price_model.pkl")
    print(f"\nSaved '{best_name}' -> car_price_model.pkl  (trained on all {len(X)} rows)")


if __name__ == "__main__":
    main()
