# 🚗 Car Price Predictor

Estimates the resale price (in ₹ lakhs) of an Indian used car from five inputs:
insurance validity, fuel type, kilometres driven, ownership, and transmission.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI for single predictions |
| `train.py` | Trains and evaluates the model, writes `car_price_model.pkl` |
| `car_price_model.pkl` | Trained `Pipeline(StandardScaler → KNeighborsRegressor)` |
| `recovered_training_data.csv` | 1,499 rows (5 features + target) recovered from the original model; used by `train.py` |
| `models (1).ipynb` | Original exploration notebook — **superseded by `train.py`** |
| `car_price_model_OLD.pkl.bak` | The previous model, kept as a backup |

## Setup

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Retrain

```bash
python train.py
```

`train.py` uses `Car Dataset Processed.csv` if it is present (mapping the text
categoricals itself); otherwise it falls back to `recovered_training_data.csv`.
It cleans data-entry errors, does an 80/20 train/test split, grid-searches KNN,
compares against Linear Regression and Random Forest, and saves the best model as
a full pipeline (so the app never has to scale inputs itself).

## Current performance (held-out test set)

| Model | Test R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.32 | ₹9.8 L | ₹15.4 L |
| **KNN (k=15, distance) — shipped** | **0.49** | **₹6.5 L** | **₹13.4 L** |
| Random Forest | 0.44 | ₹7.2 L | ₹14.0 L |

## Known limitations

- **Accuracy is modest** (~₹6–7 lakh typical error). The five available features
  carry limited signal.
- **Car age / registration year is not used.** It is the single strongest driver
  of resale price but was never captured in the recovered data. Add it by
  retraining from a source CSV that includes `registration_year`.
- Trained on ~1,500 listings only; brand, model, and city are not included.
- The old notebook's reported "0.59" was a *training* score with outliers left in
  and no train/test split — not comparable to the test scores above.

## Encoding reference (must stay in sync between `app.py` and `train.py`)

| Field | Mapping |
|---|---|
| insurance_validity | Comprehensive 0 · Third Party (insurance) 1 · Zero Dep 2 · Not Available 3 |
| fuel_type | Petrol 0 · Diesel 1 · CNG 2 |
| transmission | Manual 0 · Automatic 1 |
| ownsership | First 1 · Second 2 · Third 3 · Fourth 4 · Fifth 5 |
