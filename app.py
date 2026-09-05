import streamlit as st
import joblib
import pandas as pd

# ── encodings (must match train.py) ──────────────────────────────────────────
D_INSURANCE = {
    "Comprehensive": 0,
    "Third Party insurance": 1,
    "Third Party": 1,
    "Zero Dep": 2,
    "Not Available": 3,
}
D_FUEL = {"Petrol": 0, "Diesel": 1, "CNG": 2}
D_TRANSMISSION = {"Manual": 0, "Automatic": 1}
D_OWNERSHIP = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth Owner": 4,
    "Fifth Owner": 5,
}

# column order the model was trained on
FEATURES = ["insurance_validity", "fuel_type", "kms_driven", "ownsership", "transmission"]

# ── load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")

model = load_model()

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Predictor")
st.markdown("Fill in the details below to get an estimated resale price.")

# ── input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        fuel_type = st.selectbox("Fuel type", list(D_FUEL.keys()))
        transmission = st.selectbox("Transmission", list(D_TRANSMISSION.keys()))
        ownership = st.selectbox("Ownership", list(D_OWNERSHIP.keys()))

    with col2:
        insurance = st.selectbox("Insurance validity", list(D_INSURANCE.keys()))
        kms_driven = st.number_input(
            "Kilometres driven",
            min_value=0,
            max_value=1_000_000,
            value=30_000,
            step=1_000,
        )

    submitted = st.form_submit_button("Predict price", use_container_width=True)

# ── prediction ────────────────────────────────────────────────────────────────
if submitted:
    row = pd.DataFrame(
        [{
            "insurance_validity": D_INSURANCE[insurance],
            "fuel_type": D_FUEL[fuel_type],
            "kms_driven": kms_driven,
            "ownsership": D_OWNERSHIP[ownership],
            "transmission": D_TRANSMISSION[transmission],
        }],
        columns=FEATURES,
    )

    price = float(model.predict(row)[0])

    st.divider()
    st.metric(
        label="Estimated resale price",
        value=f"₹ {price:.2f} Lakhs",
    )

    st.caption(
        "Scaled KNN model trained on ~1,500 Indian used-car listings. "
        "Typical error is around ₹6–7 lakhs — treat this as a rough ballpark, "
        "not a valuation. Car age is not part of the model."
    )
