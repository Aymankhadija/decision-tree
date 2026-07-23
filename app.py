import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

st.title("🚗 Car Selling Price Predictor")
st.write("Enter the car details below to estimate its selling price.")

# ---------- Load trained pipeline ----------
@st.cache_resource
def load_model():
    return joblib.load("car_price_pipeline.pkl")
# ---------- unique car values----------
model = load_model()
def load_car_names():
    df = pd.read_csv("cardekho_data.csv")
    return sorted(df["Car_Name"].dropna().unique().tolist())
 
car_names = load_car_names()
# ---------- Input form ----------
with st.form("predict_form"):
    car_name = st.selectbox("Car Name", car_names)
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015, step=1)
    present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, value=5.0, step=0.1)
    kms_driven = st.number_input("Kms Driven", min_value=0, value=30000, step=1000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict Price")

# ---------- Predict ----------
if submitted:
    input_df = pd.DataFrame([{
        "Car_Name": car_name,
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner,
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Selling Price: **{prediction:.2f} Lakhs**")

    with st.expander("See input data"):
        st.dataframe(input_df)