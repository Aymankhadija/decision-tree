

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# ---------- 1. Load & clean ----------
df = pd.read_csv("cardekho_data.csv")
df.drop_duplicates(inplace=True)

cat_cols = ["Fuel_Type", "Seller_Type", "Transmission", "Car_Name"]
num_cols = [c for c in df.columns if c not in cat_cols + ["Selling_Price"]]

X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 2. Build pipeline ----------
# handle_unknown="use_encoded_value" lets new data contain categories
# (e.g. a new Car_Name) not seen during training, without crashing.
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ), cat_cols)
    ],
    remainder="passthrough"  # numeric columns pass through unchanged
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=220,
        max_depth=6,
        learning_rate=0.06,
        min_samples_leaf=4,
        random_state=42,
    ))
])

# ---------- 3. Train ----------
model.fit(X_train, y_train)
print("R² Score:", model.score(X_test, y_test))

# ---------- 4. Save the whole pipeline (preprocessing + model) ----------
joblib.dump(model, "car_price_pipeline.pkl")

# ---------- 5. Predict on a NEW dataset ----------
# new_dataset.csv must have the same raw columns as the training X
# (Car_Name, Fuel_Type, Seller_Type, Transmission, + numeric cols),
# WITHOUT Selling_Price, and with NO manual encoding needed.
unknown_df = pd.DataFrame([
    {
        "Car_Name": "ciaz",
        "Year": 2018,
        "Present_Price": 9.5,
        "Kms_Driven": 25000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
    }
])
 

# reload later in another script/session like this:
# model = joblib.load("car_price_pipeline.pkl")

predictions = model.predict(unknown_df)
unknown_df["Predicted_Selling_Price"] = predictions
print(unknown_df)







