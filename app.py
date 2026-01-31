import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")
st.write("Simple ML app using Linear Regression")

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("house_data.csv")
    return df

df = load_data()

# ===============================
# Basic Cleaning (IMPORTANT)
# ===============================
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Drop non-numeric / irrelevant columns
drop_cols = ["date", "street", "city", "statezip", "country"]
df = df.drop(columns=drop_cols, errors="ignore")

# Handle missing values
df = df.dropna()

# ===============================
# Features & Target
# ===============================
X = df.drop("price", axis=1)
y = df["price"]

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Model Training
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================
# Model Evaluation
# ===============================
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"R² Score: **{score:.2f}**")

# ===============================
# User Input
# ===============================
st.subheader("🧾 Enter House Details")

bedrooms = st.number_input("Bedrooms", min_value=0, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0)
sqft_living = st.number_input("Sqft Living", min_value=0, value=1800)
sqft_lot = st.number_input("Sqft Lot", min_value=0, value=5000)
floors = st.number_input("Floors", min_value=0.0, value=1.0)
waterfront = st.selectbox("Waterfront (0 = No, 1 = Yes)", [0, 1])
view = st.slider("View Rating", 0, 4, 0)
condition = st.slider("Condition", 1, 5, 3)
sqft_above = st.number_input("Sqft Above", min_value=0, value=1500)
sqft_basement = st.number_input("Sqft Basement", min_value=0, value=300)
yr_built = st.number_input("Year Built", min_value=1800, value=2000)
yr_renovated = st.number_input("Year Renovated (0 if none)", min_value=0, value=0)

# ===============================
# Prediction
# ===============================
if st.button("Predict Price"):
    input_data = np.array([[ 
        bedrooms, bathrooms, sqft_living, sqft_lot, floors,
        waterfront, view, condition, sqft_above,
        sqft_basement, yr_built, yr_renovated
    ]])

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated House Price: ₹ {prediction:,.0f}")
