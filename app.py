import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Trained Model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("trained_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, label_encoder = load_model()

st.set_page_config(page_title="ML Model Deployment", layout="wide")
st.title("🚀 Multi-Class Model Deployment App")

st.write("Upload test dataset to evaluate the trained model.")

# -------------------------------
# Upload Test Dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(test_data.head())

    # Separate features if target exists
    if "Action" in test_data.columns:
        X_test = test_data.drop(columns=["Action"])
    else:
        X_test = test_data

    # Predict
    predictions = model.predict(X_test)
    predicted_labels = label_encoder.inverse_transform(predictions)

    test_data["Predicted Action"] = predicted_labels

    st.subheader("📊 Predictions")
    st.dataframe(test_data)

    st.success("✅ Prediction Completed Successfully")
