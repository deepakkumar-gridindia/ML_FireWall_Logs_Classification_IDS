import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="ML Deployment App", layout="wide")
st.title("🚀 Multi-Class Machine Learning Model Deployment for Intrusion Detetction System of Firewall")

# ---------------------------------
# LOAD LABEL ENCODER
# ---------------------------------
label_encoder = joblib.load("label_encoder.pkl")

# ---------------------------------
# MODEL FILE MAP
# ---------------------------------
model_files = {
    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree Classifier": "decision_tree.pkl",
    "K-Nearest Neighbor Classifier": "knn_model.pkl",
    "Naive Bayes Classifier": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
    "XGBoost (Ensemble)": "xgboost.pkl"
}

# ---------------------------------
# MODEL SELECTION DROPDOWN (1 MARK)
# ---------------------------------
selected_model_name = st.selectbox(
    "Select Model",
    list(model_files.keys())
)

model = joblib.load(model_files[selected_model_name])

# ---------------------------------
# UPLOAD TEST DATA
# ---------------------------------
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(test_data.head())

    if "Action" in test_data.columns:
        y_true = test_data["Action"]
        X_test = test_data.drop(columns=["Action"])
    else:
        st.error("Test dataset must contain 'Action' column.")
        st.stop()

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # ---------------------------------
    # DISPLAY EVALUATION METRICS (1 MARK)
    # ---------------------------------
    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    col1.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.4f}")
    col2.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.4f}")
    col2.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
    col3.metric("AUC", f"{roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'):.4f}")
    col3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

    # ---------------------------------
    # CONFUSION MATRIX (1 MARK)
    # ---------------------------------
    st.subheader("🔢 Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    st.dataframe(cm_df)

    # ---------------------------------
    # CLASSIFICATION REPORT (1 MARK)
    # ---------------------------------
    st.subheader("📄 Classification Report")

    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )
    ).transpose()

    st.dataframe(report_df.round(4))

    st.success("✅ Evaluation Completed Successfully")
