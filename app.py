import streamlit as st
import pandas as pd
import joblib
import requests
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

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="ML-Based IDS System", layout="wide")

st.title("🚀 Machine Learning Based Intrusion Detection System (Firewall)")

# =========================================================
# CREATE TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📁 Dataset Information", "🤖 ML Models"])

# =========================================================
# TAB 1 — HOME
# =========================================================
with tab1:

    st.markdown("""
    ### 🔍 About ML-Based IDS

    This application demonstrates a Machine Learning-based Intrusion Detection System (IDS) 
    for firewall traffic classification.  
    The system uses supervised learning models to automatically classify 
    network traffic actions such as allow, deny, drop, and reset-both.
    """)

    # -------------------------------
    # DOWNLOAD SECTION
    # -------------------------------
    st.markdown("### ⬇️ Download Test Dataset")

    GITHUB_TEST_DATA_URL = "https://raw.githubusercontent.com/deepakkumar-gridindia/ML_FireWall_Logs_Classification_IDS/main/test_dataset.csv"

    if st.button("Download Test Dataset"):
        try:
            response = requests.get(GITHUB_TEST_DATA_URL)
            st.download_button(
                label="Click to Download",
                data=response.content,
                file_name="test_dataset.csv",
                mime="text/csv"
            )
        except:
            st.error("Unable to fetch dataset.")

    # -------------------------------
    # UPLOAD SECTION
    # -------------------------------
    st.markdown("### 📤 Upload Test Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    # -------------------------------
    # LOAD MODELS
    # -------------------------------
    model_files = {
        "Logistic Regression": "logistic_model.pkl",
        "Decision Tree Classifier": "decision_tree.pkl",
        "K-Nearest Neighbor Classifier": "knn_model.pkl",
        "Naive Bayes Classifier": "naive_bayes.pkl",
        "Random Forest (Ensemble)": "random_forest.pkl",
        "XGBoost (Ensemble)": "xgboost.pkl"
    }

    label_encoder = joblib.load("label_encoder.pkl")

    # -------------------------------
    # MODEL SELECTION
    # -------------------------------
    st.markdown("### ⚙️ Select Model")
    selected_model_name = st.selectbox(
        "Choose a Model for Evaluation",
        list(model_files.keys())
    )

    model = joblib.load(model_files[selected_model_name])

    # -------------------------------
    # PREDICTION & RESULTS
    # -------------------------------
    if uploaded_file is not None:

        try:
            test_data = pd.read_csv(uploaded_file)
        except Exception:
            test_data = pd.read_csv(uploaded_file, engine="python")

        if "Action" not in test_data.columns:
            st.error("Dataset must contain 'Action' column.")
            st.stop()

        y_true = test_data["Action"]
        X_test = test_data.drop(columns=["Action"])

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # -------------------------------
        # EVALUATION METRICS
        # -------------------------------
        st.markdown(f"## 📊 Evaluation Metrics – {selected_model_name}")

        accuracy_val = accuracy_score(y_true, y_pred)
        precision_val = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall_val = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_val = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        auc_val = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        mcc_val = matthews_corrcoef(y_true, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_val:.4f}")
        col1.metric("Precision", f"{precision_val:.4f}")
        col2.metric("Recall", f"{recall_val:.4f}")
        col2.metric("F1 Score", f"{f1_val:.4f}")
        col3.metric("AUC", f"{auc_val:.4f}")
        col3.metric("MCC", f"{mcc_val:.4f}")

        # -------------------------------
        # CONFUSION MATRIX
        # -------------------------------
        st.markdown(f"## 🔢 Confusion Matrix – {selected_model_name}")

        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=label_encoder.classes_,
            columns=label_encoder.classes_
        )

        st.dataframe(cm_df, use_container_width=True)

        # -------------------------------
        # CLASSIFICATION REPORT
        # -------------------------------
        st.markdown(f"## 📄 Detailed Classification Report – {selected_model_name}")

        report_dict = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report_dict).transpose()

        class_mapping = {str(i): label for i, label in enumerate(label_encoder.classes_)}
        report_df.rename(index=class_mapping, inplace=True)

        class_df = report_df.loc[label_encoder.classes_, ["precision", "recall", "f1-score", "support"]]
        st.dataframe(class_df.round(4), use_container_width=True)

        st.success("✅ Evaluation Completed Successfully")

# =========================================================
# TAB 2 — DATASET INFORMATION
# =========================================================
with tab2:

    st.markdown("## 📘 Internet Firewall Data — Dataset Information")

    st.markdown("""
    - **Instances:** 65,532  
    - **Features:** 12  
    - **Task:** Multiclass Classification  
    - **Class Labels:** allow, deny, drop, reset-both  
    """)

    st.markdown("### Feature Overview")
    st.markdown("""
    Source Port, Destination Port, NAT Source Port, NAT Destination Port,  
    Bytes, Bytes Sent, Bytes Received, Packets, Elapsed Time (sec),  
    pkts_sent, pkts_received, Action.
    """)

    st.markdown("""
    There are no missing values.  
    The target variable is **Action**.
    """)

# =========================================================
# TAB 3 — ML MODELS
# =========================================================
with tab3:

    st.markdown("## 🤖 Machine Learning Models Used")

    st.markdown("""
    **Logistic Regression** – Linear probabilistic classifier.  
    **Decision Tree** – Rule-based hierarchical classifier.  
    **kNN** – Distance-based nearest neighbor classifier.  
    **Naive Bayes** – Probabilistic model using Bayes theorem.  
    **Random Forest** – Ensemble of multiple decision trees.  
    **XGBoost** – Gradient boosting based high-performance ensemble model.
    """)
