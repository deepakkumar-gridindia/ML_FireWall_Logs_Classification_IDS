import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    roc_curve
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="ML-Based IDS System", layout="wide")

st.title("🚀 Machine Learning Based Intrusion Detection System (Firewall)")

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🔎 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Model Comparison",
        "📈 ROC Curve",
        "📁 Dataset Information",
        "🤖 ML Models"
    ]
)

# =========================================================
# COMMON VARIABLES
# =========================================================
model_files = {
    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree Classifier": "decision_tree.pkl",
    "K-Nearest Neighbor Classifier": "knn_model.pkl",
    "Naive Bayes Classifier": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
    "XGBoost (Ensemble)": "xgboost.pkl"
}

label_encoder = joblib.load("label_encoder.pkl")

# =========================================================
# HOME PAGE
# =========================================================
if page == "🏠 Home":

    st.markdown("""
    ### 🔍 About ML-Based IDS
    
    This application demonstrates a Machine Learning-based Intrusion Detection System (IDS)
    for firewall traffic classification. The system classifies network traffic actions
    such as allow, deny, drop, and reset-both.
    """)

    # -------------------------------
    # Download Section
    # -------------------------------
    st.markdown("### ⬇️ Download Test Dataset")

    GITHUB_TEST_DATA_URL = "https://raw.githubusercontent.com/deepakkumar-gridindia/ML_FireWall_Logs_Classification_IDS/main/test_dataset.csv"

    if st.button("Download Test Dataset"):
        response = requests.get(GITHUB_TEST_DATA_URL)
        st.download_button(
            label="Click to Download",
            data=response.content,
            file_name="test_dataset.csv",
            mime="text/csv"
        )

    # -------------------------------
    # Upload Section
    # -------------------------------
    # -------------------------------
    # Upload Section
    # -------------------------------
    st.markdown("### 📤 Upload Test Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    # If new file uploaded → store it
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
    
        if "Action" not in test_data.columns:
            st.error("Dataset must contain 'Action' column.")
            st.stop()
    
        st.session_state["test_data"] = test_data
    
    # -------------------------------
    # Model Selection
    # -------------------------------
    st.markdown("### ⚙️ Select Model")
    selected_model_name = st.selectbox(
        "Choose a Model",
        list(model_files.keys())
    )
    
    # =====================================================
    # IF DATA EXISTS → RUN MODEL AUTOMATICALLY
    # =====================================================
    if "test_data" in st.session_state:
    
        test_data = st.session_state["test_data"]
    
        y_true = test_data["Action"]
        X_test = test_data.drop(columns=["Action"])
    
        model = joblib.load(model_files[selected_model_name])
    
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
    
        # Store latest results
        st.session_state["y_true"] = y_true
        st.session_state["y_pred"] = y_pred
        st.session_state["y_prob"] = y_prob
        st.session_state["selected_model"] = selected_model_name
    
        # -------------------------------
        # Evaluation Metrics
        # -------------------------------
        st.markdown(f"## 📊 Evaluation Metrics – {selected_model_name}")
    
        col1, col2, col3 = st.columns(3)
    
        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
        col1.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.4f}")
    
        col2.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.4f}")
        col2.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
    
        col3.metric("AUC", f"{roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'):.4f}")
        col3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")
    
        st.success("✅ Results Update Automatically When Model Changes")
    
    else:
        st.info("Upload a dataset to begin evaluation.")


        # -------------------------------
        # Confusion Matrix
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
        # Classification Report
        # -------------------------------
        st.markdown(f"## 📄 Detailed Classification Report – {selected_model_name}")

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        class_mapping = {str(i): label for i, label in enumerate(label_encoder.classes_)}
        report_df.rename(index=class_mapping, inplace=True)

        class_df = report_df.loc[label_encoder.classes_, ["precision", "recall", "f1-score", "support"]]

        st.dataframe(class_df.round(4), use_container_width=True)

        st.success("✅ Results Persist Across Tabs")


        st.success("✅ Evaluation Completed Successfully")

# =========================================================
# MODEL COMPARISON PAGE
# =========================================================
elif page == "📊 Model Comparison":

    st.markdown("## 📊 Model Comparison")

    comparison_data = {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "kNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ],
        "Accuracy": [0.9829, 0.9978, 0.9971, 0.6918, 0.9977, 0.9982],
        "F1 Score": [0.9826, 0.9977, 0.9967, 0.7752, 0.9976, 0.9980]
    }

    comp_df = pd.DataFrame(comparison_data)

    st.bar_chart(comp_df.set_index("Model"))

# =========================================================
# ROC CURVE PAGE
# =========================================================
elif page == "📈 ROC Curve":

    st.markdown("## 📈 ROC Curve Analysis (One-vs-Rest)")

    # Check if Home tab has already run
    if "y_true" not in st.session_state:
        st.warning("⚠️ Please upload dataset and run model in Home tab first.")
        st.stop()

    y_true = st.session_state["y_true"]
    y_prob = st.session_state["y_prob"]

    from sklearn.preprocessing import label_binarize
    import numpy as np

    classes = np.arange(len(label_encoder.classes_))
    y_true_bin = label_binarize(y_true, classes=classes)

    # Independent model dropdown
    selected_model_name = st.selectbox(
        "Select Model for ROC Curve",
        list(model_files.keys()),
        key="roc_model_selector"
    )

    # Load model again only to compute probabilities fresh
    model = joblib.load(model_files[selected_model_name])

    test_data = st.session_state["test_data"]
    X_test = test_data.drop(columns=["Action"])

    y_prob = model.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(6,4))

    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])

        ax.plot(
            fpr,
            tpr,
            label=f"{label_encoder.classes_[i]} (AUC={auc_score:.3f})"
        )

    ax.plot([0,1],[0,1],'k--', linewidth=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {selected_model_name}")

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    st.pyplot(fig)






# =========================================================
# DATASET INFORMATION PAGE
# =========================================================
elif page == "📁 Dataset Information":

    st.markdown("## 📘 Internet Firewall Data — Dataset Information")

    st.markdown("""
    - **Instances:** 65,532  
    - **Features:** 12  
    - **Task:** Multiclass Classification  
    - **Class Labels:** allow, deny, drop, reset-both  
    """)

    st.markdown("""
    Source Port, Destination Port, NAT Source Port, NAT Destination Port,
    Bytes, Bytes Sent, Bytes Received, Packets,
    Elapsed Time (sec), pkts_sent, pkts_received, Action.
    """)

# =========================================================
# ML MODELS PAGE
# =========================================================
elif page == "🤖 ML Models":

    st.markdown("## 🤖 Machine Learning Models Used")

    st.markdown("""
    **Logistic Regression** – Linear probabilistic classifier.  
    **Decision Tree** – Rule-based hierarchical classifier.  
    **kNN** – Distance-based nearest neighbor classifier.  
    **Naive Bayes** – Probabilistic model.  
    **Random Forest** – Ensemble of decision trees.  
    **XGBoost** – Gradient boosting ensemble model.
    """)
