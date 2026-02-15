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
# SYSTEM DESCRIPTION
# =========================================================
st.markdown("""
## 📘 Internet Firewall Data — IDS Machine Learning Model

The **Internet Firewall Data** is a publicly available dataset from the  
**UCI Machine Learning Repository (Dataset ID: 542)**.  
It contains real network traffic records captured from a university firewall 
and is widely used for classification tasks in network security and intrusion detection research.

### 📊 Dataset Summary
- **Instances:** 65,532  
- **Features:** 12  
- **Task:** Multiclass Classification  
- **Class Labels:** allow, deny, drop, reset-both  
  (These represent the action taken by the firewall on a given traffic session.)

---

## 📊 Feature Overview

Each row in the dataset represents one firewall log entry.  
The following 12 attributes are included:

| Feature | Description |
|----------|-------------|
| Source Port | Port number initiating the connection |
| Destination Port | Receiving port number |
| NAT Source Port | Source port after NAT translation |
| NAT Destination Port | Destination port after NAT translation |
| Action | Target label (firewall decision) |
| Bytes | Total bytes transferred |
| Bytes Sent | Bytes sent by the source |
| Bytes Received | Bytes received by the destination |
| Packets | Total number of packets |
| Elapsed Time (sec) | Duration of the session |
| pkts_sent | Packets sent by the source |
| pkts_received | Packets received by the destination |

*(Attribute list adapted from the dataset documentation.)*

There are **no missing values** in the dataset.  
The class label **'Action'** is used as the target variable in supervised learning tasks.

---

## 🤖 Project: Intrusion Detection System (IDS)

This repository implements a **Machine Learning-based Intrusion Detection System (IDS)** 
trained on the Internet Firewall Data.

The objective of this IDS is to automatically classify firewall network traffic 
as benign or potentially malicious based on historical firewall actions.

---

## 📚 Citation

If you use this dataset or code in published work, please cite:

Internet Firewall Data [Dataset]. (2019).  
UCI Machine Learning Repository.  
https://doi.org/10.24432/C5131M
""")

# =========================================================
# MODEL DESCRIPTIONS
# =========================================================
st.markdown("### 🤖 Machine Learning Models Used")

st.markdown("""
**1. Logistic Regression**  
A linear classification model that estimates class probabilities using the logistic function.  
It performs well when classes are linearly separable and serves as a strong baseline model.

**2. Decision Tree Classifier**  
A tree-based model that splits data using decision rules derived from features.  
It captures non-linear patterns but may overfit without proper control.

**3. K-Nearest Neighbor (kNN)**  
A distance-based algorithm that classifies samples based on the majority class among nearest neighbors.  
It works well when similar traffic behaviors cluster together.

**4. Naive Bayes (Gaussian)**  
A probabilistic classifier based on Bayes’ theorem with independence assumptions.  
It is computationally efficient but may struggle when feature dependencies exist.

**5. Random Forest (Ensemble)**  
An ensemble method that combines multiple decision trees to improve generalization.  
It reduces overfitting and handles complex traffic patterns effectively.

**6. XGBoost (Ensemble)**  
A gradient boosting algorithm that builds trees sequentially to minimize prediction errors.  
It provides high accuracy and strong performance on structured firewall datasets.
""")

# =========================================================
# DATASET DESCRIPTION
# =========================================================
st.markdown("### 📁 Dataset Description")

st.markdown("""
The dataset used in this IDS system consists of firewall network traffic logs.  
It contains multiple numerical and categorical features such as source port, 
destination port, bytes transferred, packets, and elapsed time.

The target variable **'Action'** represents the classified firewall action for 
each network event. The dataset has been preprocessed and split into training 
and testing subsets before model deployment.
""")

# =========================================================
# DOWNLOAD TEST DATASET
# =========================================================
st.markdown("### ⬇️ Download Test Dataset")

# GITHUB_TEST_DATA_URL = "https://github.com/deepakkumar-gridindia/ML_FireWall_Logs_Classification_IDS/blob/main/test_dataset.csv"
GITHUB_TEST_DATA_URL = "https://raw.githubusercontent.com/deepakkumar-gridindia/ML_FireWall_Logs_Classification_IDS/main/test_dataset.csv"


st.markdown("""
Click below to download the prepared test dataset.  
You may upload the same dataset or a new dataset in the same format.
""")

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
        st.error("Unable to fetch dataset from GitHub. Check the link.")

# =========================================================
# UPLOAD SECTION
# =========================================================
st.markdown("""
### 📤 Upload Test Dataset

Please download the test dataset above and upload it below.  
You may also upload a new dataset with the same feature structure.  
The ML-Based IDS system will evaluate the uploaded dataset and display results.
""")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# =========================================================
# LOAD MODELS
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
# MODEL SELECTION
# =========================================================
st.markdown("### ⚙️ Select Model")

selected_model_name = st.selectbox(
    "Choose a Model for Evaluation",
    list(model_files.keys())
)

model = joblib.load(model_files[selected_model_name])

# =========================================================
# PREDICTION & RESULTS
# =========================================================
if uploaded_file is not None:

    # test_data = pd.read_csv(uploaded_file)
    try:
        test_data = pd.read_csv(uploaded_file)
    except Exception:
        test_data = pd.read_csv(uploaded_file, engine="python")

    if "Action" not in test_data.columns:
        st.error("Dataset must contain 'Action' column for evaluation.")
        st.stop()

    y_true = test_data["Action"]
    X_test = test_data.drop(columns=["Action"])

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # =====================================================
    # EVALUATION METRICS
    # =====================================================
    st.markdown(f"## 📊 Evaluation Metrics – {selected_model_name}")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    metrics_col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    metrics_col1.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.4f}")
    
    metrics_col2.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.4f}")
    metrics_col2.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    metrics_col3.metric("AUC", f"{roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'):.4f}")
    metrics_col3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    st.markdown(f"## 🔢 Confusion Matrix – {selected_model_name}")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    st.pyplot(fig)

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================
    # st.markdown(f"## 📄 Classification Report – {selected_model_name}")

    # report_df = pd.DataFrame(
    #     classification_report(
    #         y_true,
    #         y_pred,
    #         output_dict=True,
    #         zero_division=0
    #     )
    # ).transpose()

    # st.dataframe(report_df.round(4))

    import matplotlib.pyplot as plt
    import numpy as np
    
    st.markdown(f"## 📄 Classification Report – {selected_model_name}")
    
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    
    report_df = pd.DataFrame(report).transpose()
    
    # Keep only main metrics (remove support & accuracy row)
    metrics_to_plot = report_df.loc[
        [label for label in report_df.index if label not in ["accuracy"]],
        ["precision", "recall", "f1-score"]
    ]
    
    fig, ax = plt.subplots()
    
    heatmap_data = metrics_to_plot.values
    im = ax.imshow(heatmap_data)
    
    # Axis Labels
    ax.set_xticks(np.arange(len(metrics_to_plot.columns)))
    ax.set_yticks(np.arange(len(metrics_to_plot.index)))
    ax.set_xticklabels(metrics_to_plot.columns)
    ax.set_yticklabels(metrics_to_plot.index)
    
    # Rotate column headers
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Annotate values
    for i in range(len(metrics_to_plot.index)):
        for j in range(len(metrics_to_plot.columns)):
            ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                    ha="center", va="center")
    
    ax.set_title("Classification Report Heatmap")
    plt.colorbar(im)
    
    st.pyplot(fig)


    st.success("✅ Evaluation Completed Successfully")
