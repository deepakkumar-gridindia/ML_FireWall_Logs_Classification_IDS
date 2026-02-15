# ML_FireWall_Logs_Classification_IDS
Intrusion Detection System of Firewall Logs Using Machine Learning Models 

I.	Problem Statement :-> Assignment Details
Step 1: Dataset choice
Choose ONE classification dataset of your choice from any public repository - Kaggle or UCI. It may be a binary classification problem or a multi-class classification problem.
Minimum Feature Size: 12
Minimum Instance Size: 500

Step 2: Machine Learning Classification models and Evaluation metrics.
Implement the following classification models using the dataset chosen above. All the 6 ML models have to be implemented on the same dataset.
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost
For each of the models above, calculate the following evaluation metrics:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score

II.	Dataset Description :-> 
Internet Firewall Data — IDS Machine Learning Model 📘 Dataset Description

The Internet Firewall Data is a publicly-available dataset from the UCI Machine Learning Repository (Dataset ID 542). It contains real network traffic records captured from a university’s firewall and is widely used for classification tasks in network security and intrusion detection research. Instances: 65,532 Features: 12 Task: Multiclass Classification Class Labels: allow, deny, drop and reset-both (These represent the action taken by the firewall on a given traffic session.)
📊 Feature Overview
Each row in the dataset represents one firewall log entry, and the following 12 attributes are included:
Feature Description Source Port Port number initiating the connection Destination Port Receiving port number NAT Source Port Source port after NAT translation NAT Destination Port Destination port after NAT translation Action Target label (firewall decision) Bytes Total bytes transferred Bytes Sent Bytes sent by the source Bytes Received Bytes received by the destination Packets Total number of packets Elapsed Time (sec) Duration of the session pkts_sent Packets sent by the source pkts_received Packets received by the destination (Attribute list adapted from the dataset documentation)
There are no missing values in the dataset, and the class label (Action) is used as the target in supervised learning tasks.
📜 Citation
Internet Firewall Data [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5131M.

III.	🤖 Project: Intrusion Detection System (IDS)

This repository contains a machine learning-based Intrusion Detection System (IDS) trained on the Internet Firewall Data. The main goal is to automatically classify network traffic records as benign or potentially malicious based on the firewall’s historical actions.

🔧 Included ML Components
✔ Data preprocessing and feature scaling 
✔ Handling of class imbalance (if applicable) 
✔ Model training and evaluation 
✔ Performance metrics (Accuracy, F1-score, Precision, Recall, Confusion Matrix) 
✔ Trained model checkpoint and prediction interface
IV.	Model Performance Comparison Table with the evaluation metrics
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9855	0.9985	0.9852	0.9855	0.9851	0.9752
Decision Tree	0.9978	0.9989	0.9977	0.9978	0.9977	0.9962
kNN	0.9971	0.999	0.9963	0.9971	0.9967	0.995
Naive Bayes	0.9908	0.9997	0.9977	0.9908	0.9942	0.9844
Random Forest (Ensemble)	0.9977	0.9995	0.9976	0.9977	0.9976	0.996
XGBoost (Ensemble)	0.998	0.9998	0.9978	0.998	0.9979	0.9966

V.	Observations on Model Performance
ML Model Name	Observation about Model Performance
Logistic Regression	Logistic Regression shows strong overall performance with high accuracy (98.55%) and AUC (0.9985), indicating good class separability. However, its MCC is slightly lower than ensemble models, suggesting it is less effective in capturing complex non-linear relationships compared to tree-based approaches.
Decision Tree	The Decision Tree achieves very high accuracy and MCC (0.9962), demonstrating strong capability in modeling non-linear decision boundaries. It performs robustly across all classes but may still be prone to overfitting compared to ensemble techniques.
kNN	kNN performs competitively with high accuracy (99.71%) and AUC (0.9990), showing effective distance-based classification. Its slightly lower MCC compared to ensemble models indicates moderate sensitivity to local data distribution and class imbalance.
Naive Bayes	Naive Bayes now demonstrates excellent performance with high accuracy (99.08%) and very high AUC (0.9997), indicating strong probabilistic discrimination. Despite its independence assumption, it performs surprisingly well on this dataset, suggesting features are reasonably separable.
Random Forest (Ensemble)	Random Forest delivers highly stable and consistent performance across all evaluation metrics, including high MCC (0.9960). The ensemble of multiple trees effectively captures complex relationships while reducing overfitting.
XGBoost (Ensemble)	XGBoost achieves the best overall performance with the highest accuracy (99.80%), AUC (0.9998), and MCC (0.9966). Its gradient boosting mechanism efficiently captures intricate feature interactions, making it the most suitable and robust model for this dataset.


VI.	Streamlit App Link and GitHub Repository Link
🔗 Streamlit Application (Live Deployment):
https://mlfirewalllogsclassificationids-uchwn3eylxdvkjaftisrzq.streamlit.app/
💻 GitHub Repository (Source Code):
https://github.com/deepakkumar-gridindia/ML_FireWall_Logs_Classification_IDS.git
<img width="468" height="655" alt="image" src="https://github.com/user-attachments/assets/232f100b-5068-479f-8709-2733588aa75d" />



