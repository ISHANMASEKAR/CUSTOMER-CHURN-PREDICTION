📊 Customer Churn Prediction (End-to-End ML Project)

An end-to-end Machine Learning project that predicts whether a telecom customer will churn (leave the service) or not. The project includes data preprocessing, feature engineering, model training, evaluation, and a Streamlit web application for real-time predictions.

🚀 Project Overview

Customer churn is a major problem in the telecom industry. This project uses machine learning to predict churn based on customer demographics, account information, and services used.

📂 Dataset
Telco Customer Churn Dataset
Contains customer information such as:
Gender, Senior Citizen, Partner, Dependents
Tenure, Contract type
Internet services, Online security, Tech support
Monthly & Total charges
Target: Churn
⚙️ Technologies Used
Python 🐍
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
XGBoost
Imbalanced-learn (SMOTE)
Streamlit
Pickle (Model serialization)
🧹 Data Preprocessing
Removed irrelevant column (customerID)
Handled missing values in TotalCharges
Converted categorical features using Label Encoding
Encoded target variable (Churn: Yes → 1, No → 0)
Balanced dataset using SMOTE
📊 Exploratory Data Analysis (EDA)
Checked distribution of target variable (Churn vs No Churn)
Visualized outliers using boxplots
Understood feature distributions
🤖 Machine Learning Models

The following models were trained and compared:

Decision Tree Classifier 🌳
Random Forest Classifier 🌲
XGBoost Classifier ⚡

✔ Best Model: Random Forest Classifier

📈 Model Evaluation

Models were evaluated using:

Accuracy Score
Confusion Matrix
Classification Report
5-Fold Cross Validation
💾 Model Saving

The trained model and encoders were saved using Pickle:

customer_churn_model.pkl → Trained ML model
encoders.pkl → Label encoders for preprocessing
