# Cyberattack Behavior Detection

A machine learning system for detecting cyberattack behavior in network traffic using the UNSW-NB15 dataset.

The project compares Random Forest and XGBoost classifiers, evaluates their performance, selects the best-performing model, and deploys it through an interactive Streamlit web application.

---

## Overview

Cyberattacks generate network traffic patterns that can differ significantly from normal network behavior.

This project uses supervised machine learning to classify network traffic into two classes:

- Normal
- Cyberattack

The system is trained on the UNSW-NB15 dataset, a widely used benchmark dataset for network intrusion detection research.

---

## Project Pipeline

```text
UNSW-NB15 Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Random Forest ──────┐
                    │
                    ├── Model Evaluation
                    │
XGBoost ────────────┘
        │
        ▼
Best Model Selection
        │
        ▼
Model Serialization
        │
        ▼
Streamlit Deployment
        │
        ▼
Cyberattack Prediction


---

Dataset

This project uses the UNSW-NB15 dataset.

The dataset contains network traffic records generated in a controlled environment and includes both normal and malicious network activities.

Target

The target variable is:

label

where:

0 → Normal
1 → Attack

The attack_cat column is excluded from the model features because it contains attack-category information that would cause target leakage in binary classification.

Dataset source:

https://www.kaggle.com/datasets/dhoogla/unswnb15


---

Machine Learning Models

Two supervised learning algorithms were evaluated:

Random Forest

Random Forest is an ensemble learning method based on multiple decision trees.

XGBoost

XGBoost is a gradient boosting algorithm designed for efficient and accurate classification and regression tasks.

The models were evaluated using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

Classification Report


The model with the best F1 Score is selected for deployment.


---

Data Preprocessing

The preprocessing pipeline includes:

1. Loading the predefined training and testing datasets.


2. Separating features and target labels.


3. Removing label and attack_cat from the feature set.


4. Identifying categorical features.


5. Applying one-hot encoding.


6. Aligning training and testing feature columns.


7. Using the same feature order during deployment.




---

Deployment

The trained model is deployed using Streamlit.

The application allows users to:

Generate a new network traffic example.

Modify network traffic features.

Select protocol, service, and connection state.

Run the trained model.

View the predicted class.

View prediction probabilities.

Inspect the submitted feature values.


The demo examples are based on real samples from the UNSW-NB15 test set.


---

Project Structure

cyberattack-behavior-detection/
│
├── app.py
├── requirements.txt
│
├── best_cyberattack_model.pkl
├── feature_names.pkl
├── model_metadata.pkl
├── demo_samples.pkl
│
└── README.md


---

Installation

Clone the repository:

git clone https://github.com/mohammedwetwet/cyberattack-behavior-detection.git

Navigate to the project directory:

cd cyberattack-behavior-detection

Install the required dependencies:

pip install -r requirements.txt


---

Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.


---

Requirements

The project uses:

Python

Pandas

NumPy

Scikit-learn

XGBoost

Joblib

Streamlit



---

Important Note About Data Leakage

During development, the attack_cat feature was identified as a potential source of target leakage because it directly provides information about the attack class.

Therefore, attack_cat is explicitly removed before model training.

This ensures that the binary classifier learns from network traffic characteristics rather than directly receiving attack-category information.


---

Future Improvements

Potential future improvements include:

Multi-class attack classification.

Real-time network traffic integration.

Explainable AI using SHAP.

Model monitoring and performance tracking.

Automated preprocessing pipelines.

Integration with network intrusion detection systems.

Cloud deployment.



---

Technologies

Python
Pandas
NumPy
Scikit-learn
XGBoost
Joblib
Streamlit
UNSW-NB15


---

Author / Mohamed Adel Yousef Wetwet 

Developed as a machine learning project focused on cybersecurity and network intrusion detection.


---

License

This project is intended for educational and research purposes.
