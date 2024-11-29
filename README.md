# Data-Balancing-Using-SMOTE

This repository contains a Jupyter Notebook for building and analyzing a machine learning model using Python. The notebook covers key steps including data preprocessing, model training, evaluation, and feature importance analysis.

---

## Contents

### 1. Overview
The notebook focuses on implementing a machine learning pipeline with an emphasis on classification tasks. The key highlights include:
- Data preprocessing (handling missing values, scaling, encoding).
- Handling class imbalance using SMOTE.
- Training a Random Forest Classifier.
- Analyzing feature importances.

---

### 2. Dependencies
The following libraries are required to run the notebook:
- **Core Libraries**: `numpy`, `pandas`
- **Machine Learning**: 
  - `scikit-learn` for model building and evaluation
  - `imblearn` for oversampling with SMOTE
- **Data Preprocessing**:
  - `LabelEncoder`, `StandardScaler` for encoding and scaling
  - `SimpleImputer` for handling missing data
- **Evaluation**: `classification_report`, `accuracy_score`

---

### 3. Features
The notebook includes:
- **Model Training**:
  - Uses a Random Forest Classifier (`RandomForestClassifier`).
- **Feature Analysis**:
  - Calculates and displays feature importance using `feature_importances_` from the Random Forest model.
- **Data Handling**:
  - Balances imbalanced datasets using SMOTE.
  - Handles missing data with `SimpleImputer`.

---

### 4. Outputs
The notebook generates outputs such as:
- Classification metrics (accuracy, precision, recall).
- A DataFrame showing the top feature importances.

---

### 5. How to Run
1. Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn imbalanced-learn
   ```
2. Open the Jupyter Notebook and run the cells in sequence.

