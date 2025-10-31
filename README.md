# Thyroid-Cancer-Recurrence-Prediction-using-Machine-Learning
A predictive ML model that identifies thyroid cancer recurrence risk based on patient demographics, diagnosis, and treatment history using Logistic Regression, Random Forest, and XGBoost.
# 🧬 Thyroid Cancer Recurrence Detection using Machine Learning

This project predicts the **likelihood of thyroid cancer recurrence** in patients based on their medical history, diagnostic findings, and treatment details. Using supervised machine learning algorithms, the model classifies whether a patient is likely to experience a **recurrence (Yes/No)** of thyroid cancer after treatment.

---

## 🎯 Objective
To develop a machine learning model that can accurately predict the **recurrence of thyroid cancer** in previously diagnosed patients using clinical, pathological, and treatment-related parameters.  
The goal is to assist healthcare professionals in identifying high-risk patients early, enabling personalized follow-up and intervention strategies.

---

## 📘 Dataset Overview

### 📍 Source
The dataset contains information on thyroid cancer patients, their diagnosis, treatment history, and recurrence outcomes.

### 🧩 Description of Columns

| Feature | Description |
|----------|-------------|
| Age | Age at the time of diagnosis or treatment |
| Gender | Gender of the patient (Male/Female) |
| Smoking | Current smoking status |
| Hx Smoking | Smoking history (ever smoked or not) |
| Hx Radiotherapy | History of radiotherapy for any condition |
| Thyroid Function | Indicates any thyroid function abnormality |
| Physical Examination | Clinical examination findings |
| Adenopathy | Presence of enlarged lymph nodes in the neck region |
| Pathology | Type of thyroid cancer based on biopsy |
| Focality | Cancer type: Unifocal (single site) or Multifocal (multiple sites) |
| Risk | Risk category of the cancer (low, medium, or high) |
| T | Tumor size and invasion level |
| N | Lymph node involvement (nodal classification) |
| M | Distant metastasis presence |
| Stage | Overall cancer stage (I–IV) |
| Response | Response to treatment (positive/negative/stable) |
| **Recurred** | Target variable — whether cancer has recurred (`Yes` or `No`) |

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Loaded dataset and inspected structure and missing values  
- Dropped rows with missing entries  
- Converted categorical variables into numeric codes  
- Scaled continuous variables (e.g., `Age`) using **StandardScaler**  
- Encoded target variable `Recurred` as binary (1 = Yes, 0 = No)

### 2️⃣ Exploratory Data Analysis (EDA)
Visualized data distributions and relationships:
- Class distribution of recurrence vs. non-recurrence  
- Age distribution among patients  
- Correlation heatmap of features  

### 3️⃣ Model Training
Implemented and evaluated multiple models:

| Model | Description |
|--------|-------------|
| Logistic Regression | Baseline linear classifier |
| Random Forest (Tuned) | Ensemble model tuned with GridSearchCV |
| XGBoost | Gradient boosting model optimized for classification |

### 4️⃣ Model Evaluation
Each model was assessed using the following metrics:
- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrix Visualization**
- **ROC-AUC Score**

### 5️⃣ Model Comparison
Models were compared based on overall accuracy and F1-score through bar chart visualization.

---

## 📊 Model Performance Summary

| Model | Accuracy | F1-Score |
|--------|-----------|----------|
| Logistic Regression | 87% | 0.84 |
| Random Forest (Tuned) | **91%** | **0.88** |
| XGBoost | **92%** | **0.89** |

*(Note: Results may vary depending on random seed and dataset split.)*

---

## 🧠 Technologies Used
- **Python 3.8+**
- **Libraries:**
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `xgboost`
  - `pickle`

---
