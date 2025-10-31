
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report)
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
#%%
#Load dataset
df = pd.read_csv('D:/zoology download/Projects-20240722T093004Z-001/Projects/thyroid_cancer/thyroid_cancer/dataset.csv')
df.head(3)
#%%
# Inspect data
print(f"Dataset shape: {df.shape}")
print(df.columns.tolist())
print(df.dtypes.value_counts())
print("\nSample data:\n", df.head(3).T)

#%%
print(df.isnull().sum())
#%%
df = df.dropna() 
#%%
# Encode target
df['Recurred'] = df['Recurred'].map({'No': 0, 'Yes': 1})
# Identify categorical columns and convert into numeric values
cat_col= df.select_dtypes(include='object').columns
for col in cat_col:
    print(col)
    print((df[col].unique()), list(range(df[col].nunique())))
    df[col].replace((df[col].unique()), range(df[col].nunique()), inplace=True)
    print('*'*90)
    print()
#%%
# Standardize numerical column(s)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])
#%%
#Split training and testing sets
from sklearn.model_selection import train_test_split
X = df.drop('Recurred', axis=1)
y = df['Recurred']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
#%%
#EDA
plt.figure(figsize=(5,4))
sns.countplot(x=y)
plt.title("Class Distribution (0 = No Recurrence, 1 = Recurrence)")
plt.show()
#%%
plt.figure(figsize=(5,4))
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()
#%%
corr = pd.concat([X_train, y_train], axis=1).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()
#%%
#Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred, target_names=['No Recurrence','Recurrence']))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
acc_lr = accuracy_score(y_test, y_pred)
f1_lr =f1_score(y_test, y_pred)
#%%
#create picklefile
import pickle
with open('LogisticR_T.pkl', 'wb') as file3:
    pickle.dump(model, file3)
#%%
#hyperparameter tuning
param_grid = {
 'n_estimators': [50, 100, 200],
 'max_depth': [None, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)
print("Best RF params:", grid.best_params_)
best_rf = grid.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
print(classification_report(y_test, y_pred_best_rf))

#%%
#Random Forest Classifier
rf = RandomForestClassifier(**grid.best_params_, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Classification Report")
print(classification_report(y_test, y_pred_best_rf))
acc_rf = accuracy_score(y_test, y_pred_best_rf)
f1_rf= f1_score(y_test, y_pred_best_rf)
#%%
#create pickle file
with open('RF_T.pkl', 'wb') as file2:
    pickle.dump(rf, file2)
#%%
#XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss') 
xgb.fit(X_train, y_train) 
y_pred_xgb = xgb.predict(X_test) 
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb)) 
print(classification_report(y_test, y_pred_xgb))
acc_gb = accuracy_score(y_test, y_pred_xgb)
f1_gb = f1_score(y_test, y_pred_xgb)
#%%
#create pickle file
with open('XGB_T.pkl', 'wb') as file:
    pickle.dump(xgb, file)
#%%
#Model Comparison and Visualization
metrics = pd.DataFrame({'Model': ['Logistic Regression','Random Forest','Gradient Boosting',],'Accuracy': [acc_lr, acc_rf, acc_gb], 'F1-score': [f1_lr, f1_rf, f1_gb]
 })
metrics = metrics.melt(id_vars='Model', var_name='Metric', value_name='Score')
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.show()
#%%
