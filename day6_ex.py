import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# load the dataset credit card dataset from storage.googleapis.com the credit card dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# explore the dataset
print(f"Dataset Information:\n{df.info()}")

# print the class distribution
print(f"Class Distribution:\n{df['Class'].value_counts()}")

# split the dataset
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the random forest
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# predict and evaluate the model
y_pred = rf_model.predict(X_test)

# classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print(f"ROC AUC Score: {roc_auc:.4f}")

# apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# display the new class distribution
print(f"Resampled Class Distribution:\n{y_resampled.value_counts()}")

# train forest model with resampled data
rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_resampled, y_resampled)

# predict and evaluate the model with resampled data
y_pred_smote = rf_model_smote.predict(X_test)
print("\nClassification Report after SMOTE:")
print(classification_report(y_test, y_pred_smote))

roc_auc_smote = roc_auc_score(y_test, rf_model_smote.predict_proba(X_test)[:, 1])
print(f"ROC AUC Score after SMOTE: {roc_auc_smote:.4f}")