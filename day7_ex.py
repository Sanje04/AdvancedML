import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


# load the dataset
# telco churn dataset
url = "https://raw.githubusercontent.com/KimathiNewton/Telco-Customer-Churn/refs/heads/master/Datasets/telco_churn.csv"
df = pd.read_csv(url)

# print the dataset information
print(f"Dataset Information:\n{df.info()}")
# print the first few rows of the dataset
print(f"First few rows of the dataset:\n{df.head()}")
# print the class distribution
print(f"Class Distribution:\n{df['Churn'].value_counts()}")

# handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

# encode categorical variables
label_encoders = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        df[column] = label_encoders.fit_transform(df[column])

#Encode target variable
df['Churn'] = label_encoders.fit_transform(df['Churn'])

# scale numerical features
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# split the dataset, features and targets
X = df.drop('Churn')
y = df['Churn']

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Display the class distribution after resampling
print(f"Class Distribution after SMOTE:\n{y_train_resampled.value_counts()}")
print(pd.Series(y_train_resampled).value_counts())


# train the random forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# train XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

# train LightGBM
lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(X_train_resampled, y_train_resampled)

# make predictions
rf_predictions = rf_model.predict(X_test)
roc_auc_rf = roc_auc_score(y_test, rf_predictions)
xgb_predictions = xgb_model.predict(X_test)
roc_auc_xgb = roc_auc_score(y_test, xgb_predictions)
lgb_predictions = lgb_model.predict(X_test)
roc_auc_lgb = roc_auc_score(y_test, lgb_predictions)

# evaluate the models
print("\nRandom Forest Classifier:")
print(classification_report(y_test, rf_predictions))
print(f"Accuracy: {accuracy_score(y_test, rf_predictions)}")
print(f"ROC AUC: {roc_auc_rf}")

print("\nXGBoost Classifier:")
print(classification_report(y_test, xgb_predictions))
print(f"Accuracy: {accuracy_score(y_test, xgb_predictions)}")
print(f"ROC AUC: {roc_auc_xgb}")

print("\nLightGBM Classifier:")
print(classification_report(y_test, lgb_predictions))
print(f"Accuracy: {accuracy_score(y_test, lgb_predictions)}")
print(f"ROC AUC: {roc_auc_lgb}")
