import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import lightgbm as lgb # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
from catboost import CatBoostClassifier # type: ignore
from xgboost import XGBClassifier # type: ignore

# import the titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data = pd.read_csv(url)

# select our features and target
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

# Handle the missing values
titanic_data.fillna({'Age': titanic_data['Age'].mode()[0]}, inplace=True)

# Encode categorical variables
label_encoder = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    titanic_data[col] = le.fit_transform(titanic_data[col])
    label_encoder[col] = le
    
# Split the dataset into features and target variable
X = titanic_data[features]
y = titanic_data[target]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# display dataset information
print(f"Number of features: {X_train.shape}")
print(f"Classes: {X_test.shape}")

# train LightGBM model
lgb_train = lgb.LGBMClassifier()
lgb_train.fit(X_train, y_train)

# predict and evaluate the model
y_pred = lgb_train.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# train the catboost model
# specify categorical features
cat_features = ['Pclass', 'Sex', 'Embarked']
cat_model = CatBoostClassifier(cat_features=cat_features, verbose=0)
cat_model.fit(X_train, y_train)

# Predict and evaluate
cat_pred = cat_model.predict(X_test)
cat_accuracy = accuracy_score(y_test, cat_pred)

# Train the XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"CatBoost Model Accuracy: {cat_accuracy:.4f}")
print(f"XGBoost Model Accuracy: {xgb_accuracy:.4f}")

# train catboost without encoding categorical features
cat_model_native = CatBoostClassifier(cat_features=['Sex', 'Embarked'], verbose=0)
cat_model_native.fit(X_train, y_train)

# Predict and evaluate
cat_pred_native = cat_model_native.predict(X_test)
print(f"CatBoost Model (without encoding) Accuracy: {accuracy_score(y_test, cat_pred_native):.4f}")