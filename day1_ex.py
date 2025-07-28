from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()  
data = iris.data
target = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
# Fit the scaler on the training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train individual models
log_model = LogisticRegression(max_iter=200)
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()

log_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# combine the predictions
# creating our voting classifier
ensemble_model = VotingClassifier(estimators=[
    ('logistic', log_model),
    ('decision_tree', dt_model),
    ('knn', knn_model)
], voting='hard')
# majority voting

# train the ensemble model
ensemble_model.fit(X_train, y_train)

# predict with ensemble
y_pred_ensemble = ensemble_model.predict(X_test)

# evalute the accuracy of the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Model Accuracy: {accuracy_ensemble:.2f}")

# evalute individual models
y_pred_log = log_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.2f}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")