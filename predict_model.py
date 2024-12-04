import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset from CSV
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Features and target
X = df.drop(['diabetes'], axis=1)
y = df['diabetes']

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the model, scaler, and features
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')
joblib.dump(scaler, 'scaler.pkl')
