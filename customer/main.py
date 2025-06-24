# Customer Churn Prediction - Full Code

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

# Step 2: Load the dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())

# Step 3: Handle missing values and convert datatypes
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Drop customerID (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Step 4: Encode target variable (Churn: Yes -> 1, No -> 0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Step 5: Encode categorical variables using One-Hot Encoding
categorical_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Step 6: Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Step 7: Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Step 11: Print ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", round(roc_auc, 2))

# Optional: Save model for later use (if needed)
# import joblib
# joblib.dump(model, "models/logistic_model.pkl")
