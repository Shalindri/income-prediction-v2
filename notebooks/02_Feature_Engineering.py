# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering and Preprocessing

# COMMAND ----------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# COMMAND ----------
# Load data
df = pd.read_csv("/tmp/adult_raw.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace('.', '_')

print(f"✓ Data loaded: {df.shape}")

# COMMAND ----------
# Strip whitespace from strings and handle missing values
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip().replace('?', np.nan)

# Drop duplicates
df = df.drop_duplicates()

print(f"✓ After cleaning: {df.shape}")

# COMMAND ----------
# Handle missing values (simple strategy: drop rows with missing values)
df = df.dropna()

print(f"✓ After removing missing: {df.shape}")

# COMMAND ----------
# Separate features and target
X = df.drop('income', axis=1)
y = df['income'].map({'<=50K': 0, '>50K': 1})

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"✓ Target distribution: {y.value_counts().to_dict()}")

# COMMAND ----------
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# COMMAND ----------
# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

X_encoded = X.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"✓ Categorical variables encoded")

# COMMAND ----------
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

# COMMAND ----------
# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"✓ Features scaled")

# COMMAND ----------
# Save processed data
import pickle

# Save data
X_train_scaled.to_csv("/tmp/X_train.csv", index=False)
X_test_scaled.to_csv("/tmp/X_test.csv", index=False)
y_train.to_csv("/tmp/y_train.csv", index=False)
y_test.to_csv("/tmp/y_test.csv", index=False)

# Save scaler and encoders
with open('/tmp/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('/tmp/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✓ Processed data saved")

# COMMAND ----------
# MAGIC %md
# MAGIC ✅ Feature Engineering Complete!
# MAGIC
# MAGIC Next: Run notebook `03_Model_Training.py`
