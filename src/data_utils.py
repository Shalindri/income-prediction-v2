"""Data loading and preprocessing utilities"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    """Load adult income dataset"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '_')
    return df

def clean_data(df):
    """Clean dataset"""
    # Strip whitespace
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().replace('?', np.nan)

    # Drop duplicates and missing
    df = df.drop_duplicates()
    df = df.dropna()

    return df

def encode_features(X):
    """Encode categorical features"""
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    X_encoded = X.copy()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    return X_encoded, label_encoders

def scale_features(X_train, X_test, numerical_cols):
    """Scale numerical features"""
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train_scaled, X_test_scaled, scaler
