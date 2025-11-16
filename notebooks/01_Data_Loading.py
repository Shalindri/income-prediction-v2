# Databricks notebook source
# MAGIC %md
# MAGIC # Data Loading and Validation

# COMMAND ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------
# Load data
df = pd.read_csv("../data/adult.csv")

print(f"✓ Data loaded: {df.shape}")
print(f"✓ Columns: {df.columns.tolist()}")

# COMMAND ----------
# Data info
print(df.info())
print("\n" + "="*50)
print(df.describe())

# COMMAND ----------
# Check target distribution
print("Target Distribution:")
print(df['income'].value_counts())

plt.figure(figsize=(8, 5))
df['income'].value_counts().plot(kind='bar')
plt.title("Income Distribution")
plt.ylabel("Count")
plt.show()

# COMMAND ----------
# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# COMMAND ----------
# Save for next notebook
df.to_csv("/tmp/adult_raw.csv", index=False)
print("✓ Data saved for next notebook")

# COMMAND ----------
# MAGIC %md
# MAGIC ✅ Data Loaded!
# MAGIC
# MAGIC Next: Run notebook `02_Feature_Engineering.py`
