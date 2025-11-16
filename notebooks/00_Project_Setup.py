# Databricks notebook source
# MAGIC %md
# MAGIC # Adult Income Prediction - Project Setup
# MAGIC
# MAGIC **Author:** shalindri20@gmail.com
# MAGIC **Warehouse:** Serverless Starter Warehouse

# COMMAND ----------
# Install packages
%pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn shap mlflow --quiet
dbutils.library.restartPython()

# COMMAND ----------
import mlflow
import pandas as pd
import numpy as np

print("✓ Libraries imported")

# COMMAND ----------
# Configuration
PROJECT_CONFIG = {
    "project_name": "adult_income_prediction",
    "author": "shalindri20@gmail.com",
    "mlflow_experiment": "/Users/shalindri20@gmail.com/Adult_Income_MLflow_Production",
    "data_path": "../data/adult.csv",
    "random_state": 42,
    "test_size": 0.2
}

# Set MLflow experiment
mlflow.set_experiment(PROJECT_CONFIG["mlflow_experiment"])

experiment = mlflow.get_experiment_by_name(PROJECT_CONFIG["mlflow_experiment"])
print(f"✓ MLflow Experiment: {experiment.name}")
print(f"✓ Experiment ID: {experiment.experiment_id}")

# COMMAND ----------
# MAGIC %md
# MAGIC ✅ Setup Complete!
# MAGIC
# MAGIC Next: Run notebook `01_Data_Loading.py`
