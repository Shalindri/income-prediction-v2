# Databricks notebook source
# MAGIC %md
# MAGIC # Model Registration to MLflow Model Registry

# COMMAND ----------
import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------
# Initialize client
client = MlflowClient()

# COMMAND ----------
# Load best run ID
with open('/tmp/best_run_id.txt', 'r') as f:
    best_run_id = f.read().strip()

print(f"Best Run ID: {best_run_id}")

# COMMAND ----------
# Get run details
run = client.get_run(best_run_id)

print("\n" + "="*70)
print("REGISTERING MODEL")
print("="*70)
print(f"Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
print(f"Run ID: {best_run_id}")
print(f"ROC-AUC: {run.data.metrics.get('roc_auc', 'N/A'):.4f}")
print("="*70)

# COMMAND ----------
# Register model
model_name = "income_prediction_model"
model_uri = f"runs:/{best_run_id}/model"

try:
    # Register the model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"\n✓ Model registered successfully!")
    print(f"  Model Name: {model_name}")
    print(f"  Version: {model_version.version}")

except Exception as e:
    print(f"\nNote: {str(e)}")
    print("Model may already be registered. Checking versions...")

    # Get latest version
    latest_versions = client.get_latest_versions(model_name)
    if latest_versions:
        model_version = latest_versions[0]
        print(f"\n✓ Using existing model:")
        print(f"  Model Name: {model_name}")
        print(f"  Version: {model_version.version}")

# COMMAND ----------
# Add description and tags
client.update_model_version(
    name=model_name,
    version=model_version.version,
    description=f"Adult Income Prediction Model - {run.data.tags.get('mlflow.runName', 'N/A')}"
)

client.set_model_version_tag(
    name=model_name,
    version=model_version.version,
    key="task",
    value="binary_classification"
)

client.set_model_version_tag(
    name=model_name,
    version=model_version.version,
    key="dataset",
    value="adult_income"
)

print(f"\n✓ Model metadata updated")

# COMMAND ----------
# Transition to Staging
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging",
    archive_existing_versions=True
)

print(f"\n✓ Model transitioned to Staging")

# COMMAND ----------
# MAGIC %md
# MAGIC ✅ Model Registered and in Staging!
# MAGIC
# MAGIC **View in UI:**
# MAGIC - Navigate to: Machine Learning > Models
# MAGIC - Find: income_prediction_model
# MAGIC
# MAGIC **Next:** Run notebook `06_Model_Deployment.py`
