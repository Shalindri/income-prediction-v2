# Databricks notebook source
# MAGIC %md
# MAGIC # Model Deployment and Testing

# COMMAND ----------
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import requests
import json

# COMMAND ----------
# Initialize client
client = MlflowClient()

# COMMAND ----------
# Get model from registry
model_name = "income_prediction_model"

# Get staging model
staging_models = client.get_latest_versions(model_name, stages=["Staging"])

if staging_models:
    model_version = staging_models[0]
    print(f"✓ Found model in Staging:")
    print(f"  Name: {model_name}")
    print(f"  Version: {model_version.version}")
else:
    print("No model found in Staging. Please run notebook 05 first.")

# COMMAND ----------
# Transition to Production
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"\n✓ Model transitioned to Production")
print(f"  Model URI: models:/{model_name}/Production")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test Model Inference

# COMMAND ----------
# Load model for testing
model_uri = f"models:/{model_name}/Production"
loaded_model = mlflow.pyfunc.load_model(model_uri)

print(f"✓ Model loaded from: {model_uri}")

# COMMAND ----------
# Load test data
X_test = pd.read_csv("/tmp/X_test.csv")
y_test = pd.read_csv("/tmp/y_test.csv").values.ravel()

print(f"✓ Test data loaded: {X_test.shape}")

# COMMAND ----------
# Make predictions on sample
sample_data = X_test.head(5)

predictions = loaded_model.predict(sample_data)

print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

for i, pred in enumerate(predictions):
    income_label = ">50K" if pred == 1 else "<=50K"
    actual_label = ">50K" if y_test[i] == 1 else "<=50K"
    match = "✓" if pred == y_test[i] else "✗"

    print(f"Sample {i+1}: Predicted={income_label}, Actual={actual_label} {match}")

print("="*70)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Model Serving Endpoint Information

# COMMAND ----------
print("\n" + "="*70)
print("DEPLOYMENT COMPLETE")
print("="*70)
print(f"\nModel: {model_name}")
print(f"Version: {model_version.version}")
print(f"Stage: Production")
print(f"URI: models:/{model_name}/Production")
print("\n" + "="*70)
print("\nNEXT STEPS:")
print("="*70)
print("\n1. Create Model Serving Endpoint:")
print("   - Go to: Machine Learning > Serving")
print("   - Click 'Create Serving Endpoint'")
print("   - Name: income-prediction-endpoint")
print(f"   - Model: {model_name}")
print("   - Version: Production (or specific version)")
print("   - Compute: Serverless Starter Warehouse")
print("\n2. Test via API:")
print("   Use the endpoint URL provided after creation")
print("\n3. Monitor:")
print("   - View requests in Serving UI")
print("   - Check logs and metrics")
print("="*70)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Sample API Request Format

# COMMAND ----------
# Example payload for API testing
sample_input = X_test.head(1).to_dict(orient='records')

example_payload = {
    "dataframe_records": sample_input
}

print("\nExample API Request Payload:")
print(json.dumps(example_payload, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ✅ DEPLOYMENT COMPLETE!
# MAGIC
# MAGIC **What was accomplished:**
# MAGIC 1. ✓ Model moved to Production stage
# MAGIC 2. ✓ Model tested with sample predictions
# MAGIC 3. ✓ Ready for serving endpoint creation
# MAGIC
# MAGIC **To create serving endpoint:**
# MAGIC - Navigate to: Machine Learning > Serving
# MAGIC - Follow the steps outlined above
# MAGIC
# MAGIC **Project Complete!**
