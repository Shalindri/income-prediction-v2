# Databricks notebook source
# MAGIC %md
# MAGIC # Model Comparison and Selection

# COMMAND ----------
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------
# Set experiment
experiment_path = "/Users/shalindri20@gmail.com/Adult_Income_MLflow_Production"
experiment = mlflow.get_experiment_by_name(experiment_path)

# COMMAND ----------
# Get all runs
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"]
)

print(f"✓ Found {len(runs_df)} runs")

# COMMAND ----------
# Display comparison
comparison = runs_df[['tags.mlflow.runName', 'metrics.accuracy', 'metrics.f1_score', 'metrics.roc_auc']].copy()
comparison.columns = ['Model', 'Accuracy', 'F1-Score', 'ROC-AUC']
comparison = comparison.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)

print("\n" + "="*70)
print("MODEL RANKING BY ROC-AUC")
print("="*70)
print(comparison.to_string(index=False))
print("="*70)

# COMMAND ----------
# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
for idx, metric in enumerate(metrics):
    comparison.plot(
        x='Model',
        y=metric,
        kind='bar',
        ax=axes[idx],
        legend=False
    )
    axes[idx].set_title(f'{metric} Comparison')
    axes[idx].set_ylabel(metric)
    axes[idx].set_xlabel('')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------
# Best model
best_run = runs_df.iloc[0]

print("\n" + "="*70)
print("BEST MODEL")
print("="*70)
print(f"Model: {best_run['tags.mlflow.runName']}")
print(f"Run ID: {best_run['run_id']}")
print(f"")
print(f"Metrics:")
print(f"  Accuracy:  {best_run['metrics.accuracy']:.4f}")
print(f"  Precision: {best_run['metrics.precision']:.4f}")
print(f"  Recall:    {best_run['metrics.recall']:.4f}")
print(f"  F1-Score:  {best_run['metrics.f1_score']:.4f}")
print(f"  ROC-AUC:   {best_run['metrics.roc_auc']:.4f}")
print("="*70)

# Save best run ID
with open('/tmp/best_run_id.txt', 'w') as f:
    f.write(best_run['run_id'])

print(f"\n✓ Best run ID saved: {best_run['run_id']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ✅ Model Comparison Complete!
# MAGIC
# MAGIC **Best Model Identified!**
# MAGIC
# MAGIC **Next:** Run notebook `05_Model_Registration.py`
