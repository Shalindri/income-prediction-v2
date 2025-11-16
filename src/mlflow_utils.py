"""MLflow utilities"""
import mlflow

def setup_experiment(experiment_path):
    """Set up MLflow experiment"""
    mlflow.set_experiment(experiment_path)
    experiment = mlflow.get_experiment_by_name(experiment_path)
    return experiment

def get_best_run(experiment_id, metric="roc_auc"):
    """Get best run from experiment"""
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    return runs.iloc[0] if len(runs) > 0 else None
