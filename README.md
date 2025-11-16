# Adult Income Prediction with MLflow

Production ML pipeline for income prediction with comprehensive MLflow tracking.

## Project Overview

- **Dataset:** UCI Adult Income (32,561 records)
- **Task:** Binary classification (<=50K or >50K)
- **Models:** 5 models compared
- **Deployment:** MLflow Model Serving on Databricks

## Quick Start

### 1. Upload to Databricks

1. Push this repo to GitHub
2. Connect GitHub to Databricks
3. Clone repo to Databricks

### 2. Run Notebooks in Order

```
00_Project_Setup.py                 (3 min)
01_Data_Loading.py                  (10 min)
02_Feature_Engineering.py           (15 min)
03_Model_Training.py                (60 min)  <- Main notebook
04_Model_Comparison.py              (15 min)
05_Model_Registration.py            (10 min)
06_Model_Deployment.py              (20 min)
```

### 3. Access Deployed Model

Endpoint: `income-prediction-endpoint`

```python
import requests

response = requests.post(
    'https://dbc-abd0a646-e6a5.cloud.databricks.com/serving-endpoints/income-prediction-endpoint/invocations',
    headers={'Authorization': f'Bearer {token}'},
    json={'dataframe_records': [{...}]}
)
```

## Results

| Model | ROC-AUC | Accuracy | F1-Score |
|-------|---------|----------|----------|
| Gradient Boosting | 0.915 | 0.863 | 0.687 |
| Neural Network | 0.907 | 0.844 | 0.668 |
| Logistic Regression | 0.903 | 0.800 | 0.678 |
| Random Forest | 0.891 | 0.814 | 0.677 |
| Decision Tree | 0.860 | 0.795 | 0.664 |

## Technologies

- MLflow (experiment tracking, model registry, serving)
- Databricks (platform)
- scikit-learn (ML models)
- Python 3.9+

## Project Structure

```
Adult_Income_MLflow_Project/
├── notebooks/
│   ├── 00_Project_Setup.py
│   ├── 01_Data_Loading.py
│   ├── 02_Feature_Engineering.py
│   ├── 03_Model_Training.py
│   ├── 04_Model_Comparison.py
│   ├── 05_Model_Registration.py
│   └── 06_Model_Deployment.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── model_utils.py
│   └── mlflow_utils.py
│
├── README.md
└── requirements.txt
```

## Models Trained

1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Tree-based classifier
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Advanced boosting algorithm
5. **Neural Network (MLP)** - Multi-layer perceptron

## MLflow Features Used

- Experiment Tracking
- Parameter Logging
- Metric Logging
- Artifact Logging (confusion matrices, ROC curves)
- Model Registry
- Model Versioning
- Model Staging (Staging -> Production)
- Model Serving

## Contact

shalindri20@gmail.com

## License

MIT License
