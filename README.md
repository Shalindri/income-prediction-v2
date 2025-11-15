# Income Prediction Project

A machine learning project that predicts whether an individual's income exceeds $50K per year based on census data from the UCI Adult dataset.

## Overview

This project implements and compares multiple machine learning models to predict income levels based on demographic and employment features. The analysis includes comprehensive exploratory data analysis (EDA), feature engineering, and model evaluation with interpretability analysis using SHAP values.

## Dataset

**Source:** UCI Adult Dataset (Census Income)
- **Size:** 32,561 individuals
- **Features:** 15 attributes including age, education, occupation, work hours, and capital gains/losses
- **Target Variable:** Binary classification (<=50K or >50K annual income)
- **Class Distribution:** Imbalanced dataset with ~76% earning <=50K

### Features

**Demographic:**
- Age, sex, race
- Native country/continent
- Marital status
- Relationship status

**Employment:**
- Workclass (private, government, self-employed)
- Occupation type
- Hours per week
- Education level

**Financial:**
- Capital gain/loss
- Final weight (fnlwgt)

## Project Structure

```
income-prediction-v2/
├── adult.csv                          # Raw dataset
├── 01_eda_and_prediction.ipynb       # Main analysis notebook
└── README.md                          # Project documentation
```

## Methodology

### 1. Data Preprocessing

**Data Cleaning:**
- Removed duplicates
- Handled missing values in categorical features (workclass, occupation, native_country)
- Standardized column names
- Converted '?' placeholders to NaN

**Outlier Detection:**
- Applied IQR method to remove outliers in capital_gain and capital_loss (non-zero values only)

**Feature Grouping:**
- Workclass → 3 categories: private-sector, self-employed, government
- Occupation → 3 categories: white_collar, blue_collar, service
- Native country → 4 continents: North America, South America, Europe, Asia
- Marital status → Binary: married vs. not married
- Relationship → Binary: lives with family vs. not

**Feature Engineering:**
- `capital_diff`: Difference between capital gain and loss
- `has_capital_gain`, `has_capital_loss`: Binary indicators
- `has_no_capital_activity`: No capital transactions
- `age_group`: Binned into 5 categories (young, early-career, mid, senior, retired)
- `work_hours_category`: Binned into 4 categories (part-time, full-time, overtime, extreme)

**Feature Selection:**
- Dropped redundant features: fnlwgt, education (kept education_num), relationship, marital_status, native_country
- Final feature count: 31 features after one-hot encoding

### 2. Exploratory Data Analysis

**Key Findings:**
- Income distribution is imbalanced (76% <=50K vs. 24% >50K)
- Age distribution is right-skewed; majority between 25-45 years
- Most individuals work ~40 hours per week
- Higher income correlates with:
  - Older age
  - Higher education level
  - Longer work hours
  - Married status
  - Capital gains

### 3. Models Implemented

Five different machine learning models were trained and evaluated:

1. **Logistic Regression** (with StandardScaler)
2. **Decision Tree Classifier**
3. **Random Forest Classifier** (300 estimators)
4. **Gradient Boosting Classifier**
5. **Neural Network (MLP)** (128-64 hidden layers)

### 4. Model Evaluation

**Metrics Used:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve

**Threshold Optimization:**
- Applied threshold tuning to maximize F1-score
- Default threshold: 0.5
- Optimized threshold selected based on precision-recall curve

## Results

### Model Performance Summary

| Model                  | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression    | 0.800    | -         | -      | 0.678    | 0.903   |
| Decision Tree          | 0.795    | -         | -      | 0.664    | 0.860   |
| Random Forest          | 0.814    | -         | -      | 0.677    | 0.891   |
| **Gradient Boosting**  | **0.863**| -         | -      | **0.687**| **0.915**|
| Neural Network (MLP)   | 0.844    | -         | -      | 0.668    | 0.907   |

**Best Model:** Gradient Boosting Classifier
- Highest accuracy: 86.3%
- Best F1-score: 0.687
- Best ROC-AUC: 0.915

### Model Interpretability

**SHAP Analysis:**
- Global feature importance using TreeExplainer
- Local explanations for individual predictions
- Summary plots (bar and beeswarm)

**Permutation Importance:**
- Model-agnostic feature importance
- Measured by decrease in ROC-AUC when feature is shuffled
- Top 10 most important features visualized

## Technologies Used

**Python Libraries:**
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, imblearn
- **Interpretability:** SHAP
- **Evaluation:** sklearn.metrics

## Installation & Usage

### Prerequisites

```bash
Python 3.7+
```

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap
```

### Running the Analysis

1. Clone the repository
2. Ensure `adult.csv` is in the project directory
3. Open and run `01_eda_and_prediction.ipynb` in Jupyter Notebook or JupyterLab

```bash
jupyter notebook 01_eda_and_prediction.ipynb
```

## Key Insights

1. **Feature Importance:**
   - Education level, marital status, and capital gain are strong predictors
   - Age and work hours also contribute significantly
   - Occupation type and workclass matter for income prediction

2. **Class Imbalance:**
   - Dataset is imbalanced (76:24 ratio)
   - Used `class_weight='balanced'` in models to address this
   - Threshold tuning improves minority class (>50K) detection

3. **Model Selection:**
   - Gradient Boosting outperforms other models
   - Neural network shows competitive performance but requires more tuning
   - Tree-based models benefit from interpretability (SHAP)

## Future Improvements

- [ ] Implement SMOTE or other oversampling techniques for better class balance
- [ ] Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- [ ] Add XGBoost, LightGBM, or CatBoost for comparison
- [ ] Cross-validation for more robust evaluation
- [ ] Feature selection using recursive feature elimination
- [ ] Deploy model as a web application using Flask/Streamlit
- [ ] Create ensemble models combining best performers

## License

This project uses the UCI Adult Dataset, which is publicly available for research purposes.

## Acknowledgments

- UCI Machine Learning Repository for the Adult dataset
- Scikit-learn documentation and community
- SHAP library for model interpretability

## Contact

For questions or suggestions, please open an issue in the repository.
