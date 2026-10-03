# Predictive_Analysis_Thiranex

# Customer Churn Prediction

An end-to-end predictive analysis project that identifies bank customers who are likely to leave (churn), using exploratory data analysis, feature engineering and a comparison of five machine learning classifiers. The best model is saved as a reusable scikit-learn pipeline.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [What We Did](#what-we-did)
- [Model Results](#model-results)
- [Key Insights](#key-insights)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Using the Saved Model](#using-the-saved-model)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Project Overview

Retaining an existing customer is usually cheaper than acquiring a new one. The goal of this project is to:

1. Understand which customer characteristics are linked to churn.
2. Build a model that predicts whether a customer will churn and with what probability.
3. Package the final model so it can score new customers directly.

## Dataset

- **File:** `customer_data.csv`
- **Size:** 10,000 customers, 12 columns, no missing values
- **Target:** `churn` (1 = left, 0 = stayed). The overall churn rate is **20.37%**, so the data is imbalanced.

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier (dropped before modelling) |
| `credit_score` | Customer credit score (350 to 850) |
| `country` | France, Germany or Spain |
| `gender` | Male / Female |
| `age` | Customer age (18 to 92) |
| `tenure` | Years with the bank (0 to 10) |
| `balance` | Account balance |
| `products_number` | Number of bank products held (1 to 4) |
| `credit_card` | Has a credit card (1/0) |
| `active_member` | Active member (1/0) |
| `estimated_salary` | Estimated annual salary |
| `churn` | Target variable |

---

## What We Did

### 1. Data Understanding and Quality Check
- Loaded the data and checked shape, data types, summary statistics and missing values (none found).
- Checked the class distribution of `churn` (about 80% retained vs 20% churned).

### 2. Exploratory Data Analysis (EDA)
- KDE plots of `credit_score`, `age`, `tenure`, `balance`, `products_number` and `estimated_salary`, split by churn.
- Pairplot, violin plot (age vs churn) and tenure count plot.
- Churn ratio by gender (donut charts) and count plots for `country`, `gender`, `credit_card` and `active_member`.
- Correlation heatmap of the numeric features.
- Balance vs number of products scatter plot, and churn rate by number of products.

### 3. Feature Engineering
New features created to capture customer behaviour:

| Feature | Logic |
|---|---|
| `balance_per_product` | Balance divided by number of products |
| `salary_balance_ratio` | Estimated salary divided by balance (infinite values replaced, missing filled with the median) |
| `age_group` | Binned into `<25`, `25-34`, `35-44`, `45-54`, `55-64`, `65+` |
| `tenure_bucket` | Binned into `0`, `1-2`, `3-5`, `6-10`, `10+` years |
| `high_balance` | 1 if balance is above the 75th percentile |

### 4. Preprocessing Pipeline
Built a `ColumnTransformer` inside a scikit-learn `Pipeline` so preprocessing is always applied consistently:
- **Numeric features:** median imputation, then `StandardScaler`
- **Categorical features:** most-frequent imputation, then `OneHotEncoder` (`handle_unknown='ignore'`)

### 5. Train / Test Split
- 80/20 split with **stratification** on the target (random state 42).
- Train: 8,000 rows. Test: 2,000 rows. The churn proportion is preserved in both (about 20.4%).

### 6. Model Comparison
Five models were compared using **5-fold Stratified Cross-Validation** on the training set, scored by ROC AUC:

- Logistic Regression
- Random Forest
- Gradient Boosting
- AdaBoost
- Support Vector Classifier (SVC)

### 7. Final Model Evaluation
- The best model by mean CV ROC AUC (**Gradient Boosting**) was refit on the training data and evaluated on the held-out test set using accuracy, precision, recall, F1, ROC AUC, a classification report and a confusion matrix.
- Plotted the **top 20 feature importances**.

### 8. Model Export and Inference
- Saved the full pipeline (preprocessing + model) as `best_churn_pipeline.pkl` using `joblib`.
- Demonstrated prediction on a sample customer, returning the predicted class and churn probability.

---

## Model Results

### Cross-Validated ROC AUC (training set)

| Model | Mean AUC | Std |
|---|---|---|
| Logistic Regression | 0.7877 | 0.0244 |
| Random Forest | 0.8486 | 0.0130 |
| **Gradient Boosting** | **0.8628** | **0.0097** |
| AdaBoost | 0.8462 | 0.0133 |
| SVC | 0.8351 | 0.0104 |

### Test Set Performance (Gradient Boosting)

| Metric | Score |
|---|---|
| Accuracy | 0.8680 |
| Precision (churn) | 0.7804 |
| Recall (churn) | 0.4889 |
| F1-score (churn) | 0.6012 |
| ROC AUC | 0.8692 |

The model ranks churners well (AUC of about 0.87) and is precise when it flags a customer (78%). However, it currently catches only about half of the actual churners (49% recall).

---

## Key Insights

**Who churns**

1. **Age is the strongest driver.** It is the top feature in the model (about 33% importance). Churned customers are older on average (44.8 vs 37.4 years). Churn is around 8% for customers under 35 but jumps to roughly **50% for the 45 to 64 age bands**.
2. **Number of products is the second strongest driver (about 27% importance) and the relationship is not linear.**
   - Customers with 2 products churn the least (about 7.6%).
   - Customers with 1 product churn at about 27.7%.
   - Customers with 3 products churn at about 82.7%, and all customers with 4 products churned (60 of 60).
3. **Germany has the highest churn.** About 32.4% of German customers churned, roughly double France (16.2%) and Spain (16.7%).
4. **Inactive members churn far more.** The churn rate is 26.9% for inactive members vs 14.3% for active members.
5. **Female customers churn more.** The churn rate is 25.1% for women vs 16.5% for men.
6. **Higher balances are linked to churn.** Churned customers hold a higher average balance (about 91K vs 73K). The engineered `balance_per_product` and `balance` features also rank in the top five, which means valuable customers are the ones leaving.

**What does not matter much**

7. **Having a credit card has almost no effect** on churn (20.2% vs 20.8%), and its feature importance is close to zero.
8. **Estimated salary, credit score and tenure** show little separation between churners and non-churners and carry low importance.

**Modelling takeaways**

9. **Tree-based ensemble models clearly outperform linear and distance-based models.** Gradient Boosting beat Logistic Regression by about 7.5 AUC points, which indicates non-linear patterns (such as the products and age effects above).
10. **Feature engineering added value.** `balance_per_product` is the third most important feature, ahead of the raw balance.

## Business Recommendations

- **Target the 45 to 64 age group** with retention offers and relationship-manager outreach.
- **Review the 3+ product bundles.** Customers holding 3 or 4 products leave at very high rates, which suggests a product-fit or service problem.
- **Run a Germany-specific retention study** to understand why churn is double that of other markets.
- **Re-engage inactive members** early with campaigns, since inactivity is a clear warning signal.
- **Prioritise high-balance customers** when allocating retention budget, as their loss has the biggest financial impact.

---

## Repository Structure

```
├── customer_data.csv            # Dataset (10,000 customers)
├── predictive_analysis.ipynb    # Full analysis: EDA, feature engineering, modelling
├── best_churn_pipeline.pkl      # Saved pipeline (preprocessing + Gradient Boosting)
└── README.md
```

## How to Run

1. Clone the repository
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
   ```
3. Open and run the notebook
   ```bash
   jupyter notebook predictive_analysis.ipynb
   ```

## Using the Saved Model

The pipeline expects the engineered columns, so create them first (the same logic used in the notebook), then predict:

```python
import joblib
import pandas as pd
import numpy as np

pipeline = joblib.load("best_churn_pipeline.pkl")

customer = pd.DataFrame([{
    "credit_score": 650, "country": "France", "gender": "Male", "age": 40,
    "tenure": 3, "balance": 50000.0, "products_number": 2,
    "credit_card": 1, "active_member": 1, "estimated_salary": 60000.0
}])

# Feature engineering (same as in the notebook)
customer["balance_per_product"] = customer["balance"] / customer["products_number"].replace(0, np.nan)
customer["balance_per_product"] = customer["balance_per_product"].fillna(0)
customer["salary_balance_ratio"] = (customer["estimated_salary"] / customer["balance"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
customer["salary_balance_ratio"] = customer["salary_balance_ratio"].fillna(0.84)  # training median
customer["age_group"] = pd.cut(customer["age"], [0, 25, 35, 45, 55, 65, 100],
                               labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"])
customer["tenure_bucket"] = pd.cut(customer["tenure"], [-1, 0, 2, 5, 10, 100],
                                   labels=["0", "1-2", "3-5", "6-10", "10+"])
customer["high_balance"] = (customer["balance"] > 127644).astype(int)  # ~75th percentile of balance

print("Churn prediction:", pipeline.predict(customer)[0])
print("Churn probability:", round(pipeline.predict_proba(customer)[0, 1], 3))
```

Example output for the sample customer above: predicted class **0 (will stay)** with a churn probability of about **3%**.

## Limitations

- **Recall for churners is 48.9%**, so about half of the customers who will leave are missed at the default 0.5 threshold. Lowering the decision threshold or using class weighting / resampling would improve recall at some cost to precision.
- The cross-validation compared models with default hyperparameters; no hyperparameter tuning was applied to the final model.
- The dataset is a single snapshot, so the model captures associations and not causes.

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Joblib, Jupyter Notebook


