# Customer Churn Prediction Pipeline

A machine learning pipeline that predicts customer churn for a telecom company using the Telco Customer Churn dataset. The project benchmarks multiple classifiers, tunes the best-performing one, and surfaces the key drivers behind churn.

## Overview

This project builds an end-to-end pipeline to predict which customers are likely to churn. It:
- Cleans and preprocesses raw customer data (handles missing values, encodes categorical features, scales numeric ones)
- Benchmarks three classifiers: **Logistic Regression**, **Random Forest**, and **Gradient Boosting**
- Tunes the best model using **GridSearchCV** (5-fold cross-validation, optimized for F1-score)
- Identifies the top features driving churn
- Evaluates performance with accuracy, precision, recall, F1-score, and ROC-AUC
- Exports the final model as a reusable `.pkl` file for inference on new customers

## Dataset

Uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), containing ~7,000 customer records with demographic info, account details, and subscribed services.

## Tech Stack

- Python
- pandas, numpy — data handling
- scikit-learn — preprocessing, modeling, evaluation
- matplotlib, seaborn — visualization
- joblib — model persistence

## How to Run

1. Clone the repo:
```bash
   git clone https://github.com/jashanpreetsingh7570-gif/Churn-prediction-pipeline.git
   cd Churn-prediction-pipeline
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Open `churn_prediction_pipeline.ipynb` in Jupyter or Colab and run the cells in order.

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| Gradient Boosting (tuned) | — | — | — | — | — |

*(Fill in with your actual benchmark numbers from the notebook's output.)*

## Key Churn Drivers

The top features influencing churn, based on the tuned model's feature importances, include contract type, tenure, and monthly charges — see the notebook for the full breakdown and visualizations.

## License

This project is open source and available under the [MIT License](LICENSE).
