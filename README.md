# 🫀 Heart Attack Risk Prediction

An end-to-end machine learning pipeline that predicts a patient's heart attack risk from clinical measurements, cardiac stress-test results, and lifestyle factors — built with a leakage-safe preprocessing pipeline, five benchmarked models, and SHAP-based explainability so predictions stay clinically interpretable.

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://kaggle.com/zohairbaloch)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github&logoColor=white)](https://github.com/zohairbaloch-64)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/zohair-baloch-data-analyst)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-tuned-EB5E28)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-8A2BE2)](https://shap.readthedocs.io/)

---

## 📌 Overview

Cardiovascular disease is the leading cause of death worldwide, and early risk identification is one of the highest-leverage places data science can support clinical decision-making. This project builds a binary classifier on **7,000 patient records** across **21 clinical, demographic, and lifestyle features**, follows the **PACE framework** (Plan → Analyze → Construct → Execute), and closes with a patient-level risk-scoring function that returns not just a prediction but *why* the model made it.

This is not an exploratory scratch notebook — it's structured as a reproducible, portfolio-grade artifact: every claim in this README is backed by numbers computed inside the notebook.

---

##  Dataset

| | |
|---|---|
| **Rows** | 7,000 patients |
| **Features** | 21 (demographics, vitals, stress-test results, lifestyle, history) |
| **Target** | `heart_attack_risk` — binary (1 = at risk, 0 = not at risk) |
| **Class balance** | 58% no risk / 42% at risk (mild imbalance, no resampling required) |
| **Missing data** | 9 of 21 columns with 2.8%–7.2% missingness, handled via pipeline imputation |
| **Duplicates** | None (verified on both full rows and `patient_id`) |

**Feature groups:**
- **Demographics:** age, gender
- **Core vitals:** resting blood pressure, cholesterol, fasting blood sugar, BMI
- **Cardiac stress-test results:** chest pain type, resting ECG, max heart rate, exercise-induced angina, ST depression (`oldpeak`), ST slope, number of major vessels, thalassemia
- **Lifestyle & history:** smoking status, alcohol consumption, physical activity, family history, diabetes, stress level

---

##  Methodology (PACE Framework)

**1. Plan** — Defined the clinical problem, success criteria (prioritizing recall on the at-risk class, since a missed high-risk patient is costlier than a false alarm), and documented the feature dictionary.

**2. Analyze** — Full data-quality audit (missingness map, duplicate check) and exploratory analysis: target distribution, univariate distributions for every clinical/lifestyle variable, risk-rate breakdowns by categorical feature, and a full correlation matrix.

**3. Construct** — Built a **leakage-safe** `ColumnTransformer` + `Pipeline`:
- Numerical features → median imputation + standard scaling
- Categorical features → most-frequent imputation + one-hot encoding
- All fitting happens strictly on the training split (80/20 stratified split)

**4. Execute** — Benchmarked five algorithms, validated with 5-fold stratified cross-validation, tuned the strongest tree-based model with `RandomizedSearchCV`, and evaluated the final model with a confusion matrix, ROC curve, precision-recall curve, and SHAP explainability.

---

##  Model Comparison

Five algorithms were trained on identical preprocessing and evaluated with 5-fold stratified cross-validation:

| Model | CV ROC-AUC | CV Accuracy | CV F1 | CV Recall |
|---|---|---|---|---|
| **Logistic Regression** | **0.8719 ± 0.007** | 0.786 ± 0.009 | 0.737 ± 0.011 | 0.716 ± 0.011 |
| Gradient Boosting | 0.8587 ± 0.007 | 0.780 ± 0.009 | 0.724 ± 0.012 | 0.690 ± 0.012 |
| Random Forest | 0.8502 ± 0.004 | 0.771 ± 0.012 | 0.707 ± 0.016 | 0.659 ± 0.016 |
| XGBoost (baseline) | 0.8330 ± 0.006 | 0.759 ± 0.008 | 0.705 ± 0.010 | 0.687 ± 0.014 |
| K-Nearest Neighbors | 0.8254 ± 0.009 | 0.745 ± 0.007 | 0.648 ± 0.011 | 0.560 ± 0.020 |

**Honest finding:** a plain, untuned **Logistic Regression** was the strongest baseline on cross-validated ROC-AUC — a useful reminder that model complexity doesn't automatically win, and that a well-preprocessed linear model is a legitimate, highly interpretable choice on this dataset.

XGBoost was carried forward for tuning because tree-based models expose SHAP-based non-linear interaction effects that a linear model can't, and because tuning materially closed the gap with the baseline leader.

### Tuned XGBoost — Final Test Set Performance

`RandomizedSearchCV` (40 iterations, 5-fold CV, optimizing ROC-AUC) selected: `n_estimators=600`, `max_depth=3`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=1`.

| Metric | Score |
|---|---|
| **ROC-AUC** | **0.868** |
| Accuracy | 0.79 |
| Precision (At Risk) | 0.78 |
| Recall (At Risk) | 0.69 |
| F1-score (At Risk) | 0.73 |

*(Evaluated on a held-out 1,400-patient test set, never seen during training or tuning.)*

---

## 🔍 Explainability — SHAP

Rather than stopping at a leaderboard metric, the notebook uses **SHAP (SHapley Additive exPlanations)** to attribute each prediction to individual feature contributions — both globally (which features matter most across all patients) and locally (why a *specific* patient was flagged).

**Strongest linear correlations with risk** (directional sanity check ahead of SHAP):

| Feature | Correlation with risk |
|---|---|
| Age | +0.346 |
| Max heart rate | −0.324 |
| Resting blood pressure | +0.286 |
| Cholesterol | +0.250 |
| Number of major vessels | +0.249 |
| Diabetes | +0.218 |
| ST depression (`oldpeak`) | +0.211 |

The full SHAP summary (bar + beeswarm plots) inside the notebook confirms the model has learned clinically coherent relationships rather than spurious correlations.

### Patient-level risk scoring

A `score_patient()` function turns the model into something a clinician-facing tool could call directly — given one patient's record, it returns:
- A calibrated risk probability
- A risk band (🟢 Low / 🟡 Moderate / 🔴 High)
- The top 5 factors driving *that specific* prediction, with direction (↑ increases / ↓ decreases risk)

---

## 📁 Repository Structure

```
├── heart_attack_risk_prediction.ipynb   # Full notebook (EDA → modeling → SHAP)
├── heart_attack_dataset.csv             # Source dataset (7,000 patients)
└── README.md                            # This file
```

---

## 🧰 Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `SHAP` · `matplotlib` · `seaborn` · `Plotly`

---

## 🚀 Reproducing This Project

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly
jupyter nbconvert --to notebook --execute heart_attack_risk_prediction.ipynb
```

---

## 🔭 Next Steps

- Calibrate predicted probabilities (`CalibratedClassifierCV`) before using output for clinical thresholding
- Validate on a larger, multi-site cohort to test generalization
- Wrap `score_patient()` in a lightweight Streamlit app for interactive risk exploration

---

## 👤 Author

**Muhammad Zohair Baloch**
BS Artificial Intelligence, Khawaja Fareed University of Engineering & IT (KFUEIT)

[Kaggle](https://kaggle.com/zohairbaloch) · [GitHub](https://github.com/zohairbaloch-64) · [LinkedIn](https://linkedin.com/in/zohair-baloch-data-analyst)

If this project was useful, a ⭐ on the repo or an upvote on Kaggle is genuinely appreciated.
