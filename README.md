# Extrovert-vs-introverts-classification
**Introvert vs. Extrovert Personality Prediction**

## 📌 Project Overview
This repository contains an **end‑to‑end machine learning pipeline** for predicting whether a person is an **Introvert** or **Extrovert** using structured/tabular survey data.  
It follows a complete competition workflow:  
- Exploratory Data Analysis (EDA)  
- Preprocessing & Feature Engineering  
- Multiple baseline models (RandomForest, LightGBM, XGBoost)  
- Hyperparameter tuning with **Optuna**  
- Ensemble learning (weighted soft‑voting)  
- Model interpretation with SHAP  
- Kaggle‑ready submission generation

The project is based on **Kaggle's Tabular Playground Series (Season 5 Episode 7)** dataset.

---

## 📂 Dataset Structure
**Source:** Kaggle competition files:
- `train.csv` — training data (18,524 rows, 9 columns)
- `test.csv` — test data for submission
- `sample_submission.csv` — submission template

**Key Columns:**
- `Time_spent_Alone`
- `Stage_fear`
- `Social_event_attendance`
- `Going_outside`
- `Drained_after_socializing`
- `Friends_circle_size`
- `Post_frequency`
- `Personality` _(target: Introvert / Extrovert)_

---

## 🔍 EDA Insights
- **Strongest predictors:**  
  `Stage_fear` (0.91), `Drained_after_socializing` (0.91), `Time_spent_Alone` (0.78 correlation with target)
- **Clear class imbalance:** ~74% Extroverts, ~26% Introverts
- **Multicollinearity:**  
  `Stage_fear` ↔ `Drained_after_socializing` correlation ~0.99
- **Non‑random missingness** detected in some features

EDA visuals include:
- KDE, boxplots, categorical countplots
- Missing value heatmaps
- Correlation heatmap

---

## 🛠 Workflow Summary

### 1. **Preprocessing**
- Dropped `id` column
- Median imputation for numeric, most frequent for categorical
- Binary encoding for Yes/No categories
- Handled imbalance with `class_weight` or `scale_pos_weight`

### 2. **Baseline Models**
- **RandomForestClassifier:** Balanced Accuracy = `0.9596`, Macro F1 = `0.9546`
- **LightGBM:** Slight improvement over RF
- **XGBoost:** Best performer; highly stable

### 3. **Hyperparameter Tuning**
- Optuna tuning for XGBoost (Balanced Accuracy target)
- Best params:  
  `n_estimators=990, max_depth=3, learning_rate=0.072, min_child_weight=2, gamma=1.066, subsample=0.911, colsample_bytree=0.535`

### 4. **Ensemble**
- Soft‑voting ensemble of Tuned XGBoost, LightGBM, and RandomForest
- **Final Validation:** Balanced Accuracy = `0.9637`, Macro F1 = `0.9620`
- Kaggle‑ready CSV saved as `ensemble_submission.csv`

---

## 📈 Feature Importance & Explainability
- **Tree feature importance** plotted for all models
- **SHAP values** from XGBoost highlight main personality indicators:
  1. `Stage_fear`
  2. `Drained_after_socializing`
  3. `Time_spent_Alone`
  4. Low social engagement metrics reduce extroversion likelihood

/yourusername/yourrepo.git
   cd yourrepo
