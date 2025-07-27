import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV
from econml.dml import LinearDMLCateEstimator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# loading dataset
data = pd.read_csv(r"D:\PythonProjects\datasets\healthcareData\physician.csv")
print(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")

# columns
#   - treatment: binary indicator of whether physician received a specific intervention
#   - outcome: continuous measure (e.g. patient satisfaction score)
#   - covariates: adjustment variables
treatment_col  = "treatment"           # e.g. 0 = no extra training, 1 = received training
outcome_col    = "patient_satisfaction"  # e.g. score from 1–10
covariate_cols = [
    "age",
    "years_experience",
    "num_patients",
    "hospital_rating",
    "gender",            # 0 = male, 1 = female
    "avg_consult_time"
]

# sanity check
for col in [treatment_col, outcome_col] + covariate_cols:
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")

# Extracting arrays
Y = data[outcome_col].values
T = data[treatment_col].values
X = data[covariate_cols].values

# Data Split into train/test for DML
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y, test_size=0.3, random_state=42
)

# nuisance models
#    - outcome model (Y ~ X)
model_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
#    - treatment model (T ~ X)
model_t = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)

# Initializing DML estimator
est = LinearDMLCateEstimator(
    model_y = model_y,
    model_t = model_t,
    discrete_treatment = True,
    cv = 5
)

# Fitting the DML model
est.fit(Y = Y_train, T = T_train, X = X_train)

# Estimating treatment effects on test set
te_test = est.effect(X_test)
ate = np.mean(te_test)
print(f"\nEstimated ATE on test set: {ate:.4f}")

# Plot a distribution of individual effects
plt.hist(te_test, bins=25, edgecolor='k', alpha=0.7)
plt.title("Distribution of Individual Treatment Effects (Physician Training)")
plt.xlabel("Treatment Effect on Patient Satisfaction")
plt.ylabel("Count")
plt.show()

# feature‐by‐feature heterogeneity
feat_imp = est.feature_importances()
for name, imp in zip(covariate_cols, feat_imp):
    print(f"{name:20s}: {imp:.4f}")
