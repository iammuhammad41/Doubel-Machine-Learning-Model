import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from doubleml import DoubleMLData, DoubleMLPLR
import matplotlib.pyplot as plt

# data loading
data = pd.read_csv(r"D:\PythonProjects\datasets\healthcareData\physician.csv")
print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")

# specify columns
treatment_col  = "treatment"
outcome_col    = "patient_satisfaction"
covariate_cols = [
    "age",
    "years_experience",
    "num_patients",
    "hospital_rating",
    "gender",
    "avg_consult_time"
]

# sanity check
required = [outcome_col, treatment_col] + covariate_cols
data = data[required].dropna()
print(f"{data.shape[0]} rows remain after dropping missing values")

# preparing DoubleML
dml_data = DoubleMLData(
    data,
    y_col=outcome_col,
    d_cols=treatment_col,
    x_cols=covariate_cols
)

# specify learners
model_g = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
model_m = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)

# initializing model
dml_plr = DoubleMLPLR(
    data=dml_data,
    ml_g=model_g,
    ml_m=model_m,
    n_folds=5,
    score='IV-type'
)

# fitting model
dml_plr.fit()

# print ATE
ate = dml_plr.coef
print(f"Estimated ATE: {ate:.4f}")

# print SE and confidence interval
se = dml_plr.se
ci_lower, ci_upper = dml_plr.confint(level=0.95)
print(f"Standard error: {se[0]:.4f}")
print(f"95% CI: [{ci_lower[0]:.4f}, {ci_upper[0]:.4f}]")

# individual effects
effects = dml_plr.predict(data[covariate_cols])
print("First 5 individual effects:", effects[:5])

# plotting distribution of effects
plt.hist(effects, bins=30, edgecolor='k', alpha=0.7)
plt.title("Distribution of Individual Treatment Effects")
plt.xlabel("Effect on Patient Satisfaction")
plt.ylabel("Count")
plt.show()
