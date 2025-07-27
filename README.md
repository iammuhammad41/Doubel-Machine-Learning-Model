# Double Machine Learning (Double ML) on Healthcare Data

This repository demonstrates how to estimate causal treatment effects using the Double Machine Learning (DML) framework with the `double_ml.py` script. We apply DML to a physician dataset to estimate the impact of a binary treatment (e.g., extra training) on an outcome (e.g., patient satisfaction).



## 🗂 Project Structure

```
double-ml-healthcare/
├── double_ml.py          # Main script implementing DML estimation
├── README.md             # This file
└── requirements.txt      # Python dependencies
```



## 🔧 Requirements

* Python 3.7+
* pandas
* scikit-learn
* doubleml
* matplotlib

Install via pip:

```bash
pip install pandas scikit-learn doubleml matplotlib
```



## ⚙️ Configuration

1. **Dataset path**: Update the path in `double_ml.py`:

   ```python
   data = pd.read_csv(r"D:\PythonProjects\datasets\healthcareData\physician.csv")
   ```
2. **Columns**: Ensure your CSV has:

   * `treatment` (binary indicator)
   * `patient_satisfaction` (continuous outcome)
   * Covariates: `age`, `years_experience`, `num_patients`, `hospital_rating`, `gender`, `avg_consult_time`



## ▶️ Usage

Run the DML script:

```bash
python double_ml.py
python double_ml_healthcare.py
```



## 🛠️ Customization

* **Learners**: Replace `RandomForestRegressor`/`Classifier` with other scikit‑learn estimators (e.g., `LassoCV`, `GradientBoosting*`).
* **Cross‑fitting folds**: Change `n_folds` in the `DoubleMLPLR` constructor.
* **Score type**: Switch `score='IV-type'` for binary treatment or use `'partialling out'` for continuous treatment.
* **Covariates**: Add or remove columns in the `covariate_cols` list to adjust confounders.

