# COVID-19 Mortality Prediction — ML Classification Project

An end-to-end Machine Learning pipeline for predicting COVID-19 patient mortality using clinical and demographic features. Built with Python, Pandas, and scikit-learn.

> **Disclaimer:** This is an educational ML project. The models are not intended for clinical decision-making or medical diagnosis.

---

## Project Goal

Predict whether a COVID-19 patient is at risk of death based on their medical profile (age, pre-existing conditions, treatment type, etc.). The project demonstrates a complete ML workflow from raw data to final model selection.

---

## Dataset

The dataset contains **~1M patient records** from the Mexican government's COVID-19 surveillance system. Each row represents one patient with 21 features including:

| Feature | Description |
|---------|-------------|
| AGE | Patient's age |
| GENDER | Patient's gender |
| PNEUMONIA | Whether the patient had pneumonia |
| DIABETES | Whether the patient is diabetic |
| HIPERTENSION | Whether the patient has hypertension |
| OBESITY | Whether the patient is obese |
| COPD | Chronic obstructive pulmonary disease |
| TOBACCO | Whether the patient uses tobacco |
| CARDIOVASCULAR | Cardiovascular disease |
| RENAL_CHRONIC | Chronic renal disease |
| TREATMENT_TYPE | Outpatient (1) or hospitalized (2) |
| **DEATH** | **Target — 0 = survived, 1 = died** |

The dataset is **imbalanced**: approximately 7% of patients died.

---

## Preprocessing Steps

1. **Column renaming** — standardized column names for readability
2. **Target creation** — derived binary `DEATH` column from `DATE_DIED`
3. **Value encoding** — converted raw encoding (1=yes, 2=no, 97/98/99=unknown) to standard binary (1/0/NaN)
4. **Missing value handling** — dropped columns with >40% missing data; removed remaining rows with NaN
5. **Feature scaling** — applied StandardScaler for Logistic Regression (fit on training data only to prevent leakage)
6. **Class imbalance** — used `class_weight='balanced'` to give more weight to the minority class (deaths)

---

## Data Split Strategy

| Set | Size | Purpose |
|-----|------|---------|
| **Training** | 60% | Train the models |
| **Validation** | 20% | Compare models, tune threshold |
| **Test** | 20% | Final unbiased evaluation (used only once at the end) |

All splits use stratification to preserve the ~7% death rate in each set.

---

## Models Used

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Simple, interpretable linear baseline with balanced class weights |
| **Random Forest** | Ensemble of 200 decision trees with balanced class weights |
| **HistGradientBoosting** | Modern gradient boosting from scikit-learn with balanced class weights |

Logistic Regression, Random Forest, and HistGradientBoosting are configured with balanced class weights when supported by the installed scikit-learn version.

---

## Evaluation & Model Selection

### Metrics

Since this is a **medical-risk classification** problem, we focus on **recall** and **F1-score** — catching true death cases (minimizing false negatives) is more important than overall accuracy.

| Metric | What It Measures |
|--------|-----------------|
| Accuracy | Overall correctness |
| Precision | Of predicted deaths, how many actually died |
| Recall | Of actual deaths, how many were correctly predicted |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Model's ability to distinguish between classes |

### Cross-Validation

All models were validated using **5-fold StratifiedKFold cross-validation** on the training set. For Logistic Regression, a **sklearn Pipeline** wraps StandardScaler + the model to prevent data leakage between folds.

### Threshold Tuning

The default classification threshold (0.5) was compared against 0.3, 0.4, and 0.6 on the **validation set**. Lowering the threshold improves recall (catches more death cases) at the cost of some precision — an acceptable trade-off in a medical-risk context.

### Final Model Selection

The model with the highest validation ROC-AUC is selected. The best threshold (by F1-score on validation) is applied. The final evaluation is performed **once** on the held-out test set to report unbiased metrics.

---

## Results

### Validation Set — Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8931 | 0.3988 | 0.9207 | 0.5566 | 0.9531 |
| Random Forest | 0.9034 | 0.4140 | 0.7846 | 0.5420 | 0.9345 |
| **HistGradientBoosting** | 0.8835 | 0.3804 | **0.9520** | 0.5435 | **0.9599** |

### Cross-Validation (5-Fold, F1-Score)

| Model | Mean F1 | Std |
|-------|---------|-----|
| Logistic Regression | 0.5573 | 0.0021 |
| Random Forest | 0.5448 | 0.0018 |
| HistGradientBoosting | 0.5461 | 0.0006 |

### Selected Model — Final Test Results

| | |
|---|---|
| **Model** | HistGradientBoosting |
| **Threshold** | 0.6 |
| **Reason** | Highest validation ROC-AUC (0.9599); threshold chosen by best validation F1-Score |

| Metric | Test Score |
|--------|-----------|
| Accuracy | 0.8945 |
| Precision | 0.4035 |
| **Recall** | **0.9371** |
| **F1-Score** | **0.5641** |
| ROC-AUC | 0.9602 |

### Key Takeaways

- HistGradientBoosting achieved **93.7% recall** on the test set — it correctly identified the vast majority of death cases
- Cross-validation confirms stable performance across all folds (very low std)
- Top predictive features include age, pneumonia, treatment type, and diabetes
- Test results closely match validation results, confirming the model generalizes well

---

## Limitations

- This is an **educational project** — the models are not validated for clinical use
- Some columns with high missing rates (ICU, INTUBED, PREGNANT) were dropped and could carry useful signal
- No hyperparameter tuning was performed — default/simple configurations were used
- The dataset may contain biases from the original data collection process
- Real medical prediction systems require clinical validation, regulatory approval, and domain expert involvement

---

## Project Structure

```
ds project/
├── cleaning.ipynb      # Data cleaning and exploratory data analysis
├── modeling.ipynb       # ML pipeline: training, evaluation, and model selection
├── data/
│   └── Covid_Data.csv   # Raw COVID-19 dataset
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ds-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks** in order:
   - Open `cleaning.ipynb` — data cleaning and EDA
   - Open `modeling.ipynb` — ML pipeline, evaluation, and model selection

4. **Requirements:** Python 3.8+

---

## Technologies

- Python 3
- Pandas & NumPy — data manipulation
- scikit-learn — ML models, pipelines, preprocessing, evaluation, cross-validation
- Matplotlib & Seaborn — visualization
- Jupyter Notebook — interactive development

---

## Future Improvements

- Hyperparameter tuning with GridSearchCV
- Calibration curves to assess probability reliability
- Fairness analysis across demographic groups
