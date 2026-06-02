# COVID-19 Mortality Prediction — ML Classification Project

An end-to-end Machine Learning pipeline for predicting COVID-19 patient mortality using clinical and demographic features. Built with Python, Pandas, and scikit-learn.

> **Disclaimer:** This is an educational ML project. The models are not intended for clinical decision-making or medical diagnosis.

---

## Project Goal

Predict whether a COVID-19 patient is at risk of death based on their medical profile (age, pre-existing conditions, treatment type, etc.). The project demonstrates a complete ML workflow from raw data to model evaluation.

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

## Models Used

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Simple, interpretable linear baseline with balanced class weights |
| **Random Forest** | Ensemble of 200 decision trees with balanced class weights |

Both models use `class_weight='balanced'` to handle the ~7% death rate imbalance.

---

## Evaluation Metrics

Since this is a **medical-risk classification** problem, we focus on **recall** and **F1-score** — catching true death cases (minimizing false negatives) is more important than overall accuracy.

| Metric | What It Measures |
|--------|-----------------|
| Accuracy | Overall correctness |
| Precision | Of predicted deaths, how many actually died |
| Recall | Of actual deaths, how many were correctly predicted |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Model's ability to distinguish between classes |

---

## Key Results

- Both models achieve high ROC-AUC, demonstrating strong discrimination ability
- `class_weight='balanced'` significantly improves recall for the death class
- **Top predictive features** include age, pneumonia, treatment type, and diabetes
- Feature importance analysis confirms that age and respiratory conditions are the strongest mortality predictors

---

## Project Structure

```
ds project/
├── cleaning.ipynb      # Data cleaning and exploratory data analysis
├── modeling.ipynb       # ML pipeline: training, evaluation, and feature analysis
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
   - Open `modeling.ipynb` — ML pipeline and evaluation

4. **Requirements:** Python 3.8+

---

## Technologies

- Python 3
- Pandas & NumPy — data manipulation
- scikit-learn — ML models, preprocessing, evaluation
- Matplotlib & Seaborn — visualization
- Jupyter Notebook — interactive development

---

## Future Improvements

- Hyperparameter tuning with GridSearchCV
- Cross-validation for more robust evaluation
- Additional models (Gradient Boosting, XGBoost)
- Threshold optimization for recall vs. precision trade-off
- SHAP values for detailed feature interpretation
