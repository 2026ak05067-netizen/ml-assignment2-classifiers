# Breast Tumour Diagnosis – Classification Model Comparison

M.Tech (AIML) · Machine Learning · Assignment 2

## a. Problem statement
Build and compare five classification models that predict whether a breast tumour is **malignant or benign** from numeric measurements of a fine-needle-aspirate image, and expose the trained models through an interactive Streamlit app. Catching malignant cases reliably matters more than raw accuracy, so recall and MCC are tracked alongside accuracy.

## b. Dataset description
- **Source:** Breast Cancer Wisconsin (Diagnostic) – UCI Machine Learning Repository
- **Instances:** 569  |  **Features:** 30 numeric (mean, standard error and "worst" value of 10 cell-nucleus properties such as radius, texture, perimeter, area, smoothness, concavity)
- **Target:** `diagnosis` (binary) – `0` = malignant (212), `1` = benign (357)
- **Missing values:** none
- **Split:** 80 % train / 20 % test (stratified, `random_state=42`); the 114-row test split is saved as `test_data.csv`
- **Preprocessing:** `StandardScaler` fitted on the training data only, bundled with each model in one scikit-learn `Pipeline` (so the app applies identical preprocessing)

## c. GitHub repository link
https://github.com/2026ak05067-netizen/ml-assignment2-classifiers.git

Live app: `<PASTE YOUR STREAMLIT URL HERE>`

```
project-folder/
│-- app.py                  # Streamlit app
│-- train_models.py         # trains + evaluates + saves the 5 models
│-- metrics_utils.py        # shared metric function
│-- make_dataset.py         # exports the dataset to CSV
│-- requirements.txt
│-- README.md
│-- test_data.csv           # hold-out test data for the app
│-- data/dataset.csv
│-- model/                  # 5 saved pipelines (.joblib) + meta.json
│-- results/metrics.csv
```

## d. Models used

### Comparison table (test set, 114 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

*(Precision, recall and F1 are for the positive class = benign.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Top performer (accuracy 98.25 %, AUC 0.995, MCC 0.962). Classes are close to linearly separable after scaling, so a simple linear boundary works very well and the model stays easy to interpret. |
| Decision Tree | Weakest model (accuracy 90.35 %, MCC 0.797). A single depth-limited tree is sensitive to the particular training split and cannot form smooth boundaries over 30 correlated features; its AUC (0.937) is also clearly lower. |
| KNN | Very strong (accuracy 97.37 %) and the only model with recall 1.0 – it missed no benign case – but precision is lower (0.96), so a few malignant tumours were labelled benign. It depends on feature scaling and `k` (7 here). |
| Naive Bayes | Good AUC (0.987) but lower accuracy (92.98 %) and MCC (0.849). The independence assumption is violated because many features (radius, perimeter, area) are highly correlated, which hurts its calibrated decisions. |
| Support Vector Machine | Ties with Logistic Regression on accuracy, F1 and MCC (0.962); AUC 0.995. The RBF kernel handles the data well, at the cost of slower training/prediction than the linear model. |
| **Overall winner** | **Logistic Regression** – best accuracy, AUC, F1 and MCC (SVM is effectively tied), while being the simplest, fastest and most interpretable. |

## How to run
```bash
pip install -r requirements.txt
python make_dataset.py        # optional: regenerates data/dataset.csv
python train_models.py        # trains models, writes model/, test_data.csv, results/metrics.csv
streamlit run app.py
```

## Streamlit app features
- CSV upload of test data (features + `diagnosis` column)
- Model selection dropdown (5 models)
- Accuracy, AUC, Precision, Recall, F1, MCC display
- Confusion matrix and classification report
- Side-by-side metrics of all models on the uploaded data
- Models are loaded from `model/`; nothing is retrained in the app
