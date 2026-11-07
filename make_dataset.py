"""
Exports the Breast Cancer Wisconsin (Diagnostic) dataset (UCI) to data/dataset.csv.
If you choose a different Kaggle/UCI dataset, skip this file and just drop your CSV
into data/dataset.csv (last column or --target = class label).
"""
from sklearn.datasets import load_breast_cancer

bunch = load_breast_cancer(as_frame=True)
df = bunch.frame.rename(columns={"target": "diagnosis"})
# in sklearn 0 = malignant, 1 = benign -> keep as is, documented in README
df.to_csv("data/dataset.csv", index=False)
print("Saved data/dataset.csv", df.shape)
