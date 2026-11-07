"""
Trains the 5 required classifiers on ONE dataset, evaluates them on a hold-out
test split, saves each trained model (preprocessing + model in one pipeline) to model/,
and writes test_data.csv (used for the Streamlit app) + results/metrics.csv.

Usage:
    python train_models.py                       # uses data/dataset.csv, target = 'diagnosis'
    python train_models.py --data data/my.csv --target label
"""
import argparse, os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from metrics_utils import compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/dataset.csv")
parser.add_argument("--target", default="diagnosis")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

df = pd.read_csv(args.data)
X = df.drop(columns=[args.target])
y = df[args.target]
print(f"Dataset: {df.shape[0]} rows, {X.shape[1]} features, classes = {sorted(y.unique())}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.seed, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=args.seed),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=args.seed),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=args.seed),
}

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)
rows = []
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join("model", fname))

    m = compute_metrics(y_test, pipe.predict(X_test), pipe.predict_proba(X_test), pipe.classes_)
    rows.append({"ML Model Name": name, **m})
    print(name, {k: round(v, 4) for k, v in m.items()})

results = pd.DataFrame(rows)
results.to_csv("results/metrics.csv", index=False)

# test data for the Streamlit app (features + target column)
test_df = X_test.copy()
test_df[args.target] = y_test.values
test_df.to_csv("test_data.csv", index=False)

with open("model/meta.json", "w") as f:
    json.dump({"target": args.target, "features": list(X.columns),
               "classes": [int(c) if str(c).lstrip("-").isdigit() else str(c) for c in sorted(y.unique())]}, f)
print("\n", results.round(4).to_string(index=False))
