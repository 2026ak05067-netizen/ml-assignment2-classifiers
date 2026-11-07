import json, os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from metrics_utils import compute_metrics

st.set_page_config(page_title="Tumour Diagnosis - Model Comparison", page_icon="🩺", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join("model", MODEL_FILES[name]))

with open(os.path.join("model", "meta.json")) as f:
    META = json.load(f)
TARGET, FEATURES = META["target"], META["features"]

st.title("🩺 Breast Tumour Diagnosis: Classification Model Comparison")
st.caption("Pre-trained models (no retraining happens here). Upload test data, pick a model, and inspect its performance.")

# ---------------- Sidebar ----------------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV (features + target column)", type="csv")
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))
st.sidebar.markdown(f"**Target column:** `{TARGET}`")
st.sidebar.markdown(f"**Expected features:** {len(FEATURES)}")

if uploaded is None:
    st.info("Upload a CSV from the sidebar, or use the bundled `test_data.csv`.")
    if st.sidebar.checkbox("Use bundled test_data.csv", value=False):
        data = pd.read_csv("test_data.csv")
    else:
        st.stop()
else:
    data = pd.read_csv(uploaded)

missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

X, y = data[FEATURES], data[TARGET]
model = load_model(model_name)
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

st.subheader(f"Results for: {model_name}")
st.write(f"Rows evaluated: **{len(data)}**")
with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20))

# ---------------- Metrics ----------------
m = compute_metrics(y, y_pred, y_proba, model.classes_)
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.4f}")

# ---------------- Confusion matrix + report ----------------
left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("#### Classification report")
    rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3))

# ---------------- All-model comparison ----------------
st.markdown("---")
st.subheader("All models on this uploaded data")
rows = []
for name in MODEL_FILES:
    mdl = load_model(name)
    rows.append({"ML Model Name": name,
                 **compute_metrics(y, mdl.predict(X), mdl.predict_proba(X), mdl.classes_)})
comp = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(comp.style.highlight_max(axis=0, color="#d9c8f5"))
st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
