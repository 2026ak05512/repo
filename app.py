"""
Streamlit dashboard for evaluating the five saved Dry Bean classifiers.
Upload a labelled test CSV containing all 16 input features plus the 'Class'
target column. The app only loads saved pipelines and never fits/retrains them.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
TARGET = "Class"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")
st.title("🫘 Dry Bean Classification — Model Evaluation")
st.caption("BITS ML Classification Assignment | Five classifiers | Seven bean varieties")

@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded

models = load_models()
if len(models) != len(MODEL_FILES):
    missing = [f for f in MODEL_FILES.values() if not (MODEL_DIR / f).exists()]
    st.error("Saved model files are missing: " + ", ".join(missing))
    st.info("Run `python train.py` once in the project folder, then commit the generated model files.")
    st.stop()

try:
    with open(ROOT / "artifacts" / "training_summary.json", encoding="utf-8") as f:
        training_summary = json.load(f)
except FileNotFoundError:
    training_summary = {}

with st.expander("About this project", expanded=False):
    st.write(
        "The UCI Dry Bean dataset contains 13,611 records, 16 numeric shape/size "
        "features and seven bean classes. Models are trained offline by train.py. "
        "This app evaluates uploaded labelled test data; it does not retrain."
    )
    st.write("Training split:", training_summary.get("train_instances", "Run train.py first"))
    st.write("Held-out test rows:", training_summary.get("test_instances", "Run train.py first"))

st.subheader("1. Upload labelled test data")
st.write("CSV must contain the 16 feature columns and the target column named `Class`.")
uploaded = st.file_uploader("Choose test_data.csv", type=["csv"])

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

    if TARGET not in data.columns:
        st.error("The uploaded CSV must include the target column named 'Class'.")
        st.stop()

    y_true = data[TARGET].astype(str).str.strip()
    X = data.drop(columns=[TARGET])
    expected = list(next(iter(models.values())).feature_names_in_)
    missing = [c for c in expected if c not in X.columns]
    extra = [c for c in X.columns if c not in expected]
    if missing:
        st.error("Missing required feature columns: " + ", ".join(missing))
        st.stop()
    if extra:
        st.warning("Extra columns will be ignored: " + ", ".join(extra))
    X = X[expected].apply(pd.to_numeric, errors="coerce")

    if X.isna().all(axis=1).any():
        st.warning("Some rows contain only missing/non-numeric feature values. The saved pipeline imputes missing numeric values.")

    st.success(f"Loaded {len(data):,} labelled test rows.")
    model_name = st.selectbox("2. Select classifier", list(MODEL_FILES.keys()))
    model = models[model_name]

    try:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        labels = list(model.classes_)
        metrics = {
            "Accuracy": accuracy_score(y_true, predictions),
            "AUC (macro OVR)": roc_auc_score(
                y_true, probabilities, labels=labels, multi_class="ovr", average="macro"
            ),
            "Precision (macro)": precision_score(y_true, predictions, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_true, predictions, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_true, predictions, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_true, predictions),
        }
    except Exception as exc:
        st.error(f"Evaluation failed. Check target labels and feature values. Details: {exc}")
        st.stop()

    st.subheader("3. Evaluation metrics")
    cols = st.columns(3)
    for i, (metric, value) in enumerate(metrics.items()):
        cols[i % 3].metric(metric, f"{value:.4f}")

    st.subheader("4. Confusion matrix")
    cm = confusion_matrix(y_true, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted class", ylabel="Actual class",
        title=f"{model_name} — Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("5. Classification report")
    report = classification_report(y_true, predictions, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    with st.expander("Preview uploaded data"):
        st.dataframe(data.head(20), use_container_width=True)

    output = data.copy()
    output["Predicted_Class"] = predictions
    st.download_button(
        "Download predictions CSV",
        output.to_csv(index=False).encode("utf-8"),
        file_name="dry_bean_predictions.csv",
        mime="text/csv"
    )

    st.subheader("6. Compare all five saved models on this uploaded test data")
    rows = []
    for name, fitted_model in models.items():
        pred = fitted_model.predict(X)
        prob = fitted_model.predict_proba(X)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, pred),
            "AUC": roc_auc_score(y_true, prob, labels=list(fitted_model.classes_), multi_class="ovr", average="macro"),
            "Precision": precision_score(y_true, pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, pred, average="macro", zero_division=0),
            "F1": f1_score(y_true, pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_true, pred),
        })
    comparison = pd.DataFrame(rows).set_index("Model")
    st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
    st.bar_chart(comparison[["Accuracy", "AUC", "F1", "MCC"]])
else:
    st.info("Upload the generated `test_data.csv` to view metrics, confusion matrix, and the five-model comparison.")
    sample_path = ROOT / "test_data.csv"
    if sample_path.exists():
        st.download_button(
            "Download included test_data.csv",
            sample_path.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv"
        )
