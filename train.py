"""
Train five classifiers on the UCI Dry Bean dataset.

Run:
    pip install -r requirements.txt
    python train.py

The script downloads the official UCI dataset, makes one stratified 80/20 split,
trains each model on the training split only, saves fitted preprocessing+model
pipelines, writes the held-out test CSV, and calculates all required metrics.
"""
from pathlib import Path
import json
import warnings

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
RANDOM_STATE = 42
TARGET = "Class"

def load_uci_dry_bean():
    """Fetch the official UCI Dry Bean dataset using ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit(
            "Missing ucimlrepo. Run: pip install -r requirements.txt"
        ) from exc

    bean = fetch_ucirepo(id=602)
    X = bean.data.features.copy()
    y = bean.data.targets.copy()

    # UCI provides the target as a one-column DataFrame.
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.astype(str).str.strip()
    X.columns = [str(c).strip() for c in X.columns]
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y.rename(TARGET)

def metric_row(name, y_true, y_pred, y_score, labels):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true, y_score, labels=labels, multi_class="ovr", average="macro"
        ),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def build_models():
    # Scaling is included in the pipelines for models sensitive to feature scale.
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                random_state=RANDOM_STATE, min_samples_leaf=2
            ))
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf", C=3.0, gamma="scale", probability=True,
                random_state=RANDOM_STATE
            ))
        ]),
    }

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    MODEL_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)

    X, y = load_uci_dry_bean()
    if X.shape[1] < 12 or len(X) < 500:
        raise ValueError("Dataset does not meet assignment minimums.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    # Store target as the final CSV column; this is the file users upload to Streamlit.
    test_data = X_test.copy()
    test_data[TARGET] = y_test.values
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    all_labels = sorted(y.unique().tolist())
    results = []
    models = build_models()

    for display_name, pipeline in models.items():
        print(f"Training {display_name} ...")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        row = metric_row(
            display_name, y_test, predictions, probabilities, all_labels
        )
        results.append(row)

        safe_name = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "KNN": "knn",
            "Gaussian Naive Bayes": "naive_bayes",
            "Support Vector Machine": "svm",
        }[display_name]
        joblib.dump(pipeline, MODEL_DIR / f"{safe_name}.joblib")

        report = classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        )
        with open(ARTIFACT_DIR / f"{safe_name}_classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv(ARTIFACT_DIR / "model_comparison.csv")
    with open(ARTIFACT_DIR / "model_comparison.json", "w") as f:
        json.dump(results_df.reset_index().to_dict(orient="records"), f, indent=2)

    # Generate observations from the actual held-out test scores.
    observations = {}
    for model_name, row in results_df.iterrows():
        observations[model_name] = (
            f"Accuracy {row['Accuracy']:.3f}, macro F1 {row['F1']:.3f}, "
            f"macro AUC {row['AUC']:.3f}, MCC {row['MCC']:.3f}. "
            "Interpret these values together; macro averages give each bean class equal weight."
        )
    winner = results_df["F1"].idxmax()
    summary = {
        "dataset": "UCI Dry Bean",
        "instances": int(len(X)),
        "features": int(X.shape[1]),
        "classes": all_labels,
        "train_instances": int(len(X_train)),
        "test_instances": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "auc_method": "Macro one-vs-rest multiclass ROC AUC",
        "metrics": results_df.reset_index().to_dict(orient="records"),
        "observations": observations,
        "winner_by_macro_f1": winner,
    }
    with open(ARTIFACT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Update the README's result placeholders using the real measured results.
    readme_path = ROOT / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        metric_table = results_df.reset_index().to_markdown(index=False, floatfmt=".4f")
        readme = readme.replace("<!-- METRICS_TABLE -->", metric_table)
        obs_lines = "\n".join(
            f"| {name} | {text} |" for name, text in observations.items()
        )
        readme = readme.replace("<!-- OBSERVATIONS_TABLE -->", obs_lines)
        readme = readme.replace("<!-- WINNER -->", winner)
        readme_path.write_text(readme, encoding="utf-8")

    print("\nHeld-out test metrics:")
    print(results_df.round(4))
    print(f"\nOverall winner by macro F1: {winner}")
    print("\nSaved five pipelines, test_data.csv, and metrics in artifacts/.")

if __name__ == "__main__":
    main()
