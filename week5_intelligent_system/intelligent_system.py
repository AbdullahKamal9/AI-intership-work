"""
Week 5 - AI Model Implementation: Breast Cancer Diagnosis Intelligent System
Covers: data preprocessing, model training, accuracy evaluation, prediction system,
user input handling, result visualization.

Run:  python intelligent_system.py            (train + interactive prediction)
      python intelligent_system.py --demo     (train + one automatic prediction, no typing)

NOTE: educational project only. Not a medical tool.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


# ---------- 1. Data + preprocessing ----------
def load_and_preprocess():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]-1} features")
    print("Missing values:", int(df.isna().sum().sum()))
    df = df.drop_duplicates()
    X, y = df.drop(columns="target"), df["target"]
    names = list(data.target_names)            # ['malignant', 'benign']
    return X, y, names


# ---------- 2. Train + compare models ----------
def train_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "SVM": make_pipeline(StandardScaler(), SVC(random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    results = {}
    for name, m in models.items():
        cv = cross_val_score(m, X_train, y_train, cv=5).mean()
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        results[name] = {"cv": cv, "test": acc}
        print(f"{name:<20} CV accuracy: {cv*100:.2f}% | Test accuracy: {acc*100:.2f}%")
    best = max(results, key=lambda k: results[k]["cv"])   # pick by cross-validation, not by test set
    return models, results, best


# ---------- 3. Visualization ----------
def make_charts(models, results, best, X, X_test, y_test, class_names):
    # model comparison
    plt.figure(figsize=(7, 4))
    names = list(results)
    plt.bar(names, [results[n]["test"] * 100 for n in names], color=["#4c72b0", "#55a868", "#c44e52"])
    plt.ylim(90, 100); plt.ylabel("Test accuracy (%)"); plt.title("Model comparison")
    for i, n in enumerate(names):
        plt.text(i, results[n]["test"] * 100 + 0.2, f"{results[n]['test']*100:.2f}%", ha="center")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "model_comparison.png"), dpi=120); plt.close()

    # confusion matrix
    cm = confusion_matrix(y_test, models[best].predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title(f"Confusion matrix - {best}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=120); plt.close()

    # top features
    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
    imp.plot(kind="barh", figsize=(7, 4), color="#4c72b0")
    plt.title("Top 10 most important features"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=120); plt.close()


# ---------- 4. Prediction system with user input ----------
def predict_one(model, X, top_features, class_names, user_values):
    row = X.median().to_dict()                 # fill features the user did not give with median
    row.update(user_values)
    row_df = pd.DataFrame([row])[X.columns]
    pred = int(model.predict(row_df)[0])
    prob = model.predict_proba(row_df)[0][pred] if hasattr(model, "predict_proba") else None
    return class_names[pred], prob


def ask_user(X, top_features):
    print("\nEnter values for the 5 most important features (press Enter to use the median):")
    vals = {}
    for f in top_features:
        lo, hi = X[f].min(), X[f].max()
        while True:
            raw = input(f"  {f} (range {lo:.2f} - {hi:.2f}): ").strip()
            if raw == "":
                vals[f] = float(X[f].median()); break
            try:
                v = float(raw)
                if lo * 0.5 <= v <= hi * 1.5:
                    vals[f] = v; break
                print("   Value looks unrealistic, try again.")
            except ValueError:
                print("   Please enter a number.")
    return vals


def main():
    demo = "--demo" in sys.argv
    X, y, class_names = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models, results, best = train_models(X_train, y_train, X_test, y_test)
    print(f"\nBest model: {best}")
    print("\nClassification report:\n",
          classification_report(y_test, models[best].predict(X_test), target_names=class_names))
    make_charts(models, results, best, X, X_test, y_test, class_names)
    print(f"Charts saved in: {OUT}")

    imp = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)
    top5 = list(imp.sort_values(ascending=False).head(5).index)
    model = models[best]

    if demo:
        sample = {f: float(X_test[f].iloc[0]) for f in top5}
        label, conf = predict_one(model, X, top5, class_names, sample)
        conf_txt = f" (confidence {conf*100:.1f}%)" if conf is not None else ""
        print(f"\n[DEMO] Input: {sample}\n[DEMO] Prediction: {label}{conf_txt}")
        return

    while True:
        vals = ask_user(X, top5)
        label, conf = predict_one(model, X, top5, class_names, vals)
        print(f"\n>>> Prediction: {label.upper()}" + (f" (confidence {conf*100:.1f}%)" if conf is not None else ""))
        print("(Educational demo only - not medical advice.)")
        if input("\nPredict another? (y/n): ").strip().lower() != "y":
            print("Goodbye!"); break


if __name__ == "__main__":
    main()
