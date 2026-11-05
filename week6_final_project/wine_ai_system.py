"""
Week 6 - Final Project: Wine Intelligence System
An AI system that:
  1. Classifies a wine into one of 3 cultivars (ML models + a Neural Network)
  2. Recommends the most similar wines from the dataset (content-based recommender)
  3. Preprocesses data, evaluates accuracy, visualizes results
  4. Lets the user interact through a menu

Run:  python wine_ai_system.py            (interactive menu)
      python wine_ai_system.py --demo     (automatic demo, no typing)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
SEED = 42
KEY_FEATURES = ["alcohol", "flavanoids", "color_intensity", "proline", "total_phenols"]


class WineSystem:
    # ---------- Data preprocessing ----------
    def __init__(self):
        data = load_wine(as_frame=True)
        self.df = data.frame.copy()
        self.class_names = ["Cultivar 0", "Cultivar 1", "Cultivar 2"]
        print(f"Loaded {self.df.shape[0]} wines, {self.df.shape[1]-1} chemical features")
        print("Missing values:", int(self.df.isna().sum().sum()), "| Duplicates:", int(self.df.duplicated().sum()))
        self.X = self.df.drop(columns="target")
        self.y = self.df["target"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=SEED, stratify=self.y)
        # scaling is fitted on train data only (avoids data leakage)
        self.scaler = StandardScaler().fit(self.X_train)
        self.Xtr_s = self.scaler.transform(self.X_train)
        self.Xte_s = self.scaler.transform(self.X_test)

    # ---------- AI/ML models + Neural network ----------
    def train(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
            "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                                                  max_iter=1000, random_state=SEED),
        }
        self.results = {}
        print("\n=== Model training & accuracy ===")
        for name, m in self.models.items():
            cv = cross_val_score(m, self.Xtr_s, self.y_train, cv=5).mean()
            m.fit(self.Xtr_s, self.y_train)
            acc = accuracy_score(self.y_test, m.predict(self.Xte_s))
            self.results[name] = {"cv": cv, "test": acc}
            print(f"{name:<22} CV accuracy: {cv*100:.2f}% | Test accuracy: {acc*100:.2f}%")
        self.best = max(self.results, key=lambda k: self.results[k]["cv"])   # pick by cross-validation, not by test set
        print(f"Best model: {self.best}")
        # recommender: nearest neighbours in scaled feature space
        all_s = self.scaler.transform(self.X)
        self.nn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(all_s)

    # ---------- Result visualization ----------
    def visualize(self):
        names = list(self.results)
        plt.figure(figsize=(8, 4))
        plt.bar(names, [self.results[n]["test"] * 100 for n in names], color=["#4c72b0", "#55a868", "#c44e52"])
        plt.ylim(80, 102); plt.ylabel("Test accuracy (%)"); plt.title("Model comparison")
        for i, n in enumerate(names):
            plt.text(i, self.results[n]["test"] * 100 + 0.5, f"{self.results[n]['test']*100:.2f}%", ha="center")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "1_model_comparison.png"), dpi=120); plt.close()

        pred = self.models[self.best].predict(self.Xte_s)
        ConfusionMatrixDisplay(confusion_matrix(self.y_test, pred), display_labels=self.class_names).plot(cmap="Blues")
        plt.title(f"Confusion matrix - {self.best}")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "2_confusion_matrix.png"), dpi=120); plt.close()

        mlp = self.models["Neural Network (MLP)"]
        plt.figure(figsize=(6, 4)); plt.plot(mlp.loss_curve_)
        plt.title("Neural network training loss"); plt.xlabel("Iteration"); plt.ylabel("Loss")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "3_nn_loss_curve.png"), dpi=120); plt.close()

        pca = PCA(n_components=2).fit(self.scaler.transform(self.X))
        pts = pca.transform(self.scaler.transform(self.X))
        plt.figure(figsize=(7, 5))
        for c in range(3):
            plt.scatter(pts[self.y == c, 0], pts[self.y == c, 1], label=self.class_names[c], alpha=0.8)
        plt.legend(); plt.title("Wines in 2D (PCA)"); plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "4_pca_clusters.png"), dpi=120); plt.close()

        imp = pd.Series(self.models["Random Forest"].feature_importances_, index=self.X.columns).sort_values().tail(8)
        imp.plot(kind="barh", figsize=(7, 4), color="#4c72b0"); plt.title("Top features (Random Forest)")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "5_feature_importance.png"), dpi=120); plt.close()
        print(f"\nCharts saved in: {OUT}")

    # ---------- Prediction + recommendation ----------
    def _build_row(self, user_values):
        row = self.X.median().to_dict()
        row.update(user_values)
        return pd.DataFrame([row])[self.X.columns]

    def classify(self, user_values):
        row_s = self.scaler.transform(self._build_row(user_values))
        model = self.models[self.best]
        probs = model.predict_proba(row_s)[0]
        c = int(np.argmax(probs))
        return self.class_names[c], probs[c]

    def recommend(self, user_values, k=5):
        row_s = self.scaler.transform(self._build_row(user_values))
        dist, idx = self.nn.kneighbors(row_s, n_neighbors=k)
        rec = self.df.iloc[idx[0]][["target"] + KEY_FEATURES].copy()
        rec.insert(0, "wine_id", idx[0])
        rec["target"] = rec["target"].map(lambda t: self.class_names[t])
        rec = rec.rename(columns={"target": "cultivar"})
        rec["distance"] = np.round(dist[0], 2)
        return rec.reset_index(drop=True)

    # ---------- User interaction ----------
    def ask_profile(self):
        print("\nDescribe your ideal wine (Enter = dataset median):")
        vals = {}
        for f in KEY_FEATURES:
            lo, hi = self.X[f].min(), self.X[f].max()
            while True:
                raw = input(f"  {f} ({lo:.2f} - {hi:.2f}): ").strip()
                if raw == "":
                    vals[f] = float(self.X[f].median()); break
                try:
                    v = float(raw)
                    if lo <= v <= hi:
                        vals[f] = v; break
                    print("   Out of the dataset range, try again.")
                except ValueError:
                    print("   Please enter a number.")
        return vals

    def menu(self):
        while True:
            print("\n=========== WINE INTELLIGENCE SYSTEM ===========")
            print("1. Classify a wine (AI prediction)")
            print("2. Get wine recommendations")
            print("3. Show model accuracy report")
            print("4. Exit")
            choice = input("Choose (1-4): ").strip()
            if choice == "1":
                vals = self.ask_profile()
                label, conf = self.classify(vals)
                print(f"\n>>> Predicted: {label} (confidence {conf*100:.1f}%)")
            elif choice == "2":
                vals = self.ask_profile()
                print("\nTop 5 similar wines:\n", self.recommend(vals).to_string(index=False))
            elif choice == "3":
                self.report()
            elif choice == "4":
                print("Goodbye!"); break
            else:
                print("Invalid choice.")

    def report(self):
        print("\n=== Accuracy report ===")
        for n, r in self.results.items():
            print(f"{n:<22} CV {r['cv']*100:.2f}% | Test {r['test']*100:.2f}%")
        pred = self.models[self.best].predict(self.Xte_s)
        print(f"\nBest model: {self.best}\n")
        print(classification_report(self.y_test, pred, target_names=self.class_names))


def main():
    system = WineSystem()
    system.train()
    system.visualize()
    system.report()
    if "--demo" in sys.argv:
        profile = {"alcohol": 13.5, "flavanoids": 2.9, "color_intensity": 5.0,
                   "proline": 1000, "total_phenols": 2.6}
        label, conf = system.classify(profile)
        print(f"[DEMO] Profile: {profile}\n[DEMO] Predicted: {label} (confidence {conf*100:.1f}%)")
        print("\n[DEMO] Top 5 recommended wines:\n", system.recommend(profile).to_string(index=False))
    else:
        system.menu()


if __name__ == "__main__":
    main()
