"""
Week 3 - Machine Learning Intro: Basic ML model
Steps: load dataset -> train simple model -> make predictions -> show output
Dataset: Iris flowers (150 samples, 4 measurements, 3 flower types)
Model: Logistic Regression
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target
print("Dataset shape:", X.shape)
print("Features:", iris.feature_names)
print("Classes:", [str(c) for c in iris.target_names])

# 2. Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain samples: {len(X_train)} | Test samples: {len(X_test)}")

# 3. Train a simple model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Show output
print("\n--- Sample predictions (first 10 test rows) ---")
for i in range(10):
    print(f"Actual: {iris.target_names[y_test[i]]:<11} | "
          f"Predicted: {iris.target_names[y_pred[i]]:<11} | "
          f"{'OK' if y_test[i] == y_pred[i] else 'WRONG'}")

acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=iris.target_names))

# Predict one brand-new flower
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
pred = model.predict(new_flower)[0]
print("New flower [5.1, 3.5, 1.4, 0.2] ->", iris.target_names[pred])
