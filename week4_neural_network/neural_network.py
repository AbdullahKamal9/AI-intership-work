"""
Week 4 - Neural Networks Basics: simple neural network built from scratch with NumPy.
Dataset: sklearn digits (1,797 images of handwritten digits, 8x8 pixels, 10 classes)

Workflow:
 1. Load data, scale pixels to 0-1, one-hot encode labels
 2. Network: input(64) -> hidden(64, ReLU) -> output(10, Softmax)
 3. Forward pass -> compute loss (cross-entropy)
 4. Backward pass (backpropagation) -> compute gradients
 5. Update weights with gradient descent, repeat for many epochs
 6. Test on unseen data and show predictions
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.random.seed(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

# 1. Data
digits = load_digits()
X = digits.data / 16.0                      # pixel values are 0-16 -> scale to 0-1
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
Y_train = np.eye(10)[y_train]               # one-hot labels


class NeuralNetwork:
    def __init__(self, n_in=64, n_hidden=64, n_out=10, lr=0.1):
        self.lr = lr
        self.W1 = np.random.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_out) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, n_out))

    @staticmethod
    def softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)            # ReLU
        self.out = self.softmax(self.a1 @ self.W2 + self.b2)
        return self.out

    def loss(self, probs, Y):
        return -np.mean(np.sum(Y * np.log(probs + 1e-9), axis=1))

    def backward(self, X, Y):
        n = X.shape[0]
        dz2 = (self.out - Y) / n
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        dz1 = (dz2 @ self.W2.T) * (self.z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


def accuracy(model, X, y):
    return np.mean(model.predict(X) == y)


if __name__ == "__main__":
    net = NeuralNetwork()
    epochs, batch = 100, 32
    history = {"loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(len(X_train))
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            net.forward(X_train[b])
            net.backward(X_train[b], Y_train[b])
        loss = net.loss(net.forward(X_train), Y_train)
        tr, te = accuracy(net, X_train, y_train), accuracy(net, X_test, y_test)
        history["loss"].append(loss)
        history["train_acc"].append(tr)
        history["test_acc"].append(te)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss {loss:.4f} | train acc {tr*100:.2f}% | test acc {te*100:.2f}%")

    final_train = accuracy(net, X_train, y_train)
    final_test = accuracy(net, X_test, y_test)
    print(f"\nFinal Train Accuracy: {final_train*100:.2f}%")
    print(f"Final Test Accuracy : {final_test*100:.2f}%")

    # Predictions
    preds = net.predict(X_test)
    print("\nFirst 15 predictions:", preds[:15])
    print("First 15 actual     :", y_test[:15])

    # Plot 1: training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["loss"]); ax[0].set_title("Training Loss"); ax[0].set_xlabel("Epoch")
    ax[1].plot(history["train_acc"], label="Train")
    ax[1].plot(history["test_acc"], label="Test")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=120); plt.close()

    # Plot 2: sample predictions
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
    for i, a in enumerate(axes.ravel()):
        a.imshow(X_test[i].reshape(8, 8), cmap="gray_r")
        ok = preds[i] == y_test[i]
        a.set_title(f"P:{preds[i]} A:{y_test[i]}", color="green" if ok else "red", fontsize=9)
        a.axis("off")
    plt.suptitle("Predictions (P) vs Actual (A)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "sample_predictions.png"), dpi=120); plt.close()
    print(f"\nCharts saved in: {OUT}")
