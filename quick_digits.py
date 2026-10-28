# quick_digits.py
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import joblib

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

warnings.filterwarnings("ignore")

def make_pipeline(reducer_name, n_components):
    if reducer_name == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
        
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reduce", reducer),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", random_state=42))
    ])

def run_analysis(X, y):
    candidates = [5, 10, 20, 30, 40, 50, 60]
    max_comp = min(X.shape[0] - 1, X.shape[1] - 1)
    comps = [c for c in candidates if c <= max_comp]

    results = {"pca": [], "svd": []}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nComponent Analysis:")
    for reducer in ["pca", "svd"]:
        for c in comps:
            pipe = make_pipeline(reducer, c)
            t0 = time.time()
            pipe.fit(X_train, y_train)
            acc = pipe.score(X_test, y_test)
            t1 = time.time()
            results[reducer].append(acc)
            print(f"  {reducer.upper():3s} comps={c:02d} | Acc={acc:.4f} | Time={t1-t0:.3f}s")

    plt.figure(figsize=(8, 5))
    for reducer, accs in results.items():
        plt.plot(comps, accs, marker="o", label=reducer.upper())
    plt.xlabel("Number of Components")
    plt.ylabel("Test Accuracy")
    plt.title("Digit Recognition: PCA vs SVD")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("digit_pca_svd_components.png", dpi=150)
    plt.close()
    print("Saved: digit_pca_svd_components.png")

def train_and_evaluate(X, y, reducer, n_comp=30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = make_pipeline(reducer, n_comp)
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[{reducer.upper()}] Final Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(pipe, f"digit_{reducer}_model.pkl")
    print(f"Saved model: digit_{reducer}_model.pkl")
    
    cm = confusion_matrix(y_test, y_pred, labels=pipe.named_steps["clf"].classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=pipe.named_steps["clf"].classes_).plot(ax=ax, cmap="Greens", values_format="d")
    plt.title(f"{reducer.upper()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"digit_{reducer}_confusion.png", dpi=150)
    plt.close()
    print(f"Saved: digit_{reducer}_confusion.png")
    
    # Plot sample predictions
    n = 16
    preds = pipe.predict(X_test[:n])
    plt.figure(figsize=(8, 8))
    for i in range(n):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X_test[i].reshape(8, 8), cmap="gray")
        plt.title(f"Pred: {preds[i]}\nTrue: {y_test[i]}")
        plt.axis("off")
    plt.suptitle(f"Sample Predictions ({reducer.upper()})")
    plt.tight_layout()
    plt.savefig(f"digit_{reducer}_samples.png", dpi=150)
    plt.close()
    print(f"Saved: digit_{reducer}_samples.png")

if __name__ == "__main__":
    print("Loading built-in sklearn digits dataset...")
    data = load_digits()
    # Flatten 8x8 images into 64-dimensional feature vectors
    X = data.images.reshape(len(data.images), -1)
    y = data.target
    print(f"Dataset shape: {X.shape}, Classes: 0-9")
    
    run_analysis(X, y)
    
    train_and_evaluate(X, y, "pca", 30)
    train_and_evaluate(X, y, "svd", 30)
    
    print("\nDone. All files generated.")

