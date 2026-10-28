# quick_digits_best.py
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import joblib

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

warnings.filterwarnings("ignore")

def make_pipeline(reducer_name):
    if reducer_name == "pca":
        reducer = PCA(random_state=42)
    else:
        reducer = TruncatedSVD(random_state=42)
        
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reduce", reducer),
        ("clf", SVC(kernel="rbf", random_state=42))
    ])

def find_best_model(X_train, y_train, X_test, y_test, reducer_name, n_jobs=12):
    print(f"\nSearching best hyperparameters for {reducer_name.upper()} using {n_jobs} cores...")
    pipe = make_pipeline(reducer_name)
    
    # Define a grid of parameters to test
    param_grid = {
        "reduce__n_components": [5, 10, 15, 20, 30, 40, 50, 60],
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": [0.001, 0.01, 0.1, 1, "scale"]
    }
    
    # GridSearchCV will test all combinations in parallel using n_jobs
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        n_jobs=n_jobs,       # <--- This uses your 12 cores
        verbose=1,
        scoring="accuracy"
    )
    
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    t1 = time.time()
    
    print(f"\nGridSearch finished in {t1-t0:.2f} seconds")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Save model
    model_path = f"best_digit_{reducer_name}_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved model: {model_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=best_model.named_steps["clf"].classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=best_model.named_steps["clf"].classes_).plot(ax=ax, cmap="Greens", values_format="d")
    plt.title(f"Best {reducer_name.upper()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"best_digit_{reducer_name}_confusion.png", dpi=150)
    plt.close()
    
    return best_model, test_acc

if __name__ == "__main__":
    print("Loading built-in sklearn digits dataset...")
    data = load_digits()
    X = data.images.reshape(len(data.images), -1)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Use 12 cores as requested
    N_JOBS = 12
    
    find_best_model(X_train, y_train, X_test, y_test, "pca", n_jobs=N_JOBS)
    find_best_model(X_train, y_train, X_test, y_test, "svd", n_jobs=N_JOBS)
    
    print("\nDone. Best models generated using parallel processing.")

