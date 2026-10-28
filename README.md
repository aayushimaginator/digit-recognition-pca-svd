# Digit Recognition: PCA vs SVD

Handwritten digit classification on scikit-learn's built-in 8x8 digits dataset (1,797 samples, 64 features), comparing two dimensionality-reduction front ends.

**Pipeline:** `StandardScaler` -> `PCA` or `TruncatedSVD` -> RBF `SVC`

## Results
Best test accuracy with a stratified 80/20 split: **98.33 %** for both reducers.

| Reducer | Components | C | gamma | Test accuracy |
|---|---|---|---|---|
| PCA | 60 | 10 | 0.01 | 98.33 % |
| TruncatedSVD | 60 | 10 | 0.01 | 98.33 % |

- The tie is expected: after `StandardScaler` centers the data, TruncatedSVD is nearly equivalent to PCA.
- The best component count sits at the edge of the search grid (max 60 of 64 features), so the true optimum may be slightly higher.

Confusion matrices: [PCA](best_digit_pca_confusion.png) | [SVD](best_digit_svd_confusion.png)

## Scripts
| File | What it does |
|---|---|
| `digits.py` | 3-fold GridSearchCV over components, C and gamma for each reducer; saves the best model and confusion matrix |
| `quick_digits.py` | Sweeps component counts (5-60) at fixed C=10, plots PCA vs SVD accuracy, saves sample predictions |

Note: `quick_digits.py` compares component counts directly on test accuracy, which is mild test-set tuning. Fine for a demo; `digits.py` is the cleaner experiment.

## Run
pip install scikit-learn matplotlib joblib
python digits.py

Trained `.pkl` models are not committed; running the script regenerates them.

## Author
Aayush, EEE @ NIT Goa
