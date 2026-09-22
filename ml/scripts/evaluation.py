"""
Shared evaluation helpers - the Task 5 checklist, implemented once.

Task 5 asks for the same six things about every model in the project:

  1. Metrics            regression: RSS, RMSE, R2      classification: Accuracy, P, R, F1
  2. Overfit / underfit train score vs test score
  3. Cross-validation   5-fold on the TRAINING data, with the spread
  4. Comparison         one table per model family
  5. Tuning             GridSearchCV / RandomizedSearchCV on the best model
  6. Advanced models    RandomForest (bagging), AdaBoost, GradientBoosting

Rather than re-implement that per script, every model in this project is scored through the
functions here, so the numbers in the notebook, in TASK5.md and in the API all come from one
definition.

TWO DELIBERATE CHOICES
----------------------
1. Every candidate is wrapped in a `Pipeline([MinMaxScaler, model])`. Linear and logistic
   models need the scaling; trees do not care. The reason to do it for all of them is
   CROSS-VALIDATION CORRECTNESS: inside a Pipeline the scaler is re-fitted on each training
   fold, so no information from the validation fold reaches it. Scaling the whole matrix once
   before splitting - which scripts/train_models.py originally did - lets test-set minima and
   maxima leak into training. The effect is small for MinMaxScaler on this data, but it is the
   exact mistake cross-validation exists to catch, so it is fixed rather than excused.

2. The overfit / underfit verdict uses stated thresholds (below). A verdict that depends on an
   unwritten judgement call is not reproducible.
"""
import numpy as np
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42

# --- thresholds for the Task 5 step 2 verdict -------------------------------------------
# "Train score much higher than Test score -> Overfitting"
OVERFIT_GAP = 0.05        # train - test, in R2 or accuracy points
# "Both scores low -> Underfitting"
UNDERFIT_R2 = 0.50        # regression: R2 below this on BOTH splits
UNDERFIT_LIFT = 0.03      # classification: accuracy less than this above the majority baseline


def make_pipeline(model):
    """Scaler + model, so cross-validation re-fits the scaler per fold (see module docstring)."""
    return Pipeline([("scaler", MinMaxScaler()), ("model", model)])


def labels(series):
    """Class labels as a plain numpy object array.

    pandas 3 stores text columns as Arrow-backed arrays (`ArrowStringArray`), and those do not
    support the fancy indexing that scikit-learn's cross-validation performs internally -
    `y[train_index]` raises "only integer scalar arrays can be converted to a scalar index".
    Converting once, here, keeps every call site correct instead of debugging it per model.
    """
    return series.astype(str).to_numpy(dtype=object)


def candidate_regressors():
    """The regression line-up. Task 5 step 6 asks for the bottom three by name."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge (alpha=1)": Ridge(alpha=1.0, random_state=SEED),
        "DecisionTree": DecisionTreeRegressor(random_state=SEED),
        "RandomForest (300)": RandomForestRegressor(n_estimators=300, random_state=SEED,
                                                    n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
    }


def candidate_classifiers():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "RandomForest (300)": RandomForestClassifier(n_estimators=300, random_state=SEED,
                                                     n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    }


# ============================================================== regression metrics

def regression_scores(y_true, y_pred):
    """Task 5 step 1 for regression: RSS, RMSE, R2 (plus MAE, which reads in rupees)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    return {
        # RSS = sum of squared residuals. Note it grows with the number of rows, so it is
        # comparable BETWEEN models on the same split and meaningless across different splits.
        "rss": float(np.sum(residual ** 2)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def regression_verdict(train_r2, test_r2):
    """Task 5 step 2, by the stated thresholds."""
    gap = train_r2 - test_r2
    if train_r2 < UNDERFIT_R2 and test_r2 < UNDERFIT_R2:
        return "Underfitting"
    if gap > OVERFIT_GAP:
        return "Overfitting"
    return "Good fit"


def evaluate_regressor(model, X_train, y_train, X_test, y_test, cv=5, cv_scoring="r2"):
    """Fit, score both splits, run k-fold CV on the training data. Steps 1-3 in one call."""
    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)

    train = regression_scores(y_train, pipe.predict(X_train))
    test = regression_scores(y_test, pipe.predict(X_test))

    folds = KFold(n_splits=cv, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(make_pipeline(_clone_like(model)), X_train, y_train,
                                cv=folds, scoring=cv_scoring, n_jobs=-1)
    return {
        "train": train,
        "test": test,
        "verdict": regression_verdict(train["r2"], test["r2"]),
        "cv_scoring": cv_scoring,
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "cv_scores": [float(s) for s in cv_scores],
        "fitted": pipe,
    }


# ============================================================ classification metrics

def classification_scores(y_true, y_pred):
    """Task 5 step 1 for classification: Accuracy, Precision, Recall, F1.

    Macro averaging is reported as the headline because it weights every class equally - with
    an imbalanced target, weighted averaging can hide a class the model never predicts at all.
    Both are kept.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted",
                                                    zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted",
                                              zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def classification_verdict(train_acc, test_acc, baseline):
    gap = train_acc - test_acc
    if test_acc - baseline < UNDERFIT_LIFT:
        # No better than guessing the majority class - the model has learned nothing useful,
        # which is underfitting regardless of how the two splits compare.
        return "Underfitting"
    if gap > OVERFIT_GAP:
        return "Overfitting"
    return "Good fit"


def evaluate_classifier(model, X_train, y_train, X_test, y_test, cv=5):
    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)

    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)
    train = classification_scores(y_train, train_pred)
    test = classification_scores(y_test, test_pred)

    values, counts = np.unique(np.asarray(y_test), return_counts=True)
    baseline = float(counts.max() / counts.sum())

    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(make_pipeline(_clone_like(model)), X_train, y_train,
                                cv=folds, scoring="accuracy", n_jobs=-1)
    cv_f1 = cross_val_score(make_pipeline(_clone_like(model)), X_train, y_train,
                            cv=folds, scoring="f1_macro", n_jobs=-1)
    return {
        "train": train,
        "test": test,
        "baseline": baseline,
        "lift_points": (test["accuracy"] - baseline) * 100,
        "verdict": classification_verdict(train["accuracy"], test["accuracy"], baseline),
        "cv_scoring": "accuracy",
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "cv_scores": [float(s) for s in cv_scores],
        "cv_f1_mean": float(np.mean(cv_f1)),
        "cv_f1_std": float(np.std(cv_f1)),
        "classes": [str(v) for v in values],
        "fitted": pipe,
    }


def _clone_like(model):
    """A fresh unfitted copy, so the CV run never reuses the already-fitted estimator."""
    from sklearn.base import clone
    return clone(model)


# ==================================================================== presentation

def rule(title, char="="):
    print("\n" + char * 78)
    print(title)
    print(char * 78)


def print_regression_table(rows, unit="Rs"):
    """rows: list of (name, result-dict-from-evaluate_regressor)."""
    head = (f"{'model':<26}{'test R2':>10}{'RMSE':>11}{'RSS':>14}"
            f"{'train R2':>10}{'CV mean':>10}{'CV sd':>9}  verdict")
    print(head)
    print("-" * len(head))
    for name, r in rows:
        print(f"{name:<26}{r['test']['r2']:>10.6f}{r['test']['rmse']:>11.2f}"
              f"{r['test']['rss']:>14.4g}{r['train']['r2']:>10.6f}"
              f"{r['cv_mean']:>10.6f}{r['cv_std']:>9.4f}  {r['verdict']}")
    print(f"\n(RMSE in {unit}. RSS is a sum, so it is comparable within this table only.)")


def print_classification_table(rows):
    head = (f"{'model':<26}{'accuracy':>10}{'precision':>11}{'recall':>9}{'F1':>9}"
            f"{'train acc':>11}{'CV mean':>10}{'CV sd':>9}  verdict")
    print(head)
    print("-" * len(head))
    for name, r in rows:
        print(f"{name:<26}{r['test']['accuracy']:>10.4f}{r['test']['precision_macro']:>11.4f}"
              f"{r['test']['recall_macro']:>9.4f}{r['test']['f1_macro']:>9.4f}"
              f"{r['train']['accuracy']:>11.4f}{r['cv_mean']:>10.4f}{r['cv_std']:>9.4f}"
              f"  {r['verdict']}")
    if rows:
        print(f"\n(Precision / recall / F1 are macro-averaged. "
              f"Majority-class baseline = {rows[0][1]['baseline']:.4f}.)")


def strip_fitted(result):
    """JSON-safe copy - drops the fitted estimator object."""
    return {k: v for k, v in result.items() if k != "fitted"}


# How close two models must be on the held-out test error before CV gets to decide.
REG_CLOSE_PCT = 0.02      # test RMSE within 2% of the best
CLS_CLOSE_ABS = 0.005     # test accuracy within half a percentage point of the best

# Model complexity, simplest first. Used as the final tie-break: when two models cannot be
# distinguished statistically, the simpler one is preferred - it is easier to explain, faster
# to serve, and less likely to be fitting something that will not repeat. For this project it
# also matters that a linear model keeps the Week 3 gradient-descent-from-scratch code
# applicable, which no ensemble does.
COMPLEXITY = {
    "Linear, NO interaction": 0,
    "LinearRegression": 1,
    "LogisticRegression": 1,
    "Ridge (alpha=1)": 2,
    "DecisionTree": 3,
    "AdaBoost": 4,
    "GradientBoosting": 5,
    "RandomForest (300)": 6,
}


def pick_best(rows, key=None):
    """Task 5 step 4: "the model with best score AND stable cross-validation result".

    Three filters, in order:

      1. TEST-ERROR BAND. Keep models near the best held-out score. Closeness is measured on
         the ERROR, not on R2. An earlier version of this function used "R2 within 0.5%
         relative", and near R2 = 0.99 that band is wide enough to swallow a model with 15%
         more RMSE - it duly preferred a Ridge fit at 53.7 km RMSE over a random forest at
         46.6 km. Bands are therefore 2% of the best RMSE, or 0.5 accuracy points.

      2. ONE-STANDARD-ERROR RULE. Among those, keep every model whose cross-validated mean is
         within one standard error of the best cross-validated mean, where
         SE = sd(fold scores) / sqrt(k). This is the standard rule from Hastie, Tibshirani &
         Friedman, "The Elements of Statistical Learning" (section 7.10): differences smaller
         than the noise in the CV estimate itself are not real differences, so it is wrong to
         choose between those models on score at all.

      3. SIMPLICITY. Of the survivors, take the simplest by the COMPLEXITY table above.

    The result is reproducible and does not depend on which model happened to win one split.
    """
    if not rows:
        return None
    is_classifier = "accuracy" in rows[0][1]["test"]
    if key is None:
        key = ((lambda r: r["test"]["accuracy"]) if is_classifier
               else (lambda r: r["test"]["r2"]))

    # --- 1. near-best on the held-out test split -----------------------------------------
    ranked = sorted(rows, key=lambda nr: key(nr[1]), reverse=True)
    top = ranked[0][1]
    if is_classifier:
        contenders = [nr for nr in ranked
                      if top["test"]["accuracy"] - nr[1]["test"]["accuracy"] <= CLS_CLOSE_ABS]
    else:
        contenders = [nr for nr in ranked
                      if nr[1]["test"]["rmse"] <= top["test"]["rmse"] * (1 + REG_CLOSE_PCT)]

    # --- 2. statistically indistinguishable on cross-validation --------------------------
    leader = max(contenders, key=lambda nr: nr[1]["cv_mean"])[1]
    n_folds = max(1, len(leader.get("cv_scores") or [1]))
    one_se = leader["cv_std"] / np.sqrt(n_folds)
    survivors = [nr for nr in contenders if nr[1]["cv_mean"] >= leader["cv_mean"] - one_se]

    # --- 3. simplest of the survivors -----------------------------------------------------
    return min(survivors, key=lambda nr: (COMPLEXITY.get(nr[0], 99), -nr[1]["cv_mean"]))
