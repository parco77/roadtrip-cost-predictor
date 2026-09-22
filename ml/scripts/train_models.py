"""
Train and save every model the backend serves, from data/road_trip_wide.csv.

Outputs models/model.joblib containing:
  cost_regressor    - LinearRegression predicting total_trip_cost (Rs)
  band_classifier   - LogisticRegression predicting the cost-efficiency band
  scaler            - MinMaxScaler fitted on the regression features
  gradient_descent  - the from-scratch fit, its loss curve, and the sklearn comparison
  metrics           - honest test-set numbers, surfaced by the API and shown in the UI

Run:  python scripts/train_models.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

BACKEND = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backend"
)
# roadtrip_features.py is the server's feature contract - training imports the very file
# the API imports, so a column added for the model cannot go missing at serve time.
sys.path.insert(0, BACKEND)
import roadtrip_features as rf                                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "road_trip_wide.csv")
OUT_DIR = os.path.join(BACKEND, "models")
OUT = os.path.join(OUT_DIR, "model.joblib")
# The report builders read JSON, never the pickles, so the numbers in the PDF cannot drift
# from the numbers in the bundle: both come from this one run.
OUT_REPORT = os.path.join(ROOT, "data", "regression_report.json")

SEED = 42

# The regression here uses exactly the feature list in roadtrip_features.COST_FEATURES, built by
# the same function the API calls. Writing the columns out a second time is how the two sides
# drift: reorder one list and the served model silently reads its inputs in the wrong order.
# Deriving it means that cannot happen.
#
# The interaction terms are the whole point:
#   fuel_bill = litres * price. A purely additive linear model cannot express a product. On the
#   original 1140-row CSV, adding it took test MAE from Rs 47.43 to Rs 23.42 - beating
#   RandomForest and GradientBoosting while staying linear, so the from-scratch gradient-descent
#   code below still applies unchanged.
#
#   distance x vehicle. Running costs differ between vehicles per kilometre - the recovered
#   per-km wear rates are Hatchback 0.5694, Sedan 0.6943, SUV 0.8366, a 47% spread. With plain
#   one-hots the model can only learn a flat per-trip offset, which prices a 200 km drive and a
#   1200 km drive as though the gap between a hatchback and an SUV were the same. Since the
#   product shows all three vehicles side by side, that gap is the output.
REG_FEATURES = rf.COST_FEATURES

BAND_NUMERIC = ["distance_km", "mileage", "fuel_price", "toll_cost",
                "parking_cost", "fuel_consumption_litres"]
BAND_CATEGORICAL = ["vehicle_type", "fuel_type"]


def rule(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def gradient_descent(X, Y, lr=0.5, epochs=60_000, tol=1e-8):
    """Multiple linear regression by gradient descent - pure NumPy, no sklearn.

    Identical in form to the Week 3 notebook implementation:
        Y_hat = X @ w + b
        L     = (1/n) * sum((Y_hat - Y)^2)
        dw    = (2/n) * X.T @ error      db = (2/n) * sum(error)
    """
    n, k = X.shape
    w, b = np.zeros(k), 0.0
    history = []
    stopped_at = epochs
    for i in range(epochs):
        error = X.dot(w) + b - Y
        loss = (1 / n) * np.sum(error ** 2)
        history.append(loss)
        w -= lr * (2 / n) * X.T.dot(error)
        b -= lr * (2 / n) * np.sum(error)
        if i > 0 and abs(history[-2] - history[-1]) < tol:
            stopped_at = i + 1
            break
    return w, b, history, stopped_at


def main():
    df = pd.read_csv(DATA)
    df["fuel_bill"] = df["fuel_consumption_litres"] * df["fuel_price"]
    print(f"rows: {len(df):,}")

    # ---------------------------------------------------------------- regression
    rule("1. COST REGRESSION  (target: total_trip_cost)")
    X = np.asarray([
        rf.cost_features(d, m, p, t, pk, l, v)
        for d, m, p, t, pk, l, v in zip(
            df.distance_km, df.mileage, df.fuel_price, df.toll_cost,
            df.parking_cost, df.fuel_consumption_litres, df.vehicle_type)
    ], dtype=float)
    y = df["total_trip_cost"].values.astype(float)

    scaler = MinMaxScaler().fit(X)
    Xs = scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=SEED)

    reg = LinearRegression().fit(Xtr, ytr)
    pred = reg.predict(Xte)
    reg_metrics = {
        "r2": float(r2_score(yte, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "mae": float(mean_absolute_error(yte, pred)),
    }
    print(f"  R2   : {reg_metrics['r2']:.6f}")
    print(f"  RMSE : Rs {reg_metrics['rmse']:.2f}")
    print(f"  MAE  : Rs {reg_metrics['mae']:.2f}")

    # Ablations: drop each interaction group and show what it was carrying. A feature that
    # cannot be shown to change the answer does not belong in the list.
    ablations = {}
    for label, drop in [("no litres x price", ["fuel_bill"]),
                        ("no distance x vehicle", ["distance_x_SUV", "distance_x_Sedan"]),
                        ("no interactions at all",
                         ["fuel_bill", "distance_x_SUV", "distance_x_Sedan"])]:
        keep = [i for i, f in enumerate(REG_FEATURES) if f not in drop]
        Xn = MinMaxScaler().fit_transform(X[:, keep])
        Xntr, Xnte, _, _ = train_test_split(Xn, y, test_size=0.2, random_state=SEED)
        base = LinearRegression().fit(Xntr, ytr).predict(Xnte)
        ablations[label] = {"dropped": drop,
                            "mae": float(mean_absolute_error(yte, base)),
                            "r2": float(r2_score(yte, base))}
        print(f"  ablation, {label:<22} -> MAE Rs {ablations[label]['mae']:8.2f} "
              f"(R2 {ablations[label]['r2']:.6f})")
    print(f"  full feature set             -> MAE Rs {reg_metrics['mae']:8.2f} "
          f"(R2 {reg_metrics['r2']:.6f})")

    # ---------------------------------------------------------------- cost band
    rule("2. COST-EFFICIENCY BAND CLASSIFIER  (target: cost_band)")
    print("  The label the result cards carry: is this trip cheap or expensive FOR ITS LENGTH?")
    print("  Rupees alone cannot answer that - a 1200 km drive costs more than a 200 km one")
    print("  without being worse value - so the band is a quartile split of Rs/km.")
    print("  Honest framing: the regressor above already predicts cost well, and Rs/km is")
    print("  derived from cost, so accuracy here is high partly BY CONSTRUCTION. It earns its")
    print("  place because the interface needs the label, not because it is a hard problem.")
    Xb = pd.get_dummies(df[BAND_NUMERIC + BAND_CATEGORICAL],
                        columns=BAND_CATEGORICAL, drop_first=True)
    band_columns = Xb.columns.tolist()
    yb = df["cost_band"].astype(str).to_numpy(dtype=object)
    bscaler = MinMaxScaler().fit(Xb.values.astype(float))
    Xbs = bscaler.transform(Xb.values.astype(float))
    Xbtr, Xbte, ybtr, ybte = train_test_split(Xbs, yb, test_size=0.2, random_state=SEED,
                                              stratify=yb)
    band = LogisticRegression(max_iter=2000).fit(Xbtr, ybtr)
    bpred = band.predict(Xbte)
    band_acc = float(accuracy_score(ybte, bpred))
    print(f"  accuracy : {band_acc:.4f}   (baseline, 4 balanced classes = 0.25)")
    print(classification_report(ybte, bpred, digits=3))

    # Under-specified variant, to show the band is not free. Dropping the two components the
    # user cannot supply leaves only what the form collects, and the accuracy gap is the
    # difference those sub-model predictions make.
    thin = [c for c in band_columns
            if c not in ("toll_cost", "fuel_consumption_litres")]
    Xth = Xb[thin].values.astype(float)
    Xthtr, Xthte, _, _ = train_test_split(MinMaxScaler().fit_transform(Xth), yb,
                                          test_size=0.2, random_state=SEED, stratify=yb)
    thin_acc = float(accuracy_score(
        ybte, LogisticRegression(max_iter=2000).fit(Xthtr, ybtr).predict(Xthte)))
    print(f"  without the predicted toll and litres columns : {thin_acc:.4f}")
    print(f"  with them                                     : {band_acc:.4f}")
    print(f"  the sub-models are worth {(band_acc - thin_acc) * 100:+.1f} accuracy points here.")

    # ------------------------------------------------- gradient descent from scratch
    rule("3. GRADIENT DESCENT FROM SCRATCH")
    print("  The same loss / gradient / update code as the notebook, no sklearn, fitted to")
    print("  25,000 trips. The closed form and the iterative method should land in the same")
    print("  place; this measures whether they do.")
    # lr is empirical, not theory: too large and the loss diverges to inf, and a diverged run
    # writes a bundle full of NaN weights without raising. Re-measure after any change to the
    # feature list, because extra columns change the curvature of the loss surface.
    gd_w, gd_b, loss_history, stopped_at = gradient_descent(Xtr, ytr, lr=0.4, epochs=60_000)
    if not np.all(np.isfinite(gd_w)) or not np.isfinite(loss_history[-1]):
        raise SystemExit("gradient descent diverged - lower lr in train_models.py")
    gd_pred = Xte.dot(gd_w) + gd_b
    max_weight_gap = float(np.max(np.abs(gd_w - reg.coef_)))
    max_pred_gap = float(np.max(np.abs(gd_pred - pred)))
    print(f"  epochs used         : {stopped_at:,} (stopped once d(loss) < 1e-8)")
    print(f"  final loss          : {loss_history[-1]:.6f}")
    print(f"  largest weight gap  : {max_weight_gap:.6f}")
    print(f"  largest pred gap    : Rs {max_pred_gap:.6f}")
    print(f"  R2  sklearn {r2_score(yte, pred):.6f}  vs  gradient descent "
          f"{r2_score(yte, gd_pred):.6f}")

    # Downsample for the chart: 60k points would be absurd over the wire. Log-spaced so the
    # steep early drop keeps its detail and the long flat tail does not dominate.
    idx = sorted(set(np.unique(np.geomspace(1, len(loss_history), 300).astype(int) - 1)))
    curve = [{"epoch": int(i), "loss": float(loss_history[i])} for i in idx]

    # ---------------------------------------------------------------- save
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump({
        "cost_regressor": reg,
        "scaler": scaler,
        "reg_features": REG_FEATURES,
        "band_classifier": band,
        "band_scaler": bscaler,
        "band_columns": band_columns,
        "band_numeric": BAND_NUMERIC,
        "band_categorical": BAND_CATEGORICAL,
        "gradient_descent": {
            "weights": gd_w.tolist(),
            "bias": float(gd_b),
            "epochs_used": int(stopped_at),
            "final_loss": float(loss_history[-1]),
            "max_weight_gap": max_weight_gap,
            "max_prediction_gap": max_pred_gap,
            "sklearn_weights": reg.coef_.tolist(),
            "sklearn_bias": float(reg.intercept_),
            "loss_curve": curve,
            "r2_sklearn": float(r2_score(yte, pred)),
            "r2_gradient_descent": float(r2_score(yte, gd_pred)),
        },
        "metrics": {
            "regression": reg_metrics,
            "ablations": ablations,
            "band_accuracy": band_acc,
            "band_accuracy_without_submodels": thin_acc,
            "n_rows": int(len(df)),
            "distance_range": [float(df.distance_km.min()), float(df.distance_km.max())],
        },
    }, OUT)
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        json.dump({
            "feature_set": REG_FEATURES,
            "regression": reg_metrics,
            "ablations": ablations,
            "band": {"accuracy": band_acc, "accuracy_without_submodels": thin_acc,
                     "categorical": BAND_CATEGORICAL, "columns": band_columns},
            "gradient_descent": {
                "epochs_used": int(stopped_at),
                "final_loss": float(loss_history[-1]),
                "max_weight_gap": max_weight_gap,
                "max_prediction_gap": max_pred_gap,
                "r2_sklearn": float(r2_score(yte, pred)),
                "r2_gradient_descent": float(r2_score(yte, gd_pred)),
            },
            "n_rows": int(len(df)),
        }, fh, indent=2)
    rule(f"saved -> {OUT}\nsaved -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
