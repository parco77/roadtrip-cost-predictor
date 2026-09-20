"""
Train and save every model the backend serves, from data/road_trip_wide.csv.

Outputs models/model.joblib containing:
  cost_regressor    - LinearRegression predicting total_trip_cost (Rs)
  band_classifier   - LogisticRegression predicting the cost-efficiency band
  traffic_classifier- LogisticRegression predicting traffic_level from departure hour + month
  scaler            - MinMaxScaler fitted on the regression features
  metrics           - honest test-set numbers, surfaced by the API and shown in the UI

Run:  python scripts/train_models.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "road_trip_wide.csv")
OUT_DIR = os.path.join(ROOT, "models")
OUT = os.path.join(OUT_DIR, "model.joblib")

SEED = 42

# The interaction term is the whole point: the real fuel bill is litres * price, and a purely
# additive linear model cannot express a product. On the original 1140-row CSV, adding it took
# test MAE from Rs 47.43 to Rs 23.42 - beating RandomForest and GradientBoosting while staying
# linear, so the Week 3 gradient-descent-from-scratch code still applies unchanged.
# vehicle_type is one-hot encoded into the REGRESSION, not just the classifier.
#
# Week 4 recovered a per-km wear rate that splits cleanly by vehicle — Hatchback 0.5694,
# Sedan 0.6943, SUV 0.8366 — a 47% spread. Leaving it out meant the model learned a single
# average rate and returned the SAME cost for all three vehicles on an identical route. The UI
# collected the field and the model ignored it. Caught by tests/test_api.py.
REG_NUMERIC = ["distance_km", "mileage", "fuel_price", "toll_cost",
               "parking_cost", "fuel_consumption_litres", "fuel_bill"]
REG_VEHICLE = ["vehicle_type_SUV", "vehicle_type_Sedan"]   # Hatchback is the baseline
REG_FEATURES = REG_NUMERIC + REG_VEHICLE

BAND_NUMERIC = ["distance_km", "mileage", "fuel_price", "toll_cost",
                "parking_cost", "fuel_consumption_litres"]
BAND_CATEGORICAL = ["vehicle_type", "fuel_type", "traffic_level"]


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
    print(f"rows: {len(df):,}   cities: "
          f"{len(set(df.start_city) | set(df.destination_city))}")

    # ---------------------------------------------------------------- regression
    rule("1. COST REGRESSION  (target: total_trip_cost)")
    for col in REG_VEHICLE:
        df[col] = (df["vehicle_type"] == col.split("_")[-1]).astype(float)
    X = df[REG_FEATURES].values.astype(float)
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

    # ablation: prove the interaction term is what is doing the work
    noint = [f for f in REG_NUMERIC if f != "fuel_bill"]
    Xn = MinMaxScaler().fit_transform(df[noint].values.astype(float))
    Xntr, Xnte, _, _ = train_test_split(Xn, y, test_size=0.2, random_state=SEED)
    base = LinearRegression().fit(Xntr, ytr).predict(Xnte)
    print(f"  ablation, no interaction term -> MAE Rs {mean_absolute_error(yte, base):.2f} "
          f"(R2 {r2_score(yte, base):.6f})")

    # ---------------------------------------------------------------- cost band
    rule("2. COST-EFFICIENCY BAND CLASSIFIER  (target: cost_band)")
    print("  Honest framing: cost_band is a quartile split of Rs/km, which the regressor above")
    print("  already predicts well, so accuracy here is high BY CONSTRUCTION. It earns its place")
    print("  because the UI needs the label, not because it is a hard learning problem.")
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

    # ---------------------------------------------------------------- traffic
    rule("3. TRAFFIC CLASSIFIER  (target: traffic_level from departure_hour + month)")
    print("  This is the genuinely non-trivial classifier. On the ORIGINAL 1140-row CSV this task")
    print("  was impossible - traffic there is independent of hour, so a classifier scored the")
    print("  51% majority baseline. The wide dataset gives traffic a real rush-hour profile.")
    Xt = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * df.departure_hour / 24),
        "hour_cos": np.cos(2 * np.pi * df.departure_hour / 24),
        "is_peak": df.departure_hour.between(8, 11) | df.departure_hour.between(17, 21),
        "month_sin": np.sin(2 * np.pi * df.month / 12),
        "month_cos": np.cos(2 * np.pi * df.month / 12),
    }).astype(float)
    traffic_columns = Xt.columns.tolist()
    yt = df["traffic_level"].astype(str).to_numpy(dtype=object)
    Xttr, Xtte, yttr, ytte = train_test_split(Xt.values, yt, test_size=0.2,
                                              random_state=SEED, stratify=yt)
    traffic = LogisticRegression(max_iter=2000).fit(Xttr, yttr)
    tpred = traffic.predict(Xtte)
    traffic_acc = float(accuracy_score(ytte, tpred))
    majority = float(pd.Series(ytte).value_counts(normalize=True).max())
    print(f"  accuracy : {traffic_acc:.4f}   majority baseline : {majority:.4f}   "
          f"lift : +{(traffic_acc - majority) * 100:.1f} pts")
    print(classification_report(ytte, tpred, digits=3))

    # ------------------------------------------------- gradient descent from scratch
    rule("4. GRADIENT DESCENT FROM SCRATCH  (the Week 3 exercise, on the wide data)")
    print("  Same loss / gradient / update code as the notebook, no sklearn, now fitted to")
    print("  25,000 trips instead of 1,129. Proves the two methods still land in the same place.")
    # lr: measured divergence threshold on this data is ~0.45 (0.5 blows up to inf), which
    # closely matches the ~0.44 ceiling the notebook found on the original CSV. 0.4 leaves margin.
    gd_w, gd_b, loss_history, stopped_at = gradient_descent(Xtr, ytr, lr=0.4, epochs=60_000)
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
        "traffic_classifier": traffic,
        "traffic_columns": traffic_columns,
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
            "band_accuracy": band_acc,
            "traffic_accuracy": traffic_acc,
            "traffic_baseline": majority,
            "n_rows": int(len(df)),
            "n_cities": int(len(set(df.start_city) | set(df.destination_city))),
            "distance_range": [float(df.distance_km.min()), float(df.distance_km.max())],
        },
    }, OUT)
    rule(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
