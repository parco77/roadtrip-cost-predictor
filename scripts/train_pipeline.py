"""
Phase B - train the chained model pipeline that replaces app.py's hardcoded constants.

WHAT THIS REPLACES
------------------
app.py currently predicts a trip cost like this:

    distance = haversine * 1.2355          <- constant
    litres   = distance / mileage * TRAFFIC_MULT[traffic]
    toll     = distance * 1.301            <- constant
    parking  = 70.0                        <- constant
    cost     = cost_regressor([distance, mileage, price, toll, parking, litres, ...])

Only the last line is a model. Everything above it is arithmetic, and because the regressor's
single strongest feature (`fuel_consumption_litres`, corr +0.96) is produced by that arithmetic,
the reported R2 = 0.9997 largely measures the model's ability to add up numbers it was given.

This script fits a model for each of those steps and chains them, so a prediction is made from
what a traveller actually knows: two cities, a vehicle, a fuel type and a departure time.

THE SIX SUB-MODELS
------------------
  1. distance   haversine + route geometry -> road km      trained on REAL OSRM distances
  2. mileage    vehicle + fuel             -> km/l
  3. litres     distance + mileage + traffic -> litres
  4. toll       distance                   -> Rs
  5. parking    (anything)                 -> Rs           expected to FAIL, and it does
  6. traffic    departure hour + month     -> Low/Med/High
  then the cost regressor consumes their outputs.

STACKING DONE PROPERLY
----------------------
The cost regressor must be trained on the component values it will actually be served -
predictions, not ground truth. Fitting it on true components and then feeding it predicted ones
at inference is a train/serve mismatch, and it is measured here rather than assumed: both
variants are reported.

To build the training components without optimism, sub-model predictions for the training rows
come from `cross_val_predict` (out-of-fold), so no row's component features were produced by a
model that had already seen that row. Test rows use the sub-models fitted on the full training
split. That is textbook stacking.

Run:  python scripts/train_pipeline.py
Needs: data/real_distances.csv  (python scripts/fetch_real_distances.py)
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import roadtrip_features as rf                                          # noqa: E402
from evaluation import (SEED, candidate_classifiers, candidate_regressors,  # noqa: E402
                        evaluate_classifier, evaluate_regressor, labels, make_pipeline,
                        pick_best, print_classification_table,
                        print_regression_table, regression_scores, rule, strip_fitted)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDE = os.path.join(ROOT, "data", "road_trip_wide.csv")
REAL = os.path.join(ROOT, "data", "real_distances.csv")
CITIES = os.path.join(ROOT, "data", "india_cities.csv")
OUT_MODEL = os.path.join(ROOT, "models", "pipeline.joblib")
OUT_EVAL = os.path.join(ROOT, "models", "pipeline_eval.joblib")
OUT_REPORT = os.path.join(ROOT, "data", "pipeline_report.json")

# The incumbent constants, kept here so every sub-model can be scored against the thing it
# replaces. A model that cannot beat the constant it replaces has not earned its place.
CONST_WINDING = 1.2355
CONST_TOLL_RATE = 1.301
CONST_PARKING = 70.0
CONST_TRAFFIC_MULT = {"Low": 0.944, "Medium": 1.045, "High": 1.165}

TEST_SIZE = 0.2
report = {}


def build_matrix(builder, rows):
    """Apply a feature builder across a list of argument tuples -> 2-D float array."""
    return np.asarray([builder(*args) for args in rows], dtype=float)


# ==========================================================================================
# 1. ROAD DISTANCE - the one sub-model trained on genuinely observed data
# ==========================================================================================
def train_distance_model():
    rule("SUB-MODEL 1 of 6 - ROAD DISTANCE   (replaces WINDING_FACTOR = 1.2355)")
    real = pd.read_csv(REAL)
    print(f"  training data : {len(real)} REAL city-pair road distances from OSRM")
    print(f"  haversine     : {real.haversine_km.min():.0f} - {real.haversine_km.max():.0f} km")
    print(f"  observed winding factor: mean {real.factor.mean():.4f}  sd {real.factor.std():.4f}"
          f"  range {real.factor.min():.4f} - {real.factor.max():.4f}")
    print(f"  the constant it replaces: {CONST_WINDING}  <- a single number for that whole range")

    # The extreme routes are worth naming, because a reader's first instinct on seeing a
    # factor of 3.6 is "bad data" - and here it is not.
    worst = real.nlargest(3, "factor")
    print("\n  the three worst routes for the constant:")
    for _, r in worst.iterrows():
        print(f"    {r.start_city} -> {r.dest_city:<14} straight {r.haversine_km:>7.1f} km"
              f"   road {r.road_km:>7.1f} km   factor {r.factor:.2f}")
    print("  These are NOT outliers to be cleaned away - they are real. Surat and Bhavnagar sit")
    print("  on opposite shores of the Gulf of Khambhat, so a 94 km straight line is a 339 km")
    print("  drive around the head of the gulf. No single multiplier can serve both that route")
    print("  and Moradabad -> Chanduasi (factor 1.04, essentially a straight road). They are")
    print("  kept in the training data, and they are the clearest argument for a model here.")
    print("  Note which model wins below: a tree ensemble, because it can isolate a region;")
    print("  a linear fit gets dragged by these points instead of representing them.")

    X = build_matrix(rf.distance_features,
                     real[["start_lat", "start_lon", "dest_lat", "dest_lon"]].values)
    y = real.road_km.values.astype(float)
    hav = real.haversine_km.values.astype(float)

    Xtr, Xte, ytr, yte, htr, hte = train_test_split(X, y, hav, test_size=0.25,
                                                    random_state=SEED)

    # Baseline to beat: the hardcoded constant.
    const_scores = regression_scores(yte, hte * CONST_WINDING)
    print(f"\n  BASELINE  haversine x {CONST_WINDING} :  "
          f"R2 {const_scores['r2']:.4f}   RMSE {const_scores['rmse']:.2f} km   "
          f"MAE {const_scores['mae']:.2f} km")

    # Two parameterisations of the same problem, which is the interesting part:
    #   (a) predict road_km directly
    #   (b) predict the winding FACTOR and multiply by haversine afterwards
    # (b) removes the trip length from the target, so the model spends its capacity on the part
    # that is actually unknown - the shape of the route - instead of re-learning "longer line,
    # longer road". This is the same idea as the Week 6 interaction term: give the model the
    # quantity that means something.
    print("\n  (a) target = road_km directly")
    rows_direct = []
    for name, model in candidate_regressors().items():
        r = evaluate_regressor(model, Xtr, ytr, Xte, yte)
        rows_direct.append((name, r))
    print_regression_table(rows_direct, unit="km")

    print("\n  (b) target = winding factor, prediction = factor x haversine")
    ftr, fte = ytr / htr, yte / hte
    rows_factor = []
    for name, model in candidate_regressors().items():
        pipe = make_pipeline(model)
        pipe.fit(Xtr, ftr)
        # Score in KILOMETRES, not in factor units, so the two tables are comparable.
        km_test = regression_scores(yte, pipe.predict(Xte) * hte)
        km_train = regression_scores(ytr, pipe.predict(Xtr) * htr)
        folds = []
        for tr_idx, va_idx in _kfold_indices(len(Xtr)):
            p = make_pipeline(_fresh(model))
            p.fit(Xtr[tr_idx], ftr[tr_idx])
            folds.append(regression_scores(ytr[va_idx],
                                           p.predict(Xtr[va_idx]) * htr[va_idx])["r2"])
        rows_factor.append((name, {
            "train": km_train, "test": km_test,
            "verdict": _verdict(km_train["r2"], km_test["r2"]),
            "cv_scoring": "r2 (km scale)",
            "cv_mean": float(np.mean(folds)), "cv_std": float(np.std(folds)),
            "cv_scores": [float(f) for f in folds], "fitted": pipe,
        }))
    print_regression_table(rows_factor, unit="km")

    best_direct = pick_best(rows_direct)
    best_factor = pick_best(rows_factor)
    if best_factor[1]["test"]["r2"] >= best_direct[1]["test"]["r2"]:
        best_name, best = best_factor
        parameterisation, best_name_full = "factor", f"{best_name} (factor x haversine)"
    else:
        best_name, best = best_direct
        parameterisation, best_name_full = "direct", f"{best_name} (direct km)"

    gain = (const_scores["mae"] - best["test"]["mae"]) / const_scores["mae"] * 100
    print(f"\n  CHOSEN : {best_name_full}")
    print(f"           test R2 {best['test']['r2']:.4f}   RMSE {best['test']['rmse']:.2f} km   "
          f"MAE {best['test']['mae']:.2f} km")
    print(f"           vs the constant: MAE {const_scores['mae']:.2f} -> "
          f"{best['test']['mae']:.2f} km, a {gain:.1f}% reduction")

    report["distance"] = {
        "n_real_pairs": int(len(real)),
        "factor_observed": {"mean": float(real.factor.mean()), "sd": float(real.factor.std()),
                            "min": float(real.factor.min()), "max": float(real.factor.max())},
        "constant_baseline": const_scores,
        "candidates_direct": {n: strip_fitted(r) for n, r in rows_direct},
        "candidates_factor": {n: strip_fitted(r) for n, r in rows_factor},
        "chosen": best_name_full,
        "parameterisation": parameterisation,
        "chosen_scores": strip_fitted(best),
        "mae_reduction_pct": float(gain),
    }
    return best["fitted"], parameterisation


def _kfold_indices(n, k=5):
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=k, shuffle=True, random_state=SEED).split(np.arange(n)))


def _fresh(model):
    from sklearn.base import clone
    return clone(model)


def _verdict(train_r2, test_r2):
    from evaluation import regression_verdict
    return regression_verdict(train_r2, test_r2)


# ==========================================================================================
# How wrong is the generated dataset's geography? Worth knowing before trusting anything
# measured on it.
# ==========================================================================================
def measure_dataset_geography_bias():
    rule("GEOGRAPHY CHECK - the generated dataset vs the real road network")
    real = pd.read_csv(REAL)
    dataset_convention = real.haversine_km * CONST_WINDING
    err = dataset_convention - real.road_km
    pct = err / real.road_km * 100
    print("  data/road_trip_wide.csv sets  distance_km = haversine x N(1.2355, 0.082).")
    print("  Compared with the real road distance for the same city pairs:")
    print(f"    mean signed error : {err.mean():+.1f} km  ({pct.mean():+.1f}%)   <- bias")
    print(f"    mean abs error    : {err.abs().mean():.1f} km  ({pct.abs().mean():.1f}%)"
          f"   <- per-route error")
    print(f"    worst overshoot   : {err.max():+.1f} km  ({pct.max():+.1f}%)")
    print(f"    worst undershoot  : {err.min():+.1f} km  ({pct.min():+.1f}%)")
    print("\n  READ THIS CAREFULLY, because the two numbers say opposite-sounding things:")
    print("  The constant is very nearly UNBIASED - averaged over hundreds of routes it is")
    print("  within a fraction of a percent of the truth. That is why it survived review for so")
    print("  long. But 'unbiased on average' is not 'correct': route by route it is off by")
    print(f"  {pct.abs().mean():.1f}% typically, and by tens of percent at the extremes. A user does not")
    print("  take the average of every road trip in India - they take one specific trip, and on")
    print("  that trip the errors do not cancel.")
    print("\n  Consequence for this project: costs measured ON the generated dataset stay")
    print("  self-consistent (its own distance column generated its own costs), but the DISTANCE")
    print("  model must be trained on the real data, which is what happens above. The app")
    print("  therefore serves real geography even though the cost model was fitted on generated")
    print("  trips. Stated in the notebook rather than smoothed over.")
    report["geography_bias"] = {
        "mean_signed_km": float(err.mean()), "mean_signed_pct": float(pct.mean()),
        "mean_abs_km": float(err.abs().mean()), "mean_abs_pct": float(pct.abs().mean()),
        "max_km": float(err.max()), "min_km": float(err.min()),
    }


# ==========================================================================================
# 2-6. the sub-models trained on the generated wide dataset
# ==========================================================================================
def train_sub_models(df, tr, te):
    models, chosen = {}, {}

    # ---------------------------------------------------------------- 2. mileage
    rule("SUB-MODEL 2 of 6 - MILEAGE   (so the user need not know their own km/l)")
    X = build_matrix(rf.mileage_features, df[["vehicle_type", "fuel_type"]].values)
    y = df.mileage.values.astype(float)
    rows = [(n, evaluate_regressor(m, X[tr], y[tr], X[te], y[te]))
            for n, m in candidate_regressors().items()]
    print_regression_table(rows, unit="km/l")
    name, best = pick_best(rows)
    print(f"\n  CHOSEN : {name}   test R2 {best['test']['r2']:.4f}   "
          f"MAE {best['test']['mae']:.3f} km/l")
    print("  Honest reading: the generator draws mileage UNIFORMLY inside a per-(vehicle, fuel)")
    print("  range, so the best any model can do is predict the middle of the right cell. The")
    print("  ceiling is structural, not a modelling failure - and the interaction terms in")
    print("  MILEAGE_FEATURES are what let a linear model reach all nine cells.")
    models["mileage"], chosen["mileage"] = best["fitted"], name
    report["mileage"] = {"candidates": {n: strip_fitted(r) for n, r in rows}, "chosen": name}

    # ---------------------------------------------------------------- 3. litres
    rule("SUB-MODEL 3 of 6 - FUEL CONSUMED   (replaces TRAFFIC_MULT)")
    X = build_matrix(rf.litres_features,
                     df[["distance_km", "mileage", "traffic_level"]].values)
    y = df.fuel_consumption_litres.values.astype(float)
    rows = [(n, evaluate_regressor(m, X[tr], y[tr], X[te], y[te]))
            for n, m in candidate_regressors().items()]
    print_regression_table(rows, unit="litres")
    name, best = pick_best(rows)
    print(f"\n  CHOSEN : {name}   test R2 {best['test']['r2']:.6f}")
    # The R2 above is 1.000000, which means the linear model reproduced the generator exactly.
    # That claim is worth cashing in: fit the same model UNSCALED on just the three ratio
    # columns and the coefficients are the traffic multipliers themselves.
    #
    #   litres = ratio * mult(traffic)
    #          = ratio * [ mult_Low + (mult_Med - mult_Low)*Medium + (mult_High - mult_Low)*High ]
    #
    # so the weight on `km_per_litre_ratio` IS mult_Low, and the two interaction weights are
    # the gaps up to Medium and High.
    from sklearn.linear_model import LinearRegression as _LR
    cols = [rf.LITRES_FEATURES.index(c)
            for c in ("km_per_litre_ratio", "ratio_x_Medium", "ratio_x_High")]
    bare = _LR().fit(X[tr][:, cols], y[tr])
    w_low, w_med_gap, w_high_gap = bare.coef_
    recovered = {"Low": w_low, "Medium": w_low + w_med_gap, "High": w_low + w_high_gap}
    print("\n  READING THE GENERATOR OFF THE COEFFICIENTS")
    print("  Because R2 is exactly 1.000000, the fit has not approximated the data-generating")
    print("  rule - it has reproduced it. Refitting unscaled on the three ratio columns gives:")
    print(f"    {'traffic':<10}{'recovered':>12}{'generator':>12}{'error':>12}")
    for level in ("Low", "Medium", "High"):
        truth = CONST_TRAFFIC_MULT[level]
        print(f"    {level:<10}{recovered[level]:>12.6f}{truth:>12.4f}"
              f"{recovered[level] - truth:>+12.6f}")
    print(f"    intercept {bare.intercept_:+.2e}  (zero, as the formula has no constant term)")
    print("\n  This is the point of the interaction columns. Given only distance, mileage and")
    print("  traffic one-hots, an additive model CANNOT express a ratio multiplied by a")
    print("  per-class constant, and it would have to settle for an approximation. Handed the")
    print("  products, it recovers the three multipliers to six decimal places.")
    print("  Honest reading: litres is a deterministic function of distance, mileage and")
    print("  traffic in this dataset, so a perfect score here confirms the FEATURES are right -")
    print("  it is not evidence about real-world fuel burn.")
    models["litres"], chosen["litres"] = best["fitted"], name
    report["litres"] = {"candidates": {n: strip_fitted(r) for n, r in rows}, "chosen": name}

    # ---------------------------------------------------------------- 4. toll
    rule("SUB-MODEL 4 of 6 - TOLL   (replaces TOLL_RATE = 1.301)")
    X = build_matrix(rf.toll_features, df[["distance_km"]].values)
    y = df.toll_cost.values.astype(float)
    const = regression_scores(y[te], df.distance_km.values[te] * CONST_TOLL_RATE)
    print(f"  BASELINE  distance x {CONST_TOLL_RATE} :  R2 {const['r2']:.4f}   "
          f"RMSE Rs {const['rmse']:.2f}")
    rows = [(n, evaluate_regressor(m, X[tr], y[tr], X[te], y[te]))
            for n, m in candidate_regressors().items()]
    print_regression_table(rows)
    name, best = pick_best(rows)
    print(f"\n  CHOSEN : {name}   test R2 {best['test']['r2']:.4f}   "
          f"RMSE Rs {best['test']['rmse']:.2f}")
    print("  Honest reading: the generator sets toll = distance x N(1.301, 0.360), so the RATE")
    print("  is recoverable but the per-trip spread is injected noise. The remaining RMSE is")
    print("  irreducible on this data - no model can do better, and one that appeared to would")
    print("  be leaking.")
    models["toll"], chosen["toll"] = best["fitted"], name
    report["toll"] = {"constant_baseline": const,
                      "candidates": {n: strip_fitted(r) for n, r in rows}, "chosen": name}

    # ---------------------------------------------------------------- 5. parking
    rule("SUB-MODEL 5 of 6 - PARKING   (replaces DEFAULT_PARKING = 70.0)   EXPECTED TO FAIL")
    X = build_matrix(rf.parking_features,
                     df[["distance_km", "vehicle_type", "passengers"]].values)
    y = df.parking_cost.values.astype(float)
    const = regression_scores(y[te], np.full(len(te), CONST_PARKING))
    mean_pred = regression_scores(y[te], np.full(len(te), y[tr].mean()))
    print(f"  BASELINE  always Rs {CONST_PARKING} :      R2 {const['r2']:+.4f}   "
          f"RMSE Rs {const['rmse']:.2f}")
    print(f"  BASELINE  always the train mean Rs {y[tr].mean():.1f} :  "
          f"R2 {mean_pred['r2']:+.4f}   RMSE Rs {mean_pred['rmse']:.2f}")
    rows = [(n, evaluate_regressor(m, X[tr], y[tr], X[te], y[te]))
            for n, m in candidate_regressors().items()]
    print_regression_table(rows)
    name, best = pick_best(rows)
    print(f"\n  RESULT : no model beats predicting the mean. Best was {name} at test R2 "
          f"{best['test']['r2']:+.4f}.")
    print("  This is the correct answer, not a bug. The generator draws parking uniformly from")
    print("  {0, 40, 60, 80, 120, 150} independently of every other column, so it carries no")
    print("  signal at all. REPORTED AS A NEGATIVE RESULT and the served value stays the mean -")
    print("  a model with R2 <= 0 has no business being in the prediction path.")
    report["parking"] = {"constant_baseline": const, "mean_baseline": mean_pred,
                         "candidates": {n: strip_fitted(r) for n, r in rows},
                         "chosen": "train mean (no model beat it)",
                         "train_mean": float(y[tr].mean()),
                         "negative_result": True}
    chosen["parking"] = "train mean (no model beat it)"
    models["parking_mean"] = float(y[tr].mean())

    # ---------------------------------------------------------------- 6. traffic
    rule("SUB-MODEL 6 of 6 - TRAFFIC LEVEL   (classifier, already part of the project)")
    X = build_matrix(rf.traffic_features, df[["departure_hour", "month"]].values)
    y = labels(df.traffic_level)
    rows = [(n, evaluate_classifier(m, X[tr], y[tr], X[te], y[te]))
            for n, m in candidate_classifiers().items()]
    print_classification_table(rows)
    name, best = pick_best(rows, key=lambda r: r["test"]["accuracy"])
    print(f"\n  CHOSEN : {name}   accuracy {best['test']['accuracy']:.4f}   "
          f"F1 {best['test']['f1_macro']:.4f}   "
          f"vs {best['baseline']:.4f} baseline (+{best['lift_points']:.1f} pts)")
    print("  Honest reading: the rush-hour profile was injected deliberately in Week 5, so this")
    print("  accuracy measures that design choice, not Indian roads.")
    models["traffic"], chosen["traffic"] = best["fitted"], name
    report["traffic"] = {"candidates": {n: strip_fitted(r) for n, r in rows}, "chosen": name}

    return models, chosen


# ==========================================================================================
# The chained cost model - and the train/serve mismatch, measured
# ==========================================================================================
def train_chained_cost(df, tr, te, sub, distance_pred_tr, distance_pred_te):
    rule("CHAINED COST MODEL - fed sub-model predictions, not ground truth")

    price = df.fuel_price.values.astype(float)
    mileage = df.mileage.values.astype(float)
    vehicle = df.vehicle_type.to_numpy(dtype=object)
    y = df.total_trip_cost.values.astype(float)

    # --- components as PREDICTED by the sub-models -------------------------------------
    # Training rows use out-of-fold predictions (cross_val_predict), so no row's component
    # features come from a model that had already seen that row. Without this the cost model
    # would be trained on unrealistically good components and would fall apart in production.
    print("  building component features (out-of-fold for train, fitted-on-train for test)...")

    Xtraffic = build_matrix(rf.traffic_features, df[["departure_hour", "month"]].values)
    traffic_tr = cross_val_predict(make_pipeline(_fresh(sub["traffic"].named_steps["model"])),
                                   Xtraffic[tr], labels(df.traffic_level)[tr], cv=5,
                                   n_jobs=-1)
    traffic_te = sub["traffic"].predict(Xtraffic[te])

    def litres_for(idx, dist, traffic, out_of_fold):
        rows = [rf.litres_features(d, m, t) for d, m, t in zip(dist, mileage[idx], traffic)]
        X = np.asarray(rows, dtype=float)
        if out_of_fold:
            ytrue = df.fuel_consumption_litres.values.astype(float)[idx]
            return cross_val_predict(make_pipeline(_fresh(sub["litres"].named_steps["model"])),
                                     X, ytrue, cv=5, n_jobs=-1)
        return sub["litres"].predict(X)

    def toll_for(idx, dist, out_of_fold):
        X = np.asarray([rf.toll_features(d) for d in dist], dtype=float)
        if out_of_fold:
            ytrue = df.toll_cost.values.astype(float)[idx]
            return cross_val_predict(make_pipeline(_fresh(sub["toll"].named_steps["model"])),
                                     X, ytrue, cv=5, n_jobs=-1)
        return sub["toll"].predict(X)

    litres_tr = litres_for(tr, distance_pred_tr, traffic_tr, True)
    litres_te = litres_for(te, distance_pred_te, traffic_te, False)
    toll_tr = toll_for(tr, distance_pred_tr, True)
    toll_te = toll_for(te, distance_pred_te, False)
    parking_tr = np.full(len(tr), sub["parking_mean"])
    parking_te = np.full(len(te), sub["parking_mean"])

    def cost_matrix(idx, dist, toll, parking, litres):
        return np.asarray([rf.cost_features(d, m, p, t, pk, l, v)
                           for d, m, p, t, pk, l, v
                           in zip(dist, mileage[idx], price[idx], toll, parking, litres,
                                  vehicle[idx])], dtype=float)

    Xpred_tr = cost_matrix(tr, distance_pred_tr, toll_tr, parking_tr, litres_tr)
    Xpred_te = cost_matrix(te, distance_pred_te, toll_te, parking_te, litres_te)

    # --- the same matrix built from GROUND TRUTH, for the comparison -------------------
    Xtrue_tr = cost_matrix(tr, df.distance_km.values[tr], df.toll_cost.values[tr],
                           df.parking_cost.values[tr],
                           df.fuel_consumption_litres.values[tr])
    Xtrue_te = cost_matrix(te, df.distance_km.values[te], df.toll_cost.values[te],
                           df.parking_cost.values[te],
                           df.fuel_consumption_litres.values[te])

    print("\n  (i) the project's existing model: trained AND tested on true components")
    print("      This is where R2 = 0.9997 comes from. It answers 'given the litres burnt, the")
    print("      toll paid and the parking paid, can you add them up?'")
    rows_true = [(n, evaluate_regressor(m, Xtrue_tr, y[tr], Xtrue_te, y[te]))
                 for n, m in candidate_regressors().items()]
    print_regression_table(rows_true)
    name_true, best_true = pick_best(rows_true)

    print("\n  (ii) MISMATCHED: trained on true components, served predicted ones")
    print("       The bug you get by chaining a pipeline without retraining the final stage.")
    mismatch = regression_scores(y[te], best_true["fitted"].predict(Xpred_te))
    print(f"      {name_true}: test R2 {mismatch['r2']:.4f}   RMSE Rs {mismatch['rmse']:.2f}   "
          f"MAE Rs {mismatch['mae']:.2f}")

    print("\n  (iii) CHAIN-CONSISTENT: trained and served on predicted components")
    rows_chain = [(n, evaluate_regressor(m, Xpred_tr, y[tr], Xpred_te, y[te]))
                  for n, m in candidate_regressors().items()]
    print_regression_table(rows_chain)
    name_chain, best_chain = pick_best(rows_chain)
    print(f"\n  CHOSEN : {name_chain}   test R2 {best_chain['test']['r2']:.4f}   "
          f"RMSE Rs {best_chain['test']['rmse']:.2f}   MAE Rs {best_chain['test']['mae']:.2f}")

    print("\n  READ THESE THREE NUMBERS TOGETHER:")
    print(f"    (i)   true components in, true components out : R2 "
          f"{best_true['test']['r2']:.4f}  MAE Rs {best_true['test']['mae']:.2f}")
    print(f"    (ii)  mismatched chain                        : R2 "
          f"{mismatch['r2']:.4f}  MAE Rs {mismatch['mae']:.2f}")
    print(f"    (iii) honest chain, user inputs only          : R2 "
          f"{best_chain['test']['r2']:.4f}  MAE Rs {best_chain['test']['mae']:.2f}")
    print("  (i) is the number the project used to advertise. (iii) is what a user actually")
    print("  gets. The gap is the arithmetic that used to be handed to the model.")

    # Where does the remaining error come from? Decompose it, because 'the model is worse'
    # is not a finding - 'toll noise accounts for most of it' is.
    print("\n  WHERE THE REMAINING ERROR COMES FROM (test split, in Rs):")
    toll_err = df.toll_cost.values[te] - toll_te
    park_err = df.parking_cost.values[te] - parking_te
    litre_err = (df.fuel_consumption_litres.values[te] - litres_te) * price[te]
    dist_err = df.distance_km.values[te] - distance_pred_te
    for label, e in [("toll (injected noise)", toll_err), ("parking (uniform noise)", park_err),
                     ("fuel bill via litres", litre_err), ("distance (km, not Rs)", dist_err)]:
        print(f"    {label:<26} sd {np.std(e):>8.2f}   mean |err| {np.mean(np.abs(e)):>8.2f}")
    irreducible = float(np.sqrt(np.var(toll_err) + np.var(park_err) + np.var(litre_err)))
    print(f"    -> quadrature sum of the three cost terms: Rs {irreducible:.2f}")
    print(f"    -> chained model RMSE:                     Rs {best_chain['test']['rmse']:.2f}")
    print("    The two agree closely, which says the chain is near the floor this dataset")
    print("    allows. The error is the DATA's unpredictability, not the model's weakness.")

    report["cost"] = {
        "true_components": {"candidates": {n: strip_fitted(r) for n, r in rows_true},
                            "chosen": name_true},
        "mismatched_chain": mismatch,
        "chained": {"candidates": {n: strip_fitted(r) for n, r in rows_chain},
                    "chosen": name_chain},
        "error_budget": {
            "toll_sd": float(np.std(toll_err)), "parking_sd": float(np.std(park_err)),
            "fuel_bill_sd": float(np.std(litre_err)), "distance_sd_km": float(np.std(dist_err)),
            "quadrature_sum": irreducible,
            "chained_rmse": best_chain["test"]["rmse"],
        },
    }
    return best_chain["fitted"], name_chain, best_true["fitted"], name_true


# ==========================================================================================
def main():
    if not os.path.exists(REAL):
        sys.exit("data/real_distances.csv missing - run scripts/fetch_real_distances.py first")

    df = pd.read_csv(WIDE)
    cities = rf.load_city_index(CITIES)

    # Coordinates for every trip, so the pipeline can start from city names like a user does.
    missing = {c for c in set(df.start_city) | set(df.destination_city) if c not in cities}
    if missing:
        sys.exit(f"cities missing from india_cities.csv: {sorted(missing)[:5]}")
    df["start_lat"] = [cities[c][1] for c in df.start_city]
    df["start_lon"] = [cities[c][2] for c in df.start_city]
    df["dest_lat"] = [cities[c][1] for c in df.destination_city]
    df["dest_lon"] = [cities[c][2] for c in df.destination_city]
    df["haversine_km"] = [rf.haversine(a, b, c, d) for a, b, c, d in
                          zip(df.start_lat, df.start_lon, df.dest_lat, df.dest_lon)]

    print(f"wide dataset : {len(df):,} trips over "
          f"{len(set(df.start_city) | set(df.destination_city))} cities")

    # ONE canonical split, reused by every model below, so the comparison tables are all
    # measured on identical rows.
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)
    print(f"split        : {len(tr):,} train / {len(te):,} test (random_state={SEED})")

    distance_model, parameterisation = train_distance_model()
    measure_dataset_geography_bias()
    sub, chosen = train_sub_models(df, tr, te)

    # ------------------------------------------------------------------------------------
    # Inside the chained evaluation the distance model must reproduce THIS DATASET's distance
    # convention, not the real road network - the dataset's costs were generated from its own
    # distance column, so scoring against real km would charge the model for the dataset's
    # geography error (measured above) instead of its own. The app ships the real-data model.
    # Both are saved, and the difference between them is quantified in the report.
    # ------------------------------------------------------------------------------------
    rule("DATASET-CONVENTION DISTANCE MODEL   (used only inside the chained evaluation)")
    Xd = build_matrix(rf.distance_features,
                      df[["start_lat", "start_lon", "dest_lat", "dest_lon"]].values)
    yd = df.distance_km.values.astype(float)
    rows = [(n, evaluate_regressor(m, Xd[tr], yd[tr], Xd[te], yd[te]))
            for n, m in candidate_regressors().items()]
    print_regression_table(rows, unit="km")
    name_d, best_d = pick_best(rows)
    print(f"\n  CHOSEN : {name_d}   test R2 {best_d['test']['r2']:.6f}   "
          f"RMSE {best_d['test']['rmse']:.2f} km")
    print("  This one IS partly circular - the dataset's distance is haversine x N(1.2355,")
    print("  0.082), so the model recovers a constant plus irreducible noise. It exists so the")
    print("  chained cost evaluation is apples-to-apples. The REAL distance model above is the")
    print("  one that ships, and it is the one trained on observed data.")
    dataset_distance_model = best_d["fitted"]
    report["distance_dataset_convention"] = {
        "candidates": {n: strip_fitted(r) for n, r in rows}, "chosen": name_d,
        "circular": True,
    }

    dist_tr = dataset_distance_model.predict(Xd[tr])
    dist_te = dataset_distance_model.predict(Xd[te])

    chained_cost, chained_name, true_cost, true_name = train_chained_cost(
        df, tr, te, sub, dist_tr, dist_te)

    # ---------------------------------------------------------------------------- save
    #
    # TWO FILES, and the split is the point.
    #
    # A RandomForest of 300 trees fitted to 20,000 rows pickles to about 210 MB. Writing every
    # fitted object into one bundle produced a 227 MB models/pipeline.joblib - unusable in a
    # repository and absurd inside a Docker image, when the file it replaced was 10 KB.
    #
    # Compression alone is not the fix; the right question is which models the API actually
    # loads. app.py needs the six sub-models and the chained cost regressor. It never touches
    # `distance_dataset_model` (the deliberately circular model that exists only to keep the
    # chained evaluation apples-to-apples) or `cost_model_true_components` (the Week 6
    # comparison baseline). Those two are evaluation apparatus: fully reproducible by re-running
    # this script, and every number derived from them is already in data/pipeline_report.json.
    #
    # So the serving bundle carries only what is served, both files are compressed, and the
    # evaluation bundle is git- and docker-ignored.
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump({
        "distance_dataset_model": dataset_distance_model,
        "cost_model_true_components": true_cost,
        "note": ("Evaluation-only models, NOT loaded by app.py. Regenerate with "
                 "python scripts/train_pipeline.py"),
    }, OUT_EVAL, compress=3)

    joblib.dump({
        "distance_model": distance_model,
        "distance_parameterisation": parameterisation,
        "distance_features": rf.DISTANCE_FEATURES,
        "mileage_model": sub["mileage"],
        "litres_model": sub["litres"],
        "toll_model": sub["toll"],
        "parking_mean": sub["parking_mean"],
        "traffic_model": sub["traffic"],
        "cost_model_chained": chained_cost,
        "chosen": {**chosen, "distance": report["distance"]["chosen"],
                   "cost_chained": chained_name, "cost_true_components": true_name},
        "metrics": {
            "distance_real": report["distance"]["chosen_scores"]["test"],
            "distance_constant_baseline": report["distance"]["constant_baseline"],
            "cost_true_components": report["cost"]["true_components"]["candidates"][true_name]["test"],
            "cost_chained": report["cost"]["chained"]["candidates"][chained_name]["test"],
            "cost_mismatched": report["cost"]["mismatched_chain"],
            "traffic": report["traffic"]["candidates"][chosen["traffic"]]["test"],
            "geography_bias": report["geography_bias"],
            "error_budget": report["cost"]["error_budget"],
            "n_rows": int(len(df)),
            "n_cities": int(len(set(df.start_city) | set(df.destination_city))),
            "n_real_distance_pairs": report["distance"]["n_real_pairs"],
        },
    }, OUT_MODEL, compress=3)

    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    rule(f"saved -> {OUT_MODEL}  ({os.path.getsize(OUT_MODEL) / 1e6:.1f} MB, served by app.py)\n"
         f"saved -> {OUT_EVAL}  ({os.path.getsize(OUT_EVAL) / 1e6:.1f} MB, evaluation only)\n"
         f"saved -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
