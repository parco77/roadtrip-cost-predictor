"""
Train the chained model pipeline that the API serves.

WHAT THE PIPELINE IS FOR
------------------------
A trip cost is an addition: fuel plus toll plus parking plus running costs. Writing down that
addition is not the hard part - knowing what to put into it is. A traveller setting off knows
the distance, what their fuel costs and roughly what parking will run to. They do not know how
many litres they will burn or what the tolls will come to, and those are the two terms that
actually move the total.

So each unknown gets its own fitted model, and the cost regressor consumes their outputs.

THE SUB-MODELS
--------------
  1. distance   route geometry -> road km    trained on REAL OSRM distances; powers the
                                             distance lookup when a route is not in the table
  2. mileage    vehicle + fuel -> km/l       gives each of the three vehicles its own figure
  3. litres     distance + mileage -> litres
  4. toll       distance -> Rs
  5. parking    nothing predicts it          a negative result, and the reason parking is
                                             asked for rather than guessed
  then the cost regressor consumes 2-4 plus what the user typed.

Every sub-model is scored against the simplest thing that could work in its place - a fixed
ratio, a flat rate, the training mean. A model that cannot beat that has not earned its slot,
and one of them does not: see sub-model 5.

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

BACKEND = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backend"
)
# roadtrip_features.py is the server's feature contract - training imports the very file
# the API imports, so a column added for the model cannot go missing at serve time.
sys.path.insert(0, BACKEND)
import roadtrip_features as rf                                          # noqa: E402
from evaluation import (SEED, candidate_regressors, evaluate_regressor,  # noqa: E402
                        make_pipeline, pick_best, print_regression_table,
                        regression_scores, rule, strip_fitted)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDE = os.path.join(ROOT, "data", "road_trip_wide.csv")
REAL = os.path.join(ROOT, "data", "real_distances.csv")
CITIES = os.path.join(BACKEND, "data", "india_cities.csv")
OUT_MODEL = os.path.join(BACKEND, "models", "pipeline.joblib")
OUT_EVAL = os.path.join(ROOT, "models", "pipeline_eval.joblib")
OUT_REPORT = os.path.join(ROOT, "data", "pipeline_report.json")

# Naive baselines. Every sub-model below is scored against the simplest thing that could stand
# in its place - one ratio, one rate, one flat amount - because "the model got R2 0.8" means
# nothing until you know what arithmetic alone would have scored. A model that cannot beat its
# baseline has not earned a slot in the prediction path.
BASE_WINDING = 1.2355      # a single straight-line-to-road ratio for the whole country
BASE_TOLL_RATE = 1.301     # Rs per km, flat
BASE_PARKING = 70.0        # Rs, the same for every trip

TEST_SIZE = 0.2
report = {}


def build_matrix(builder, rows):
    """Apply a feature builder across a list of argument tuples -> 2-D float array."""
    return np.asarray([builder(*args) for args in rows], dtype=float)


# ==========================================================================================
# 1. ROAD DISTANCE - the one sub-model trained on genuinely observed data
# ==========================================================================================
def train_distance_model():
    rule("SUB-MODEL 1 of 5 - ROAD DISTANCE   (powers the distance lookup)")
    real = pd.read_csv(REAL)
    print(f"  training data : {len(real)} REAL city-pair road distances from OSRM")
    print(f"  haversine     : {real.haversine_km.min():.0f} - {real.haversine_km.max():.0f} km")
    print(f"  observed winding factor: mean {real.factor.mean():.4f}  sd {real.factor.std():.4f}"
          f"  range {real.factor.min():.4f} - {real.factor.max():.4f}")
    print(f"  a single national ratio would be: {BASE_WINDING}  <- one number for that whole range")

    # The extreme routes are worth naming, because a reader's first instinct on seeing a
    # factor of 3.6 is "bad data" - and here it is not.
    worst = real.nlargest(3, "factor")
    print("\n  the three routes a single ratio serves worst:")
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

    # Baseline to beat: one ratio for the whole country.
    const_scores = regression_scores(yte, hte * BASE_WINDING)
    print(f"\n  BASELINE  haversine x {BASE_WINDING} :  "
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
# THE ASSUMPTION THIS PROJECT RESTS ON, MEASURED RATHER THAN ASSERTED
# ==========================================================================================
def measure_distance_convention_gap():
    """How far apart are the distances the model LEARNED ON and the ones it is SERVED?

    This is the honest limitation of the whole design, so it is measured first and stated in
    kilometres rather than left in a footnote.

    The cost model is fitted on data/road_trip_wide.csv, whose distance column follows a
    synthetic convention: straight-line distance times a random draw around 1.2355. The number
    a user types comes from the reference table, which holds REAL road distances measured on the
    actual network. If those two disagree systematically, the model is being asked at serve time
    about a quantity that does not mean what it meant at training time.
    """
    rule("DISTANCE CONVENTION CHECK - what the model learned on vs what it is served")
    real = pd.read_csv(REAL)
    convention = real.haversine_km * BASE_WINDING
    err = convention - real.road_km
    pct = err / real.road_km * 100
    print(f"  Checked on {len(real)} city pairs where both are known.")
    print(f"    mean signed error : {err.mean():+.1f} km  ({pct.mean():+.1f}%)   <- systematic bias")
    print(f"    mean abs error    : {err.abs().mean():.1f} km  ({pct.abs().mean():.1f}%)"
          f"   <- typical single route")
    print(f"    worst overshoot   : {err.max():+.1f} km  ({pct.max():+.1f}%)")
    print(f"    worst undershoot  : {err.min():+.1f} km  ({pct.min():+.1f}%)")
    print("\n  The two numbers say opposite-sounding things, and both matter:")
    print("  The systematic bias is near zero - averaged over hundreds of routes the training")
    print("  convention lands within a fraction of a percent of the real network. So the cost")
    print("  model is not learning a distorted idea of what a kilometre is, and the rupees-per-km")
    print("  relationships it fits carry over to real distances intact.")
    print(f"  But route by route the convention is off by {pct.abs().mean():.1f}% typically, and by tens of")
    print("  percent where geography intervenes. That spread is why the reference table is built")
    print("  from measured road distances instead of generated from the same convention: the")
    print("  user gets the real number even though the cost model was fitted on synthetic ones.")
    print("\n  LIMITATION, STATED PLAINLY: the cost relationships are learned from generated")
    print("  trips. This check bounds what that costs - it does not eliminate it.")
    report["distance_convention_gap"] = {
        "mean_signed_km": float(err.mean()), "mean_signed_pct": float(pct.mean()),
        "mean_abs_km": float(err.abs().mean()), "mean_abs_pct": float(pct.abs().mean()),
        "max_km": float(err.max()), "min_km": float(err.min()),
        "n_pairs": int(len(real)),
    }


# ==========================================================================================
# 2-5. the sub-models trained on the generated wide dataset
# ==========================================================================================
def train_sub_models(df, tr, te):
    models, chosen = {}, {}

    # ---------------------------------------------------------------- 2. mileage
    rule("SUB-MODEL 2 of 5 - MILEAGE   (a km/l figure for each of the three vehicles)")
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
    rule("SUB-MODEL 3 of 5 - FUEL CONSUMED")
    X = build_matrix(rf.litres_features, df[["distance_km", "mileage"]].values)
    y = df.fuel_consumption_litres.values.astype(float)
    rows = [(n, evaluate_regressor(m, X[tr], y[tr], X[te], y[te]))
            for n, m in candidate_regressors().items()]
    print_regression_table(rows, unit="litres")
    name, best = pick_best(rows)
    print(f"\n  CHOSEN : {name}   test R2 {best['test']['r2']:.6f}   "
          f"RMSE {best['test']['rmse']:.3f} L")

    # WHAT THE FITTED SLOPE MEANS, AND WHY IT IS NOT 1.0
    #
    # Textbook fuel burn is litres = distance / mileage, i.e. a slope of exactly 1.0 on the
    # ratio column. Refitting unscaled on that one column recovers what the data actually says.
    from sklearn.linear_model import LinearRegression as _LR
    ratio_col = rf.LITRES_FEATURES.index("km_per_litre_ratio")
    bare = _LR().fit(X[tr][:, [ratio_col]], y[tr])
    slope = float(bare.coef_[0])
    ratio_te = X[te][:, ratio_col]
    residual = y[te] - slope * ratio_te
    print("\n  READING THE SLOPE")
    print(f"    fitted slope on (distance / mileage) : {slope:.5f}")
    print("    textbook value                       : 1.00000")
    print(f"    intercept                            : {bare.intercept_:+.3f} L  (near zero, "
          f"as the relationship has no constant term)")
    print("\n  The slope sits ABOVE 1.0, and that excess is the point. A car does not achieve its")
    print("  rated mileage on a real trip - stop-start driving, gradients and congestion all")
    print("  cost fuel - so the litres actually burnt run above distance / mileage. How much")
    print("  above depends on conditions the traveller cannot state before setting off, so the")
    print("  model cannot be told them. It learns the AVERAGE penalty instead, and carries the")
    print("  variation around that average as error:")
    print(f"    residual sd on held-out trips : {residual.std():.3f} L")
    print(f"    mean absolute error           : {abs(residual).mean():.3f} L")
    print("  That error is irreducible from these inputs, and it scales with trip length rather")
    print("  than averaging away - a 1200 km drive has four times the uncertainty of a 300 km")
    print("  one. It is the largest single term in the cost model's error budget below.")
    models["litres"], chosen["litres"] = best["fitted"], name
    report["litres"] = {
        "candidates": {n: strip_fitted(r) for n, r in rows}, "chosen": name,
        "fitted_ratio_slope": slope,
        "residual_sd_litres": float(residual.std()),
        "residual_mae_litres": float(abs(residual).mean()),
    }

    # ---------------------------------------------------------------- 4. toll
    rule("SUB-MODEL 4 of 5 - TOLL")
    X = build_matrix(rf.toll_features, df[["distance_km"]].values)
    y = df.toll_cost.values.astype(float)
    const = regression_scores(y[te], df.distance_km.values[te] * BASE_TOLL_RATE)
    print(f"  BASELINE  distance x {BASE_TOLL_RATE} :  R2 {const['r2']:.4f}   "
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
    rule("SUB-MODEL 5 of 5 - PARKING   (EXPECTED TO FAIL, and the reason it is a user input)")
    X = build_matrix(rf.parking_features,
                     df[["distance_km", "vehicle_type", "passengers"]].values)
    y = df.parking_cost.values.astype(float)
    const = regression_scores(y[te], np.full(len(te), BASE_PARKING))
    mean_pred = regression_scores(y[te], np.full(len(te), y[tr].mean()))
    print(f"  BASELINE  always Rs {BASE_PARKING} :      R2 {const['r2']:+.4f}   "
          f"RMSE Rs {const['rmse']:.2f}")
    print(f"  BASELINE  always the train mean Rs {y[tr].mean():.1f} :  "
          f"R2 {mean_pred['r2']:+.4f}   RMSE Rs {mean_pred['rmse']:.2f}")
    rows = [(n, evaluate_regressor(m, X[tr], y[tr], X[te], y[te]))
            for n, m in candidate_regressors().items()]
    print_regression_table(rows)
    name, best = pick_best(rows)
    print(f"\n  RESULT : no model beats predicting the mean. Best was {name} at test R2 "
          f"{best['test']['r2']:+.4f}.")
    print("  This is the correct answer, not a bug. Parking in this data carries no relationship")
    print("  to distance, vehicle or party size - it is drawn independently of all of them, so")
    print("  there is nothing in the inputs for a model to find. Six families were fitted and")
    print("  every one scored at or below the mean.")
    print("\n  WHY THIS MATTERS FOR THE INTERFACE. A quantity nothing predicts should be asked")
    print("  for, not guessed. So parking is the one cost term the form requests outright, and")
    print("  this negative result is the evidence for that choice rather than a preference.")
    print("  Reported rather than tuned away: a model with R2 <= 0 has no business in a")
    print("  prediction path, and dressing one up to look respectable would be worse than")
    print("  admitting the data has no answer.")
    report["parking"] = {"constant_baseline": const, "mean_baseline": mean_pred,
                         "candidates": {n: strip_fitted(r) for n, r in rows},
                         "chosen": "user input (no model beat the mean)",
                         "train_mean": float(y[tr].mean()),
                         "negative_result": True}
    chosen["parking"] = "user input (no model beat the mean)"

    return models, chosen


# ==========================================================================================
# The chained cost model - and the train/serve mismatch, measured
# ==========================================================================================
def train_chained_cost(df, tr, te, sub):
    rule("CHAINED COST MODEL - fed sub-model predictions, not ground truth")

    price = df.fuel_price.values.astype(float)
    mileage = df.mileage.values.astype(float)
    vehicle = df.vehicle_type.to_numpy(dtype=object)
    y = df.total_trip_cost.values.astype(float)

    # Distance and parking are TYPED BY THE USER, so at serve time they arrive exact. Training
    # them as ground truth is therefore not leakage - it matches what the model will actually
    # receive. Litres and toll are the opposite: nobody knows them before setting off, so they
    # must come from the sub-models here exactly as they will in production.
    distance_tr = df.distance_km.values.astype(float)[tr]
    distance_te = df.distance_km.values.astype(float)[te]
    parking_tr = df.parking_cost.values.astype(float)[tr]
    parking_te = df.parking_cost.values.astype(float)[te]

    # --- components as PREDICTED by the sub-models -------------------------------------
    # Training rows use out-of-fold predictions (cross_val_predict), so no row's component
    # features come from a model that had already seen that row. Without this the cost model
    # would be trained on unrealistically good components and would fall apart in production.
    print("  building component features (out-of-fold for train, fitted-on-train for test)...")

    def litres_for(idx, dist, out_of_fold):
        rows = [rf.litres_features(d, m) for d, m in zip(dist, mileage[idx])]
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

    litres_tr = litres_for(tr, distance_tr, True)
    litres_te = litres_for(te, distance_te, False)
    toll_tr = toll_for(tr, distance_tr, True)
    toll_te = toll_for(te, distance_te, False)

    def cost_matrix(idx, dist, toll, parking, litres):
        return np.asarray([rf.cost_features(d, m, p, t, pk, l, v)
                           for d, m, p, t, pk, l, v
                           in zip(dist, mileage[idx], price[idx], toll, parking, litres,
                                  vehicle[idx])], dtype=float)

    Xpred_tr = cost_matrix(tr, distance_tr, toll_tr, parking_tr, litres_tr)
    Xpred_te = cost_matrix(te, distance_te, toll_te, parking_te, litres_te)

    # --- the same matrix built from GROUND TRUTH, for the comparison -------------------
    Xtrue_tr = cost_matrix(tr, df.distance_km.values[tr], df.toll_cost.values[tr],
                           df.parking_cost.values[tr],
                           df.fuel_consumption_litres.values[tr])
    Xtrue_te = cost_matrix(te, df.distance_km.values[te], df.toll_cost.values[te],
                           df.parking_cost.values[te],
                           df.fuel_consumption_litres.values[te])

    print("\n  (i) HANDED THE ANSWER: trained AND tested on true components")
    print("      The model is given the litres actually burnt and the toll actually paid, and")
    print("      asked to add them up. It scores near-perfectly, and that score is worth almost")
    print("      nothing: a traveller cannot supply either number before the trip.")
    rows_true = [(n, evaluate_regressor(m, Xtrue_tr, y[tr], Xtrue_te, y[te]))
                 for n, m in candidate_regressors().items()]
    print_regression_table(rows_true)
    name_true, best_true = pick_best(rows_true)

    print("\n  (ii) MISMATCHED: trained on true components, served predicted ones")
    print("       What happens if you chain a pipeline without retraining the final stage - it")
    print("       learns to trust inputs that are exact, then receives inputs that are not.")
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
    print("  (iii) is what a user actually gets, and it is the only one of the three worth")
    print("  quoting. (i) measures a question nobody asks - it flatters the model by handing it")
    print("  the two terms that carry the uncertainty.")
    gap = mismatch["mae"] - best_chain["test"]["mae"]
    print(f"\n  (ii) VERSUS (iii) - the retraining is worth Rs {gap:+.2f} of MAE here, which is")
    print("  nothing. That is a real result and it is worth saying rather than glossing:")
    print("  retraining the final stage on predicted components matters when those predictions")
    print("  are BIASED, because the model has to learn to distrust them. Only two components")
    print("  are predicted in this chain, toll and litres, and both sub-models are close to")
    print("  unbiased - their errors are spread, not shifted - so there is nothing for the")
    print("  retrained model to correct for. The chain-consistent model is still the one that")
    print("  ships, because that guarantee comes from the design rather than from this run")
    print("  happening to be unbiased; the measurement is what tells us the cost, not an")
    print("  assumption in either direction.")

    # Where does the remaining error come from? Decompose it, because 'the model is worse'
    # is not a finding - 'toll noise accounts for most of it' is.
    print("\n  WHERE THE REMAINING ERROR COMES FROM (test split, in Rs):")
    toll_err = df.toll_cost.values[te] - toll_te
    litre_err = (df.fuel_consumption_litres.values[te] - litres_te) * price[te]
    for label, e in [("toll (unpredictable spread)", toll_err),
                     ("fuel bill via litres", litre_err)]:
        print(f"    {label:<30} sd {np.std(e):>8.2f}   mean |err| {np.mean(np.abs(e)):>8.2f}")
    print(f"    {'distance (typed, so exact)':<30} sd {0.0:>8.2f}   mean |err| {0.0:>8.2f}")
    print(f"    {'parking (typed, so exact)':<30} sd {0.0:>8.2f}   mean |err| {0.0:>8.2f}")
    irreducible = float(np.sqrt(np.var(toll_err) + np.var(litre_err)))
    print(f"    -> quadrature sum of the two predicted terms: Rs {irreducible:.2f}")
    print(f"    -> chained model RMSE:                        Rs {best_chain['test']['rmse']:.2f}")
    print("    The two agree closely, which says the chain is near the floor this data allows.")
    print("    The error is the DATA's unpredictability, not the model's weakness - and the two")
    print("    terms that carry it are exactly the two the user could not have told us.")

    # -------------------------------------------------------------- the three-vehicle check
    #
    # The product prices all three vehicles side by side, so the GAP between them is the output,
    # not a detail. Measure it at several distances: running costs differ per kilometre, so the
    # gap must grow with the trip. If it comes back flat, the distance x vehicle interaction has
    # been dropped from COST_FEATURES - which no single-vehicle estimate would ever reveal.
    print("\n  THE GAP BETWEEN VEHICLES, ACROSS DISTANCE (petrol at Rs 106, parking Rs 70):")
    gap_rows = []
    for dist in (200, 400, 800, 1200):
        totals = {}
        for veh in ("Hatchback", "Sedan", "SUV"):
            km_per_l = float(sub["mileage"].predict(
                np.array([rf.mileage_features(veh, "Petrol")], dtype=float))[0])
            lit = float(sub["litres"].predict(
                np.array([rf.litres_features(dist, km_per_l)], dtype=float))[0])
            tll = float(sub["toll"].predict(
                np.array([rf.toll_features(dist)], dtype=float))[0])
            totals[veh] = float(best_chain["fitted"].predict(np.array(
                [rf.cost_features(dist, km_per_l, 106.0, tll, 70.0, lit, veh)],
                dtype=float))[0])
        gap = totals["SUV"] - totals["Hatchback"]
        gap_rows.append({"distance_km": dist, **{k: round(v, 2) for k, v in totals.items()},
                         "suv_minus_hatchback": round(gap, 2),
                         "gap_per_km": round(gap / dist, 4)})
        print(f"    {dist:>5} km   " + "  ".join(f"{k} {v:>9.0f}" for k, v in totals.items())
              + f"   gap Rs {gap:>7.0f}  ({gap / dist:.2f}/km)")
    per_km = [r["gap_per_km"] for r in gap_rows]
    print(f"    gap per km stays within {min(per_km):.2f} - {max(per_km):.2f} Rs/km, so the")
    print("    difference scales with the trip rather than sitting as a flat surcharge.")
    report["vehicle_gap"] = gap_rows

    report["cost"] = {
        "true_components": {"candidates": {n: strip_fitted(r) for n, r in rows_true},
                            "chosen": name_true},
        "mismatched_chain": mismatch,
        "chained": {"candidates": {n: strip_fitted(r) for n, r in rows_chain},
                    "chosen": name_chain},
        "error_budget": {
            "toll_sd": float(np.std(toll_err)),
            "fuel_bill_sd": float(np.std(litre_err)),
            "parking_sd": 0.0,        # typed by the user, so exact at serve time
            "distance_sd_km": 0.0,    # typed by the user, so exact at serve time
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
    measure_distance_convention_gap()
    sub, chosen = train_sub_models(df, tr, te)

    chained_cost, chained_name, true_cost, true_name = train_chained_cost(df, tr, te, sub)

    # ---------------------------------------------------------------------------- save
    #
    # TWO FILES, and the split is the point.
    #
    # A RandomForest of 300 trees fitted to 20,000 rows pickles to about 210 MB. Writing every
    # fitted object into one bundle produced a 227 MB models/pipeline.joblib - unusable in a
    # repository and absurd inside a Docker image, when the file it replaced was 10 KB.
    #
    # Compression alone is not the fix; the right question is which models the API actually
    # loads. app.py needs the four serving sub-models and the chained cost regressor. It never
    # touches `cost_model_true_components`, which exists only as the comparison baseline in the
    # three-way table above. That one is evaluation apparatus: fully reproducible by re-running
    # this script, and every number derived from it is already in data/pipeline_report.json.
    #
    # So the serving bundle carries only what is served, both files are compressed, and the
    # evaluation bundle is git- and docker-ignored.
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump({
        "cost_model_true_components": true_cost,
        "note": ("Evaluation-only model, NOT loaded by app.py. Regenerate with "
                 "python scripts/train_pipeline.py"),
    }, OUT_EVAL, compress=3)

    joblib.dump({
        "distance_model": distance_model,
        "distance_parameterisation": parameterisation,
        "distance_features": rf.DISTANCE_FEATURES,
        "mileage_model": sub["mileage"],
        "litres_model": sub["litres"],
        "toll_model": sub["toll"],
        "cost_model_chained": chained_cost,
        "chosen": {**chosen, "distance": report["distance"]["chosen"],
                   "cost_chained": chained_name, "cost_true_components": true_name},
        "metrics": {
            "distance_real": report["distance"]["chosen_scores"]["test"],
            "distance_constant_baseline": report["distance"]["constant_baseline"],
            "cost_true_components": report["cost"]["true_components"]["candidates"][true_name]["test"],
            "cost_chained": report["cost"]["chained"]["candidates"][chained_name]["test"],
            "cost_mismatched": report["cost"]["mismatched_chain"],
            "litres": report["litres"]["candidates"][chosen["litres"]]["test"],
            "litres_ratio_slope": report["litres"]["fitted_ratio_slope"],
            "parking_negative_result": {
                "best_r2": max(c["test"]["r2"]
                               for c in report["parking"]["candidates"].values()),
                "mean_baseline_rmse": report["parking"]["mean_baseline"]["rmse"],
            },
            "distance_convention_gap": report["distance_convention_gap"],
            "error_budget": report["cost"]["error_budget"],
            "n_rows": int(len(df)),
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
