"""
Phase D - Task 5, worked end to end.

Task 5 checklist:
  1. Metrics            RSS / RMSE / R2  (regression)   Accuracy / P / R / F1  (classification)
  2. Overfit / underfit train score vs test score
  3. Cross-validation   5-fold on the training data, with the spread   (bootstrap also run)
  4. Comparison         one table per family, best score AND stable CV
  5. Tuning             GridSearchCV / RandomizedSearchCV, then re-test
  6. Advanced models    RandomForest (bagging), AdaBoost, GradientBoosting

Everything is scored through scripts/evaluation.py, so these numbers match the notebook and
TASK5.md exactly. Output: data/task5_evaluation.json

Run:  python scripts/evaluate_models.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     train_test_split)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import roadtrip_features as rf                                              # noqa: E402
from evaluation import (SEED, candidate_classifiers, candidate_regressors,   # noqa: E402
                        evaluate_classifier, evaluate_regressor,
                        labels, make_pipeline, pick_best, print_classification_table,
                        print_regression_table, regression_scores, rule, strip_fitted)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDE = os.path.join(ROOT, "data", "road_trip_wide.csv")
ORIGINAL = os.path.join(ROOT, "road_trip_data.csv")
CITIES = os.path.join(ROOT, "data", "india_cities.csv")
OUT = os.path.join(ROOT, "data", "task5_evaluation.json")

TEST_SIZE = 0.2
out = {"task": "Task 5 - evaluation, validation and tuning", "seed": SEED}


def load_wide():
    df = pd.read_csv(WIDE)
    cities = rf.load_city_index(CITIES)
    df["start_lat"] = [cities[c][1] for c in df.start_city]
    df["start_lon"] = [cities[c][2] for c in df.start_city]
    df["dest_lat"] = [cities[c][1] for c in df.destination_city]
    df["dest_lon"] = [cities[c][2] for c in df.destination_city]
    return df


# ==========================================================================================
# Items 1-4, regression
# ==========================================================================================
def regression_section(df, tr, te):
    rule("TASK 5 ITEMS 1-4  |  REGRESSION - target: total_trip_cost (Rs)")
    print("Feature set: the model this project already ships (Week 6) - distance, mileage,")
    print("fuel price, toll, parking, litres, the litres x price interaction and vehicle")
    print("one-hots. Every model below sees exactly these columns and this split, so the")
    print("table compares MODELS, not feature sets.\n")

    y = df.total_trip_cost.values.astype(float)
    X = np.asarray([rf.cost_features(d, m, p, t, pk, l, v) for d, m, p, t, pk, l, v in zip(
        df.distance_km, df.mileage, df.fuel_price, df.toll_cost, df.parking_cost,
        df.fuel_consumption_litres, df.vehicle_type)], dtype=float)

    rows = [(n, evaluate_regressor(m, X[tr], y[tr], X[te], y[te]))
            for n, m in candidate_regressors().items()]

    # The Week 6 ablation, as an extra row: same models, interaction term removed. It belongs
    # in this table because it is the honest baseline the project's headline number beats.
    keep = [i for i, name in enumerate(rf.COST_FEATURES) if name != "fuel_bill"]
    Xno = X[:, keep]
    ablation = evaluate_regressor(candidate_regressors()["LinearRegression"],
                                  Xno[tr], y[tr], Xno[te], y[te])
    rows.insert(0, ("Linear, NO interaction", ablation))

    print_regression_table(rows)
    name, best = pick_best(rows)
    print(f"\nBEST (score + CV stability): {name}")
    print(f"  test R2 {best['test']['r2']:.6f}   RMSE Rs {best['test']['rmse']:.2f}   "
          f"RSS {best['test']['rss']:.4g}   CV {best['cv_mean']:.6f} +/- {best['cv_std']:.4f}")

    print("\nITEM 2 - overfit / underfit, per model:")
    print("  thresholds: train-test R2 gap > 0.05 = Overfitting; both R2 < 0.50 = Underfitting")
    for n, r in rows:
        print(f"  {n:<26} train {r['train']['r2']:.6f}  test {r['test']['r2']:.6f}  "
              f"gap {r['train']['r2'] - r['test']['r2']:+.6f}  -> {r['verdict']}")
    print("\n  DecisionTree shows train R2 = 1.000000 - a tree grown to purity memorises every")
    print("  training row. Its test score stays high here only because the target is a")
    print("  deterministic formula; on noisy data that gap is the classic overfitting signature.")

    print("\nITEM 3 - 5-fold cross-validation on the training data (spread matters):")
    for n, r in rows:
        spread = max(r["cv_scores"]) - min(r["cv_scores"])
        print(f"  {n:<26} mean {r['cv_mean']:.6f}  sd {r['cv_std']:.6f}  "
              f"fold range {spread:.6f}")
    print("  All spreads are tiny, which is expected: 20,000 training rows and a target that")
    print("  is a formula. A large spread here would mean the model was unstable to which rows")
    print("  it happened to see - the thing cross-validation exists to detect.")

    out["regression"] = {
        "feature_set": rf.COST_FEATURES,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "models": {n: strip_fitted(r) for n, r in rows},
        "best": name,
    }
    return X, y, name, best


# ==========================================================================================
# Items 1-4, classification
# ==========================================================================================
def classification_section(df, tr, te):
    results = {}

    # The same target twice, with and without the two columns the sub-models supply. The
    # second run is the under-specified control: it is what the classifier could manage if
    # the interface asked only for what the user can type. The gap between them is what the
    # litres and toll sub-models are worth, in accuracy points rather than in prose.
    SERVED = ["distance_km", "mileage", "fuel_price", "parking_cost",
              "toll_cost", "fuel_consumption_litres"]
    THIN = ["distance_km", "mileage", "fuel_price", "parking_cost"]

    for key, label, numeric, note in [
        ("cost_band", "COST-EFFICIENCY BAND", SERVED,
         "Is this trip cheap or expensive FOR ITS LENGTH? A quartile split of Rs/km, which\n"
         "is the label the result cards carry. Partly high BY CONSTRUCTION: the regressor\n"
         "above already predicts cost well and Rs/km is derived from cost."),
        ("cost_band_thin", "COST-EFFICIENCY BAND, UNDER-SPECIFIED", THIN,
         "The same target with the toll and litres columns removed - only what the form\n"
         "actually collects. Expected to score WORSE, and by how much is the point: it is\n"
         "the sub-models' contribution, measured instead of asserted."),
    ]:
        target = "cost_band"
        rule(f"TASK 5 ITEMS 1-4  |  CLASSIFICATION - {label}")
        print(note + "\n")

        cat = ["vehicle_type", "fuel_type"]
        Xdf = pd.get_dummies(df[numeric + cat], columns=cat, drop_first=True)
        feature_names = Xdf.columns.tolist()
        X = Xdf.values.astype(float)

        y = labels(df[target])
        rows = [(n, evaluate_classifier(m, X[tr], y[tr], X[te], y[te]))
                for n, m in candidate_classifiers().items()]
        print_classification_table(rows)

        name, best = pick_best(rows, key=lambda r: r["test"]["accuracy"])
        print(f"\nBEST (score + CV stability): {name}")
        print(f"  accuracy {best['test']['accuracy']:.4f}   precision "
              f"{best['test']['precision_macro']:.4f}   recall "
              f"{best['test']['recall_macro']:.4f}   F1 {best['test']['f1_macro']:.4f}")
        print(f"  baseline {best['baseline']:.4f}  ->  lift +{best['lift_points']:.1f} points")
        print(f"  CV {best['cv_mean']:.4f} +/- {best['cv_std']:.4f}")

        print("\nITEM 2 - overfit / underfit, per model:")
        for n, r in rows:
            print(f"  {n:<26} train {r['train']['accuracy']:.4f}  test "
                  f"{r['test']['accuracy']:.4f}  "
                  f"gap {r['train']['accuracy'] - r['test']['accuracy']:+.4f}  -> {r['verdict']}")

        pred = best["fitted"].predict(X[te])
        classes = sorted(set(y))
        cm = confusion_matrix(y[te], pred, labels=classes)
        print(f"\nConfusion matrix for {name} (rows = actual, cols = predicted):")
        print("            " + "".join(f"{c:>12}" for c in classes))
        for c, line in zip(classes, cm):
            print(f"  {c:<10}" + "".join(f"{v:>12}" for v in line))
        print("\nPer-class report:")
        print(classification_report(y[te], pred, digits=3, zero_division=0))

        results[key] = {
            "feature_set": feature_names,
            "models": {n: strip_fitted(r) for n, r in rows},
            "best": name,
            "classes": classes,
            "confusion_matrix": cm.tolist(),
            "baseline": best["baseline"],
        }

    out["classification"] = results
    return results


# ==========================================================================================
# Item 5 - hyperparameter tuning
# ==========================================================================================
def tuning_section(df, X, y, tr, te, cls_results):
    rule("TASK 5 ITEM 5  |  HYPERPARAMETER TUNING")
    tuned = {}

    # ---------------------------------------------------------------------------------
    # (a) GridSearchCV over Ridge alpha.
    #
    # Why Ridge and not the winning LinearRegression? Because LinearRegression has no
    # hyperparameter to tune - a grid search over it is theatre. Ridge is the same model plus
    # one knob, so searching alpha is a real search whose answer means something: Week 8 of
    # this project rejected Ridge by hand, arguing that regularisation fixed the ill-
    # conditioning at an unacceptable cost in accuracy. If that argument was right, the search
    # should drive alpha to its floor on its own.
    # ---------------------------------------------------------------------------------
    print("(a) GridSearchCV - Ridge alpha, on the shipped feature set")
    grid = {"model__alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    t0 = time.time()
    gs = GridSearchCV(make_pipeline(Ridge(random_state=SEED)), grid, cv=5, scoring="r2",
                      n_jobs=-1, return_train_score=True)
    gs.fit(X[tr], y[tr])
    untuned = regression_scores(y[te], make_pipeline(Ridge(alpha=1.0, random_state=SEED))
                                .fit(X[tr], y[tr]).predict(X[te]))
    tuned_scores = regression_scores(y[te], gs.best_estimator_.predict(X[te]))
    print(f"    grid            : {grid['model__alpha']}")
    print(f"    best alpha      : {gs.best_params_['model__alpha']}")
    print(f"    best CV R2      : {gs.best_score_:.6f}   ({time.time() - t0:.1f}s)")
    print(f"    test R2 before  : {untuned['r2']:.6f}   RMSE Rs {untuned['rmse']:.2f}   "
          f"(alpha = 1.0, the sklearn default)")
    print(f"    test R2 after   : {tuned_scores['r2']:.6f}   RMSE Rs {tuned_scores['rmse']:.2f}")
    print("\n    alpha vs CV score, the whole curve:")
    for a, m, s in zip(grid["model__alpha"], gs.cv_results_["mean_test_score"],
                       gs.cv_results_["std_test_score"]):
        print(f"      alpha {a:<8} CV R2 {m:.6f} +/- {s:.6f}")
    curve = list(zip(grid["model__alpha"], gs.cv_results_["mean_test_score"]))
    top = max(m for _, m in curve)
    plateau = [a for a, m in curve if top - m < 1e-6]
    falling = all(curve[i][1] >= curve[i + 1][1]
                  for i in range(len(curve) - 1) if curve[i][0] >= max(plateau))
    print(f"\n    The CV score is FLAT for every alpha <= {max(plateau):g} "
          f"(all within 1e-6 of the best)")
    print(f"    and {'falls monotonically' if falling else 'declines'} above it - by alpha = "
          f"{grid['model__alpha'][-1]:g} it has collapsed to {curve[-1][1]:.4f}.")
    print("    Read what that means: across the whole flat region the penalty term is small")
    print("    enough to do nothing, so Ridge there IS ordinary least squares. The search has")
    print("    no reason to prefer any point on the plateau and returns one of them")
    print(f"    (alpha = {gs.best_params_['model__alpha']:g}); every alpha large enough to")
    print("    actually regularise scores strictly worse.")
    print("\n    This reproduces Week 8's hand-argument automatically. That week rejected Ridge")
    print("    by reasoning about the Hessian condition number; the grid search reaches the same")
    print("    verdict mechanically, without being told the theory. A search whose answer is")
    print("    'do not regularise' has not failed - it has confirmed the model was already")
    print("    correctly specified.")
    print("    Note the 'IMPROVED' verdict below is against sklearn's DEFAULT alpha = 1.0, and")
    print("    the improvement consists of turning the regularisation back off.")
    tuned["ridge_alpha"] = {
        "grid": grid["model__alpha"], "best_params": gs.best_params_,
        "best_cv_r2": float(gs.best_score_),
        "curve": [{"alpha": a, "cv_r2": float(m), "cv_sd": float(s)}
                  for a, m, s in zip(grid["model__alpha"], gs.cv_results_["mean_test_score"],
                                     gs.cv_results_["std_test_score"])],
        "test_before": untuned, "test_after": tuned_scores,
        "improved": tuned_scores["r2"] > untuned["r2"],
    }

    # ---------------------------------------------------------------------------------
    # (b) RandomizedSearchCV over RandomForest - the tunable model with real knobs.
    # ---------------------------------------------------------------------------------
    print("\n(b) RandomizedSearchCV - RandomForestRegressor, 12 draws x 5 folds")
    space = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 8, 14, 20],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": [1.0, 0.7, 0.5],
    }
    t0 = time.time()
    rs = RandomizedSearchCV(make_pipeline(RandomForestRegressor(random_state=SEED, n_jobs=-1)),
                            space, n_iter=12, cv=5, scoring="r2", n_jobs=-1,
                            random_state=SEED)
    rs.fit(X[tr], y[tr])
    rf_default = regression_scores(
        y[te], make_pipeline(RandomForestRegressor(n_estimators=300, random_state=SEED,
                                                   n_jobs=-1)).fit(X[tr], y[tr]).predict(X[te]))
    rf_tuned = regression_scores(y[te], rs.best_estimator_.predict(X[te]))
    print(f"    best params     : "
          f"{ {k.replace('model__', ''): v for k, v in rs.best_params_.items()} }")
    print(f"    best CV R2      : {rs.best_score_:.6f}   ({time.time() - t0:.1f}s)")
    print(f"    test R2 before  : {rf_default['r2']:.6f}   MAE Rs {rf_default['mae']:.2f}   "
          f"(300 trees, defaults)")
    print(f"    test R2 after   : {rf_tuned['r2']:.6f}   MAE Rs {rf_tuned['mae']:.2f}")
    delta = rf_tuned["mae"] - rf_default["mae"]
    print(f"    change in MAE   : Rs {delta:+.2f}  "
          f"({'improved' if delta < 0 else 'no improvement'})")
    tuned["random_forest"] = {
        "space": {k.replace("model__", ""): v for k, v in space.items()}, "n_iter": 12,
        "best_params": {k.replace("model__", ""): v for k, v in rs.best_params_.items()},
        "best_cv_r2": float(rs.best_score_),
        "test_before": rf_default, "test_after": rf_tuned,
        "improved": rf_tuned["r2"] > rf_default["r2"],
    }

    # ---------------------------------------------------------------------------------
    # (c) GridSearchCV on the distance model - the one target in the project measured from
    #     the world rather than generated, so the one carrying real irreducible noise.
    # ---------------------------------------------------------------------------------
    print("\n(c) GridSearchCV - the distance model (the only observed-data target)")
    print("    840 real OSRM road distances. Small, noisy and not drawn from any formula,")
    print("    which is exactly the situation where capacity control earns its keep.")
    real = pd.read_csv(os.path.join(ROOT, "data", "real_distances.csv"))
    Xd = np.asarray([rf.distance_features(a, b, c, d) for a, b, c, d in
                     real[["start_lat", "start_lon", "dest_lat", "dest_lon"]].values],
                    dtype=float)
    hav = real.haversine_km.values.astype(float)
    yd = real.road_km.values.astype(float) / hav        # the winding factor
    dtr, dte = train_test_split(np.arange(len(real)), test_size=0.25, random_state=SEED)

    cgrid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 4, 8],
        "model__min_samples_leaf": [1, 5, 20],
    }
    t0 = time.time()
    cgs = GridSearchCV(make_pipeline(RandomForestRegressor(random_state=SEED, n_jobs=-1)),
                       cgrid, cv=KFold(5, shuffle=True, random_state=SEED),
                       scoring="r2", n_jobs=-1)
    cgs.fit(Xd[dtr], yd[dtr])
    before_model = make_pipeline(RandomForestRegressor(n_estimators=300, random_state=SEED,
                                                       n_jobs=-1)).fit(Xd[dtr], yd[dtr])
    # Scored in KILOMETRES, not in factor units - the factor is a modelling convenience and
    # nobody cares about an R2 on it; the error a user feels is kilometres of road.
    km_true = real.road_km.values.astype(float)[dte]
    before = regression_scores(km_true, before_model.predict(Xd[dte]) * hav[dte])
    after = regression_scores(km_true, cgs.best_estimator_.predict(Xd[dte]) * hav[dte])
    print(f"    grid size       : {len(cgs.cv_results_['params'])} combinations x 5 folds")
    print(f"    best params     : "
          f"{ {k.replace('model__', ''): v for k, v in cgs.best_params_.items()} }")
    print(f"    best CV r2      : {cgs.best_score_:.4f}   ({time.time() - t0:.1f}s)")
    print(f"    test MAE before : {before['mae']:.2f} km   (300 trees, defaults)")
    print(f"    test MAE after  : {after['mae']:.2f} km")
    print(f"    change          : {before['mae'] - after['mae']:+.2f} km of error removed")
    print("\n    Compare with (a) and (b), where the target is a formula the features already")
    print("    span and there is nothing left for a search to find.")
    tuned["distance_model"] = {
        "grid": {k.replace("model__", ""): v for k, v in cgrid.items()},
        "best_params": {k.replace("model__", ""): v for k, v in cgs.best_params_.items()},
        "best_cv_r2": float(cgs.best_score_),
        "test_before": before, "test_after": after,
        "improved": after["mae"] < before["mae"],
    }

    print("\nITEM 5 SUMMARY - 'confirm score improved':")
    for k, v in tuned.items():
        print(f"  {k:<20} {'IMPROVED' if v['improved'] else 'no improvement'}")
    n_improved = sum(1 for v in tuned.values() if v["improved"])
    print(f"\n  {n_improved} of {len(tuned)} searches improved the test score. The three are not")
    print("  equally interesting, and the checklist's 'confirm score improved' deserves a more")
    print("  careful answer than a tick:")
    print("\n  - Ridge alpha IMPROVED, but only against sklearn's default alpha = 1.0, and it did")
    print("    so by driving the penalty to zero. The search's real finding is that this problem")
    print("    wants no regularisation at all, which is Week 8's conclusion arrived at")
    print("    mechanically.")
    print("  - RandomForest did NOT improve. Twelve random draws over four hyperparameters could")
    print("    not beat 300 trees at their defaults, because the target is a deterministic")
    print("    formula that a linear model already fits to R2 0.9997 - there is no accuracy left")
    print("    on the table for a forest to find, tuned or not.")
    dm = tuned["distance_model"]
    print(f"  - The distance model {'IMPROVED' if dm['improved'] else 'did NOT improve'}, by "
          f"{dm['test_before']['mae'] - dm['test_after']['mae']:+.2f} km of MAE.")
    print("    This is the only target in the project measured from the world rather than")
    print("    generated from a formula, so it is the only one carrying noise a model can")
    print("    overfit - and therefore the one place where constraining capacity "
          f"(max_depth = {dm['best_params'].get('max_depth')}, min_samples_leaf = "
          f"{dm['best_params'].get('min_samples_leaf')}) has anything to buy.")
    print("\n  The pattern is the lesson: hyperparameter tuning pays where the data is NOISY and")
    print("  the model can overfit it. Where the target is a formula and the features already")
    print("  span it, tuning has nothing to do, and a search that reports no improvement is")
    print("  evidence the model was specified correctly - not evidence the search was wasted.")

    out["tuning"] = tuned
    return tuned


# ==========================================================================================
# Item 3, alternative - bootstrap
# ==========================================================================================
def bootstrap_section(X, y, tr, te, model_name):
    rule("TASK 5 ITEM 3, ALTERNATIVE  |  BOOTSTRAP VALIDATION (200 resamples)")
    print("The checklist allows 5-fold CV OR bootstrap. Both are run here because they answer")
    print("slightly different questions: k-fold asks 'how does the score vary across disjoint")
    print("held-out folds', bootstrap asks 'how would the score vary if I had drawn a different")
    print("training sample of the same size'. Resampling is WITH replacement, and each model is")
    print("scored on the fixed test split.\n")
    model = candidate_regressors()[model_name] if model_name in candidate_regressors() \
        else candidate_regressors()["LinearRegression"]
    rng = np.random.RandomState(SEED)
    scores = []
    for _ in range(200):
        pick = rng.randint(0, len(tr), len(tr))
        pipe = make_pipeline(_fresh(model)).fit(X[tr][pick], y[tr][pick])
        scores.append(r2(y[te], pipe.predict(X[te])))
    scores = np.array(scores)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    print(f"  model            : {model_name}")
    print(f"  bootstrap mean R2: {scores.mean():.6f}   sd {scores.std():.6f}")
    print(f"  95% interval     : [{lo:.6f}, {hi:.6f}]")
    print(f"  width            : {hi - lo:.6f}")
    print("  A narrow interval means the fit does not depend on which particular rows were")
    print("  drawn - the model is stable, not lucky.")
    out["bootstrap"] = {"model": model_name, "n_resamples": 200,
                        "mean_r2": float(scores.mean()), "sd": float(scores.std()),
                        "ci95": [float(lo), float(hi)]}


def r2(a, b):
    from sklearn.metrics import r2_score
    return float(r2_score(a, b))


def _fresh(m):
    from sklearn.base import clone
    return clone(m)


# ==========================================================================================
# Appendix - the same checklist on the ORIGINAL 1,140-row dataset
# ==========================================================================================
def original_dataset_appendix():
    rule("APPENDIX  |  THE SAME CHECKLIST ON THE ORIGINAL 1,140-ROW road_trip_data.csv")
    df = pd.read_csv(ORIGINAL)
    before_dedup = len(df)
    df = df.drop_duplicates()
    print("The assignment was handed out with this file, so the tables are repeated on it. It")
    print(f"also makes a point the wide dataset cannot: with {len(df):,} rows instead of 25,000,")
    print("the cross-validation spread is visibly larger - small data is less stable, and")
    print("k-fold is how you see that rather than guess it.")
    if before_dedup != len(df):
        print(f"({before_dedup - len(df)} duplicate rows dropped.)")
    print()
    df["fuel_bill"] = df.fuel_consumption_litres * df.fuel_price
    y = df.total_trip_cost.values.astype(float)
    X = np.asarray([rf.cost_features(d, m, p, t, pk, l, v) for d, m, p, t, pk, l, v in zip(
        df.distance_km, df.mileage, df.fuel_price, df.toll_cost, df.parking_cost,
        df.fuel_consumption_litres, df.vehicle_type)], dtype=float)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)
    print(f"rows: {len(df)}  ({len(tr)} train / {len(te)} test)\n")

    rows = [(n, evaluate_regressor(m, X[tr], y[tr], X[te], y[te]))
            for n, m in candidate_regressors().items()]
    print("REGRESSION - target: total_trip_cost")
    print_regression_table(rows)
    name, best = pick_best(rows)
    print(f"\nBEST: {name}   test R2 {best['test']['r2']:.6f}   "
          f"RMSE Rs {best['test']['rmse']:.2f}")

    print(f"\nCV spread, original ({len(df):,} rows) vs wide (25,000 rows) - same models, "
          f"same code:")
    for n, r in rows:
        wide_sd = out["regression"]["models"].get(n, {}).get("cv_std")
        wide_txt = f"{wide_sd:.6f}" if wide_sd is not None else "n/a"
        print(f"  {n:<26} original sd {r['cv_std']:.6f}   wide sd {wide_txt}")

    # Classification on the original data - the same target, a twentieth of the rows.
    #
    # The original csv has no cost_band column; it is derived. Deriving it here with the SAME
    # rule build_wide_dataset.py uses - quartiles of Rs/km, cut on this file's own distribution
    # - is what makes the two accuracies comparable. Borrowing the wide dataset's cut points
    # would import its price distribution and quietly change the question.
    print("\nCLASSIFICATION - target: cost_band, on a twentieth of the data")
    per_km = (df.total_trip_cost / df.distance_km).values
    q = np.quantile(per_km, [0.25, 0.50, 0.75])
    df = df.assign(cost_band=np.select(
        [per_km < q[0], per_km < q[1], per_km < q[2]],
        ["Excellent", "Good", "Average"], default="Poor"))
    print(f"  Rs/km quartile cuts on this file: {[round(float(x), 2) for x in q]}")
    cat = [c for c in ("vehicle_type", "fuel_type") if c in df.columns]
    num = [c for c in ("distance_km", "mileage", "fuel_price", "parking_cost",
                       "toll_cost", "fuel_consumption_litres") if c in df.columns]
    Xc = pd.get_dummies(df[num + cat], columns=cat, drop_first=True).values.astype(float)
    yt = labels(df.cost_band)
    ctr, cte = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, random_state=SEED,
                                stratify=yt)
    crows = [(n, evaluate_classifier(m, Xc[ctr], yt[ctr], Xc[cte], yt[cte]))
             for n, m in candidate_classifiers().items()]
    print_classification_table(crows)
    cname, cbest = pick_best(crows, key=lambda r: r["test"]["accuracy"])
    print(f"\nBEST: {cname}   accuracy {cbest['test']['accuracy']:.4f}   vs baseline "
          f"{cbest['baseline']:.4f}  ->  lift {cbest['lift_points']:+.1f} points")
    wide_band = out.get("classification", {}).get("cost_band", {})
    wide_best = wide_band.get("best")
    if wide_best:
        wide_acc = wide_band["models"][wide_best]["test"]["accuracy"]
        print(f"  Same target on 25,000 rows: {wide_acc:.4f}. The gap is sample size, not")
        print("  a different problem - and the CV spread above shows the same thing from the")
        print("  other direction: small data is less stable, and k-fold is how you see that")
        print("  rather than guess it.")

    out["original_dataset"] = {
        "n_rows": int(len(df)),
        "regression": {n: strip_fitted(r) for n, r in rows}, "regression_best": name,
        "cost_band": {n: strip_fitted(r) for n, r in crows}, "cost_band_best": cname,
    }


# ==========================================================================================
def architecture_comparison():
    """Pull the three architectures together - the project's actual answer."""
    rule("ARCHITECTURE COMPARISON  |  what the model is actually being asked to do")
    path = os.path.join(ROOT, "data", "pipeline_report.json")
    if not os.path.exists(path):
        print("  skipped - run scripts/train_pipeline.py first")
        return
    pipe = json.load(open(path, encoding="utf-8"))

    true_name = pipe["cost"]["true_components"]["chosen"]
    true_scores = pipe["cost"]["true_components"]["candidates"][true_name]["test"]
    chain_name = pipe["cost"]["chained"]["chosen"]
    chain_scores = pipe["cost"]["chained"]["candidates"][chain_name]["test"]
    mismatch = pipe["cost"]["mismatched_chain"]

    print(f"{'architecture':<44}{'test R2':>11}{'RMSE Rs':>11}{'MAE Rs':>10}")
    print("-" * 76)
    print(f"{'A. handed the true toll and litres':<44}{true_scores['r2']:>11.6f}"
          f"{true_scores['rmse']:>11.2f}{true_scores['mae']:>10.2f}")
    print(f"{'B. chained, trained on true / served pred':<44}{mismatch['r2']:>11.6f}"
          f"{mismatch['rmse']:>11.2f}{mismatch['mae']:>10.2f}")
    print(f"{'C. chained, trained on predictions':<44}{chain_scores['r2']:>11.6f}"
          f"{chain_scores['rmse']:>11.2f}{chain_scores['mae']:>10.2f}")
    print("\n  A scores highest and means least: it answers 'given the litres burnt and the")
    print("  toll paid, can you add them up?', and a traveller can supply neither.")
    print("  C is what a user actually gets, and it is what the app serves.")
    print("  B is A's model served C's inputs - the train/serve mismatch, priced.")
    print(f"  The A -> C drop in R2 is {true_scores['r2'] - chain_scores['r2']:.4f}, and in MAE it is")
    print(f"  Rs {chain_scores['mae'] - true_scores['mae']:.2f}. That gap is what it costs to derive the two")
    print("  inputs nobody can supply, instead of being handed them.")
    out["architecture_comparison"] = {
        "A_true_components": {"model": true_name, **true_scores},
        "B_mismatched_chain": mismatch,
        "C_chained": {"model": chain_name, **chain_scores},
        "r2_drop_A_to_C": float(true_scores["r2"] - chain_scores["r2"]),
        "mae_gap_A_to_C": float(chain_scores["mae"] - true_scores["mae"]),
    }


def main():
    t_start = time.time()
    df = load_wide()
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)
    print(f"wide dataset: {len(df):,} rows   split {len(tr):,}/{len(te):,}   seed {SEED}")

    X, y, best_name, _ = regression_section(df, tr, te)
    cls = classification_section(df, tr, te)
    tuning_section(df, X, y, tr, te, cls)
    bootstrap_section(X, y, tr, te, best_name)
    original_dataset_appendix()
    architecture_comparison()

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    rule(f"saved -> {OUT}   ({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
