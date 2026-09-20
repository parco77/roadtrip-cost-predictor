"""
Phase C - the end-to-end model, which exists to be honest rather than to win.

THE QUESTION IT ANSWERS
-----------------------
The project advertises R2 = 0.9997 for its cost regressor. But that regressor is handed
`fuel_consumption_litres`, `toll_cost` and `parking_cost` as inputs - and those are three of the
four terms in the cost formula. A fair examiner asks: predicting WHAT, exactly?

This script trains a model on ONLY what a traveller could actually know before the trip:

    start city, destination city        -> real latitude / longitude (a lookup, not a formula)
    vehicle type, fuel type
    departure hour, month
    passengers
    mileage, fuel price                 -> things you can read off your own car and pump

No distance. No toll. No litres. No parking. The model has to do all of it.

The gap between this model's R2 and the 0.9997 is the headline number of the whole project: it
is how much of that 0.9997 was arithmetic being handed back to the model.

IT ALSO REPEATS THE WEEK 6 LESSON
---------------------------------
The true cost contains  (distance / mileage) * fuel_price  - a product of three inputs. An
additive model cannot express a product, so two feature sets are compared:

    base          the raw inputs only
    base + 2      plus  haversine/mileage  and  (haversine/mileage) * fuel_price

If Week 6's argument is right, those two extra columns should matter more than switching to a
tree ensemble does. That is a prediction this script tests rather than assumes.

Run:  python scripts/train_end_to_end.py
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import roadtrip_features as rf                                              # noqa: E402
from evaluation import (SEED, candidate_regressors, evaluate_regressor,      # noqa: E402
                        pick_best, print_regression_table, rule, strip_fitted)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDE = os.path.join(ROOT, "data", "road_trip_wide.csv")
CITIES = os.path.join(ROOT, "data", "india_cities.csv")
OUT_MODEL = os.path.join(ROOT, "models", "end_to_end.joblib")
OUT_REPORT = os.path.join(ROOT, "data", "end_to_end_report.json")

TEST_SIZE = 0.2


def main():
    df = pd.read_csv(WIDE)
    cities = rf.load_city_index(CITIES)
    df["start_lat"] = [cities[c][1] for c in df.start_city]
    df["start_lon"] = [cities[c][2] for c in df.start_city]
    df["dest_lat"] = [cities[c][1] for c in df.destination_city]
    df["dest_lon"] = [cities[c][2] for c in df.destination_city]

    y = df.total_trip_cost.values.astype(float)
    idx = np.arange(len(df))
    # Identical split to scripts/train_pipeline.py, so the three architectures are compared on
    # exactly the same test rows.
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)

    args = df[["start_lat", "start_lon", "dest_lat", "dest_lon", "mileage", "fuel_price",
               "passengers", "departure_hour", "month", "vehicle_type", "fuel_type"]].values

    report = {"n_rows": int(len(df)), "n_train": int(len(tr)), "n_test": int(len(te))}
    results = {}

    for label, interactions, names in [
            ("base inputs only", False, rf.END_TO_END_BASE),
            ("base + 2 engineered", True, rf.END_TO_END_FEATURES)]:
        rule(f"END-TO-END MODEL - {label}   ({len(names)} features)")
        X = np.asarray([rf.end_to_end_features(*a, interactions=interactions) for a in args],
                       dtype=float)
        rows = [(n, evaluate_regressor(m, X[tr], y[tr], X[te], y[te]))
                for n, m in candidate_regressors().items()]
        print_regression_table(rows)
        name, best = pick_best(rows)
        print(f"\n  CHOSEN : {name}   test R2 {best['test']['r2']:.6f}   "
              f"RMSE Rs {best['test']['rmse']:.2f}   MAE Rs {best['test']['mae']:.2f}")
        results[label] = {"features": names,
                          "candidates": {n: strip_fitted(r) for n, r in rows},
                          "chosen": name, "chosen_scores": strip_fitted(best),
                          "fitted": best["fitted"], "X": X}

    # ------------------------------------------------------------------ the comparison
    rule("DOES THE WEEK 6 ARGUMENT HOLD END-TO-END?")
    base = results["base inputs only"]
    eng = results["base + 2 engineered"]
    lin_base = base["candidates"]["LinearRegression"]["test"]
    lin_eng = eng["candidates"]["LinearRegression"]["test"]
    rf_base = base["candidates"]["RandomForest (300)"]["test"]
    print(f"  linear, base features        : R2 {lin_base['r2']:.6f}   MAE Rs {lin_base['mae']:.2f}")
    print(f"  linear, + 2 engineered       : R2 {lin_eng['r2']:.6f}   MAE Rs {lin_eng['mae']:.2f}")
    print(f"  RandomForest, base features  : R2 {rf_base['r2']:.6f}   MAE Rs {rf_base['mae']:.2f}")
    from_features = lin_base["mae"] - lin_eng["mae"]
    from_model = lin_base["mae"] - rf_base["mae"]
    print(f"\n  MAE saved by two engineered columns : Rs {from_features:.2f}")
    print(f"  MAE saved by switching to 300 trees : Rs {from_model:.2f}")
    verdict = ("features win - Week 6 holds end-to-end" if from_features > from_model
               else "the ensemble wins here - Week 6's claim does NOT generalise to this setup")
    print(f"  VERDICT: {verdict}")
    report["week6_check"] = {
        "linear_base": lin_base, "linear_engineered": lin_eng, "rf_base": rf_base,
        "mae_saved_by_features": float(from_features),
        "mae_saved_by_model": float(from_model), "verdict": verdict,
    }

    # ------------------------------------------------------- what the model actually uses
    rule("FEATURE IMPORTANCE - which inputs carry the prediction")
    best_overall_label = max(results, key=lambda k: results[k]["chosen_scores"]["test"]["r2"])
    best_pipe = results[best_overall_label]["fitted"]
    names = results[best_overall_label]["features"]
    model = best_pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        kind = "tree impurity importance"
    else:
        # For a linear model on MinMax-scaled columns, |coefficient| is directly comparable
        # across features because every column spans the same 0-1 range.
        imp = np.abs(model.coef_)
        kind = "|coefficient| on 0-1 scaled columns"
    order = np.argsort(imp)[::-1]
    print(f"  from {results[best_overall_label]['chosen']} ({best_overall_label}) - {kind}\n")
    total = imp.sum() or 1.0
    for i in order[:12]:
        bar = "#" * max(1, int(round(imp[i] / imp[order[0]] * 40)))
        print(f"    {names[i]:<26}{imp[i] / total * 100:>6.2f}%  {bar}")
    shares = {names[i]: imp[i] / total * 100 for i in range(len(names))}
    dead = [names[i] for i in order if imp[i] / total < 0.001]
    if dead:
        print(f"\n  carrying essentially nothing (<0.1%): {', '.join(dead)}")

    # Read the ranking carefully - two entries need explaining, and one of them is a trap.
    print("\n  WHAT THE RANKING SAYS")
    print(f"  Distance dominates: haversine and its log together carry "
          f"{shares.get('haversine_km', 0) + shares.get('log_haversine', 0):.0f}% of the")
    print("  prediction, which is right - cost is mostly fuel, and fuel is mostly distance.")
    print(f"  mileage is next at {shares.get('mileage', 0):.1f}%: it divides into the fuel term.")

    vf = [c for c in ("vehicle_Sedan", "vehicle_SUV", "fuel_Diesel", "fuel_CNG") if c in shares]
    if vf and max(shares[c] for c in vf) < 0.5:
        print("\n  TRAP: the vehicle and fuel one-hots score near zero, and it would be wrong to")
        print("  conclude vehicle type does not matter. Week 4 recovered a real per-km")
        print("  maintenance rate that differs by vehicle (Hatchback 0.5694, Sedan 0.6943, SUV")
        print("  0.8366 Rs/km), and Week 6 showed that dropping those columns made the app quote")
        print("  an identical price for an SUV and a hatchback on the same route. The rate")
        print("  spread is about 0.27 Rs/km - roughly Rs 108 on a 400 km trip - and end-to-end")
        print("  that signal sits underneath the unpredictable toll and parking noise, which is")
        print("  hundreds of rupees. So the importance is low because the NOISE here is large,")
        print("  not because the effect is absent. It is exactly why the app serves the chained")
        print("  pipeline, where distance and litres are pinned down first and the vehicle term")
        print("  becomes visible again.")

    if "passengers" in shares:
        print(f"\n  passengers: {shares['passengers']:.2f}%. The recovered cost formula has no")
        print("  passenger term at all, so anything above zero here is the forest fitting noise")
        print("  - a useful reminder that impurity importance is never exactly zero for a")
        print("  column with many distinct values.")
    report["feature_importance"] = {
        "source": f"{results[best_overall_label]['chosen']} / {best_overall_label}",
        "kind": kind,
        "ranked": [{"feature": names[i], "share_pct": float(imp[i] / total * 100)}
                   for i in order],
        "near_zero": dead,
    }

    # ------------------------------------------------------------------------------ save
    winner_label = best_overall_label
    winner = results[winner_label]

    # The winner here is a 300-tree RandomForest fitted to 20,000 rows, which pickles to about
    # 546 MB uncompressed. Nothing needs that file: this model is a BENCHMARK, not a service.
    # app.py serves the chained pipeline, and the notebook's Week 10 reads its numbers from the
    # JSON report below. So it is written compressed, and the two callers that might want it are
    # told plainly that re-running this script rebuilds it in about two minutes.
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump({
        "model": winner["fitted"],
        "features": winner["features"],
        "interactions": winner_label == "base + 2 engineered",
        "chosen": winner["chosen"],
        "metrics": winner["chosen_scores"],
        "note": ("Benchmark model - NOT served by app.py, and not needed to read Week 10 of "
                 "the notebook (that uses data/end_to_end_report.json). Regenerate with "
                 "python scripts/train_end_to_end.py"),
    }, OUT_MODEL, compress=3)

    report["winner"] = {"feature_set": winner_label, "model": winner["chosen"],
                        "scores": winner["chosen_scores"]}
    for k in results:
        results[k].pop("fitted", None)
        results[k].pop("X", None)
    report["variants"] = results
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    rule(f"saved -> {OUT_MODEL}\nsaved -> {OUT_REPORT}")
    print(f"\nHONEST HEADLINE: from user inputs alone, {winner['chosen']} reaches "
          f"R2 {winner['chosen_scores']['test']['r2']:.4f}, "
          f"MAE Rs {winner['chosen_scores']['test']['mae']:.2f}.")


if __name__ == "__main__":
    main()
