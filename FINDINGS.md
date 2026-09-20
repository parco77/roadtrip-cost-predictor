# Findings — working notes

Evidence and dead ends behind the decisions in the notebook. [`README.md`](README.md) is the
front door; this is the log of how the numbers were arrived at, including the things that
did not work.

## 1. The cost formula is fully recovered (R² = 0.999162, mean abs err ₹18.44)

`road_trip_data.csv` is **synthetic**, generated from:

```
total_trip_cost = fuel_consumption_litres × fuel_price
                + toll_cost
                + parking_cost
                + distance_km × maintenance_rate(vehicle_type)

maintenance_rate (₹/km, ±~20% noise):  Hatchback 0.5694 · Sedan 0.6943 · SUV 0.8366
fuel_consumption_litres = (distance_km / mileage) × traffic_multiplier
traffic_multiplier:  Low 0.944 · Medium 1.045 · High 1.165   (weather has NO effect)
estimated_time_minutes/km:  Low 0.919 · Medium 1.059 · High 1.263
toll ≈ 1.30 ₹/km (σ 0.36)      parking ∈ {0, 40, 60, 80, 120, 150}
```

**Implication:** R² = 0.9928 measures formula recovery, not predictive skill. It also means a wide
dataset over 500+ real cities can be generated *physically consistently* using this same formula.

## 2. One engineered feature beats every black-box model

**Measured on the ORIGINAL 1,140-row dataset** — all four rows from the same data, so they are
comparable. (The wide-dataset version of this table is in §7; do not mix the two.)

| Model | Test R² | Test MAE |
|---|---|---|
| LinearRegression | 0.993377 | ₹47.43 |
| RandomForest (300 trees) | 0.989125 | ₹62.26 |
| GradientBoosting | 0.993483 | ₹46.37 |
| **Linear + `fuel_consumption_litres × fuel_price`** | **0.998438** | **₹23.42** |

Adding the interaction term the notebook already names as the structural blind spot ("a linear model
can only add") **halves MAE and beats both ensembles**, while staying linear — so the Week 3
gradient-descent-from-scratch code still applies unchanged. Do this instead of switching to an ensemble.

## 3. Traffic level is NOT predictable from departure_hour

P(High) by hour ranges 0.13–0.35 with no rush-hour structure. Base rates: Medium 0.511, Low 0.249,
High 0.240. A classifier would score ≈ the 51% majority baseline. **Traffic classification on the
current data is decoration — reject it** unless rush-hour realism is deliberately built into the
generated wide dataset (and disclosed).

## 4. Verified available data

- Network access works from this environment.
- `https://raw.githubusercontent.com/nshntarora/Indian-Cities-JSON/master/cities.json`
  — **1221 Indian cities, verified fetched**, fields: `id`, `name`, `state`. **No lat/lon** — so it
  is not sufficient on its own; a coordinate source is still needed.

## 5. Route constraint in current data

9 fixed pairs, distances 110–460 km:
Delhi→Jaipur · Pune→Mumbai · Ahmedabad→Surat · Ahmedabad→Udaipur · Ahmedabad→Vadodara ·
Rajkot→Ahmedabad · Bengaluru→Mysuru · Surat→Mumbai · Vadodara→Mumbai

---

## 6. Data sources — checked and resolved

| Need | Source | Status |
|---|---|---|
| Cities + coordinates | GeoNames `cities15000.zip` (CC BY 4.0) | ✅ **3739 Indian cities**, 35 states, real lat/lon |
| Road distance | OSRM public demo (`router.project-osrm.org`) | ✅ works; agreed with the project's own routes within 5–9% |
| Road distance fallback | haversine × **1.2355** | ✅ factor measured on the project's own 9 routes (σ 0.082) |
| Fuel prices | no free current feed exists | ⚠️ 6 states verified, 29 estimated — see `data/fuel_prices.csv` `source` column |
| Toll | modelled from the original data | ✅ **1.301 ₹/km**, uniform across vehicle classes |

Rejected fuel-price sources: `anshikakaythwas/fuel-prices-india-api` (stale — Pune petrol ₹87.13,
a ~2021 price), Zyla API (paid, needs a key), energy.thecore.in (JS-rendered, not machine-readable).

## 7. Built and verified

```
data/india_cities.csv    3739 cities, real lat/lon
data/fuel_prices.csv     35 states
data/road_trip_wide.csv  25,000 trips over 534 cities, 50-1500 km (median 412)
models/model.joblib      regressor + 2 classifiers + scalers + metrics
app.py                   FastAPI: /api/cities, /api/predict, /api/metrics, /api/health
```

**Model results on the wide dataset (test set):**

| Model | Result | Baseline |
|---|---|---|
| Cost regression | R² **0.999707** · RMSE ₹54.16 · MAE **₹39.66** | ₹143.85 without the interaction term |
| Cost-band classifier | **85.8%** | 25% (4 balanced classes) |
| Traffic classifier | **53.9%** | 38.8% majority → **+15.1 pts** |

**Ablation re-run on the wide dataset** (all four from the 25,000-row data — the numbers the UI
and the notebook both quote):

| Model | Test R² | Test MAE |
|---|---|---|
| Linear, no interaction | 0.994666 | ₹143.85 |
| RandomForest (300 trees) | 0.998543 | ₹69.97 |
| GradientBoosting | 0.998560 | ₹73.50 |
| **Linear + `litres × price`** | **0.999707** | **₹39.66** |

The conclusion survives the change of dataset: the engineered term still beats both ensembles.

Backend verified end to end: live OSRM routing, rush-hour traffic inference, correct 400/404/422
error handling, extrapolation warnings firing beyond 1500 km, and the predicted breakdown
reconciling against the recovered formula (Sedan wear-and-tear ₹214.00 predicted vs ₹211.13 exact).

## 8. Frontend (React 19 + Vite 7 + Tailwind v4)

Built and verified in-browser. `frontend/` builds into `static/`, which `app.py` mounts at `/`.

```bash
uvicorn app:app          # then open http://127.0.0.1:8000
```

Verified live: city autocomplete (keyboard + aria), prediction end to end
(Mumbai→Pune ₹1,580 = 1217 fuel + 189 toll + 104 wear + 70 parking), extrapolation and
estimated-fuel-price warnings both rendering, no horizontal scroll at 375/768/1440, every touch
target ≥ 44px, contrast 7.36:1 on muted text and 17.85:1 on headings, zero console errors.

### The non-obvious bug worth remembering

`gsap.from(el, {opacity: 0})` writes its START state synchronously but advances on
`requestAnimationFrame`, which browsers do not fire in a tab that is not compositing. Loading the
page in a **background tab left the hero headline, driver columns and torn edge permanently
invisible**, and froze the cost figure at ₹0.

Fix: `frontend/src/lib/motion.js` exports `shouldAnimate()`, which returns false for
`prefers-reduced-motion` **or** `document.hidden`. When false, components render the finished
state directly instead of animating toward it. Every entrance animation must go through it.

## 9. Gradient descent on the wide dataset — an honest non-result

Re-running the Week 3 from-scratch gradient descent on 25,000 rows:

| | Value |
|---|---|
| R² sklearn vs. gradient descent | 0.999707 vs **0.999701** |
| Largest prediction gap | **₹82.81** |
| Largest weight gap | **682.72** |
| Epochs | 60,000 (did **not** early-stop) |

Predictions agree; **weights do not converge**. `fuel_bill` is by construction ≈
`fuel_consumption_litres × fuel_price`, so those columns are near-collinear, the loss surface is a
long narrow valley, and gradient descent drifts slowly along the flat direction while sklearn jumps
straight to one solution via the normal equations. Both are correct — they pick different points in
the same valley. This is exactly the ill-conditioning the notebook's §3.2 flags, now demonstrated
rather than argued. Surfaced in the UI via `/api/loss-curve`, not hidden.

Measured divergence threshold on this data is **lr ≈ 0.45** (0.5 → `inf`), closely matching the
~0.44 ceiling the notebook found on the original CSV. Training uses 0.4.

---

## 10. The winding factor is nearly unbiased and still wrong (840 real routes)

`scripts/fetch_real_distances.py` collected **840 real road distances** from OSRM over a
stratified sample of city pairs, 40–1,248 km straight-line. This is the only *observed* data in
the project — everything in `road_trip_wide.csv` is generated.

Observed winding factor (road ÷ straight): **mean 1.2520, sd 0.1448, range 1.0437 – 3.6015**,
against the hardcoded `1.2355`.

| | Value |
|---|---|
| mean signed error of the constant | **−3.3 km (−0.4%)** ← nearly unbiased |
| mean **absolute** error | **33.4 km (6.3%)** ← per-route |
| worst overshoot / undershoot | +155.3 km / −235.3 km (−65.7%) |

**Both numbers matter, and they say opposite-sounding things.** Averaged over hundreds of routes
the constant is within half a percent of the truth, which is why it survived review. But a user
takes *one* trip, and on that trip the errors do not cancel.

The extremes are real, not dirty data. **Surat → Bhavnagar is 94.2 km straight and 339.4 km by
road** (factor 3.60) because the Gulf of Khambhat is in between; Moradabad → Chanduasi is 1.04.
No single multiplier serves both, so the rows are kept and are the clearest argument for a model.

Result: `RandomForest`, predicting the *factor* and multiplying by haversine, reaches
**R² 0.9889, MAE 30.67 km** against the constant's 34.33 km — a **10.7%** reduction. Deliberately
not oversold: endpoint coordinates only partly determine route shape, since the rest depends on
where roads and bridges happen to be.

Parameterising on the factor beat predicting kilometres directly. Removing trip length from the
target lets the model spend capacity on the unknown part (route shape) instead of re-learning
"longer line, longer road" — the same idea as the Week 6 interaction term.

## 11. Parking is unpredictable — a negative result worth reporting

Six model families fitted to `parking_cost`, **all with test R² ≤ 0**:

| Model | Test R² | RMSE |
|---|---|---|
| LinearRegression / Ridge | −0.0009 | ₹50.09 |
| DecisionTree | **−1.0046** | ₹70.88 |
| RandomForest (300) | −0.2602 | ₹56.20 |
| AdaBoost | −0.0014 | ₹50.10 |
| GradientBoosting | −0.0062 | ₹50.22 |
| *always the train mean (₹75.09)* | *−0.0010* | *₹50.09* |

The generator draws parking uniformly from {0, 40, 60, 80, 120, 150} independently of every other
column, so it carries no signal. **Do not tune this — report it.** The API serves the training
mean and discloses the failure via `/api/metrics`. Note `DecisionTree` at train R² 0.9888 against
test −1.00: pure memorisation of noise, and the cleanest overfitting example in the project.

Incidentally, the old hardcoded ₹70.0 was not even the mean of the training data.

## 12. The chained pipeline, and what R² 0.9997 was measuring

Replacing the five constants with sub-models and chaining them, using out-of-fold
`cross_val_predict` for the training components (i.e. proper stacking):

| Architecture | Test R² | RMSE | MAE |
|---|---|---|---|
| A. fed true toll / parking / litres | 0.999707 | ₹54.16 | ₹39.66 |
| B. chained, trained on true / served predicted | 0.964602 | ₹595.37 | ₹390.56 |
| **C. chained, trained on predictions — shipped** | **0.964542** | **₹595.88** | **₹391.16** |
| D. end-to-end, user inputs only | 0.850960 | ₹1221.67 | ₹644.65 |

**A → C is the project's honest headline.** The R² drop of 0.0352 (MAE ×10) is the price of the
model deriving its own inputs instead of being handed three of the four terms of the cost formula.

**B vs C is a result I expected to go the other way.** The train/serve mismatch — fitting on true
components and serving predicted ones — cost essentially *nothing* (₹390.56 vs ₹391.16, with B
marginally ahead). The reason is specific: the sub-model errors are close to zero-mean, so the
linear cost model's coefficients barely change when refitted on them. C still ships because it is
the correct construction, but **on this data the mismatch did not bite**, and claiming otherwise
would be inventing a result.

### The error is the data, not the model

| Source | sd |
|---|---|
| fuel bill, via litres | ₹479.22 |
| toll (injected noise) | ₹218.77 |
| parking (uniform noise) | ₹50.06 |
| **quadrature sum** | **₹529.17** |
| **chained model RMSE** | **₹595.88** |

Ratio 1.13 — the chain sits near the floor this dataset allows. The largest term (fuel bill)
inherits the distance error, so it is the only one a better distance model could still improve;
toll and parking are drawn from random distributions by construction.
`tests/test_api.py::test_metrics_error_budget_explains_the_remaining_error` pins that ratio to
0.7–1.6, so a regression in the chain fails a test instead of quietly getting worse.

## 13. "Features beat complexity" has a boundary condition

Week 6 showed one engineered feature beating two ensembles. End-to-end (architecture D) it
**reverses**:

| | MAE saved |
|---|---|
| two engineered columns (`hav/mileage`, `× price`) | ₹67.05 |
| switching to a 300-tree RandomForest | **₹474.03** |

Not a contradiction — a boundary. With distance *given*, the only inexpressible thing was a
product, so handing over the product closed the one gap. End-to-end, the model must infer road
distance from four coordinates, which is a non-linear function of geography (coastlines, the Gulf
of Khambhat). No finite set of hand-made interactions encodes a map; partitioning space does.

**Feature engineering wins when the missing structure is something you can write down. When the
missing structure is a map, it does not.**

One trap in the importance ranking: the vehicle and fuel one-hots score **below 0.1%**
end-to-end. That is *not* evidence vehicle type is irrelevant — the per-km maintenance spread is
about 0.27 ₹/km (≈₹108 on 400 km), which sits underneath hundreds of rupees of toll and parking
noise. Pin distance and litres down first (the chained pipeline) and the vehicle term is visible
again.

## 14. Task 5 — two evaluation bugs found in this project's own code

Both were in the selection and reporting logic, not in the models:

1. **`pick_best` compared models on R² with a relative tolerance.** Near R² = 0.99 a 0.5%
   relative band is wide enough to swallow a model with 15% more RMSE; it preferred a Ridge fit
   at 53.7 km RMSE over a random forest at 46.6 km. Fixed by judging closeness on the **error**
   (2% of the best RMSE, or 0.5 accuracy points), then applying the **one-standard-error rule**
   (Hastie, Tibshirani & Friedman §7.10) and preferring the simplest survivor.
2. **`/api/predict` quoted an unachievable error bar.** `typical_error` was ₹39.66 — the MAE
   measured with the true toll, parking and litres handed in. No real prediction could achieve
   it. It is now ₹391.16, the chained pipeline's own held-out MAE, with a regression test
   (`test_the_error_bar_beside_a_prediction_describes_that_prediction`) asserting it tracks the
   served figure and exceeds the fed-components one.

Also fixed: `MinMaxScaler` was fitted on the whole matrix before `train_test_split`, leaking
test-set minima and maxima into training. Every model is now wrapped in
`Pipeline([MinMaxScaler, model])`, so the scaler is re-fitted on each CV fold.

Tuning outcome, honestly: **2 of 3 searches improved.** `GridSearchCV` over Ridge α drove α to
the bottom of the grid — reproducing Week 8's hand-argument mechanically, so its "improvement" is
against sklearn's default α = 1.0 and consists of turning regularisation off.
`RandomizedSearchCV` on RandomForest could **not** improve on 300 default trees. Only the traffic
classifier gained anything real (+0.92 accuracy points, at `max_depth=4`) — the one target with
genuine noise to overfit.

## 15. Model artefacts: 227 MB → 5.1 MB

The first `models/pipeline.joblib` came out at **227 MB**, replacing a 10 KB file. A 300-tree
RandomForest on 20,000 rows pickles to about 210 MB, and `distance_dataset_model` — the
deliberately circular model that exists only to keep the chained evaluation apples-to-apples —
was essentially all of it.

Compression alone was not the answer; the question was which models `app.py` actually loads. It
needs the six sub-models and the chained regressor, and never touches the two evaluation-only
models. Splitting them out and compressing gives **5.1 MB served**, plus 63 MB of evaluation
apparatus that is git- and docker-ignored and rebuilt by re-running the script. The end-to-end
benchmark (546 MB uncompressed) is treated the same way — Week 10 of the notebook reads its
numbers from `data/end_to_end_report.json`, not from the fitted forest.
