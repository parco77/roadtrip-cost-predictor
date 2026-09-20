"""
Append Weeks 4-8 to RoadTripCost.ipynb.

The notebook ended at Week 3, so none of the project's actual findings were in the graded
artifact. This adds them as documented, runnable cells. Idempotent: re-running replaces the
appended chapters rather than duplicating them.

Run:  python scripts/build_notebook_chapters.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "RoadTripCost.ipynb")
MARKER = "<!-- WEEK4PLUS -->"


# nbformat stores `source` as a list of lines where every line KEEPS its trailing newline
# (except the last). Splitting on "\n" and dropping them makes Jupyter render the whole cell
# as a single concatenated line — markdown headings, tables and code all collapse.
def _lines(text):
    return text.splitlines(keepends=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip())}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text.strip("\n")),
    }


CELLS = []

# ============================================================ WEEK 4
CELLS += [
    md(f"""
{MARKER}
---

# Week 4: The Dataset Is Synthetic — Recovering the Cost Formula

Week 3 ended with `R2 = 0.9928` and a note that the remaining 0.7% was structural. That number
was suspiciously high, so this week asks a different question: **not "how well can we predict
the target" but "what actually generated it?"**

The approach is deliberate. Subtract the cost components we already know about and study what
is left. If the residual is structured rather than random, the data came from a formula.
"""),
    code("""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

raw = pd.read_csv('road_trip_data.csv')

# Peel off the three components we can compute exactly
raw['fuel_cost'] = raw.fuel_consumption_litres * raw.fuel_price
raw['residual']  = raw.total_trip_cost - raw.fuel_cost - raw.toll_cost - raw.parking_cost

print(raw.residual.describe().round(2).to_string())
print(f"\\nresidual vs distance correlation: {raw.residual.corr(raw.distance_km):.4f}")
"""),
    md("""
The residual is never negative, never zero, and correlates **+0.85 with distance**. That is not
noise — something proportional to distance is missing. Dividing the residual by distance gives a
per-kilometre rate, and grouping that by each categorical column shows which one controls it.
"""),
    code("""
raw['residual_per_km'] = raw.residual / raw.distance_km

for col in ['vehicle_type', 'fuel_type', 'traffic_level', 'weather_condition']:
    stats = raw.groupby(col).residual_per_km.agg(['mean', 'std']).round(4)
    print(f"--- {col} ---"); print(stats.to_string()); print()
"""),
    md("""
**Found it.** The rate is flat across fuel, traffic and weather, but splits cleanly by
`vehicle_type`:

| Vehicle | ₹/km |
|---|---|
| Hatchback | 0.5694 |
| Sedan | 0.6943 |
| SUV | 0.8366 |

That is a per-kilometre wear-and-tear charge that scales with vehicle size. The full formula:

$$\\text{total} = \\text{litres} \\times \\text{price} + \\text{toll} + \\text{parking} + \\text{distance} \\times \\text{rate}(\\text{vehicle})$$

Let's verify it reconstructs the target.
"""),
    code("""
RATE = {'Hatchback': 0.5694, 'Sedan': 0.6943, 'SUV': 0.8366}

reconstructed = (raw.fuel_consumption_litres * raw.fuel_price
                 + raw.toll_cost + raw.parking_cost
                 + raw.distance_km * raw.vehicle_type.map(RATE))

err = raw.total_trip_cost - reconstructed
r2  = 1 - (err**2).sum() / ((raw.total_trip_cost - raw.total_trip_cost.mean())**2).sum()

print(f"R2 of reconstruction : {r2:.6f}")
print(f"mean abs error       : Rs {err.abs().mean():.2f}")
print(f"max abs error        : Rs {err.abs().max():.2f}")
"""),
    md("""
**R² = 0.999162 from arithmetic alone — no model fitted.**

This changes how Week 3's result should be read. `R² = 0.9928` was not evidence of a strong
model; the target is a near-deterministic function of the features, so almost any reasonable
regressor scores high. The dataset is **synthetic**.

One more relationship falls out the same way — fuel consumption itself is derived:
"""),
    code("""
raw['ideal_litres'] = raw.distance_km / raw.mileage
raw['burn_ratio']   = raw.fuel_consumption_litres / raw.ideal_litres

print(raw.groupby('traffic_level').burn_ratio.agg(['mean','min','max']).round(4).to_string())
print()
print("weather has no effect:")
print(raw.groupby('weather_condition').burn_ratio.mean().round(4).to_string())
"""),
    md("""
$$\\text{litres} = \\frac{\\text{distance}}{\\text{mileage}} \\times m(\\text{traffic}),
\\qquad m = \\{\\text{Low}: 0.944,\\ \\text{Medium}: 1.045,\\ \\text{High}: 1.165\\}$$

Weather is irrelevant — all four conditions sit at 1.04–1.05.

**Why this matters practically.** `fuel_consumption_litres` is the strongest predictor
(corr +0.96), but no user planning a trip could ever supply it. Because it is *derived*, an
application can compute it from distance, mileage and traffic instead of asking. That single
insight is what makes a usable product possible.
"""),
]

# ============================================================ WEEK 5
CELLS += [
    md("""
---

# Week 5: Scaling from 9 Routes to 537 Cities

The original data covers **9 fixed city pairs**, 110–460 km. A model trained on it cannot serve
an arbitrary Indian route. Since Week 4 recovered the generating formula, we can regenerate the
same physics over **real geography**.

| Layer | Source | Real or modelled? |
|---|---|---|
| Cities + coordinates | GeoNames `cities15000` (CC BY 4.0) | **Real** — 3,739 Indian cities |
| Road distance | haversine × winding factor | **Calibrated** on our own 9 routes |
| Fuel price | per-state table | 6 states verified, 29 estimated |
| Cost | Week 4 formula | Recovered, R² 0.999162 |

The winding factor is the interesting part: rather than assume a value, we measure it against
the 9 routes we already have ground truth for.
"""),
    code("""
import math

cities = pd.read_csv('data/india_cities.csv')

def haversine(a, b):
    R = 6371.0088
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(h))

# Key by name only, but keep the LARGEST match. India has several Udaipurs (Rajasthan and
# Tripura among them); taking whichever row happened to come last put Ahmedabad -> Udaipur at
# 1,929 km instead of 208. This is the same rule the API uses to resolve an ambiguous name.
biggest = cities.sort_values('population', ascending=False).drop_duplicates('city')
coords = {r.city: (r.lat, r.lon) for r in biggest.itertuples()}
factors = []
print(f"{'route':<28}{'straight':>10}{'actual':>10}{'factor':>9}")
for (s, t), g in raw.groupby(['start_city', 'destination_city']):
    straight = haversine(coords[s], coords[t]); actual = g.distance_km.mean()
    factors.append(actual / straight)
    print(f"{s+' -> '+t:<28}{straight:>10.1f}{actual:>10.1f}{actual/straight:>9.3f}")

print(f"\\nwinding factor: mean {np.mean(factors):.4f}  sd {np.std(factors):.4f}")
"""),
    md("""
**1.2355**, standard deviation 0.082 — squarely inside the 1.2–1.3 range reported for real road
networks, and cross-checked against the OSRM routing engine, which agreed within 5–9% on the
same routes.

`scripts/build_wide_dataset.py` applies this to 537 cities (population ≥ 100k) and produces
**25,000 trips across ~5,000 routes**. Two deliberate choices are documented there:

1. Routes are sampled with a short-trip bias, because uniformly random Indian city pairs are
   mostly 900 km apart while real road trips are not. 48% of the result falls inside the
   original 110–460 km band.
2. Traffic is given a **real rush-hour profile**. In the original data traffic is statistically
   independent of departure hour, which made it unpredictable. This is a modelling choice, made
   openly, and it is what turns traffic classification into a genuine task rather than decoration.
"""),
    code("""
wide = pd.read_csv('data/road_trip_wide.csv')
print(f"trips  : {len(wide):,}")
print(f"cities : {len(set(wide.start_city) | set(wide.destination_city))}")
print(f"\\n{wide.distance_km.describe(percentiles=[.25,.5,.75]).round(1).to_string()}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(wide.distance_km, bins=50, color='steelblue', edgecolor='white')
ax[0].axvspan(110, 460, color='orange', alpha=0.25, label='original data range')
ax[0].set_xlabel('distance (km)'); ax[0].set_ylabel('trips'); ax[0].legend()
ax[0].set_title('Wide dataset covers, and extends past, the original range')

hourly = pd.crosstab(wide.departure_hour, wide.traffic_level, normalize='index')
hourly.plot(ax=ax[1]); ax[1].set_title('P(traffic | departure hour) — the injected rush-hour profile')
ax[1].set_xlabel('departure hour'); ax[1].set_ylabel('probability')
plt.tight_layout(); plt.show()
"""),
]

# ============================================================ WEEK 6
CELLS += [
    md("""
---

# Week 6: One Engineered Feature Beats Two Ensemble Models

Week 3 identified the linear model's structural blind spot:

> the real fuel bill is the *product* `litres × price`, and a linear model can only **add** its
> features together, never multiply them.

That is a testable claim. If the missing product is really the bottleneck, then supplying it as
a feature should beat throwing a nonlinear model at the problem — because a tree ensemble has to
*approximate* a product it can never represent exactly, whereas we can just hand it over.
"""),
    code("""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

BASE = ['distance_km','mileage','fuel_price','toll_cost','parking_cost','fuel_consumption_litres']
wide['fuel_bill'] = wide.fuel_consumption_litres * wide.fuel_price

Xb, y = wide[BASE], wide.total_trip_cost
Xb_tr, Xb_te, y_tr, y_te = train_test_split(Xb, y, test_size=0.2, random_state=42)

results = []
for name, model in [('Linear (no interaction)', LinearRegression()),
                    ('RandomForest (300)', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
                    ('GradientBoosting', GradientBoostingRegressor(random_state=42))]:
    model.fit(Xb_tr, y_tr); p = model.predict(Xb_te)
    results.append((name, r2_score(y_te, p), mean_absolute_error(y_te, p)))

# same linear model, one extra column
Xi = wide[BASE + ['fuel_bill']]
Xi_tr, Xi_te, _, _ = train_test_split(Xi, y, test_size=0.2, random_state=42)
m = LinearRegression().fit(Xi_tr, y_tr); p = m.predict(Xi_te)
results.append(('Linear + litres x price', r2_score(y_te, p), mean_absolute_error(y_te, p)))

print(f"{'model':<26}{'test R2':>12}{'test MAE':>12}")
for n, r, mae in results:
    print(f"{n:<26}{r:>12.6f}{mae:>12.2f}")
"""),
    md("""
**The prediction holds.** One engineered feature takes MAE from ₹143.85 to ₹42.60 — a **3.4×
improvement** — and beats both ensembles, which had access to the same information but had to
learn the product implicitly.

### 6.1 One more feature the model was missing

Week 4 found that the per-km wear rate splits by vehicle: Hatchback 0.5694, Sedan 0.6943,
SUV 0.8366 — a 47% spread. None of the models above can see it, because `vehicle_type` is
categorical and was never encoded. That means they return the **same cost for an SUV and a
hatchback on an identical route**, which is both wrong and obviously wrong to any user."""),
    code("""
# One-hot, with Hatchback as the baseline
wide['vehicle_type_SUV']   = (wide.vehicle_type == 'SUV').astype(float)
wide['vehicle_type_Sedan'] = (wide.vehicle_type == 'Sedan').astype(float)

DEPLOYED = BASE + ['fuel_bill', 'vehicle_type_SUV', 'vehicle_type_Sedan']
Xd = wide[DEPLOYED]
Xd_tr, Xd_te, _, _ = train_test_split(Xd, y, test_size=0.2, random_state=42)

final = LinearRegression().fit(Xd_tr, y_tr); pd_ = final.predict(Xd_te)
print(f"with vehicle one-hots : R2 {r2_score(y_te, pd_):.6f}   MAE Rs {mean_absolute_error(y_te, pd_):.2f}")
print()
print("learned per-vehicle offsets (relative to Hatchback):")
for name, coef in zip(['SUV','Sedan'], final.coef_[-2:]):
    print(f"  {name:<6} Rs {coef:+8.2f}")
"""),
    md("""
Encoding it improves MAE again **and** makes the model respond to a choice the interface was
already collecting. This is the model that ships.""" + """

Two things worth drawing out:

1. **Domain understanding beat model complexity.** The win came from knowing what the number
   *means*, not from a bigger hypothesis class.
2. **The model stays linear.** Everything from Week 3 — the closed-form solution, the
   from-scratch gradient descent, the interpretable coefficients — still applies. A Random
   Forest would have cost all of that *and* performed worse.

This is the model the application serves.
"""),
]

# ============================================================ WEEK 7
CELLS += [
    md("""
---

# Week 7: Classification — and Being Honest About Which One Is Real

The task asked for classification where it adds value. Two targets are available, and they are
**not** of equal quality. Saying so is part of the work.

### 7.1 Traffic level — the genuinely non-trivial one

First, why this was impossible on the original dataset:
"""),
    code("""
orig_ct = pd.crosstab(raw.departure_hour, raw.traffic_level, normalize='index')
print("P(traffic | hour) in the ORIGINAL data — range across hours:")
print(orig_ct.agg(['min','max']).round(3).to_string())
print(f"\\nmajority-class baseline: {raw.traffic_level.value_counts(normalize=True).max():.3f}")
"""),
    md("""
P(High) wanders between 0.13 and 0.35 with no rush-hour structure — traffic is independent of
departure hour, so no classifier could beat the 51% majority baseline. Any accuracy reported on
that data would have been meaningless.

The wide dataset injects a real rush-hour profile (Week 5), which makes the task learnable:
"""),
    code("""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

Xt = pd.DataFrame({
    'hour_sin':  np.sin(2*np.pi*wide.departure_hour/24),
    'hour_cos':  np.cos(2*np.pi*wide.departure_hour/24),
    'is_peak':   (wide.departure_hour.between(8,11) | wide.departure_hour.between(17,21)).astype(float),
    'month_sin': np.sin(2*np.pi*wide.month/12),
    'month_cos': np.cos(2*np.pi*wide.month/12),
})
yt = wide.traffic_level
Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(Xt, yt, test_size=0.2, random_state=42, stratify=yt)

clf = LogisticRegression(max_iter=2000).fit(Xt_tr, yt_tr)
pred = clf.predict(Xt_te)
baseline = yt_te.value_counts(normalize=True).max()

print(f"accuracy : {accuracy_score(yt_te, pred):.4f}")
print(f"baseline : {baseline:.4f}   lift: +{(accuracy_score(yt_te, pred)-baseline)*100:.1f} points")
print()
print(classification_report(yt_te, pred, digits=3))
"""),
    md("""
**53.9% against a 38.8% baseline — a real signal, but a modest one.** That is the honest result
and it is the correct one: departure hour genuinely only *partly* determines traffic. A
classifier claiming 95% here would be a sign of leakage, not skill.

Note the feature encoding: hour is cyclical, so it is fed in as `sin`/`cos` rather than as the
integer 0–23. Otherwise the model would treat 23:00 and 00:00 as maximally distant.

### 7.2 Cost band — useful, but high by construction

The second target is a quartile split of ₹/km. **Cost per kilometre, not absolute cost** —
absolute cost is essentially determined by distance, so classifying it would tell a user nothing.
Efficiency isolates the effect of vehicle, mileage, fuel and traffic.
"""),
    code("""
from sklearn.preprocessing import MinMaxScaler

BAND_NUM = BASE
BAND_CAT = ['vehicle_type','fuel_type','traffic_level']
Xb2 = pd.get_dummies(wide[BAND_NUM + BAND_CAT], columns=BAND_CAT, drop_first=True)
# Scale before logistic regression, matching the deployed pipeline in scripts/train_models.py.
# Without it the features span wildly different ranges and the reported accuracy drifts.
Xb2 = pd.DataFrame(MinMaxScaler().fit_transform(Xb2.values.astype(float)), columns=Xb2.columns)
yb  = wide.cost_band

Xb2_tr, Xb2_te, yb_tr, yb_te = train_test_split(Xb2, yb, test_size=0.2, random_state=42, stratify=yb)
band = LogisticRegression(max_iter=2000).fit(Xb2_tr, yb_tr)
print(f"accuracy : {accuracy_score(yb_te, band.predict(Xb2_te)):.4f}   (4 balanced classes, baseline 0.25)")
"""),
    md("""
**85.8% against a 25% baseline** — but this is high *by construction*, not because the problem is
hard. `cost_band` is a discretisation of a quantity the Week 6 regressor already predicts to
within ₹42. It earns its place because the interface needs the label, not because it demonstrates
difficult learning. **The traffic classifier is the one that shows real modelling.**
"""),
]

# ============================================================ WEEK 8
CELLS += [
    md("""
---

# Week 8: Why Gradient Descent Stopped Converging

Week 3 found that sklearn and from-scratch gradient descent agreed to within 0.03% on 1,129
rows. Re-running the identical algorithm on the 25,000-row model tells a different story.
"""),
    code("""
from sklearn.preprocessing import MinMaxScaler

FEATURES = DEPLOYED
Xs = MinMaxScaler().fit_transform(wide[FEATURES].values.astype(float))
ys = wide.total_trip_cost.values.astype(float)
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.2, random_state=42)

def gradient_descent(X, Y, lr=0.4, epochs=20_000, alpha=0.0):
    '''Identical to the Week 3 implementation, with an optional L2 term.'''
    n, k = X.shape; w = np.zeros(k); b = 0.0
    for _ in range(epochs):
        error = X.dot(w) + b - Y
        w -= lr * ((2/n) * X.T.dot(error) + 2*alpha*w/n)
        b -= lr * (2/n) * error.sum()
    return w, b

closed_form = LinearRegression().fit(Xs_tr, ys_tr)
w_gd, b_gd  = gradient_descent(Xs_tr, ys_tr, epochs=20_000)

print(f"largest weight gap vs closed form : {np.abs(w_gd - closed_form.coef_).max():.2f}")
print(f"largest |weight| in the solution   : {np.abs(closed_form.coef_).max():.0f}")
print(f"predictions still agree? max gap   : Rs {np.abs(Xs_te.dot(w_gd)+b_gd - closed_form.predict(Xs_te)).max():.2f}")
"""),
    md("""
The **predictions** agree closely, but the **weights** are thousands apart. Gradient descent has
not converged, and the reason is visible in the data: `fuel_bill` is by construction almost
exactly `litres × price`.
"""),
    code("""
print(f"corr(litres, fuel_bill) = {np.corrcoef(wide.fuel_consumption_litres, wide.fuel_bill)[0,1]:.6f}")

# GD's convergence rate is governed by the condition number of the objective's Hessian,
# which for the ridge objective is (2/n)(X^T X + alpha*I).
sv = np.linalg.svd(Xs_tr, compute_uv=False)
kappa = lambda a: (sv[0]**2 + a) / (sv[-1]**2 + a)
print(f"condition number of the OLS Hessian: {kappa(0.0):,.0f}")
"""),
    md("""
**κ ≈ 11,000.** Gradient descent's error shrinks by roughly a factor of $(1 - 1/\\kappa)$ per
step along the slowest direction, so a condition number of that size means tens of thousands of
epochs just to traverse it — while the closed-form solution, which inverts the matrix directly,
does not care about conditioning at all.

Ridge regression adds $\\alpha\\|w\\|^2$ to the loss, which adds $\\alpha$ to every eigenvalue of
the Hessian and pulls the condition number down. The question is what that costs.
"""),
    code("""
from sklearn.linear_model import Ridge

rows = []
for a in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
    ref = LinearRegression().fit(Xs_tr, ys_tr) if a == 0 else Ridge(alpha=a).fit(Xs_tr, ys_tr)
    w, b = gradient_descent(Xs_tr, ys_tr, epochs=20_000, alpha=a)
    rows.append({'alpha': a, 'cond(H)': kappa(a),
                 'GD gap @20k': np.abs(w - ref.coef_).max(),
                 'test MAE': mean_absolute_error(ys_te, ref.predict(Xs_te)),
                 'max|w|': np.abs(ref.coef_).max()})

sweep = pd.DataFrame(rows)
print(sweep.round(2).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(sweep.alpha + 1e-3, sweep['cond(H)'], 'o-', color='crimson')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('alpha'); ax[0].set_ylabel('condition number'); ax[0].grid(alpha=.3)
ax[0].set_title('Regularisation conditions the problem')
ax[1].plot(sweep.alpha + 1e-3, sweep['GD gap @20k'] + 1e-3, 'o-', label='GD gap (lower = converged)')
ax[1].plot(sweep.alpha + 1e-3, sweep['test MAE'], 's-', label='test MAE (lower = accurate)')
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('alpha')
ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title('...but accuracy is the price')
plt.tight_layout(); plt.show()
"""),
    md("""
## Conclusion — and why we do *not* ship Ridge

The two curves move in opposite directions. Raising `alpha` drives the condition number from
~11,000 down to ~220 and lets gradient descent converge essentially exactly — while test MAE
climbs from **₹39.66 to ₹256.41**, six times worse.

So Ridge does exactly what the theory promises, and it is still the wrong choice here:

| Question | Answer |
|---|---|
| Does regularisation fix the conditioning? | **Yes** — κ falls ~50× |
| Does it make gradient descent converge? | **Yes** — gap 4113 → 0.00 |
| Should we use it for this model? | **No** |

The reason is that we do not actually need gradient descent. The normal equations solve this
model exactly, in one step, and are indifferent to the condition number. Gradient descent was an
*exercise* in understanding what `fit()` does — and the exercise has now taught something more
useful than "the two methods agree":

> **An ill-conditioned problem is not a broken model.** It is a problem where iterative solvers
> struggle and direct solvers do not. Reaching for Ridge here would trade real accuracy to fix a
> difficulty that only exists because of the solver we chose.

The model this week settles on is therefore ordinary least squares with the Week 6 interaction
term, solved in closed form: **R² 0.9997, MAE ₹39.66** across 537 cities.

> **Read on before quoting that number.** Week 9 evaluates it properly and Week 10 shows what it
> was measuring: this regressor is handed the true `toll_cost`, `parking_cost` and
> `fuel_consumption_litres` as input features, and a user supplies none of them. The figure that
> describes a real prediction is **R² 0.9645, MAE ₹391**, and that is what the app now serves.

---

## Project summary — as of Week 8

> These are the Week 8 figures. Weeks 9 and 10 supersede several of them: the traffic and
> cost-band accuracies are re-measured on a single common split, and the headline cost accuracy
> is restated as the chained pipeline's **R² 0.9645 / MAE ₹391**. See
> [`TASK5.md`](TASK5.md) for the consolidated tables.


| | |
|---|---|
| Original data | 1,140 trips, 9 city pairs, 110–460 km |
| Recovered formula | R² **0.999162** from arithmetic alone — data is synthetic |
| Wide dataset | **25,000** trips, **537** cities, 50–1500 km |
| Best regressor | Linear + `litres × price` — R² **0.9996**, MAE **₹39.66** |
| Beat | RandomForest (₹69.97) and GradientBoosting (₹73.50) |
| Traffic classifier | 53.9% vs 38.8% baseline — real but modest |
| Cost-band classifier | 85.8% vs 25% baseline — high by construction |
| Diagnosis | κ ≈ 11,000; GD unsuited, closed form correct |

### Known limitations

1. The wide dataset is **generated**, not observed — physically consistent with the original
   data, but not independent evidence.
2. Fuel prices: **6 states verified, 29 estimated** (`data/fuel_prices.csv` records which).
3. The rush-hour traffic profile was **injected deliberately** in Week 5; the traffic
   classifier's accuracy is a measure of that design, not of Indian roads.
4. Predictions beyond 1,500 km extrapolate; the application warns when this happens.
"""),
]


def main():
    nb = json.load(open(NB, encoding="utf-8"))

    # idempotent: drop any previously appended chapters
    cut = next((i for i, c in enumerate(nb["cells"])
                if MARKER in "".join(c.get("source", []))), None)
    if cut is not None:
        removed = len(nb["cells"]) - cut
        nb["cells"] = nb["cells"][:cut]
        print(f"removed {removed} previously appended cells")

    before = len(nb["cells"])
    nb["cells"].extend(CELLS)
    json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"cells: {before} -> {len(nb['cells'])}  (+{len(CELLS)})")
    print(f"wrote {NB}")


if __name__ == "__main__":
    main()
