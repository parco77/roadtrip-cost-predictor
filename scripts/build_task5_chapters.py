"""
Append Weeks 9 and 10 to RoadTripCost.ipynb.

  Week 9  - Task 5: model evaluation, validation and hyperparameter tuning
  Week 10 - Removing the magic numbers: a chained model pipeline

Idempotent: re-running replaces the appended chapters rather than duplicating them.

ORDER MATTERS
-------------
scripts/build_notebook_chapters.py rebuilds everything from Week 4 onwards, so it wipes these
chapters. Run them in this order:

    python scripts/build_notebook_chapters.py     # Weeks 4-8
    python scripts/build_task5_chapters.py        # Weeks 9-10

The cells here read the JSON reports written by the training scripts, so those must have run:

    python scripts/fetch_real_distances.py
    python scripts/train_pipeline.py       -> data/pipeline_report.json
    python scripts/train_end_to_end.py     -> data/end_to_end_report.json
    python scripts/evaluate_models.py      -> data/task5_evaluation.json

Run:  python scripts/build_task5_chapters.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "RoadTripCost.ipynb")
MARKER = "<!-- WEEK9PLUS -->"
WEEK4_MARKER = "<!-- WEEK4PLUS -->"


def _lines(text):
    # nbformat keeps the trailing newline on every line except the last; splitting them off
    # makes Jupyter render the whole cell as one concatenated line.
    return text.splitlines(keepends=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip())}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": _lines(text.strip("\n"))}


CELLS = []

# ==========================================================================================
#                                        WEEK 9
# ==========================================================================================
CELLS += [
    md(f"""
{MARKER}
---

# Week 9 — Task 5: Evaluation, Validation and Tuning

Weeks 1–8 built models. This week measures them properly, using the same six steps for every
model in the project:

| Step | Regression | Classification |
|---|---|---|
| 1. Metrics | RSS, RMSE, R² | Accuracy, Precision, Recall, F1 |
| 2. Fit diagnosis | train score vs test score | same |
| 3. Validation | 5-fold cross-validation (+ bootstrap) | 5-fold stratified |
| 4. Comparison | one table, best score **and** stable CV | same |
| 5. Tuning | `GridSearchCV` / `RandomizedSearchCV` | same |
| 6. Advanced models | RandomForest (bagging), AdaBoost, GradientBoosting | same |

Two things are set up once and reused, because a comparison is only meaningful if everything
in it was measured the same way.

**One split for everything.** `train_test_split(..., test_size=0.2, random_state=42)` gives
20,000 training rows and 5,000 test rows, and every table below is scored on those same 5,000
rows.

**Every model wrapped in a `Pipeline([MinMaxScaler, model])`.** Linear and logistic models need
the scaling and trees do not care, so the reason to do it for all of them is cross-validation
correctness: inside a `Pipeline`, the scaler is re-fitted on each training fold, so nothing
from the validation fold reaches it. Scaling the whole matrix once *before* splitting — which
`scripts/train_models.py` originally did — lets test-set minima and maxima leak into training.
The effect is small for `MinMaxScaler` on this data, but it is exactly the mistake
cross-validation exists to catch, so it is fixed rather than excused.

All of the scoring lives in [`scripts/evaluation.py`](scripts/evaluation.py) so the numbers
here, in `TASK5.md` and in the API are the same numbers.
"""),
    code("""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
sys.path.insert(0, os.getcwd())
import roadtrip_features as rf
from evaluation import (SEED, candidate_regressors, candidate_classifiers,
                        evaluate_regressor, evaluate_classifier, labels,
                        regression_scores, make_pipeline, pick_best)

plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.25,
                     'axes.spines.top': False, 'axes.spines.right': False})

# The reports written by the training scripts. Re-deriving all of them in the notebook would
# take about 8 minutes; the expensive searches are loaded, the tables below are recomputed live.
TASK5    = json.load(open('data/task5_evaluation.json', encoding='utf-8'))
PIPELINE = json.load(open('data/pipeline_report.json', encoding='utf-8'))
E2E      = json.load(open('data/end_to_end_report.json', encoding='utf-8'))

print('reports loaded:', list(TASK5))
"""),

    md("""
## 9.1 The metrics, and what each one is for

Four numbers for regression, and they are not interchangeable.

$$\\text{RSS} = \\sum_i (y_i - \\hat{y}_i)^2
\\qquad
\\text{RMSE} = \\sqrt{\\frac{\\text{RSS}}{n}}
\\qquad
R^2 = 1 - \\frac{\\text{RSS}}{\\sum_i (y_i - \\bar{y})^2}$$

- **RSS** is the raw quantity least squares actually minimises. It is a *sum*, so it grows
  with the number of rows: comparable between models on the same split, meaningless across
  splits of different sizes. That is why it is reported here but never used to compare the
  25,000-row dataset with the 1,140-row one.
- **RMSE** is RSS per row, square-rooted, so it reads in rupees. Squaring inside means one
  ₹2,000 miss counts as much as a hundred ₹200 misses — it is the metric to quote when large
  errors are what hurt.
- **MAE** is the average error in rupees with no squaring. It is the number to tell a user.
- **R²** is the fraction of variance explained. Its weakness is exactly what this project ran
  into: when the target is nearly deterministic, R² sits so close to 1 that differences which
  matter in rupees become invisible. Two models at R² 0.9997 and 0.9992 look identical and
  differ by 60% in RMSE.

For classification, accuracy alone is not a result — it needs its baseline:

$$\\text{Precision} = \\frac{TP}{TP+FP}
\\qquad
\\text{Recall} = \\frac{TP}{TP+FN}
\\qquad
F_1 = 2\\cdot\\frac{P \\cdot R}{P + R}$$

Precision, recall and F1 are **macro-averaged** (each class weighted equally) rather than
weighted, because weighted averaging can hide a class the model never predicts at all.
"""),

    md("""
## 9.2 Step 1 & 4 — the regression comparison table

Target: `total_trip_cost`. Features: the model this project already ships (Week 6) — distance,
mileage, fuel price, toll, parking, litres, the `litres × price` interaction and the vehicle
one-hots. Every row of the table sees exactly those columns, so this compares **models**, not
feature sets.

The Week 6 ablation is included as the first row — the same linear model with the interaction
term removed — because it is the honest baseline the project's headline number beats.
"""),
    code("""
wide = pd.read_csv('data/road_trip_wide.csv')
cities = rf.load_city_index('data/india_cities.csv')
for side, col in [('start', 'start_city'), ('dest', 'destination_city')]:
    wide[f'{side}_lat'] = [cities[c][1] for c in wide[col]]
    wide[f'{side}_lon'] = [cities[c][2] for c in wide[col]]

from sklearn.model_selection import train_test_split
y = wide.total_trip_cost.values.astype(float)
X = np.asarray([rf.cost_features(d, m, p, t, pk, l, v) for d, m, p, t, pk, l, v in zip(
    wide.distance_km, wide.mileage, wide.fuel_price, wide.toll_cost, wide.parking_cost,
    wide.fuel_consumption_litres, wide.vehicle_type)], dtype=float)
tr, te = train_test_split(np.arange(len(wide)), test_size=0.2, random_state=SEED)
print(f'{len(tr):,} train / {len(te):,} test')

reg_rows = []
keep = [i for i, n in enumerate(rf.COST_FEATURES) if n != 'fuel_bill']
reg_rows.append(('Linear, NO interaction',
                 evaluate_regressor(candidate_regressors()['LinearRegression'],
                                    X[:, keep][tr], y[tr], X[:, keep][te], y[te])))
for name, model in candidate_regressors().items():
    reg_rows.append((name, evaluate_regressor(model, X[tr], y[tr], X[te], y[te])))

table = pd.DataFrame([{
    'model': n, 'test R2': r['test']['r2'], 'RMSE': r['test']['rmse'],
    'MAE': r['test']['mae'], 'RSS': r['test']['rss'],
    'train R2': r['train']['r2'], 'CV mean': r['cv_mean'], 'CV sd': r['cv_std'],
    'verdict': r['verdict'],
} for n, r in reg_rows])
display(table.style.format({'test R2': '{:.6f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}',
                            'RSS': '{:.4g}', 'train R2': '{:.6f}', 'CV mean': '{:.6f}',
                            'CV sd': '{:.6f}'}).hide(axis='index'))
best_name, best = pick_best(reg_rows)
print(f'BEST: {best_name}  (best test score with a stable CV result)')
"""),
    md("""
**Read the table from the RMSE column, not the R² column.** Every model except AdaBoost scores
R² > 0.998, which invites the conclusion that the choice does not matter. In rupees it clearly
does: ₹54 for the linear model against ₹88 for GradientBoosting and ₹399 for AdaBoost — the
worst model is **seven times** worse than the best while looking like a rounding difference in
R².

The single largest effect in the table is not a model at all. Removing one engineered column
(`fuel_bill = litres × price`) takes RMSE from ₹54 to ₹228. That gap is larger than the gap
between the best and worst *model*, which is Week 6's argument restated: an additive model
cannot express a product, and no amount of model complexity substitutes for handing it over.
"""),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))

names = [n for n, _ in reg_rows]
rmse  = [r['test']['rmse'] for _, r in reg_rows]
order = np.argsort(rmse)
colours = ['#c0392b' if names[i] == 'Linear, NO interaction'
           else ('#2e7d32' if names[i] == best_name else '#7f8c8d') for i in order]
ax[0].barh([names[i] for i in order], [rmse[i] for i in order], color=colours)
ax[0].set_xlabel('test RMSE (Rs)  -  lower is better')
ax[0].set_title('Same R2 to three decimals, sevenfold apart in rupees')
ax[0].invert_yaxis()
for yi, i in enumerate(order):
    ax[0].text(rmse[i] + 6, yi, f'{rmse[i]:.0f}', va='center', fontsize=9)

# R2 on its own axis, to show why it is the wrong lens here
r2 = [r['test']['r2'] for _, r in reg_rows]
ax[1].barh([names[i] for i in order], [r2[i] for i in order], color=colours)
ax[1].set_xlim(0.98, 1.0)
ax[1].set_xlabel('test R2  -  note the axis starts at 0.98')
ax[1].set_title('The same models judged by R2: almost indistinguishable')
ax[1].invert_yaxis()
plt.tight_layout(); plt.show()
"""),

    md("""
## 9.3 Step 2 — overfitting and underfitting

Task 5 gives three verdicts. To be reproducible they need stated thresholds, so
`scripts/evaluation.py` fixes them:

| Condition | Verdict |
|---|---|
| train R² − test R² > 0.05 | Overfitting |
| both R² < 0.50 | Underfitting |
| otherwise | Good fit |

For classification, "underfitting" additionally covers a model that beats the majority-class
baseline by less than 3 points — a model that has learned nothing useful is underfit no matter
how well its two splits agree.
"""),
    code("""
fig, ax = plt.subplots(figsize=(9, 4.4))
idx = np.arange(len(reg_rows))
w = 0.38
ax.bar(idx - w/2, [r['train']['r2'] for _, r in reg_rows], w, label='train R2', color='#5b8dd6')
ax.bar(idx + w/2, [r['test']['r2']  for _, r in reg_rows], w, label='test R2',  color='#e08a3c')
ax.set_xticks(idx); ax.set_xticklabels([n for n, _ in reg_rows], rotation=30, ha='right')
ax.set_ylim(0.97, 1.001); ax.set_ylabel('R2  (axis starts at 0.97)')
ax.set_title('Train vs test - the gap is what matters, not the height')
ax.legend(loc='lower right')
for i, (n, r) in enumerate(reg_rows):
    gap = r['train']['r2'] - r['test']['r2']
    if gap > 0.0005:
        ax.annotate(f'gap {gap:.4f}', (i, r['train']['r2']), textcoords='offset points',
                    xytext=(0, 6), ha='center', fontsize=8, color='#c0392b')
plt.tight_layout(); plt.show()

for n, r in reg_rows:
    print(f"{n:<26} train {r['train']['r2']:.6f}  test {r['test']['r2']:.6f}  "
          f"gap {r['train']['r2'] - r['test']['r2']:+.6f}  -> {r['verdict']}")
"""),
    md("""
Every model on this target reads "Good fit", and the reason is worth stating because it is a
property of the **data**, not evidence of careful modelling.

`DecisionTree` reaches **train R² = 1.000000**. A tree grown to purity has memorised all 20,000
training rows — the textbook overfitting signature. It nevertheless holds test R² = 0.9986,
because the target here is a deterministic formula: memorising the training set and learning
the rule produce nearly the same predictions when there is almost no noise to memorise.

So on this target the train/test comparison is close to uninformative. It becomes informative
the moment there is real noise, which is exactly what happens in two places later in this
notebook — the toll sub-model (§10.4) and the `cost_band` classifier below, where the same
trees show gaps of 0.16–0.21 and are correctly flagged.
"""),

    md("""
## 9.4 Step 3 — 5-fold cross-validation, and why the spread is the point

A single train/test split gives one number, and that number depends on which rows happened to
land in the test set. K-fold splits the *training* data into k parts, trains on k−1 and
validates on the remaining one, k times. Two things come out: a mean (a better estimate than
one split) and a **standard deviation across folds** — which is what says whether the model is
stable or merely lucky.
"""),
    code("""
names = [n for n, _ in reg_rows]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))

# Left: the unexplained variance (1 - R2) per fold, on a log axis. Plotting R2 directly is
# useless here - every value rounds to 1.00 on any sane axis. 1 - R2 spreads them out and is
# still exactly the same information, just read as "how much is left over".
ax[0].boxplot([[1 - s for s in r['cv_scores']] for _, r in reg_rows],
              tick_labels=names, widths=0.55, medianprops=dict(color='#c0392b'))
ax[0].set_yscale('log'); ax[0].set_ylabel('1 - R2 per fold  (log, lower is better)')
ax[0].set_xticklabels(names, rotation=30, ha='right')
ax[0].set_title('5-fold CV, per-fold unexplained variance')

# Right: the spread itself, which is what step 3 is actually asking about.
ax[1].bar(names, [r['cv_std'] for _, r in reg_rows], color='#5b8dd6')
ax[1].set_yscale('log'); ax[1].set_ylabel('CV standard deviation (log)')
ax[1].set_xticklabels(names, rotation=30, ha='right')
ax[1].set_title('Stability: spread across the 5 folds')
plt.tight_layout(); plt.show()

print(f"{'model':<26}{'CV mean':>11}{'CV sd':>11}{'fold range':>13}")
for n, r in reg_rows:
    print(f"{n:<26}{r['cv_mean']:>11.6f}{r['cv_std']:>11.6f}"
          f"{max(r['cv_scores']) - min(r['cv_scores']):>13.6f}")
"""),
    md("""
Every spread is tiny — CV standard deviations of 1e-5 to 4e-4. With 20,000 training rows and a
near-deterministic target that is the expected result, and it is worth saying plainly: **this
cross-validation confirms stability, it does not discriminate between the models.**

It earns its place twice over anyway. §9.9 repeats it on the original 1,140-row dataset, where
the spreads are 4–20× wider and small-sample instability becomes visible instead of assumed.
And the spread is what settles which model to ship, via the rule below.

### Choosing between models that are statistically tied

Task 5 asks for "best score **and** stable cross-validation result", which needs to be made
precise. `pick_best` applies three filters in order:

1. **Test-error band** — keep models within 2% of the best RMSE (or 0.5 accuracy points).
   Closeness is judged on *error*, never on R²: a first version of this function used "R²
   within 0.5% relative", and near R² = 0.99 that band is wide enough to swallow a model with
   15% more RMSE. It duly preferred a Ridge fit at 53.7 km RMSE over a random forest at
   46.6 km.
2. **One-standard-error rule** — of those, keep every model whose CV mean is within one
   standard error ($\\mathrm{sd}/\\sqrt{k}$) of the best CV mean. This is the standard rule from
   Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* §7.10: a difference
   smaller than the noise in the CV estimate itself is not a real difference, so it is wrong to
   choose on score at all.
3. **Simplicity** — of the survivors, take the simplest. Here that is not just Occam's razor:
   a linear model keeps the Week 3 gradient-descent-from-scratch code applicable, which no
   ensemble does.
"""),

    md("""
## 9.5 Step 3, alternative — bootstrap validation

The checklist allows 5-fold CV **or** bootstrap. Both are run, because they answer different
questions. K-fold asks *how does the score vary across disjoint held-out folds*. Bootstrap asks
*how would the score vary if I had drawn a different training sample of the same size* —
resampling the training rows with replacement, refitting, and scoring each fit on the fixed
test split.
"""),
    code("""
boot = TASK5['bootstrap']
print(f"model            : {boot['model']}")
print(f"resamples        : {boot['n_resamples']}")
print(f"bootstrap mean R2: {boot['mean_r2']:.6f}   sd {boot['sd']:.8f}")
print(f"95% interval     : [{boot['ci95'][0]:.6f}, {boot['ci95'][1]:.6f}]")
print(f"width            : {boot['ci95'][1] - boot['ci95'][0]:.8f}")
print()
print('The interval is about 3e-6 wide. The fit does not depend in any meaningful way on')
print('which particular rows were drawn - the model is stable, not lucky.')
"""),

    md("""
## 9.6 Steps 1–4 for classification

Two targets, and they are **not** of equal quality. Saying which is which is part of the work.

- **`traffic_level`** — the genuinely non-trivial one. Departure hour only partly determines
  traffic, so the ceiling is low and an honest result looks modest.
- **`cost_band`** — a quartile split of ₹/km, which the Week 6 regressor already predicts well.
  Its high accuracy is **by construction**. It earns its place because the interface needs the
  label, not because it demonstrates difficult learning.
"""),
    code("""
cls_tables = {}
for target, Xc in [
        ('traffic_level',
         np.asarray([rf.traffic_features(h, m)
                     for h, m in zip(wide.departure_hour, wide.month)], dtype=float)),
        ('cost_band', None)]:
    if Xc is None:
        num = ['distance_km','mileage','fuel_price','toll_cost','parking_cost',
               'fuel_consumption_litres']
        cat = ['vehicle_type','fuel_type','traffic_level']
        Xc = pd.get_dummies(wide[num + cat], columns=cat, drop_first=True).values.astype(float)
    yc = labels(wide[target])
    rows = [(n, evaluate_classifier(m, Xc[tr], yc[tr], Xc[te], yc[te]))
            for n, m in candidate_classifiers().items()]
    cls_tables[target] = (rows, Xc, yc)

    t = pd.DataFrame([{
        'model': n, 'accuracy': r['test']['accuracy'],
        'precision': r['test']['precision_macro'], 'recall': r['test']['recall_macro'],
        'F1': r['test']['f1_macro'], 'train acc': r['train']['accuracy'],
        'CV mean': r['cv_mean'], 'CV sd': r['cv_std'], 'verdict': r['verdict'],
    } for n, r in rows])
    print(f'=== {target}   (majority baseline {rows[0][1]["baseline"]:.4f}) ===')
    display(t.style.format({c: '{:.4f}' for c in
                            ['accuracy','precision','recall','F1','train acc','CV mean','CV sd']}
                          ).hide(axis='index'))
    nm, bst = pick_best(rows, key=lambda r: r['test']['accuracy'])
    print(f'BEST: {nm}   accuracy {bst["test"]["accuracy"]:.4f}   '
          f'F1 {bst["test"]["f1_macro"]:.4f}   '
          f'lift +{bst["lift_points"]:.1f} points over baseline\\n')
"""),
    md("""
**`traffic_level`: 56.96% against a 37.66% baseline — +19.3 points.** Modest, and correct to be
modest: departure hour genuinely only partly determines traffic. A classifier claiming 95% here
would be evidence of leakage, not skill. Note the feature encoding — hour goes in as
`sin`/`cos` rather than as the integer 0–23, so the model does not treat 23:00 and 00:00 as
maximally distant.

**`cost_band`: 85.62% against a 25.68% baseline**, and here the train/test comparison finally
does some work. `DecisionTree` and `RandomForest` both hit train accuracy 1.0000 against test
0.7918 and 0.8354 — gaps of 0.21 and 0.16, flagged **Overfitting**. `LogisticRegression`, with
almost no capacity to memorise, wins outright at 0.8562. This is the cleanest demonstration in
the project of why step 2 exists.
"""),
    code("""
from sklearn.metrics import confusion_matrix, classification_report

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
for axis, (target, (rows, Xc, yc)) in zip(axes, cls_tables.items()):
    nm, bst = pick_best(rows, key=lambda r: r['test']['accuracy'])
    classes = sorted(set(yc))
    cm = confusion_matrix(yc[te], bst['fitted'].predict(Xc[te]), labels=classes)
    # Row-normalised: what fraction of each ACTUAL class went where.
    norm = cm / cm.sum(axis=1, keepdims=True)
    im = axis.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    axis.set_xticks(range(len(classes))); axis.set_xticklabels(classes, rotation=20, ha='right')
    axis.set_yticks(range(len(classes))); axis.set_yticklabels(classes)
    axis.set_xlabel('predicted'); axis.set_ylabel('actual')
    axis.set_title(f'{target} - {nm}\\naccuracy {bst["test"]["accuracy"]:.3f}')
    axis.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            axis.text(j, i, f'{cm[i, j]}\\n{norm[i, j]:.0%}', ha='center', va='center',
                      fontsize=8, color='white' if norm[i, j] > 0.5 else '#333')
plt.tight_layout(); plt.show()

rows, Xc, yc = cls_tables['traffic_level']
nm, bst = pick_best(rows, key=lambda r: r['test']['accuracy'])
print(f'Per-class report - traffic_level, {nm}:')
print(classification_report(yc[te], bst['fitted'].predict(Xc[te]), digits=3, zero_division=0))
"""),
    md("""
The traffic confusion matrix shows *how* the classifier is limited rather than just how much.
Errors are concentrated between adjacent levels — Low confused with Medium, Medium with High —
and the classifier is best at the extremes. That is the right failure mode for an ordered
target: it has learned the rush-hour structure and is uncertain in the middle, which is where
the data itself is genuinely ambiguous.
"""),

    md("""
## 9.7 Step 5 — hyperparameter tuning

Three searches, chosen to answer three different questions.

**Why not grid-search the winning model?** The best regressor is `LinearRegression`, which has
no hyperparameter worth tuning — a grid search over it is theatre. Ridge is the same model plus
exactly one knob, so searching α is a real search whose answer means something: **Week 8 of
this project rejected Ridge by hand**, arguing from the Hessian condition number that
regularisation fixed the ill-conditioning at an unacceptable cost in accuracy. If that argument
was right, the search should drive α to the bottom of the grid on its own, without being told
the theory.
"""),
    code("""
tune = TASK5['tuning']
ridge = tune['ridge_alpha']
alphas = [c['alpha'] for c in ridge['curve']]
scores = [c['cv_r2'] for c in ridge['curve']]

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4))
ax[0].semilogx(alphas, scores, 'o-', color='#2e5f8a')
ax[0].axvline(ridge['best_params']['model__alpha'], color='#c0392b', ls='--',
              label=f"chosen alpha = {ridge['best_params']['model__alpha']:g}")
ax[0].set_xlabel('Ridge alpha (log scale)'); ax[0].set_ylabel('5-fold CV R2')
ax[0].set_title('GridSearchCV: the answer is "do not regularise"')
ax[0].legend()

ax[1].semilogx(alphas, scores, 'o-', color='#2e5f8a')
ax[1].set_ylim(0.9995, 1.0); ax[1].set_xlabel('Ridge alpha (log scale)')
ax[1].set_title('Same curve, zoomed: flat to alpha ~ 0.01, then it falls')
plt.tight_layout(); plt.show()

for c in ridge['curve']:
    print(f"  alpha {c['alpha']:<9g} CV R2 {c['cv_r2']:.6f} +/- {c['cv_sd']:.6f}")
print(f"\\n  test R2 at sklearn's default alpha=1.0 : {ridge['test_before']['r2']:.6f}"
      f"   RMSE Rs {ridge['test_before']['rmse']:.2f}")
print(f"  test R2 after the search              : {ridge['test_after']['r2']:.6f}"
      f"   RMSE Rs {ridge['test_after']['rmse']:.2f}")
"""),
    md("""
The CV score is **flat for every α ≤ 0.01** and falls monotonically above it, collapsing to
R² 0.835 by α = 1000. Across the flat region the penalty is too small to do anything, so Ridge
there *is* ordinary least squares; the search has no reason to prefer any point on the plateau
and returns one of them. Every α large enough to actually regularise scores strictly worse.

That is Week 8's hand-argument reproduced mechanically. **A search whose answer is "do not
regularise" has not failed — it has confirmed the model was already correctly specified.**
"""),
    code("""
rfs = tune['random_forest']
trf = tune['traffic_classifier']

print('(b) RandomizedSearchCV - RandomForestRegressor, 12 draws x 5 folds')
print(f"    search space : {rfs['space']}")
print(f"    best params  : {rfs['best_params']}")
print(f"    best CV R2   : {rfs['best_cv_r2']:.6f}")
print(f"    test MAE before (300 trees, defaults) : Rs {rfs['test_before']['mae']:.2f}")
print(f"    test MAE after  the search            : Rs {rfs['test_after']['mae']:.2f}")
print(f"    -> {'IMPROVED' if rfs['improved'] else 'NO IMPROVEMENT'}"
      f"  ({rfs['test_after']['mae'] - rfs['test_before']['mae']:+.2f} Rs)")

print('\\n(c) GridSearchCV - the traffic classifier (18 combinations x 5 folds)')
print(f"    grid         : {trf['grid']}")
print(f"    best params  : {trf['best_params']}")
print(f"    best CV acc  : {trf['best_cv_accuracy']:.4f}")
print(f"    test accuracy before : {trf['test_before']['accuracy']:.4f}"
      f"   F1 {trf['test_before']['f1_macro']:.4f}")
print(f"    test accuracy after  : {trf['test_after']['accuracy']:.4f}"
      f"   F1 {trf['test_after']['f1_macro']:.4f}")
print(f"    -> {'IMPROVED' if trf['improved'] else 'NO IMPROVEMENT'}  "
      f"({(trf['test_after']['accuracy'] - trf['test_before']['accuracy']) * 100:+.2f} points)")
print(f"    majority baseline: {trf['baseline']:.4f}")
"""),
    md("""
### What the three searches say together

| Search | Outcome | Why |
|---|---|---|
| Ridge α | improved | …by turning regularisation **off**. Real finding: this problem wants none. |
| RandomForest | **no improvement** | Nothing left to find — a linear model already fits the formula to R² 0.9997. |
| Traffic classifier | improved, +0.92 pts | The one target with genuine noise, so capping `max_depth` at 4 helps. |

The checklist says *"re-test the tuned model on the test set — confirm score improved"*. The
honest answer is that one of the three did not, and the pattern is the lesson:

> **Hyperparameter tuning pays where the data is noisy and the model can overfit it.** Where the
> target is a formula and the features already span it, tuning has nothing to do — and a search
> that reports no improvement is evidence the model was specified correctly, not evidence the
> search was wasted.

Notice the traffic search chose `max_depth=4` over an unrestricted tree. That is the same
finding as §9.6 from the other direction: the unconstrained forest was memorising, and forcing
it to generalise raised the *test* score even as it lowered the training one.
"""),

    md("""
## 9.8 Step 6 — the advanced models, in one place

RandomForest (bagging), AdaBoost and GradientBoosting appear in every table above rather than in
a section of their own, because the point of including them is comparison. Collecting the
verdict:

- **RandomForest** — the strongest ensemble here, and still beaten by a linear model on the
  cost target (₹87 vs ₹54 RMSE). It wins where the relationship is genuinely non-linear: the
  road-distance sub-model (§10.3) and the end-to-end model (§10.7), both of which have to
  learn geography.
- **AdaBoost** — consistently the weakest, and it is worth knowing why rather than just
  reporting it. AdaBoost fits shallow stumps and re-weights the rows it got wrong. On a smooth
  additive target that is a poor fit: it spends its capacity chasing the noisiest rows instead
  of representing the smooth part, which is exactly the wrong instinct on a formula with
  injected noise.
- **GradientBoosting** — close behind RandomForest throughout, and it wins the traffic
  classification outright, where the signal is weak and the boosted residual fitting helps.

**None of the three beats a correctly specified linear model on the cost target.** That is the
project's recurring result, and it survived every test in this chapter.
"""),

    md("""
## 9.9 The same checklist on the original 1,140-row dataset

The assignment was handed out with `road_trip_data.csv`, so the tables are repeated on it. It
makes two points the wide dataset cannot.
"""),
    code("""
orig = TASK5['original_dataset']
reg_o = orig['regression']

comp = pd.DataFrame([{
    'model': n,
    'original test R2': reg_o[n]['test']['r2'],
    'original RMSE': reg_o[n]['test']['rmse'],
    'original CV sd': reg_o[n]['cv_std'],
    'wide CV sd': TASK5['regression']['models'].get(n, {}).get('cv_std', np.nan),
} for n in reg_o])
comp['CV sd ratio'] = comp['original CV sd'] / comp['wide CV sd']
display(comp.style.format({'original test R2': '{:.6f}', 'original RMSE': '{:.2f}',
                           'original CV sd': '{:.6f}', 'wide CV sd': '{:.6f}',
                           'CV sd ratio': '{:.1f}x'}).hide(axis='index'))

fig, ax = plt.subplots(figsize=(8.5, 3.8))
m = comp.dropna(subset=['wide CV sd'])
i = np.arange(len(m)); w = 0.38
ax.bar(i - w/2, m['original CV sd'], w, label=f"original ({orig['n_rows']:,} rows)",
       color='#c0392b')
ax.bar(i + w/2, m['wide CV sd'], w, label='wide (25,000 rows)', color='#2e7d32')
ax.set_yscale('log'); ax.set_ylabel('CV standard deviation (log)')
ax.set_xticks(i); ax.set_xticklabels(m['model'], rotation=30, ha='right')
ax.set_title('Small data is less stable - and k-fold is how you see it')
ax.legend(); plt.tight_layout(); plt.show()

print('\\nCLASSIFICATION on the original data - traffic_level:')
for n, r in orig['traffic'].items():
    print(f"  {n:<26} acc {r['test']['accuracy']:.4f}  F1 {r['test']['f1_macro']:.4f}  "
          f"-> {r['verdict']}")
print(f"\\n  majority baseline: {list(orig['traffic'].values())[0]['baseline']:.4f}")
"""),
    md("""
**First: small data is measurably less stable.** With 1,140 rows instead of 25,000, the
cross-validation standard deviation is 4–20× wider for the same models and the same code. This
is the clearest argument in the project for why step 3 is on the checklist at all — with one
split you would simply never see it.

**Second: on the original data, traffic classification is impossible, and the metrics say so
unmistakably.** Every one of the five classifiers lands on *exactly* 0.5088 — the majority-class
baseline to four decimals — with macro F1 of 0.2248 and a lift of +0.0 points. All five are
flagged **Underfitting**.

That is not five models failing. It is five models all discovering the same thing: in the
original data `traffic_level` is statistically independent of `departure_hour`, so the best
available strategy is to always predict the majority class, and each of them found it. Macro F1
of 0.2248 is the giveaway — a model predicting one class out of three scores about
$\\frac{1}{3}$ on macro recall no matter how good its accuracy looks.

**This is why Week 5 injected a rush-hour profile before the traffic classifier was allowed
into the project**, and why the 56.96% reported in §9.6 measures that design decision rather
than Indian roads. The comparison between these two tables is the honest disclosure.
"""),

    md("""
## Week 9 conclusion

| Step | Result |
|---|---|
| 1. Metrics | Best regressor: `LinearRegression`, R² 0.999707, RMSE ₹54.16, RSS 1.467e7. Best classifiers: `LogisticRegression` on `cost_band` (85.6%), `GradientBoosting` on `traffic_level` (57.0%) |
| 2. Overfit / underfit | "Good fit" everywhere on the cost target — because the target is a formula. Real overfitting appears on `cost_band` (trees, gaps of 0.16–0.21) and on the toll sub-model |
| 3. Validation | 5-fold CV sd ≤ 4e-4 on 25,000 rows; 4–20× wider on 1,140 rows. Bootstrap 95% CI width 3e-6 |
| 4. Comparison | One split, one scoring module, one selection rule (error band → one-SE → simplicity) |
| 5. Tuning | 2 of 3 searches improved. Ridge α → 0 reproduces Week 8 mechanically; RandomForest could not be improved; the traffic classifier gained 0.92 points |
| 6. Advanced models | RandomForest, AdaBoost and GradientBoosting all tested; none beats a correctly specified linear model on cost |

The chapter's own finding is about **what the metrics were measuring**. R² 0.999707 is real, but
it was obtained with the true toll, parking and litres supplied as input features — and a user
supplies none of those. Week 10 takes them away.
"""),
]

# ==========================================================================================
#                                        WEEK 10
# ==========================================================================================
CELLS += [
    md("""
---

# Week 10 — Removing the Magic Numbers: a Chained Model Pipeline

## 10.1 The problem

`app.py` served a prediction like this:

```python
distance = haversine * 1.2355                          # constant
litres   = distance / mileage * TRAFFIC_MULT[traffic]  # constant lookup
toll     = distance * 1.301                            # constant
parking  = 70.0                                        # constant
cost     = cost_regressor([distance, mileage, price, toll, parking, litres, ...])
```

Only the last line is a model. Everything above it is arithmetic — and it is not incidental
arithmetic, because it produces `fuel_consumption_litres`, the regressor's single strongest
feature at correlation +0.96 with the target. The model was being handed three of the four
terms of the cost formula and asked to add them up.

That reframes the headline. **R² 0.999707 is a true number answering the wrong question**:
*"given the litres burnt and the toll paid, can you total the bill?"* The user's question is
*"what will this trip cost?"*, and they know neither.

This week replaces each constant with a fitted sub-model, chains them, and measures the cost of
doing so honestly.

| Constant | Replaced by | Trained on |
|---|---|---|
| `WINDING_FACTOR = 1.2355` | sub-model 1 — road distance | **840 real OSRM road distances** |
| (user had to know their mileage) | sub-model 2 — mileage | generated data |
| `TRAFFIC_MULT = {...}` | sub-model 3 — litres | generated data |
| `TOLL_RATE = 1.301` | sub-model 4 — toll | generated data |
| `DEFAULT_PARKING = 70.0` | sub-model 5 — parking | **fails; mean served instead** |
| (already a model) | sub-model 6 — traffic | generated data |
"""),

    md("""
## 10.2 The one piece of genuinely observed data

Everything in `data/road_trip_wide.csv` is generated. To fit a distance model that is not
circular, this week collected real data: `scripts/fetch_real_distances.py` samples city pairs
stratified across six distance bands and asks the OSRM routing engine for the actual driving
distance.

**840 real routes**, 40–1,248 km straight-line. This is the only observed data in the project.
"""),
    code("""
real = pd.read_csv('data/real_distances.csv')
CONST = 1.2355

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
ax[0].scatter(real.haversine_km, real.road_km, s=9, alpha=0.45, color='#2e5f8a',
              label='840 real routes')
xs = np.linspace(0, real.haversine_km.max(), 50)
ax[0].plot(xs, xs * CONST, 'r--', lw=1.8, label=f'the constant: haversine x {CONST}')
ax[0].plot(xs, xs, color='#999', lw=1, ls=':', label='straight line (x1.0)')
ax[0].set_xlabel('straight-line distance (km)'); ax[0].set_ylabel('real road distance (km)')
ax[0].set_title('Road vs straight-line distance'); ax[0].legend(fontsize=8)

ax[1].scatter(real.haversine_km, real.factor, s=9, alpha=0.45, color='#2e5f8a')
ax[1].axhline(CONST, color='r', ls='--', lw=1.8, label=f'the constant = {CONST}')
ax[1].set_xlabel('straight-line distance (km)'); ax[1].set_ylabel('winding factor (road / straight)')
ax[1].set_title('The factor the constant claims is one number'); ax[1].legend(fontsize=8)
worst = real.nlargest(2, 'factor')
for _, r in worst.iterrows():
    ax[1].annotate(f'{r.start_city} -> {r.dest_city}\\nfactor {r.factor:.2f}',
                   (r.haversine_km, r.factor), textcoords='offset points', xytext=(12, -6),
                   fontsize=8, color='#c0392b',
                   arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8))
plt.tight_layout(); plt.show()

print(f"observed winding factor: mean {real.factor.mean():.4f}  sd {real.factor.std():.4f}"
      f"  range {real.factor.min():.4f} - {real.factor.max():.4f}")
print(f"the constant it replaces: {CONST}\\n")
print(real.nlargest(3, 'factor')[['start_city','dest_city','haversine_km','road_km','factor']]
      .to_string(index=False))
"""),
    md("""
### The extreme routes are real, and they are the whole argument

A factor of **3.60** looks like bad data. It is not. **Surat and Bhavnagar face each other
across the Gulf of Khambhat**: 94 km of straight line, 339 km of road around the head of the
gulf. Valsad → Bhavnagar (2.68) is the same water. At the other end, Moradabad → Chanduasi has
a factor of 1.04 — essentially a straight road.

No single multiplier can serve both. These points are kept in the training data rather than
cleaned away, because they are precisely the cases a constant cannot represent and a model
can — and note which model wins below: a **tree ensemble**, because it can isolate a region of
the map. A linear fit gets dragged by these points instead of representing them.
"""),
    code("""
bias = PIPELINE['geography_bias']
print('How wrong is the GENERATED dataset\\'s geography?')
print('data/road_trip_wide.csv sets distance_km = haversine x N(1.2355, 0.082).')
print('Against the real road distance for the same city pairs:\\n')
print(f"  mean signed error : {bias['mean_signed_km']:+.1f} km  "
      f"({bias['mean_signed_pct']:+.2f}%)   <- bias")
print(f"  mean ABS error    : {bias['mean_abs_km']:.1f} km  "
      f"({bias['mean_abs_pct']:.1f}%)   <- per-route error")
print(f"  worst overshoot   : {bias['max_km']:+.1f} km")
print(f"  worst undershoot  : {bias['min_km']:+.1f} km")
"""),
    md("""
These two numbers say opposite-sounding things, and both matter.

The constant is very nearly **unbiased**: averaged over hundreds of routes it is within half a
percent of the truth. That is why it survived review for so long. But *unbiased on average* is
not *correct*: route by route it is off by **6.3%** typically, and by tens of percent at the
extremes.

**A user does not take the average of every road trip in India. They take one specific trip,
and on that trip the errors do not cancel.**
"""),

    md("""
## 10.3 Sub-model 1 — road distance

Two parameterisations of the same problem were compared, and the difference is instructive:

- **(a) predict `road_km` directly**
- **(b) predict the winding *factor*, then multiply by haversine**

(b) removes trip length from the target. The model then spends its capacity on the part that is
genuinely unknown — the *shape* of the route — instead of re-learning "longer straight line,
longer road". It is the same idea as Week 6's interaction term: hand the model the quantity
that actually means something.
"""),
    code("""
d = PIPELINE['distance']
base = d['constant_baseline']
rows = []
for label, key in [('(a) direct km', 'candidates_direct'), ('(b) factor x haversine', 'candidates_factor')]:
    for n, r in d[key].items():
        rows.append({'parameterisation': label, 'model': n,
                     'test R2': r['test']['r2'], 'RMSE km': r['test']['rmse'],
                     'MAE km': r['test']['mae'], 'CV mean': r['cv_mean'], 'CV sd': r['cv_std']})
t = pd.DataFrame(rows)
display(t.style.format({'test R2': '{:.6f}', 'RMSE km': '{:.2f}', 'MAE km': '{:.2f}',
                        'CV mean': '{:.6f}', 'CV sd': '{:.4f}'}).hide(axis='index'))

print(f"BASELINE  haversine x {CONST}: R2 {base['r2']:.4f}  RMSE {base['rmse']:.2f} km  "
      f"MAE {base['mae']:.2f} km")
print(f"CHOSEN    {d['chosen']}: R2 {d['chosen_scores']['test']['r2']:.4f}  "
      f"RMSE {d['chosen_scores']['test']['rmse']:.2f} km  "
      f"MAE {d['chosen_scores']['test']['mae']:.2f} km")
print(f"          -> {d['mae_reduction_pct']:.1f}% lower MAE than the constant it replaces")
"""),
    md("""
**The model beats the constant, by 10.7% on MAE (34.33 → 30.67 km) and 12% on RMSE.** Real, and
deliberately not oversold: endpoint coordinates only partly determine route shape, because the
rest depends on where roads and bridges happen to be — information not in the features.

Parameterisation (b) wins, exactly as predicted, and the winner is `RandomForest`. On the raw
kilometre target the ensembles have to spend their splits re-deriving proportionality to
haversine; given the scale-free target they can spend them on geography instead.

**A sub-model that could not beat the constant it replaces would not have earned its place**,
which is why every table in this chapter carries the constant as its baseline row.
"""),

    md("""
## 10.4 Sub-models 2–6

Three of these succeed, one succeeds trivially, and one **fails** — reported as a result rather
than quietly dropped.
"""),
    code("""
for key, title, unit in [('mileage', 'SUB-MODEL 2: mileage (vehicle + fuel -> km/l)', 'km/l'),
                         ('litres',  'SUB-MODEL 3: litres (replaces TRAFFIC_MULT)', 'litres'),
                         ('toll',    'SUB-MODEL 4: toll (replaces TOLL_RATE = 1.301)', 'Rs'),
                         ('parking', 'SUB-MODEL 5: parking (replaces DEFAULT_PARKING = 70)', 'Rs')]:
    sec = PIPELINE[key]
    print('=' * 78); print(title); print('=' * 78)
    if 'constant_baseline' in sec:
        cb = sec['constant_baseline']
        print(f"  BASELINE (the constant): R2 {cb['r2']:+.4f}  RMSE {cb['rmse']:.2f} {unit}")
    if 'mean_baseline' in sec:
        mb = sec['mean_baseline']
        print(f"  BASELINE (train mean)  : R2 {mb['r2']:+.4f}  RMSE {mb['rmse']:.2f} {unit}")
    t = pd.DataFrame([{'model': n, 'test R2': r['test']['r2'], f'RMSE {unit}': r['test']['rmse'],
                       'train R2': r['train']['r2'], 'CV sd': r['cv_std'],
                       'verdict': r['verdict']}
                      for n, r in sec['candidates'].items()])
    display(t.style.format({'test R2': '{:.6f}', f'RMSE {unit}': '{:.3f}',
                            'train R2': '{:.6f}', 'CV sd': '{:.4f}'}).hide(axis='index'))
    print(f"  CHOSEN: {sec['chosen']}\\n")
"""),
    md("""
### Sub-model 2, mileage — a structural ceiling, not a modelling failure

Test R² **0.673**, and every model family lands on the same number to three decimals. That is
the correct answer: the generator draws mileage *uniformly* inside a per-(vehicle, fuel) range,
so the best any model can do is name the middle of the right cell and the residual is uniform
noise by construction.

Two details worth noting. The nine (vehicle, fuel) cells cannot be reproduced by additive
one-hots alone — 1 + 2 + 2 = 5 parameters cannot fit nine independent means — so
`MILEAGE_FEATURES` includes the four vehicle × fuel interaction terms, and with them the design
matrix spans all nine cells exactly. And this sub-model changes the *interface*: the app used to
default every car to 15.5 km/l, and now predicts 19.5 for a diesel hatchback and 10.8 for a CNG
SUV.

### Sub-model 3, litres — reading the generator off the coefficients

**Test R² = 1.000000.** The model has not approximated the data-generating rule; it has
reproduced it. That claim is worth cashing in, because
`litres = (distance / mileage) × traffic_multiplier` is a *product*, and an additive model
cannot express a product. Handed the ratio and its interactions with the traffic one-hots, the
coefficients become the multipliers themselves:
"""),
    code("""
# Refit UNSCALED on just the three ratio columns, so the coefficients are readable directly:
#   litres = ratio * [ mult_Low + (mult_Med - mult_Low)*Medium + (mult_High - mult_Low)*High ]
from sklearn.linear_model import LinearRegression

Xl = np.asarray([rf.litres_features(dd, mm, tt) for dd, mm, tt in
                 zip(wide.distance_km, wide.mileage, wide.traffic_level)], dtype=float)
yl = wide.fuel_consumption_litres.values.astype(float)
cols = [rf.LITRES_FEATURES.index(c)
        for c in ('km_per_litre_ratio', 'ratio_x_Medium', 'ratio_x_High')]
bare = LinearRegression().fit(Xl[tr][:, cols], yl[tr])
w_low, med_gap, high_gap = bare.coef_
recovered = {'Low': w_low, 'Medium': w_low + med_gap, 'High': w_low + high_gap}
generator = {'Low': 0.944, 'Medium': 1.045, 'High': 1.165}   # Week 4, from the original CSV

print(f"{'traffic':<10}{'recovered':>13}{'Week 4':>10}{'error':>13}")
for lvl in ('Low', 'Medium', 'High'):
    print(f"{lvl:<10}{recovered[lvl]:>13.6f}{generator[lvl]:>10.4f}"
          f"{recovered[lvl] - generator[lvl]:>+13.6f}")
print(f"\\nintercept {bare.intercept_:+.2e}  (zero - the formula has no constant term)")
print('\\nSix decimal places. This is the Week 6 lesson in its purest form: the features were')
print('the whole problem, and with the right ones a linear model IS the generating rule.')
"""),
    md("""
### Sub-model 4, toll — and the first honest overfitting in the project

Test R² **0.795**, RMSE **₹212**, and the fitted model does not beat the constant it replaces
(R² 0.7950 vs 0.7949). The generator sets `toll = distance × N(1.301, 0.360)`, so the *rate* is
recoverable but the per-trip spread is injected noise. **That ₹212 is irreducible on this data,
and a model that appeared to beat it would be leaking.**

Look at the trees in that table, though — this is the first target in the project with real
noise, and they behave exactly as theory says:

| Model | train R² | test R² | verdict |
|---|---|---|---|
| LinearRegression | 0.8046 | 0.7949 | Good fit |
| DecisionTree | **0.9856** | **0.6109** | **Overfitting** |
| RandomForest (300) | **0.9509** | **0.7117** | **Overfitting** |

The tree fits the *noise* — a gap of 0.37 between train and test. Week 9's step 2 was
uninformative on the deterministic cost target; here it immediately identifies the wrong model.
So the toll sub-model is kept for a different reason than accuracy: it is fitted, it reports its
own error bar, and it is evaluated like everything else instead of being asserted.

### Sub-model 5, parking — a negative result, reported

**Every one of the six model families scores test R² ≤ 0** — worse than predicting the mean.
`DecisionTree` reaches R² −1.00 (train 0.9888: pure memorisation of noise).

This is the correct answer, not a bug. The generator draws parking uniformly from
{0, 40, 60, 80, 120, 150} **independently of every other column**, so it carries no signal at
all. A model with negative R² has no business in a prediction path, so the app serves the
training mean (₹75.09) and the negative result is disclosed through `/api/metrics`.

Worth noting: the old hardcoded ₹70.0 was not even the mean of the training data.
"""),
    code("""
sec = PIPELINE['traffic']
print('=' * 78); print('SUB-MODEL 6: traffic level (classifier)'); print('=' * 78)
t = pd.DataFrame([{'model': n, 'accuracy': r['test']['accuracy'],
                   'precision': r['test']['precision_macro'],
                   'recall': r['test']['recall_macro'], 'F1': r['test']['f1_macro'],
                   'baseline': r['baseline'], 'lift (pts)': r['lift_points'],
                   'verdict': r['verdict']}
                  for n, r in sec['candidates'].items()])
display(t.style.format({'accuracy': '{:.4f}', 'precision': '{:.4f}', 'recall': '{:.4f}',
                        'F1': '{:.4f}', 'baseline': '{:.4f}',
                        'lift (pts)': '{:+.1f}'}).hide(axis='index'))
print(f"  CHOSEN: {sec['chosen']}")
"""),

    md("""
## 10.5 Chaining them properly — stacking, and a bug worth measuring

Six sub-models feeding a seventh introduces a failure mode that is easy to miss.

The cost regressor was trained on the **true** toll, parking and litres. If it is then served
**predicted** ones, it has learned to trust inputs that are exact and receives inputs that are
not. That is a **train/serve mismatch**, and rather than assume it matters, both ways are
measured.

Building the training components correctly needs care too. Sub-model predictions on their own
training rows are optimistic — the model has already seen those rows — so the component features
for training come from **`cross_val_predict` (out-of-fold)**: no row's features were produced by
a model that had seen that row. Test rows use the sub-models fitted on the full training split.
That is textbook stacking.
"""),
    code("""
arch = TASK5['architecture_comparison']
rows = [
    ('A. fed true toll / parking / litres (Week 6)', arch['A_true_components']),
    ('B. chained, trained on true / served predicted', arch['B_mismatched_chain']),
    ('C. chained, trained on predictions (SHIPPED)',   arch['C_chained']),
    ('D. end-to-end, user inputs only',                arch['D_end_to_end']),
]
t = pd.DataFrame([{'architecture': n, 'model': v.get('model', '-'),
                   'test R2': v['r2'], 'RMSE Rs': v['rmse'], 'MAE Rs': v['mae']}
                  for n, v in rows])
display(t.style.format({'test R2': '{:.6f}', 'RMSE Rs': '{:.2f}',
                        'MAE Rs': '{:.2f}'}).hide(axis='index'))

fig, ax = plt.subplots(figsize=(9, 3.8))
labels_ = [n.split('.')[0] for n, _ in rows]
maes = [v['mae'] for _, v in rows]
bars = ax.bar(labels_, maes, color=['#7f8c8d', '#e08a3c', '#2e7d32', '#2e5f8a'])
ax.set_ylabel('test MAE (Rs)'); ax.set_title('What the model is actually being asked to do')
for b, v in zip(bars, maes):
    ax.text(b.get_x() + b.get_width()/2, v + 12, f'Rs {v:.0f}', ha='center', fontsize=9)
ax.set_ylim(0, max(maes) * 1.2)
plt.tight_layout(); plt.show()

print(f"R2 drop from A to C: {arch['r2_drop_A_to_C']:.4f}")
print(f"MAE goes from Rs {arch['A_true_components']['mae']:.2f} to "
      f"Rs {arch['C_chained']['mae']:.2f} - roughly ten times larger.")
"""),
    md("""
### Reading the four rows

**A → C is the honest headline of this project.** R² falls 0.9997 → 0.9645 and MAE rises ₹40 →
₹391. That gap is not a regression in quality; it is the price of the model deriving its own
inputs instead of being handed them. **C is what a user actually receives, so C is what the app
now reports.**

**B vs C is the train/serve mismatch, and the measurement is a surprise.** B (trained on truth,
served predictions) scores MAE ₹390.56; C (trained on its own predictions) scores ₹391.16 —
B is *marginally better*. The mismatch cost essentially nothing here, and the reason is
specific: the sub-model errors are close to **zero-mean noise**, so the linear cost model's
coefficients are almost unchanged by retraining on them.

That is worth stating precisely rather than claiming a win. The chain-consistent model is still
the one shipped — it is the correct construction, and the property that rescues B (unbiased
sub-model errors) is a fact about this dataset, not something to rely on. But the honest report
is that **on this data the mismatch did not bite**, and pretending otherwise would be inventing
a result.
"""),

    md("""
## 10.6 Where the remaining error comes from

"The honest model is worse" is not a finding. *"Toll and parking noise account for nearly all of
it"* is. Decomposing the test-set error into its sources:
"""),
    code("""
eb = PIPELINE['cost']['error_budget']
parts = [('toll (injected noise)', eb['toll_sd']),
         ('parking (uniform noise)', eb['parking_sd']),
         ('fuel bill, via litres', eb['fuel_bill_sd'])]

fig, ax = plt.subplots(figsize=(8.5, 3.6))
names_ = [p[0] for p in parts] + ['quadrature sum', 'chained model RMSE']
vals   = [p[1] for p in parts] + [eb['quadrature_sum'], eb['chained_rmse']]
cols   = ['#7f8c8d'] * 3 + ['#e08a3c', '#2e7d32']
b = ax.barh(names_, vals, color=cols)
ax.set_xlabel('standard deviation / RMSE (Rs)')
ax.set_title('The error is the data\\'s unpredictability, not the model\\'s weakness')
ax.invert_yaxis()
for bb, v in zip(b, vals):
    ax.text(v + 8, bb.get_y() + bb.get_height()/2, f'{v:.0f}', va='center', fontsize=9)
plt.tight_layout(); plt.show()

print(f"  distance error (km, not Rs)  : {eb['distance_sd_km']:.2f} km")
print(f"  quadrature sum of cost terms : Rs {eb['quadrature_sum']:.2f}")
print(f"  chained model RMSE           : Rs {eb['chained_rmse']:.2f}")
print(f"  ratio                        : {eb['chained_rmse'] / eb['quadrature_sum']:.2f}")
"""),
    md("""
The quadrature sum of the irreducible terms is **₹529** against a model RMSE of **₹596** — a
ratio of 1.13. The two agree closely enough to say that the chain is near the floor this dataset
allows, and that the residual is **the data's unpredictability rather than the model's
weakness**.

The largest single term is the fuel bill (sd ₹479), and it is the one that could still be
improved: it inherits the distance error, so a better distance model propagates straight through
to a better cost estimate. Toll (₹219) and parking (₹50) cannot be improved by any model —
those are drawn from random distributions by construction.

There is a corresponding test in the suite (`test_metrics_error_budget_explains_the_remaining_error`)
asserting that ratio stays between 0.7 and 1.6, so a future regression in the chain shows up as
a failing test rather than a quietly worse number.
"""),

    md("""
## 10.7 The end-to-end model — and where Week 6's lesson reverses

Architecture D removes even the chain. One model, fed **only** what a traveller could know:
two city names (as real latitude/longitude), vehicle, fuel type, departure hour, month,
passengers, mileage and fuel price. No distance, no toll, no litres, no parking.

It also re-runs Week 6's experiment in a harder setting. The true cost contains
`(distance / mileage) × fuel_price`, so two feature sets are compared: the raw inputs, and the
raw inputs plus `haversine/mileage` and `(haversine/mileage) × fuel_price`.
"""),
    code("""
chk = E2E['week6_check']
print('END-TO-END: features vs model complexity\\n')
print(f"  linear, base features        : R2 {chk['linear_base']['r2']:.6f}   "
      f"MAE Rs {chk['linear_base']['mae']:.2f}")
print(f"  linear, + 2 engineered       : R2 {chk['linear_engineered']['r2']:.6f}   "
      f"MAE Rs {chk['linear_engineered']['mae']:.2f}")
print(f"  RandomForest, base features  : R2 {chk['rf_base']['r2']:.6f}   "
      f"MAE Rs {chk['rf_base']['mae']:.2f}")
print(f"\\n  MAE saved by two engineered columns : Rs {chk['mae_saved_by_features']:.2f}")
print(f"  MAE saved by switching to 300 trees : Rs {chk['mae_saved_by_model']:.2f}")
print(f"  VERDICT: {chk['verdict']}")

fi = E2E['feature_importance']
top = fi['ranked'][:12]
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.barh([f['feature'] for f in top][::-1], [f['share_pct'] for f in top][::-1],
        color='#2e5f8a')
ax.set_xlabel('share of importance (%)')
ax.set_title(f"What the end-to-end model leans on\\n({fi['source']})")
plt.tight_layout(); plt.show()
"""),
    md("""
### Week 6's claim does not generalise, and that is the interesting part

Week 6 argued — and demonstrated — that **domain understanding beat model complexity**: one
engineered feature beat two ensembles. End-to-end, the result **reverses**:

| | MAE saved |
|---|---|
| Two engineered columns | **₹67** |
| Switching to 300 trees | **₹474** |

The ensemble wins by a factor of seven. This is not a contradiction of Week 6, it is a boundary
condition on it, and the reason is identifiable. When distance was *given*, the only thing the
linear model could not express was a product — so handing it the product fixed the one gap.
End-to-end, the model must first infer road distance from four coordinates, and that is a
genuinely non-linear function of geography: coastlines, mountains, the Gulf of Khambhat. No
finite set of hand-made interaction terms captures it, and partitioning space is exactly what
trees do.

**Feature engineering beats model complexity when the missing structure is something you can
write down. When the missing structure is a map, it does not.**

### One trap in the importance ranking

`haversine` and its log carry **69%** of the prediction, mileage 11% — all sensible. But the
vehicle and fuel one-hots score **below 0.1%**, and it would be wrong to conclude vehicle type
does not matter.

Week 4 recovered a real per-km maintenance rate that differs by vehicle (Hatchback 0.5694,
Sedan 0.6943, SUV 0.8366 ₹/km) and Week 6 showed that dropping those columns made the app quote
an identical price for an SUV and a hatchback. The rate spread is about 0.27 ₹/km — roughly
₹108 on a 400 km trip — and end-to-end that signal sits *underneath* the toll and parking noise,
which is hundreds of rupees.

**The importance is low because the noise here is large, not because the effect is absent.**
It is precisely why the app serves the chained pipeline: pin distance and litres down first, and
the vehicle term becomes visible again. `passengers`, at 0.33%, is the genuine null — the
recovered formula has no passenger term at all, so anything above zero there is the forest
fitting noise.
"""),

    md("""
## 10.8 What ships

`app.py` now loads `models/pipeline.joblib` and the five constants are gone. Live OSRM routing
is still preferred for distance — a real answer beats a predicted one — but the *fallback* is
now the fitted model rather than `haversine × 1.2355`.

Two constants remain, deliberately and labelled:

- `TRAFFIC_MIN_PER_KM` — travel **time**, which this project never modelled. It is not part of
  the cost target, so fitting it would be inventing scope.
- `MIN/MAX_TRAINED_KM` — the training range, used only to warn about extrapolation.

The change with the most user-visible consequence is not a model at all. `/api/predict` returns
`typical_error` beside every estimate, and that number used to be **₹39.66** — the MAE measured
with the true toll, parking and litres handed in. No real prediction could achieve it. It is now
**₹391.16**, the chained pipeline's own held-out error, and `/api/metrics` reports `served`
alongside `fed_true_components` with the gap between them named.

Quoting an unachievable error bar next to a prediction was the single least honest thing in the
project, and it is the thing this week actually fixed.

## Week 10 conclusion

| | |
|---|---|
| Constants removed from `app.py` | 5 |
| Sub-models fitted | 6 (1 trained on real observed data) |
| Sub-models that failed | 1 — parking, R² ≤ 0, reported as a negative result |
| Real data collected | 840 OSRM road distances |
| Distance model vs its constant | MAE 34.33 → 30.67 km (−10.7%) |
| Honest cost accuracy | **R² 0.9645, MAE ₹391** (was R² 0.9997, MAE ₹40 with inputs handed in) |
| Error budget | ₹529 of the ₹596 RMSE is irreducible noise in the data |
| Tests | 65, all passing |

### The three things this week established

1. **A near-perfect score is a question about the question.** R² 0.9997 was true and
   uninformative; the number that describes a user's prediction is 0.9645.
2. **A model that fails is a result.** Six model families could not predict parking, because
   nothing in the data predicts parking. Reporting that is worth more than tuning until it
   looks respectable.
3. **"Features beat complexity" has a boundary.** It held when the missing structure was a
   product you could write down. It reversed when the missing structure was a map.
"""),
]


def main():
    if not os.path.exists(NB):
        sys.exit(f"notebook not found: {NB}")
    nb = json.load(open(NB, encoding="utf-8"))

    if not any(WEEK4_MARKER in "".join(c.get("source", [])) for c in nb["cells"]):
        sys.exit("Weeks 4-8 are missing - run scripts/build_notebook_chapters.py first")

    # idempotent: drop any previously appended Week 9+ chapters
    cut = next((i for i, c in enumerate(nb["cells"])
                if MARKER in "".join(c.get("source", []))), None)
    if cut is not None:
        print(f"removed {len(nb['cells']) - cut} previously appended cells")
        nb["cells"] = nb["cells"][:cut]

    before = len(nb["cells"])
    nb["cells"].extend(CELLS)
    json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"cells: {before} -> {len(nb['cells'])}  (+{len(CELLS)})")
    print(f"wrote {NB}")
    print("\nNow execute the appended cells:  python scripts/execute_new_chapters.py")


if __name__ == "__main__":
    main()
