# Task 5 — Model Evaluation, Validation and Tuning

**Road Trip Cost Prediction** · Semester 5 Machine Learning · Param Kotadiya

Every number in this document is generated from the saved evaluation reports by
`scripts/build_task5_doc.py`, so it cannot drift from the models. Full working, with plots,
is in **Week 9** of [`RoadTripCost.ipynb`](RoadTripCost.ipynb); the model rework it evaluates
is **Week 10**.

```bash
python scripts/evaluate_models.py     # regenerates data/task5_evaluation.json
python scripts/build_task5_doc.py     # regenerates this document
```

---

## How everything was measured

- **Dataset**: `data/road_trip_wide.csv` — 25,000 trips. 537 Indian cities are in-domain (population ≥ 100k, which is what the app accepts); **534** of them actually appear in the sampled routes.
- **Split**: `train_test_split(test_size=0.2, random_state=42)` → **20,000 train / 5,000 test**. Every table is scored on those same test rows.
- **Every model wrapped in `Pipeline([MinMaxScaler, model])`**, so cross-validation re-fits the scaler on each training fold. Scaling the whole matrix before splitting — which the project originally did — leaks test-set minima and maxima into training.
- **All scoring in one module**, `scripts/evaluation.py`, so the notebook, this document and the API report identical figures.

---

## 1 & 4 — Metrics and model comparison

### Regression — target `total_trip_cost` (₹)

Feature set: the shipped Week 6 model — distance, mileage, fuel price, toll, parking, litres, the `litres × price` interaction, and vehicle one-hots. Every row sees the same columns, so this compares **models**, not feature sets. The first row is the Week 6 ablation (interaction term removed) and belongs here as the honest baseline.

| Model | RSS | RMSE (₹) | R² | MAE (₹) |
|---|---:|---:|---:|---:|
| Linear, NO interaction | 2.606e+08 | 228.31 | 0.994795 | 139.75 |
| **LinearRegression** | 1.467e+07 | 54.16 | 0.999707 | 39.66 |
| Ridge (alpha=1) | 2.843e+07 | 75.40 | 0.999432 | 52.63 |
| DecisionTree | 6.948e+07 | 117.88 | 0.998612 | 79.52 |
| RandomForest (300) | 3.762e+07 | 86.75 | 0.999249 | 52.92 |
| AdaBoost | 7.944e+08 | 398.59 | 0.984135 | 313.86 |
| GradientBoosting | 3.871e+07 | 87.99 | 0.999227 | 56.76 |

**Best: `LinearRegression`** — chosen by the rule in section 4 below.

**Read this table from the RMSE column, not R².** Every model except AdaBoost scores R² > 0.998, which invites the conclusion that the choice does not matter. In rupees it clearly does: the worst model is about seven times worse than the best while looking like a rounding difference in R². When a target is nearly deterministic, R² compresses differences that matter.

The largest single effect in the table is not a model at all: removing one engineered column (`fuel_bill = litres × price`) moves RMSE from ₹54.16 to ₹228.31 — a bigger gap than between the best and worst *model*.

### Classification

#### Target: `traffic_level`

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| LogisticRegression | 0.5456 | 0.5393 | 0.5634 | 0.5373 |
| DecisionTree | 0.5604 | 0.5639 | 0.5696 | 0.5641 |
| RandomForest (300) | 0.5604 | 0.5639 | 0.5696 | 0.5641 |
| AdaBoost | 0.5692 | 0.5718 | 0.5795 | 0.5718 |
| **GradientBoosting** | 0.5696 | 0.5729 | 0.5797 | 0.5724 |

Majority-class baseline **0.3766** → best model `GradientBoosting` gains **+19.3 points**. Precision, recall and F1 are macro-averaged, so every class counts equally — weighted averaging can hide a class the model never predicts.

#### Target: `cost_band`

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| **LogisticRegression** | 0.8562 | 0.8580 | 0.8574 | 0.8577 |
| DecisionTree | 0.7918 | 0.7931 | 0.7936 | 0.7934 |
| RandomForest (300) | 0.8354 | 0.8385 | 0.8367 | 0.8375 |
| AdaBoost | 0.6340 | 0.6655 | 0.6328 | 0.6383 |
| GradientBoosting | 0.8194 | 0.8237 | 0.8210 | 0.8222 |

Majority-class baseline **0.2568** → best model `LogisticRegression` gains **+59.9 points**. Precision, recall and F1 are macro-averaged, so every class counts equally — weighted averaging can hide a class the model never predicts.

**The two targets are not of equal quality, and saying so is part of the answer.** `traffic_level` is the genuinely non-trivial one: departure hour only partly determines traffic, so an honest result looks modest, and a classifier claiming 95% here would be evidence of leakage. `cost_band` is a quartile split of ₹/km that the regressor above already predicts well, so its high accuracy is **by construction** — it earns its place because the interface needs the label, not because it is hard.

---

## 2 — Overfitting / underfitting

Verdicts need stated thresholds to be reproducible. `scripts/evaluation.py` fixes them:

| Condition | Verdict |
|---|---|
| train R² − test R² > 0.05 | Overfitting |
| both R² < 0.50 | Underfitting |
| accuracy less than 3 points above the majority baseline | Underfitting (classification) |
| otherwise | Good fit |

### Regression

| Model | Train R² | Test R² | Gap | Verdict |
|---|---:|---:|---:|---|
| Linear, NO interaction | 0.994853 | 0.994795 | +0.000058 | Good fit |
| LinearRegression | 0.999715 | 0.999707 | +0.000008 | Good fit |
| Ridge (alpha=1) | 0.999457 | 0.999432 | +0.000025 | Good fit |
| DecisionTree | 1.000000 | 0.998612 | +0.001388 | Good fit |
| RandomForest (300) | 0.999906 | 0.999249 | +0.000658 | Good fit |
| AdaBoost | 0.984114 | 0.984135 | -0.000020 | Good fit |
| GradientBoosting | 0.999462 | 0.999227 | +0.000236 | Good fit |

Every model reads **Good fit**, and that is a property of the *data*, not evidence of careful modelling. `DecisionTree` reaches train R² = 1.000000 — a tree grown to purity has memorised all 20,000 training rows — yet still holds test R² ≈ 0.9986, because the target is a deterministic formula. Memorising and learning the rule give nearly the same predictions when there is almost no noise to memorise.

**So on this target the train/test comparison is close to uninformative.** It becomes informative the moment real noise appears — which is exactly what the next two tables show.

### Classification — where the diagnosis actually bites

`traffic_level`:

| Model | Train acc | Test acc | Gap | Verdict |
|---|---:|---:|---:|---|
| LogisticRegression | 0.5277 | 0.5456 | -0.0179 | Good fit |
| DecisionTree | 0.5531 | 0.5604 | -0.0073 | Good fit |
| RandomForest (300) | 0.5531 | 0.5604 | -0.0073 | Good fit |
| AdaBoost | 0.5504 | 0.5692 | -0.0188 | Good fit |
| GradientBoosting | 0.5517 | 0.5696 | -0.0179 | Good fit |

`cost_band`:

| Model | Train acc | Test acc | Gap | Verdict |
|---|---:|---:|---:|---|
| LogisticRegression | 0.8563 | 0.8562 | +0.0001 | Good fit |
| DecisionTree | 1.0000 | 0.7918 | +0.2082 | Overfitting |
| RandomForest (300) | 1.0000 | 0.8354 | +0.1646 | Overfitting |
| AdaBoost | 0.6297 | 0.6340 | -0.0043 | Good fit |
| GradientBoosting | 0.8470 | 0.8194 | +0.0276 | Good fit |

On `cost_band`, `DecisionTree` and `RandomForest` both hit train accuracy 1.0000 with test accuracy well below it — gaps of roughly 0.16–0.21, correctly flagged **Overfitting**. `LogisticRegression`, with almost no capacity to memorise, wins outright. This is the clearest demonstration in the project of why this step is on the checklist.

The same thing happens on the toll sub-model (Week 10), the first regression target in the project with genuine noise:

| Model | Train R² | Test R² | Verdict |
|---|---:|---:|---|
| LinearRegression | 0.8046 | 0.7949 | Good fit |
| DecisionTree | 0.9856 | 0.6109 | Overfitting |
| RandomForest (300) | 0.9509 | 0.7117 | Overfitting |

The tree fits the injected noise and its test score collapses. Given a noisy target, the diagnosis identifies the wrong model immediately.

---

## 3 — Cross-validation (5-fold) and bootstrap

5-fold CV on the **training** data. The mean is a better estimate than one split; the **standard deviation across folds** is what says whether a model is stable or merely lucky.

| Model | CV mean R² | CV std | Fold range |
|---|---:|---:|---:|
| Linear, NO interaction | 0.994846 | 0.000125 | 0.000322 |
| LinearRegression | 0.999714 | 0.000010 | 0.000026 |
| Ridge (alpha=1) | 0.999387 | 0.000030 | 0.000078 |
| DecisionTree | 0.998292 | 0.000092 | 0.000244 |
| RandomForest (300) | 0.999266 | 0.000080 | 0.000233 |
| AdaBoost | 0.983787 | 0.000447 | 0.001347 |
| GradientBoosting | 0.999267 | 0.000083 | 0.000239 |

Every spread is tiny — CV standard deviations of 1e-5 to 4e-4. With 20,000 training rows and a near-deterministic target that is expected, and worth stating plainly: **this cross-validation confirms stability, it does not discriminate between the models.** Appendix A repeats it on 1,140 rows, where it does.

### Bootstrap (the checklist's alternative — both were run)

K-fold asks *how does the score vary across disjoint held-out folds*. Bootstrap asks *how would the score vary given a different training sample of the same size* — resampling training rows with replacement, refitting, scoring on the fixed test split.

| Quantity | Value |
|---|---|
| Model | `LinearRegression` |
| Resamples | 200 |
| Mean R² | 0.999707 |
| Std | 0.00000062 |
| 95% interval | [0.999705, 0.999708] |
| Interval width | 0.00000252 |

The interval is about 3e-6 wide: the fit does not depend in any meaningful way on which rows were drawn.

---

## 4 — Choosing between models

The checklist asks for *"the model with best score **and** stable cross-validation result"*, which has to be made precise. `pick_best` applies three filters in order:

1. **Test-error band** — keep models within 2% of the best RMSE (or 0.5 accuracy points). Closeness is judged on *error*, never on R². A first version of this function used "R² within 0.5% relative", and near R² = 0.99 that band is wide enough to swallow a model with 15% more RMSE — it duly preferred a Ridge fit at 53.7 km RMSE over a random forest at 46.6 km.
2. **One-standard-error rule** — of those, keep every model whose CV mean is within one standard error (`sd/√k`) of the best. Standard practice from Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* §7.10: a difference smaller than the noise in the CV estimate itself is not a real difference.
3. **Simplicity** — of the survivors, take the simplest. Here that is not only Occam's razor: a linear model keeps the Week 3 gradient-descent-from-scratch code applicable, which no ensemble does.

Result: **`LinearRegression`** for cost regression, **`GradientBoosting`** for `traffic_level`, **`LogisticRegression`** for `cost_band`.

---

## 5 — Hyperparameter tuning

**Why not grid-search the winning model?** The best regressor is `LinearRegression`, which has no hyperparameter worth tuning — a grid search over it is theatre. Ridge is the same model plus exactly one knob, so searching α is a real search whose answer means something: **Week 8 rejected Ridge by hand**, arguing from the Hessian condition number that regularisation fixed the ill-conditioning at an unacceptable cost in accuracy. If that was right, the search should drive α to the bottom of the grid unaided.

### (a) `GridSearchCV` — Ridge α

| α | CV R² | CV sd |
|---:|---:|---:|
| 0.0001 | 0.999714 | 0.000015 |
| 0.001 | 0.999714 | 0.000015 |
| 0.01 | 0.999714 | 0.000015 |
| 0.1 | 0.999704 | 0.000016 |
| 1 | 0.999387 | 0.000019 |
| 10 | 0.997173 | 0.000095 |
| 100 | 0.983673 | 0.000500 |
| 1000 | 0.835241 | 0.002508 |

- Best α: **0.001** · best CV R² 0.999714
- Test R² at sklearn's default α = 1.0: 0.999432 (RMSE ₹75.40)
- Test R² after the search: **0.999707** (RMSE ₹54.16)

The score is **flat for every α ≤ 0.01** and falls monotonically above it. Across that flat region the penalty is too small to do anything, so Ridge there *is* ordinary least squares; every α large enough to actually regularise scores strictly worse. **This reproduces Week 8's hand-argument mechanically** — a search whose answer is "do not regularise" has confirmed the model was already correctly specified. Note the "improved" verdict is against the *default* α = 1.0, and the improvement consists of turning regularisation off.

### (b) `RandomizedSearchCV` — RandomForestRegressor

Search space: `{'n_estimators': [100, 200, 300, 500], 'max_depth': [None, 8, 14, 20], 'min_samples_leaf': [1, 2, 5, 10], 'max_features': [1.0, 0.7, 0.5]}` · 12 draws × 5 folds

|  | Test R² | Test MAE (₹) |
|---|---:|---:|
| 300 trees, defaults | 0.999249 | 52.92 |
| tuned | 0.999187 | 53.42 |

Best params: `{'n_estimators': 300, 'min_samples_leaf': 1, 'max_features': 0.7, 'max_depth': None}` · best CV R² 0.999223

**No improvement** (+0.50 ₹ MAE).

### (c) `GridSearchCV` — the traffic classifier

Grid: `{'n_estimators': [100, 300], 'max_depth': [None, 4, 8], 'min_samples_leaf': [1, 10, 50]}`

|  | Accuracy | F1 (macro) |
|---|---:|---:|
| 300 trees, defaults | 0.5604 | 0.5641 |
| tuned | 0.5696 | 0.5724 |

Best params: `{'max_depth': 4, 'min_samples_leaf': 1, 'n_estimators': 100}` · best CV accuracy 0.5517 · majority baseline 0.3766

**Improved by +0.92 accuracy points.** This is the one search with room to work: an unrestricted forest memorises the training rows, and capping `max_depth` at 4 raised the *test* score while lowering the training one.

### "Confirm score improved" — 2 of 3 did

| Search | Outcome | Why |
|---|---|---|
| Ridge α | improved | …by turning regularisation **off**. Real finding: this problem wants none. |
| RandomForest | **no improvement** | Nothing left to find — a linear model already fits the formula to R² 0.9997. |
| Traffic classifier | improved | The one target with genuine noise, so capping depth helps. |

> **Hyperparameter tuning pays where the data is noisy and the model can overfit it.** Where the target is a formula and the features already span it, tuning has nothing to do — and a search reporting no improvement is evidence the model was specified correctly, not evidence the search was wasted.

---

## 6 — Advanced models

RandomForest (bagging), AdaBoost and GradientBoosting appear in every table above rather than in a section of their own, because the point of including them is comparison.

| Model | Test R² | RMSE (₹) | MAE (₹) | Verdict |
|---|---:|---:|---:|---|
| RandomForest (300) | 0.999249 | 86.75 | 52.92 | Good fit |
| AdaBoost | 0.984135 | 398.59 | 313.86 | Good fit |
| GradientBoosting | 0.999227 | 87.99 | 56.76 | Good fit |
| _LinearRegression (for comparison)_ | 0.999707 | 54.16 | 39.66 | Good fit |

- **RandomForest** — the strongest ensemble here, still beaten by a linear model on the cost target. It wins where the relationship is genuinely non-linear: the road-distance sub-model and the end-to-end model, both of which must learn geography.
- **AdaBoost** — consistently weakest, and worth knowing why. It fits shallow stumps and re-weights the rows it got wrong; on a smooth additive target that spends capacity chasing the noisiest rows instead of representing the smooth part.
- **GradientBoosting** — close behind RandomForest throughout, and it wins the traffic classification outright, where the signal is weak.

**None of the three beats a correctly specified linear model on the cost target.** That is the project's recurring result, and it survived every test here.

---

## Appendix A — the same checklist on the original 1,140-row dataset

The assignment was handed out with `road_trip_data.csv` (1,140 rows), so the tables are repeated on it. It makes two points the wide dataset cannot.

### A.1 Small data is measurably less stable

| Model | Test R² | RMSE (₹) | CV sd (1,140 rows) | CV sd (25,000 rows) | Ratio |
|---|---:|---:|---:|---:|---:|
| LinearRegression | 0.998902 | 25.32 | 0.000037 | 0.000010 | 3.6× |
| Ridge (alpha=1) | 0.997435 | 38.69 | 0.000371 | 0.000030 | 12.5× |
| DecisionTree | 0.987789 | 84.42 | 0.001811 | 0.000092 | 19.7× |
| RandomForest (300) | 0.992900 | 64.37 | 0.000907 | 0.000080 | 11.4× |
| AdaBoost | 0.973382 | 124.64 | 0.002752 | 0.000447 | 6.2× |
| GradientBoosting | 0.996449 | 45.53 | 0.000824 | 0.000083 | 10.0× |

Best: **`LinearRegression`**. The cross-validation standard deviation is **4–20× wider** for the same models and the same code. This is the clearest argument in the project for why step 3 is on the checklist at all — with one split you would never see it.

### A.2 On the original data, traffic classification is impossible — and the metrics say so

| Model | Accuracy | Precision | Recall | F1-score | Verdict |
|---|---:|---:|---:|---:|---|
| LogisticRegression | 0.5088 | 0.1696 | 0.3333 | 0.2248 | Underfitting |
| DecisionTree | 0.5088 | 0.1696 | 0.3333 | 0.2248 | Underfitting |
| RandomForest (300) | 0.5088 | 0.1696 | 0.3333 | 0.2248 | Underfitting |
| AdaBoost | 0.5088 | 0.1696 | 0.3333 | 0.2248 | Underfitting |
| GradientBoosting | 0.5088 | 0.1696 | 0.3333 | 0.2248 | Underfitting |

Every one of the five classifiers lands on **exactly 0.5088** — the majority-class baseline to four decimals — with macro F1 of 0.2248 and a lift of **+0.0 points**. All five are flagged **Underfitting**.

That is not five models failing. It is five models all discovering the same thing: in the original data `traffic_level` is statistically independent of `departure_hour`, so the best available strategy is to always predict the majority class, and each of them found it. Macro F1 ≈ 0.22 is the giveaway — predicting one class out of three scores about ⅓ on macro recall however good the accuracy looks.

**This is why Week 5 injected a rush-hour profile before the traffic classifier was allowed into the project**, and why the accuracy reported in section 1 measures that design decision rather than Indian roads. The comparison between these two tables is the honest disclosure.

---

## Appendix B — what the metrics were actually measuring

The single most important result to come out of this evaluation is not in the checklist. The regression above scores R² 0.9997 — but it is handed the true `toll_cost`, `parking_cost` and `fuel_consumption_litres` as input features, and those are three of the four terms of the cost formula. A user supplies none of them.

So R² 0.9997 is a true number answering the wrong question: *"given the litres burnt and the toll paid, can you total the bill?"* Week 10 rebuilt the prediction path so every intermediate value comes from its own sub-model, and measured the difference.

| Architecture | Test R² | RMSE (₹) | MAE (₹) |
|---|---:|---:|---:|
| A. fed true toll / parking / litres (Week 6) | 0.999707 | 54.16 | 39.66 |
| B. chained, trained on true / served predicted | 0.964602 | 595.37 | 390.56 |
| C. chained, trained on predictions — **shipped** | 0.964542 | 595.88 | 391.16 |
| D. end-to-end, user inputs only | 0.850960 | 1221.67 | 644.65 |

**A → C is the honest headline.** R² falls by 0.0352 and MAE rises from ₹39.66 to ₹391.16. That gap is not a regression in quality — it is the price of the model deriving its own inputs instead of being handed them. **C is what a user receives, so C is what the API now reports.**

And the residual is accounted for rather than left unexplained:

| Source | Std dev (₹) |
|---|---:|
| toll (injected noise) | 218.78 |
| parking (uniform noise) | 50.06 |
| fuel bill, via litres | 479.22 |
| **quadrature sum** | **529.17** |
| **chained model RMSE** | **595.88** |

Ratio 1.13 — close enough to say the chain sits near the floor this dataset allows. **The error is the data's unpredictability, not the model's weakness.**

### A model that failed, reported as a result

Six model families were fitted to `parking_cost` and **every one scored test R² ≤ 0** — worse than predicting the mean:

| Model | Test R² | RMSE (₹) | Verdict |
|---|---:|---:|---|
| LinearRegression | -0.000911 | 50.09 | Underfitting |
| Ridge (alpha=1) | -0.000911 | 50.09 | Underfitting |
| DecisionTree | -1.004573 | 70.88 | Overfitting |
| RandomForest (300) | -0.260242 | 56.20 | Overfitting |
| AdaBoost | -0.001358 | 50.10 | Underfitting |
| GradientBoosting | -0.006166 | 50.22 | Underfitting |

This is the correct answer, not a bug: the generator draws parking uniformly from {0, 40, 60, 80, 120, 150} independently of every other column, so it carries no signal. A model with negative R² has no business in a prediction path, so the API serves the training mean (₹75.09) and the negative result is disclosed through `/api/metrics`. **Reporting it is worth more than tuning until it looks respectable.**

### Where "features beat complexity" stops being true

Week 6 demonstrated that domain understanding beat model complexity: one engineered feature beat two ensembles. End-to-end, that **reverses**:

|  | MAE saved (₹) |
|---|---:|
| Two engineered columns | 67.05 |
| Switching to 300 trees | 474.03 |

Not a contradiction — a boundary condition. When distance was *given*, the only thing the linear model could not express was a product, so handing it the product fixed the one gap. End-to-end, the model must first infer road distance from four coordinates, and that is a genuinely non-linear function of geography — coastlines, mountains, the Gulf of Khambhat. No finite set of hand-made interaction terms captures a map, and partitioning space is what trees do.

> **Feature engineering beats model complexity when the missing structure is something you can write down. When the missing structure is a map, it does not.**

---

## Summary

| Step | Result |
|---|---|
| 1. Metrics | Best regressor `LinearRegression`: R² 0.999707, RMSE ₹54.16, RSS 1.467e+07. `GradientBoosting` on `traffic_level` 57.0%; `LogisticRegression` on `cost_band` 85.6% |
| 2. Overfit / underfit | "Good fit" throughout on the cost target — because it is a formula. Real overfitting appears on `cost_band` and the toll sub-model, where trees show gaps of 0.16–0.37 |
| 3. Validation | 5-fold CV sd ≤ 4e-4 on 25,000 rows, 4–20× wider on 1,140. Bootstrap 95% CI width 2.5e-06 |
| 4. Comparison | One split, one scoring module, one selection rule: error band → one-standard-error → simplicity |
| 5. Tuning | 2 of 3 searches improved. Ridge α → 0 reproduces Week 8 mechanically; RandomForest could not be improved; the traffic classifier gained 0.92 points |
| 6. Advanced models | RandomForest, AdaBoost and GradientBoosting all evaluated; none beats a correctly specified linear model on cost |

**And the finding the checklist did not ask for:** the headline R² was measuring the wrong question. The number that describes a real prediction is **R² 0.9645, MAE ₹391.16** — and that is now what the API reports.
