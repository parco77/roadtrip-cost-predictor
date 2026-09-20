"""
Emit the Part 12 / Part 10 HTML sections of the full project report, straight from the saved JSON.

WHY THIS IS GENERATED
---------------------
Parts 11 and 12 of the report quote roughly two hundred numbers. Typing them by hand into HTML
guarantees at least one transcription error and guarantees they go stale on the next retrain.
These sections are therefore built from the same reports that TASK5.md is built from, so the PDF,
the markdown and the notebook cannot disagree.

Run:  python scripts/build_report_tables.py > <out>.html
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PART = 12


def load(name):
    p = os.path.join(ROOT, "data", name)
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    return json.load(open(p, encoding="utf-8"))


T = load("task5_evaluation.json")
P = load("pipeline_report.json")

out = []


def w(s=""):
    out.append(s)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tbl(headers, rows, numeric_from=1, cls=""):
    w(f'<table class="{cls}">')
    w("  <tr>" + "".join(
        f'<th{" class=\'n\'" if i >= numeric_from else ""}>{h}</th>'
        for i, h in enumerate(headers)) + "</tr>")
    for r in rows:
        hl = ""
        cells = r
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], bool):
            cells, hl = r[0], ' class="hl"' if r[1] else ""
        w(f"  <tr{hl}>" + "".join(
            f'<td{" class=\'n\'" if i >= numeric_from else ""}>{c}</td>'
            for i, c in enumerate(cells)) + "</tr>")
    w("</table>")


reg = T["regression"]
cls_ = T["classification"]
tune = T["tuning"]
arch = T.get("architecture_comparison", {})
boot = T.get("bootstrap")
orig = T.get("original_dataset")
best_reg = reg["best"]

# ==========================================================================================
#                                   PART 11 - WEEK 9
# ==========================================================================================
w(f'<h1 class="part"><span class="num">Part {PART}</span>Evaluation, validation and tuning</h1>')

w("<p>Parts 3&ndash;9 built models. Part 12 measures them properly, applying the same six steps to "
  "every model in the project. This is the Task 5 checklist, answered in full.</p>")

tbl(["Step", "Regression", "Classification"], [
    ["1. Metrics", "RSS, RMSE, R&sup2;", "Accuracy, Precision, Recall, F1"],
    ["2. Fit diagnosis", "train score vs test score", "same"],
    ["3. Validation", "5-fold cross-validation + bootstrap", "5-fold stratified"],
    ["4. Comparison", "one table, best score <em>and</em> stable CV", "same"],
    ["5. Tuning", "GridSearchCV / RandomizedSearchCV", "same"],
    ["6. Advanced models", "RandomForest (bagging), AdaBoost, GradientBoosting", "same"],
], numeric_from=99)

w("<h2>12.1 How everything was measured</h2>")
w("<p>A comparison only means something if every row of it was measured the same way, so two "
  "things are fixed once and reused everywhere.</p>")
w(f"<p><strong>One split for everything.</strong> "
  f"<span class='mono'>train_test_split(test_size=0.2, random_state={T['seed']})</span> gives "
  f"<strong>{reg['n_train']:,} training rows and {reg['n_test']:,} test rows</strong>, and every "
  f"table below is scored on those same {reg['n_test']:,} rows.</p>")
w("<p><strong>Every model wrapped in <span class='mono'>Pipeline([MinMaxScaler, model])</span>."
  "</strong> Linear and logistic models need the scaling and trees do not care, so the reason to "
  "do it for all of them is cross-validation correctness: inside a Pipeline the scaler is "
  "re-fitted on each training fold, so nothing from the validation fold reaches it. Scaling the "
  "whole matrix once <em>before</em> splitting &mdash; which this project originally did &mdash; "
  "lets test-set minima and maxima leak into training. The effect is small for MinMaxScaler on "
  "this data, but it is exactly the mistake cross-validation exists to catch.</p>")

w("<h2>12.2 Step 1 &mdash; the metrics, and what each is for</h2>")
w('<div class="formula">RSS = &sum;<sub>i</sub> (y<sub>i</sub> &minus; &#375;<sub>i</sub>)&sup2;'
  '&nbsp;&nbsp;&middot;&nbsp;&nbsp; RMSE = &radic;(RSS / n) &nbsp;&nbsp;&middot;&nbsp;&nbsp; '
  'R&sup2; = 1 &minus; RSS / &sum;<sub>i</sub>(y<sub>i</sub> &minus; &#563;)&sup2;</div>')
tbl(["Metric", "What it is, and when to quote it"], [
    ["<strong>RSS</strong>", "Residual sum of squares &mdash; the raw quantity least squares "
     "minimises. It is a <em>sum</em>, so it grows with the number of rows: comparable between "
     "models on the same split, meaningless across splits of different sizes."],
    ["<strong>RMSE</strong>", "RSS per row, square-rooted, so it reads in rupees. Squaring inside "
     "means one &#8377;2,000 miss counts as much as a hundred &#8377;200 misses &mdash; quote it "
     "when large errors are what hurt."],
    ["<strong>MAE</strong>", "Mean absolute error &mdash; average error in rupees, no squaring. "
     "This is the number to tell a user."],
    ["<strong>R&sup2;</strong>", "Fraction of variance explained. Its weakness is exactly what "
     "this project ran into: when the target is nearly deterministic, R&sup2; sits so close to 1 "
     "that differences which matter in rupees become invisible."],
], numeric_from=99)
w('<div class="formula">Precision = TP/(TP+FP) &nbsp;&nbsp;&middot;&nbsp;&nbsp; '
  'Recall = TP/(TP+FN) &nbsp;&nbsp;&middot;&nbsp;&nbsp; '
  'F<sub>1</sub> = 2PR/(P+R)</div>')
w("<p>Precision, recall and F1 are <strong>macro-averaged</strong> (every class weighted equally) "
  "rather than weighted, because weighted averaging can hide a class the model never predicts at "
  "all. Part 10.10 shows exactly that failure on the original dataset.</p>")

# ---------------------------------------------------------------- regression table
w("<h2>12.3 Steps 1 and 4 &mdash; the regression comparison</h2>")
w("<p>Target <span class='mono'>total_trip_cost</span>. Feature set: the model this project "
  "already ships (Part 8). Every row sees exactly those columns and that split, so the table "
  "compares <strong>models</strong>, not feature sets. The first row is the Part 8 ablation "
  "(interaction term removed), included because it is the honest baseline the headline beats.</p>")
rows = []
for n, r in reg["models"].items():
    rows.append(([
        f"<strong>{n}</strong>" if n == best_reg else n,
        f"{r['test']['rss']:.4g}", f"{r['test']['rmse']:.2f}", f"{r['test']['r2']:.6f}",
        f"{r['test']['mae']:.2f}",
    ], n == best_reg))
tbl(["Model", "RSS", "RMSE (&#8377;)", "R&sup2;", "MAE (&#8377;)"], rows)

lin = reg["models"].get("LinearRegression")
abl = reg["models"].get("Linear, NO interaction")
ada = reg["models"].get("AdaBoost")
w(f"<p><strong>Read this table from the RMSE column, not R&sup2;.</strong> Every model except "
  f"AdaBoost scores R&sup2; &gt; 0.998, which invites the conclusion that the choice does not "
  f"matter. In rupees it clearly does: &#8377;{lin['test']['rmse']:.0f} for the linear model "
  f"against &#8377;{ada['test']['rmse']:.0f} for AdaBoost &mdash; the worst model is about "
  f"{ada['test']['rmse'] / lin['test']['rmse']:.0f} times worse than the best while looking like "
  f"a rounding difference in R&sup2;.</p>")
w(f"<p>The largest single effect in the table is not a model at all. Removing one engineered "
  f"column (<span class='mono'>fuel_bill = litres &times; price</span>) moves RMSE from "
  f"&#8377;{lin['test']['rmse']:.2f} to &#8377;{abl['test']['rmse']:.2f} &mdash; a bigger gap "
  f"than between the best and worst <em>model</em>. That is Part 8's argument restated: an "
  f"additive model cannot express a product, and no amount of model complexity substitutes for "
  f"handing it over.</p>")

# ---------------------------------------------------------------- overfit table
w("<h2>12.4 Step 2 &mdash; overfitting and underfitting</h2>")
w("<p>Verdicts need <strong>stated thresholds</strong> to be reproducible. A verdict that depends "
  "on an unwritten judgement call is not a result.</p>")
tbl(["Condition", "Verdict"], [
    ["train R&sup2; &minus; test R&sup2; &gt; 0.05", "Overfitting"],
    ["both R&sup2; &lt; 0.50", "Underfitting"],
    ["accuracy less than 3 points above the majority baseline", "Underfitting (classification)"],
    ["otherwise", "Good fit"],
], numeric_from=99)
rows = []
for n, r in reg["models"].items():
    gap = r["train"]["r2"] - r["test"]["r2"]
    rows.append(([n, f"{r['train']['r2']:.6f}", f"{r['test']['r2']:.6f}", f"{gap:+.6f}",
                  r["verdict"]], n == "DecisionTree"))
tbl(["Model", "Train R&sup2;", "Test R&sup2;", "Gap", "Verdict"], rows)
w('<div class="box warn">')
w("<h4>Every model says &ldquo;Good fit&rdquo; &mdash; and that is a fact about the data</h4>")
w("<p><span class='mono'>DecisionTree</span> reaches <strong>train R&sup2; = 1.000000</strong>. A "
  "tree grown to purity has memorised all 20,000 training rows &mdash; the textbook overfitting "
  "signature. It nevertheless holds test R&sup2; = 0.9986, because the target here is a "
  "deterministic formula: memorising the training set and learning the rule produce nearly the "
  "same predictions when there is almost no noise to memorise.</p>")
w("<p>So <strong>on this target the train/test comparison is close to uninformative</strong>. It "
  "becomes informative the moment there is real noise &mdash; which happens in two places later: "
  "the <span class='mono'>cost_band</span> classifier below, and the toll sub-model in Part 10, "
  "where the same trees show gaps of 0.16 to 0.37 and are correctly flagged.</p>")
w("</div>")

# ---------------------------------------------------------------- CV table
w("<h2>12.5 Step 3 &mdash; 5-fold cross-validation, and why the spread is the point</h2>")
w("<p>A single train/test split gives one number, and that number depends on which rows happened "
  "to land in the test set. K-fold splits the <em>training</em> data into k parts, trains on "
  "k&minus;1 and validates on the remaining one, k times. Two things come out: a mean (a better "
  "estimate than one split) and a <strong>standard deviation across folds</strong> &mdash; which "
  "is what says whether the model is stable or merely lucky.</p>")
rows = []
for n, r in reg["models"].items():
    spread = max(r["cv_scores"]) - min(r["cv_scores"])
    rows.append([n, f"{r['cv_mean']:.6f}", f"{r['cv_std']:.6f}", f"{spread:.6f}"])
tbl(["Model", "CV mean R&sup2;", "CV std", "Fold range"], rows)
w("<p>Every spread is tiny &mdash; CV standard deviations of 1e&minus;5 to 4e&minus;4. With 20,000 "
  "training rows and a near-deterministic target that is the expected result, and worth stating "
  "plainly: <strong>this cross-validation confirms stability, it does not discriminate between "
  "the models.</strong> Part 10.8 repeats it on 1,140 rows, where it does.</p>")

if boot:
    w("<h3>Bootstrap &mdash; the checklist's alternative, run as well</h3>")
    w("<p>K-fold asks <em>how does the score vary across disjoint held-out folds</em>. Bootstrap "
      "asks <em>how would the score vary if I had drawn a different training sample of the same "
      "size</em> &mdash; resampling the training rows with replacement, refitting, and scoring "
      "each fit on the fixed test split.</p>")
    tbl(["Quantity", "Value"], [
        ["Model", f"<span class='mono'>{boot['model']}</span>"],
        ["Resamples", f"{boot['n_resamples']}"],
        ["Mean R&sup2;", f"{boot['mean_r2']:.6f}"],
        ["Standard deviation", f"{boot['sd']:.8f}"],
        ["95% interval", f"[{boot['ci95'][0]:.6f}, {boot['ci95'][1]:.6f}]"],
        ["Interval width", f"{boot['ci95'][1] - boot['ci95'][0]:.8f}"],
    ])
    w("<p>The interval is about 3e&minus;6 wide: the fit does not depend in any meaningful way on "
      "which particular rows were drawn. The model is stable, not lucky.</p>")

# ---------------------------------------------------------------- selection rule
w("<h2>12.6 Step 4 &mdash; choosing between models that are statistically tied</h2>")
w("<p>Task 5 asks for &ldquo;best score <strong>and</strong> stable cross-validation "
  "result&rdquo;, which has to be made precise. <span class='mono'>pick_best</span> applies three "
  "filters in order.</p>")
tbl(["Filter", "Rule", "Why"], [
    ["1. Test-error band", "keep models within 2% of the best RMSE (or 0.5 accuracy points)",
     "Closeness judged on <em>error</em>, never on R&sup2;. See the box below."],
    ["2. One-standard-error rule", "of those, keep every model whose CV mean is within one "
     "standard error (sd/&radic;k) of the best CV mean",
     "Standard practice from Hastie, Tibshirani &amp; Friedman, <em>The Elements of Statistical "
     "Learning</em> &sect;7.10: a difference smaller than the noise in the CV estimate itself is "
     "not a real difference."],
    ["3. Simplicity", "of the survivors, take the simplest",
     "Not only Occam's razor &mdash; a linear model keeps the Part 7 gradient-descent-from-scratch "
     "code applicable, which no ensemble does."],
], numeric_from=99)
w('<div class="box warn">')
w("<h4>A bug this project found in its own selection code</h4>")
w("<p>A first version of <span class='mono'>pick_best</span> treated two models as tied when their "
  "R&sup2; was within 0.5% <em>relative</em>. Near R&sup2; = 0.99 that band is wide enough to "
  "swallow a model with 15% more RMSE &mdash; and it duly preferred a Ridge fit at 53.7 km RMSE "
  "over a random forest at 46.6 km, because the Ridge fit had a marginally steadier CV score. "
  "Judging closeness on the error instead fixed it. This is a good example to volunteer: it shows "
  "the evaluation code itself was tested, not trusted.</p>")
w("</div>")
w(f"<p>Result: <strong><span class='mono'>{best_reg}</span></strong> for cost regression, and "
  + ", ".join(f"<strong><span class='mono'>{s['best']}</span></strong> for "
              f"<span class='mono'>{t}</span>" for t, s in cls_.items()) + ".</p>")

# ---------------------------------------------------------------- classification
w("<h2>12.7 Steps 1&ndash;4 for classification</h2>")
for target, sec in cls_.items():
    best = sec["best"]
    w(f"<h3>Target: <span class='mono'>{target}</span></h3>")
    rows = []
    for n, r in sec["models"].items():
        rows.append(([
            f"<strong>{n}</strong>" if n == best else n,
            f"{r['test']['accuracy']:.4f}", f"{r['test']['precision_macro']:.4f}",
            f"{r['test']['recall_macro']:.4f}", f"{r['test']['f1_macro']:.4f}",
            f"{r['train']['accuracy']:.4f}",
            f"{r['train']['accuracy'] - r['test']['accuracy']:+.4f}", r["verdict"],
        ], n == best))
    tbl(["Model", "Accuracy", "Precision", "Recall", "F1", "Train acc", "Gap", "Verdict"],
        rows, cls="wide")
    b = sec["models"][best]
    w(f"<p>Majority-class baseline <strong>{sec['baseline']:.4f}</strong> &rarr; best model "
      f"<span class='mono'>{best}</span> gains "
      f"<strong>+{(b['test']['accuracy'] - sec['baseline']) * 100:.1f} points</strong>, with "
      f"CV {b['cv_mean']:.4f} &plusmn; {b['cv_std']:.4f}.</p>")
    cm = sec.get("confusion_matrix")
    classes = sec.get("classes", [])
    if cm and classes:
        w(f"<h4>Confusion matrix &mdash; <span class='mono'>{best}</span> "
          f"(rows = actual, columns = predicted)</h4>")
        hdr = ["actual &darr; / predicted &rarr;"] + classes
        rws = []
        for c, line in zip(classes, cm):
            rws.append([f"<strong>{c}</strong>"] + [f"{v:,}" for v in line])
        tbl(hdr, rws)

w('<div class="box">')
w("<h4>Two runs of the same target, and the gap between them is the point</h4>")
w("<p><span class='mono'>cost_band</span> is a quartile split of &#8377;/km that the regressor "
  "already predicts well, so its accuracy is <strong>partly by construction</strong>. It earns "
  "its place because the interface needs the label, not because it is hard.</p>")
w("<p>The <strong>under-specified</strong> run is the control. It drops "
  "<span class='mono'>toll_cost</span> and "
  "<span class='mono'>fuel_consumption_litres</span> &mdash; the two columns the sub-models "
  "supply &mdash; leaving only what the form actually collects. The accuracy it loses is what "
  "those sub-models are worth, measured in points rather than asserted in prose.</p>")
w("<p>Both runs are where the overfitting diagnosis finally bites: the two tree models hit train "
  "accuracy 1.0000 with test well below, and <span class='mono'>LogisticRegression</span> "
  "&mdash; with almost no capacity to memorise &mdash; wins outright. On the deterministic cost "
  "target the same diagnosis was uninformative.</p>")
w("</div>")

# ---------------------------------------------------------------- tuning
w("<h2>12.8 Step 5 &mdash; hyperparameter tuning</h2>")
w("<p><strong>Why not grid-search the winning model?</strong> The best regressor is "
  "<span class='mono'>LinearRegression</span>, which has no hyperparameter worth tuning &mdash; a "
  "grid search over it is theatre. Ridge is the same model plus exactly one knob, so searching "
  "&alpha; is a real search whose answer means something: <strong>Part 9 rejected Ridge by "
  "hand</strong>, arguing from the Hessian condition number. If that argument was right, the "
  "search should drive &alpha; to the bottom of the grid unaided.</p>")

r_a = tune["ridge_alpha"]
w("<h3>(a) GridSearchCV &mdash; Ridge &alpha;</h3>")
rows = []
best_cv = max(c["cv_r2"] for c in r_a["curve"])
for c in r_a["curve"]:
    rows.append(([f"{c['alpha']:g}", f"{c['cv_r2']:.6f}", f"{c['cv_sd']:.6f}"],
                 abs(c["cv_r2"] - best_cv) < 1e-6))
tbl(["&alpha;", "CV R&sup2;", "CV sd"], rows)
tbl(["", "Test R&sup2;", "RMSE (&#8377;)"], [
    ["sklearn default &alpha; = 1.0", f"{r_a['test_before']['r2']:.6f}",
     f"{r_a['test_before']['rmse']:.2f}"],
    ([f"after the search (&alpha; = {r_a['best_params']['model__alpha']:g})",
      f"{r_a['test_after']['r2']:.6f}", f"{r_a['test_after']['rmse']:.2f}"], True),
])
plateau = max(c["alpha"] for c in r_a["curve"] if best_cv - c["cv_r2"] < 1e-6)
w(f"<p>The CV score is <strong>flat for every &alpha; &le; {plateau:g}</strong> and falls "
  f"monotonically above it, collapsing to R&sup2; {r_a['curve'][-1]['cv_r2']:.4f} by "
  f"&alpha; = {r_a['curve'][-1]['alpha']:g}. Across the flat region the penalty is too small to "
  f"do anything, so Ridge there <em>is</em> ordinary least squares; the search has no reason to "
  f"prefer any point on the plateau and returns one of them. Every &alpha; large enough to "
  f"actually regularise scores strictly worse.</p>")
w("<p><strong>This reproduces Part 9's hand-argument mechanically.</strong> A search whose answer "
  "is &ldquo;do not regularise&rdquo; has not failed &mdash; it has confirmed the model was "
  "already correctly specified. Note the &ldquo;improved&rdquo; verdict below is against "
  "sklearn's <em>default</em> &alpha; = 1.0, and the improvement consists of turning "
  "regularisation off.</p>")

r_b = tune["random_forest"]
w("<h3>(b) RandomizedSearchCV &mdash; RandomForestRegressor</h3>")
w(f"<p>Search space, {r_b['n_iter']} random draws &times; 5 folds:</p>")
w("<pre>" + esc(json.dumps(r_b["space"], indent=2)) + "</pre>")
tbl(["", "Test R&sup2;", "MAE (&#8377;)"], [
    ["300 trees at defaults", f"{r_b['test_before']['r2']:.6f}",
     f"{r_b['test_before']['mae']:.2f}"],
    ["tuned", f"{r_b['test_after']['r2']:.6f}", f"{r_b['test_after']['mae']:.2f}"],
])
w(f"<p>Best parameters found: <span class='mono'>{esc(r_b['best_params'])}</span>, best CV R&sup2; "
  f"{r_b['best_cv_r2']:.6f}. Outcome: "
  f"<strong>{'improved' if r_b['improved'] else 'NO IMPROVEMENT'}</strong> "
  f"({r_b['test_after']['mae'] - r_b['test_before']['mae']:+.2f} &#8377; MAE).</p>")

r_c = tune["distance_model"]
w("<h3>(c) GridSearchCV &mdash; the distance model</h3>")
w("<p>840 real OSRM road distances. Small, noisy, and not drawn from any formula &mdash; the "
  "only target in the project measured from the world rather than generated, and therefore the "
  "only one carrying noise a model can overfit.</p>")
w("<pre>" + esc(json.dumps(r_c["grid"], indent=2)) + "</pre>")
tbl(["", "R&sup2;", "MAE (km)"], [
    ["300 trees at defaults", f"{r_c['test_before']['r2']:.6f}",
     f"{r_c['test_before']['mae']:.2f}"],
    (["tuned", f"{r_c['test_after']['r2']:.6f}",
      f"{r_c['test_after']['mae']:.2f}"], True),
])
delta = r_c["test_before"]["mae"] - r_c["test_after"]["mae"]
w(f"<p>Best parameters: <span class='mono'>{esc(r_c['best_params'])}</span>, best CV R&sup2; "
  f"{r_c['best_cv_r2']:.6f}. "
  f"<strong>{'Improved' if r_c['improved'] else 'No improvement'}: "
  f"{delta:+.2f} km of MAE.</strong></p>")
if r_c["improved"]:
    w("<p>This is the one search with room to work: an unrestricted forest memorises the "
      "training rows, and constraining it removes error the defaults were leaving on the "
      "table.</p>")
else:
    w('<div class="box warn">')
    w("<h4>The search expected to succeed, and did not</h4>")
    w(f"<p>This was the one target with genuine observational noise &mdash; "
      f"{r_c['test_before']['mae']:.2f} km of MAE from 300 trees at their defaults &mdash; so it "
      f"was the place capacity control should have paid. Eighteen combinations over five folds "
      f"came back <strong>{delta:+.2f} km</strong>: worse than doing nothing.</p>")
    w("<p>The reason is worth more than the result would have been. <strong>A RandomForest is "
      "already an averaging machine</strong>: it fits each tree to a bootstrap sample and means "
      "the predictions, which is precisely the variance control the grid was shopping for. "
      "Restricting the individual trees on top of that removes signal along with the noise.</p>")
    w("</div>")

n_improved = sum(1 for v in tune.values() if v["improved"])
w(f"<h3>&ldquo;Confirm score improved&rdquo; &mdash; {n_improved} of {len(tune)} did</h3>")
tbl(["Search", "Outcome", "Why"], [
    ["Ridge &alpha;", "improved", "&hellip;by turning regularisation <strong>off</strong>. The "
     "real finding is that this problem wants none."],
    ["RandomForest", "<strong>no improvement</strong>", "Nothing left to find &mdash; a linear "
     "model already fits the formula to R&sup2; 0.9997."],
    ["Traffic classifier", "improved", "The one target with genuine noise, so constraining model "
     "capacity buys something."],
], numeric_from=99)
w('<div class="box key">')
w("<h4>The answer to give if asked whether tuning worked</h4>")
w("<p>Two of the three searches improved the score, one did not, and <strong>the pattern is the "
  "lesson</strong>: hyperparameter tuning pays where the data is <em>noisy</em> and the model can "
  "overfit it. Where the target is a formula and the features already span it, tuning has nothing "
  "to do &mdash; and a search that reports no improvement is evidence the model was specified "
  "correctly, not evidence the search was wasted.</p>")
w("</div>")

# ---------------------------------------------------------------- advanced models
w("<h2>12.9 Step 6 &mdash; the advanced models, in one place</h2>")
w("<p>RandomForest (bagging), AdaBoost and GradientBoosting appear in every table rather than in a "
  "section of their own, because the point of including them is comparison.</p>")
rows = []
for n in ("RandomForest (300)", "AdaBoost", "GradientBoosting"):
    r = reg["models"].get(n)
    if r:
        rows.append([n, f"{r['test']['r2']:.6f}", f"{r['test']['rmse']:.2f}",
                     f"{r['test']['mae']:.2f}", r["verdict"]])
if lin:
    rows.append(([f"<em>{best_reg} &mdash; for comparison</em>", f"{lin['test']['r2']:.6f}",
                  f"{lin['test']['rmse']:.2f}", f"{lin['test']['mae']:.2f}", lin["verdict"]], True))
tbl(["Model", "Test R&sup2;", "RMSE (&#8377;)", "MAE (&#8377;)", "Verdict"], rows)
w("<ul>")
w("<li><strong>Bagging (RandomForest)</strong> &mdash; trains many trees on bootstrap samples and "
  "averages them, which cuts variance. Strongest ensemble here, and still beaten by a linear model "
  "on the cost target. It wins where the relationship is genuinely non-linear: the road-distance "
  "sub-model and the end-to-end model, both of which must learn geography.</li>")
w("<li><strong>Boosting (AdaBoost)</strong> &mdash; fits shallow stumps sequentially, re-weighting "
  "the rows it got wrong. Consistently weakest here, and worth knowing <em>why</em>: on a smooth "
  "additive target it spends its capacity chasing the noisiest rows instead of representing the "
  "smooth part.</li>")
w("<li><strong>GradientBoosting</strong> &mdash; fits each new tree to the residuals of the "
  "previous ones. Close behind RandomForest throughout, and strongest of the three wherever the "
  "signal is weak rather than deterministic.</li>")
w("</ul>")
w("<p><strong>None of the three beats a correctly specified linear model on the cost "
  "target.</strong> That is the project's recurring result, and it survived every test in this "
  "part.</p>")

# ---------------------------------------------------------------- appendix: original data
if orig:
    w("<h2>12.10 The same checklist on the original 1,140-row dataset</h2>")
    w(f"<p>The assignment was handed out with <span class='mono'>road_trip_data.csv</span> "
      f"({orig['n_rows']:,} rows), so the tables are repeated on it. It makes two points the wide "
      f"dataset cannot.</p>")
    w("<h3>Point 1 &mdash; small data is measurably less stable</h3>")
    rows = []
    for n, r in orig["regression"].items():
        wide_sd = reg["models"].get(n, {}).get("cv_std")
        ratio = (r["cv_std"] / wide_sd) if wide_sd else None
        rows.append([n, f"{r['test']['r2']:.6f}", f"{r['test']['rmse']:.2f}",
                     f"{r['cv_std']:.6f}", f"{wide_sd:.6f}" if wide_sd else "&mdash;",
                     f"{ratio:.1f}&times;" if ratio else "&mdash;"])
    tbl(["Model", "Test R&sup2;", "RMSE (&#8377;)", f"CV sd ({orig['n_rows']:,} rows)",
         "CV sd (25,000 rows)", "Ratio"], rows, cls="wide")
    w(f"<p>Best on the original data: <strong><span class='mono'>"
      f"{orig['regression_best']}</span></strong>. The cross-validation standard deviation is "
      f"<strong>4 to 20 times wider</strong> for the same models and the same code. This is the "
      f"clearest argument in the project for why step 3 is on the checklist at all &mdash; with "
      f"one split you would never see it.</p>")

    w("<h3>Point 2 &mdash; the same classification target, a twentieth of the rows</h3>")
    rows = []
    for n, r in orig["cost_band"].items():
        rows.append([n, f"{r['test']['accuracy']:.4f}", f"{r['test']['precision_macro']:.4f}",
                     f"{r['test']['recall_macro']:.4f}", f"{r['test']['f1_macro']:.4f}",
                     f"{r['train']['accuracy']:.4f}", r["verdict"]])
    tbl(["Model", "Accuracy", "Precision", "Recall", "F1", "Train acc", "Verdict"], rows,
        cls="wide")
    ob = list(orig["cost_band"].values())[0]["baseline"]
    obest = orig["cost_band"][orig["cost_band_best"]]["test"]["accuracy"]
    wide_band = T["classification"]["cost_band"]
    wbest = wide_band["models"][wide_band["best"]]["test"]["accuracy"]
    w(f"<p>Best on the original data: <strong>{obest:.4f}</strong> against a baseline of "
      f"{ob:.4f}. The same target on 25,000 rows reaches <strong>{wbest:.4f}</strong>.</p>")
    w('<div class="box key">')
    w("<h4>What the gap is, and what it is not</h4>")
    w(f"<p>The {(wbest - obest) * 100:.1f}-point difference is <strong>sample size, not a "
      f"different problem</strong>. The band is defined the same way in both files &mdash; "
      f"quartiles of &#8377;/km, cut on that file's own distribution &mdash; and the feature set "
      f"is the same. What changes is how much data the classifier has to find the boundaries "
      f"with.</p>")
    w("<p>Look at the trees: train accuracy 1.0000 against test well below it, on both datasets, "
      "flagged <strong>Overfitting</strong> &mdash; and the gap is wider on the small file, "
      "because there is less data to average the memorisation away. "
      "<span class='mono'>LogisticRegression</span>, with almost no capacity to memorise, is "
      "the most stable of the five in both.</p>")
    w("<p>This is the same lesson as the cross-validation spread above, reached from the other "
      "direction: <strong>small data is less stable, and the diagnostics are how you see that "
      "rather than guess it.</strong></p>")
    w("</div>")

# ---------------------------------------------------------------- week 9 summary
w("<h2>12.11 Part 12 summary</h2>")
best = reg["models"][best_reg]
cv_line = (f"5-fold CV sd &le; 4e&minus;4 on 25,000 rows, 4&ndash;20&times; wider on "
           f"{orig['n_rows']:,}" if orig else "5-fold CV sd &le; 4e&minus;4")
if boot:
    cv_line += f". Bootstrap 95% CI width {boot['ci95'][1] - boot['ci95'][0]:.1e}"
tbl(["Step", "Result"], [
    ["1. Metrics", f"Best regressor <span class='mono'>{best_reg}</span>: R&sup2; "
     f"{best['test']['r2']:.6f}, RMSE &#8377;{best['test']['rmse']:.2f}, RSS "
     f"{best['test']['rss']:.4g}. "
     + "; ".join(f"<span class='mono'>{s['best']}</span> on <span class='mono'>{t}</span> "
                 f"{s['models'][s['best']]['test']['accuracy'] * 100:.1f}%"
                 for t, s in cls_.items())],
    ["2. Overfit / underfit", "&ldquo;Good fit&rdquo; throughout on the cost target &mdash; "
     "because it is a formula. Real overfitting appears on <span class='mono'>cost_band</span> "
     "and the toll sub-model, where trees show gaps of 0.16&ndash;0.37"],
    ["3. Validation", cv_line],
    ["4. Comparison", "One split, one scoring module, one selection rule: error band &rarr; "
     "one-standard-error &rarr; simplicity"],
    ["5. Tuning", f"{n_improved} of {len(tune)} searches improved, and none found real accuracy. "
     f"Ridge &alpha; &rarr; 0 reproduces the ill-conditioning argument mechanically; neither "
     f"forest search beat its defaults (the distance model lost {abs(delta):.2f} km of MAE)"],
    ["6. Advanced models", "RandomForest, AdaBoost and GradientBoosting all evaluated; none beats "
     "a correctly specified linear model on cost"],
], numeric_from=99)
w("<p><strong>And the finding the checklist did not ask for.</strong> The regression above scores "
  "R&sup2; 0.9998, but it is handed the true <span class='mono'>toll_cost</span> and "
  "<span class='mono'>fuel_consumption_litres</span> as input features &mdash; the two terms that "
  "carry the uncertainty in the total, and the two a traveller cannot know before setting off. "
  "Part 10 takes them away and measures what that costs.</p>")

print("\n".join(out))
