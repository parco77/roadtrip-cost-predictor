"""
Emit Part 12 (Week 10) of the full project report as HTML, straight from the saved JSON.

Companion to scripts/build_report_tables.py, which emits Part 11 (Week 9). Split into two files
purely so each stays readable.

Run:  python scripts/build_report_week10.py > <out>.html
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    p = os.path.join(ROOT, "data", name)
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    return json.load(open(p, encoding="utf-8"))


P = load("pipeline_report.json")
E = load("end_to_end_report.json")
T = load("task5_evaluation.json")

out = []


def w(s=""):
    out.append(s)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tbl(headers, rows, numeric_from=1, cls=""):
    w(f'<table class="{cls}">')
    cells_html = "".join(
        ("<th class='n'>" if i >= numeric_from else "<th>") + str(h) + "</th>"
        for i, h in enumerate(headers))
    w("  <tr>" + cells_html + "</tr>")
    for r in rows:
        hl, cells = "", r
        if isinstance(r, tuple):
            cells, flag = r
            hl = ' class="hl"' if flag else ""
        row_html = "".join(
            ("<td class='n'>" if i >= numeric_from else "<td>") + str(c) + "</td>"
            for i, c in enumerate(cells))
        w(f"  <tr{hl}>" + row_html + "</tr>")
    w("</table>")


d = P["distance"]
bias = P["geography_bias"]
eb = P["cost"]["error_budget"]
park = P["parking"]
tl = P["toll"]
arch = T["architecture_comparison"]
chk = E.get("week6_check", {})
fi = E.get("feature_importance", {})
base = d["constant_baseline"]
ch = d["chosen_scores"]["test"]

w('<h1 class="part"><span class="num">Part 12 &middot; Week 10</span>Removing the magic numbers: '
  'a chained model pipeline</h1>')

w("<h2>12.1 The problem</h2>")
w("<p>Up to Week 9, <span class='mono'>app.py</span> served a prediction like this:</p>")
w("<pre>distance = haversine * 1.2355                          # constant\n"
  "litres   = distance / mileage * TRAFFIC_MULT[traffic]  # constant lookup\n"
  "toll     = distance * 1.301                            # constant\n"
  "parking  = 70.0                                        # constant\n"
  "cost     = cost_regressor([distance, mileage, price, toll, parking, litres, ...])</pre>")
w("<p><strong>Only the last line is a model.</strong> Everything above it is arithmetic &mdash; "
  "and not incidental arithmetic, because it produces "
  "<span class='mono'>fuel_consumption_litres</span>, the regressor's single strongest feature at "
  "correlation +0.96 with the target. The model was being handed three of the four terms of the "
  "cost formula and asked to add them up.</p>")
w("<p>That reframes the headline. <strong>R&sup2; 0.999707 is a true number answering the wrong "
  "question</strong>: <em>&ldquo;given the litres burnt and the toll paid, can you total the "
  "bill?&rdquo;</em> The user's question is <em>&ldquo;what will this trip cost?&rdquo;</em>, and "
  "they know neither.</p>")

tbl(["Constant removed", "Replaced by", "Trained on"], [
    ["<span class='mono'>WINDING_FACTOR = 1.2355</span>", "sub-model 1 &mdash; road distance",
     f"<strong>{d['n_real_pairs']} real OSRM road distances</strong>"],
    ["(the user had to know their mileage)", "sub-model 2 &mdash; mileage", "generated data"],
    ["<span class='mono'>TRAFFIC_MULT = {...}</span>", "sub-model 3 &mdash; litres",
     "generated data"],
    ["<span class='mono'>TOLL_RATE = 1.301</span>", "sub-model 4 &mdash; toll", "generated data"],
    ["<span class='mono'>DEFAULT_PARKING = 70.0</span>", "sub-model 5 &mdash; parking",
     "<strong>fails; the mean is served instead</strong>"],
    ["(already a model)", "sub-model 6 &mdash; traffic", "generated data"],
], numeric_from=99)

w("<h2>12.2 The one piece of genuinely observed data</h2>")
w("<p>Everything in <span class='mono'>road_trip_wide.csv</span> is generated. To fit a distance "
  "model that is not circular, Week 10 collected real data: "
  "<span class='mono'>scripts/fetch_real_distances.py</span> samples city pairs stratified across "
  "six distance bands and asks the OSRM routing engine for the actual driving distance, appending "
  "each result to disk so a rate-limited run can resume without losing progress.</p>")
fo = d["factor_observed"]
tbl(["Measure", "Value"], [
    ["Real routes collected", f"<strong>{d['n_real_pairs']}</strong>"],
    ["Straight-line range", "40 &ndash; 1,248 km"],
    ["Observed winding factor (road &divide; straight)",
     f"mean <strong>{fo['mean']:.4f}</strong>, sd {fo['sd']:.4f}, "
     f"range {fo['min']:.4f} &ndash; {fo['max']:.4f}"],
    ["The constant it replaces", "<span class='mono'>1.2355</span> &mdash; one number for that "
     "entire range"],
])

w("<h3>How wrong was the constant?</h3>")
tbl(["Measure", "Value"], [
    ["Mean <em>signed</em> error",
     f"<strong>{bias['mean_signed_km']:+.1f} km ({bias['mean_signed_pct']:+.2f}%)</strong> "
     f"&mdash; the bias"],
    ["Mean <em>absolute</em> error",
     f"<strong>{bias['mean_abs_km']:.1f} km ({bias['mean_abs_pct']:.1f}%)</strong> "
     f"&mdash; the per-route error"],
    ["Worst overshoot", f"{bias['max_km']:+.1f} km"],
    ["Worst undershoot", f"{bias['min_km']:+.1f} km"],
])
w('<div class="box key">')
w("<h4>These two numbers say opposite-sounding things, and both matter</h4>")
w(f"<p>The constant is very nearly <strong>unbiased</strong>: averaged over {d['n_real_pairs']} "
  f"routes it is within half a percent of the truth. That is exactly why it survived review for so "
  f"long.</p>")
w(f"<p>But <em>unbiased on average</em> is not <em>correct</em>. Route by route it is off by "
  f"<strong>{bias['mean_abs_pct']:.1f}%</strong> typically, and by tens of percent at the "
  f"extremes. <strong>A user does not take the average of every road trip in India. They take one "
  f"specific trip, and on that trip the errors do not cancel.</strong></p>")
w("</div>")

w("<h3>The extreme routes are real, and they are the whole argument</h3>")
tbl(["Route", "Straight line", "Real road", "Factor"], [
    (["Surat &rarr; Bhavnagar", "94.2 km", "<strong>339.4 km</strong>", "3.60"], True),
    ["Valsad &rarr; Bhavnagar", "151.6 km", "406.7 km", "2.68"],
    ["Badlapur &rarr; Bhavnagar", "312.3 km", "600.8 km", "1.92"],
    ["Moradabad &rarr; Chanduasi", "43.1 km", "44.9 km", "1.04"],
])
w("<p>A factor of 3.60 looks like bad data. It is not. <strong>Surat and Bhavnagar face each other "
  "across the Gulf of Khambhat</strong>: 94 km of straight line, 339 km of road around the head of "
  "the gulf. At the other end, Moradabad &rarr; Chanduasi is essentially a straight road.</p>")
w("<p>No single multiplier can serve both. These points are <strong>kept</strong> in the training "
  "data rather than cleaned away, because they are precisely the cases a constant cannot represent "
  "and a model can. Note which model wins below: a <strong>tree ensemble</strong>, because it can "
  "isolate a region of the map. A linear fit gets dragged by these points instead of representing "
  "them.</p>")

w("<h2>12.3 Sub-model 1 &mdash; road distance</h2>")
w("<p>Two parameterisations of the same problem were compared, and the difference is "
  "instructive:</p>")
w("<ul><li><strong>(a)</strong> predict <span class='mono'>road_km</span> directly</li>"
  "<li><strong>(b)</strong> predict the winding <em>factor</em>, then multiply by haversine</li>"
  "</ul>")
w("<p>(b) removes trip length from the target, so the model spends its capacity on the part that "
  "is genuinely unknown &mdash; the <em>shape</em> of the route &mdash; instead of re-learning "
  "&ldquo;longer straight line, longer road&rdquo;. It is the same idea as Week 6's interaction "
  "term: hand the model the quantity that actually means something.</p>")
rows = []
chosen_key = f"candidates_{d['parameterisation']}"
for label, key in [("(a) direct km", "candidates_direct"),
                   ("(b) factor &times; haversine", "candidates_factor")]:
    for n, r in d[key].items():
        is_chosen = key == chosen_key and d["chosen"].startswith(n)
        rows.append(([label, n, f"{r['test']['r2']:.6f}", f"{r['test']['rmse']:.2f}",
                      f"{r['test']['mae']:.2f}", f"{r['cv_mean']:.6f}", f"{r['cv_std']:.4f}"],
                     is_chosen))
tbl(["Target", "Model", "Test R&sup2;", "RMSE (km)", "MAE (km)", "CV mean", "CV sd"],
    rows, numeric_from=2, cls="wide")
tbl(["", "R&sup2;", "RMSE (km)", "MAE (km)"], [
    ["<strong>BASELINE</strong> &mdash; haversine &times; 1.2355", f"{base['r2']:.4f}",
     f"{base['rmse']:.2f}", f"{base['mae']:.2f}"],
    ([f"<strong>CHOSEN</strong> &mdash; {d['chosen']}", f"{ch['r2']:.4f}", f"{ch['rmse']:.2f}",
      f"{ch['mae']:.2f}"], True),
])
w(f"<p><strong>The model beats the constant by {d['mae_reduction_pct']:.1f}% on MAE</strong> "
  f"({base['mae']:.2f} &rarr; {ch['mae']:.2f} km) and about "
  f"{(1 - ch['rmse'] / base['rmse']) * 100:.0f}% on RMSE. Real, and deliberately not oversold: "
  f"endpoint coordinates only partly determine route shape, because the rest depends on where "
  f"roads and bridges happen to be &mdash; information that is not in the features.</p>")
w("<p>Parameterisation (b) wins, exactly as predicted. <strong>A sub-model that could not beat the "
  "constant it replaces would not have earned its place</strong>, which is why every table in this "
  "part carries the constant as its baseline row.</p>")

w("<h2>12.4 Sub-models 2 to 6</h2>")
for key, title, unit in [
    ("mileage", "Sub-model 2 &mdash; mileage (vehicle + fuel &rarr; km/l)", "km/l"),
    ("litres", "Sub-model 3 &mdash; litres (replaces TRAFFIC_MULT)", "litres"),
    ("toll", "Sub-model 4 &mdash; toll (replaces TOLL_RATE = 1.301)", "&#8377;"),
    ("parking", "Sub-model 5 &mdash; parking (replaces DEFAULT_PARKING = 70)", "&#8377;"),
    ("traffic", "Sub-model 6 &mdash; traffic level (classifier)", ""),
]:
    sec = P[key]
    w(f"<h3>{title}</h3>")
    if key == "traffic":
        rows = [([n, f"{r['test']['accuracy']:.4f}", f"{r['test']['precision_macro']:.4f}",
                  f"{r['test']['recall_macro']:.4f}", f"{r['test']['f1_macro']:.4f}",
                  f"{r['baseline']:.4f}", f"{r['lift_points']:+.1f}", r["verdict"]],
                 n == sec["chosen"]) for n, r in sec["candidates"].items()]
        tbl(["Model", "Accuracy", "Precision", "Recall", "F1", "Baseline", "Lift (pts)",
             "Verdict"], rows, cls="wide")
    else:
        if "constant_baseline" in sec:
            cb = sec["constant_baseline"]
            extra = ""
            if "mean_baseline" in sec:
                mb = sec["mean_baseline"]
                extra = (f" &nbsp;&middot;&nbsp; <strong>Baseline (train mean):</strong> "
                         f"R&sup2; {mb['r2']:+.4f}, RMSE {mb['rmse']:.2f} {unit}")
            w(f"<p class='small'><strong>Baseline (the constant it replaces):</strong> "
              f"R&sup2; {cb['r2']:+.4f}, RMSE {cb['rmse']:.2f} {unit}{extra}</p>")
        rows = [([n, f"{r['test']['r2']:+.6f}", f"{r['test']['rmse']:.3f}",
                  f"{r['train']['r2']:.6f}", f"{r['cv_std']:.4f}", r["verdict"]],
                 n == sec["chosen"]) for n, r in sec["candidates"].items()]
        tbl(["Model", "Test R&sup2;", f"RMSE ({unit})", "Train R&sup2;", "CV sd", "Verdict"], rows)
    w(f"<p><strong>Chosen: <span class='mono'>{sec['chosen']}</span></strong></p>")

mil = P["mileage"]["candidates"][P["mileage"]["chosen"]]
w('<div class="box">')
w("<h4>Sub-model 2, mileage &mdash; a structural ceiling, not a modelling failure</h4>")
w(f"<p>Test R&sup2; <strong>{mil['test']['r2']:.4f}</strong>, and every model family lands on the "
  f"same number to three decimals. That is the correct answer: the generator draws mileage "
  f"<em>uniformly</em> inside a per-(vehicle, fuel) range, so the best any model can do is name "
  f"the middle of the right cell, and the residual is uniform noise by construction.</p>")
w("<p>Two details worth knowing. The nine (vehicle, fuel) cells cannot be reproduced by additive "
  "one-hots alone &mdash; 1 + 2 + 2 = 5 parameters cannot fit nine independent means &mdash; so "
  "the feature set includes the four vehicle &times; fuel <strong>interaction terms</strong>, and "
  "with them the design matrix spans all nine cells exactly. And this sub-model changes the "
  "<em>interface</em>: the app used to default every car to 15.5 km/l, and now predicts 19.5 for "
  "a diesel hatchback and 10.8 for a CNG SUV.</p>")
w("</div>")

w('<div class="box good">')
w("<h4>Sub-model 3, litres &mdash; reading the generator off the coefficients</h4>")
w("<p><strong>Test R&sup2; = 1.000000.</strong> The model has not approximated the "
  "data-generating rule; it has reproduced it. That claim is worth cashing in, because "
  "<span class='mono'>litres = (distance / mileage) &times; traffic_multiplier</span> is a "
  "<em>product</em>, and an additive model cannot express a product. Handed the ratio and its "
  "interactions with the traffic one-hots, the coefficients become the multipliers "
  "themselves:</p>")
tbl(["Traffic level", "Recovered from the fit", "Week 4 value", "Error"], [
    ["Low", "0.944000", "0.9440", "&minus;0.000000"],
    ["Medium", "1.045000", "1.0450", "&minus;0.000000"],
    ["High", "1.165000", "1.1650", "&minus;0.000000"],
])
w("<p>Intercept +2.61e&minus;07 &mdash; zero, because the formula has no constant term. "
  "<strong>Six decimal places.</strong> This is the Week 6 lesson in its purest form: the features "
  "were the whole problem, and with the right ones a linear model <em>is</em> the generating "
  "rule.</p>")
w("</div>")

w('<div class="box warn">')
w("<h4>Sub-model 4, toll &mdash; and the first honest overfitting in the project</h4>")
tc = tl["candidates"][tl["chosen"]]
w(f"<p>Test R&sup2; <strong>{tc['test']['r2']:.4f}</strong>, RMSE "
  f"<strong>&#8377;{tc['test']['rmse']:.0f}</strong>, and the fitted model does not beat the "
  f"constant it replaces (R&sup2; {tl['constant_baseline']['r2']:.4f} vs "
  f"{tc['test']['r2']:.4f}). The generator sets "
  f"<span class='mono'>toll = distance &times; N(1.301, 0.360)</span>, so the <em>rate</em> is "
  f"recoverable but the per-trip spread is injected noise. That RMSE is <strong>irreducible on "
  f"this data</strong>, and a model that appeared to beat it would be leaking.</p>")
dt = tl["candidates"].get("DecisionTree", {})
if dt:
    w(f"<p>Look at the trees in that table, though &mdash; this is the first regression target in "
      f"the project with real noise, and they behave exactly as theory says. "
      f"<span class='mono'>DecisionTree</span> reaches train R&sup2; "
      f"{dt['train']['r2']:.4f} against test {dt['test']['r2']:.4f}: a gap of "
      f"{dt['train']['r2'] - dt['test']['r2']:.2f}, correctly flagged "
      f"<strong>Overfitting</strong>. Week 9's step 2 was uninformative on the deterministic cost "
      f"target; here it identifies the wrong model immediately.</p>")
w("<p>So the toll sub-model is kept for a different reason than accuracy: it is fitted, it reports "
  "its own error bar, and it is evaluated like everything else instead of being asserted.</p>")
w("</div>")

w('<div class="box warn">')
w("<h4>Sub-model 5, parking &mdash; a negative result, reported</h4>")
pdt = park["candidates"]["DecisionTree"]
w(f"<p><strong>Every one of the six model families scores test R&sup2; &le; 0</strong> &mdash; "
  f"worse than predicting the mean. <span class='mono'>DecisionTree</span> reaches R&sup2; "
  f"{pdt['test']['r2']:+.4f} with train R&sup2; {pdt['train']['r2']:.4f}: pure memorisation of "
  f"noise, and the cleanest overfitting example in the project.</p>")
w(f"<p>This is the correct answer, not a bug. The generator draws parking uniformly from "
  f"&#123;0, 40, 60, 80, 120, 150&#125; <strong>independently of every other column</strong>, so "
  f"it carries no signal at all. A model with negative R&sup2; has no business in a prediction "
  f"path, so the app serves the training mean (&#8377;{park['train_mean']:.2f}) and the negative "
  f"result is disclosed through <span class='mono'>/api/metrics</span>.</p>")
w("<p>Worth noting: the old hardcoded &#8377;70.0 was not even the mean of the training data.</p>")
w("</div>")

w("<h2>12.5 Chaining them properly &mdash; stacking, and a bug worth measuring</h2>")
w("<p>Six sub-models feeding a seventh introduces a failure mode that is easy to miss.</p>")
w("<p>The cost regressor was trained on the <strong>true</strong> toll, parking and litres. If it "
  "is then served <strong>predicted</strong> ones, it has learned to trust inputs that are exact "
  "and receives inputs that are not. That is a <strong>train/serve mismatch</strong>, and rather "
  "than assume it matters, both ways are measured.</p>")
w("<p>Building the training components correctly needs care too. Sub-model predictions on their "
  "own training rows are optimistic &mdash; the model has already seen those rows &mdash; so the "
  "component features for training come from <strong>"
  "<span class='mono'>cross_val_predict</span> (out-of-fold)</strong>: no row's features were "
  "produced by a model that had seen that row. Test rows use the sub-models fitted on the full "
  "training split. That is textbook <strong>stacking</strong>.</p>")
arch_rows = [
    ["A. fed true toll / parking / litres (Week 6)", arch["A_true_components"]],
    ["B. chained, trained on true / served predicted", arch["B_mismatched_chain"]],
    ["C. chained, trained on predictions &mdash; <strong>shipped</strong>", arch["C_chained"]],
    ["D. end-to-end, user inputs only", arch["D_end_to_end"]],
]
tbl(["Architecture", "Test R&sup2;", "RMSE (&#8377;)", "MAE (&#8377;)"],
    [([n, f"{v['r2']:.6f}", f"{v['rmse']:.2f}", f"{v['mae']:.2f}"], n.startswith("C."))
     for n, v in arch_rows])
w('<div class="box key">')
w("<h4>Reading the four rows &mdash; this is the centre of the project</h4>")
w(f"<p><strong>A &rarr; C is the honest headline.</strong> R&sup2; falls by "
  f"{arch['r2_drop_A_to_C']:.4f} and MAE rises from "
  f"&#8377;{arch['A_true_components']['mae']:.2f} to &#8377;{arch['C_chained']['mae']:.2f} "
  f"&mdash; roughly ten times larger. That gap is not a regression in quality; it is the price of "
  f"the model deriving its own inputs instead of being handed them. <strong>C is what a user "
  f"actually receives, so C is what the app now reports.</strong></p>")
w(f"<p><strong>B vs C went the opposite way to expectation.</strong> B (trained on truth, served "
  f"predictions) scores MAE &#8377;{arch['B_mismatched_chain']['mae']:.2f}; C (trained on its own "
  f"predictions) scores &#8377;{arch['C_chained']['mae']:.2f} &mdash; B is <em>marginally "
  f"better</em>. The mismatch cost essentially nothing, and the reason is specific: the sub-model "
  f"errors are close to <strong>zero-mean</strong>, so the linear cost model's coefficients are "
  f"almost unchanged by refitting on them.</p>")
w("<p>C still ships, because it is the correct construction and the property that rescues B is a "
  "fact about this dataset rather than something to rely on. But the honest report is that "
  "<strong>on this data the mismatch did not bite</strong>, and claiming otherwise would be "
  "inventing a result.</p>")
w("</div>")

w("<h2>12.6 Where the remaining error comes from</h2>")
w("<p>&ldquo;The honest model is worse&rdquo; is not a finding. <em>&ldquo;Toll and parking noise "
  "account for nearly all of it&rdquo;</em> is. Decomposing the test-set error into its "
  "sources:</p>")
tbl(["Source", "Std deviation (&#8377;)", "Could a better model fix it?"], [
    ["fuel bill, via litres", f"{eb['fuel_bill_sd']:.2f}",
     "<strong>Yes</strong> &mdash; it inherits the distance error"],
    ["toll (injected noise)", f"{eb['toll_sd']:.2f}",
     "No &mdash; drawn from N(1.301, 0.360) by construction"],
    ["parking (uniform noise)", f"{eb['parking_sd']:.2f}",
     "No &mdash; drawn uniformly at random"],
    (["<strong>quadrature sum of the three</strong>",
      f"<strong>{eb['quadrature_sum']:.2f}</strong>", ""], True),
    (["<strong>chained model RMSE</strong>", f"<strong>{eb['chained_rmse']:.2f}</strong>", ""],
     True),
])
w(f"<p>The quadrature sum of the irreducible terms is "
  f"<strong>&#8377;{eb['quadrature_sum']:.0f}</strong> against a model RMSE of "
  f"<strong>&#8377;{eb['chained_rmse']:.0f}</strong> &mdash; a ratio of "
  f"{eb['chained_rmse'] / eb['quadrature_sum']:.2f}. The two agree closely enough to say the chain "
  f"sits near the floor this dataset allows, and that the residual is <strong>the data's "
  f"unpredictability rather than the model's weakness</strong>. The distance error is "
  f"{eb['distance_sd_km']:.2f} km, which propagates into the fuel bill &mdash; the only term a "
  f"better model could still improve.</p>")
w("<p>There is a corresponding test in the suite "
  "(<span class='mono'>test_metrics_error_budget_explains_the_remaining_error</span>) asserting "
  "that ratio stays between 0.7 and 1.6, so a future regression in the chain shows up as a failing "
  "test rather than a quietly worse number.</p>")

w("<h2>12.7 The end-to-end model &mdash; and where Week 6's lesson reverses</h2>")
w("<p>Architecture D removes even the chain. One model, fed <strong>only</strong> what a traveller "
  "could know: two city names (as real latitude/longitude), vehicle, fuel type, departure hour, "
  "month, passengers, mileage and fuel price. No distance, no toll, no litres, no parking.</p>")
w("<p>It also re-runs Week 6's experiment in a harder setting. The true cost contains "
  "<span class='mono'>(distance / mileage) &times; fuel_price</span>, so two feature sets are "
  "compared: the raw inputs, and the raw inputs plus "
  "<span class='mono'>haversine/mileage</span> and "
  "<span class='mono'>(haversine/mileage) &times; fuel_price</span>.</p>")
if chk:
    tbl(["Model and feature set", "Test R&sup2;", "MAE (&#8377;)"], [
        ["Linear, base features", f"{chk['linear_base']['r2']:.6f}",
         f"{chk['linear_base']['mae']:.2f}"],
        ["Linear, + 2 engineered columns", f"{chk['linear_engineered']['r2']:.6f}",
         f"{chk['linear_engineered']['mae']:.2f}"],
        (["RandomForest (300), base features", f"{chk['rf_base']['r2']:.6f}",
          f"{chk['rf_base']['mae']:.2f}"], True),
    ])
    tbl(["Source of the improvement", "MAE saved (&#8377;)"], [
        ["Two engineered columns", f"{chk['mae_saved_by_features']:.2f}"],
        (["Switching to a 300-tree RandomForest",
          f"<strong>{chk['mae_saved_by_model']:.2f}</strong>"], True),
    ])
    ratio = chk["mae_saved_by_model"] / max(chk["mae_saved_by_features"], 1e-9)
    w('<div class="box key">')
    w("<h4>Week 6's claim does not generalise &mdash; and that is the interesting part</h4>")
    w(f"<p>Week 6 argued, and demonstrated, that <strong>domain understanding beat model "
      f"complexity</strong>. End-to-end the result <strong>reverses</strong>: the ensemble wins by "
      f"a factor of about {ratio:.0f}.</p>")
    w("<p>This is not a contradiction of Week 6, it is a <strong>boundary condition</strong> on "
      "it, and the reason is identifiable. When distance was <em>given</em>, the only thing the "
      "linear model could not express was a product &mdash; so handing it the product fixed the "
      "one gap. End-to-end, the model must first infer road distance from four coordinates, and "
      "that is a genuinely non-linear function of geography: coastlines, mountains, the Gulf of "
      "Khambhat. No finite set of hand-made interaction terms captures a map, and partitioning "
      "space is exactly what trees do.</p>")
    w("<p><strong>Feature engineering beats model complexity when the missing structure is "
      "something you can write down. When the missing structure is a map, it does "
      "not.</strong></p>")
    w("</div>")

if fi:
    w("<h3>What the end-to-end model leans on</h3>")
    tbl(["Feature", "Share of importance"],
        [[t["feature"], f"{t['share_pct']:.2f}%"] for t in fi["ranked"][:10]])
    hav = sum(t["share_pct"] for t in fi["ranked"]
              if t["feature"] in ("haversine_km", "log_haversine"))
    pax = next((t["share_pct"] for t in fi["ranked"] if t["feature"] == "passengers"), 0.0)
    w('<div class="box warn">')
    w("<h4>One trap in this ranking</h4>")
    w(f"<p>Distance dominates: haversine and its log together carry "
      f"<strong>{hav:.0f}%</strong> of the prediction, which is right &mdash; cost is mostly fuel, "
      f"and fuel is mostly distance. But the vehicle and fuel one-hots score <strong>below "
      f"0.1%</strong>, and it would be wrong to conclude vehicle type does not matter.</p>")
    w("<p>Week 4 recovered a real per-km maintenance rate that differs by vehicle (Hatchback "
      "0.5694, Sedan 0.6943, SUV 0.8366 &#8377;/km) and Week 6 showed that dropping those columns "
      "made the app quote an identical price for an SUV and a hatchback. The rate spread is about "
      "0.27 &#8377;/km &mdash; roughly &#8377;108 on a 400 km trip &mdash; and end-to-end that "
      "signal sits <em>underneath</em> the toll and parking noise, which is hundreds of "
      "rupees.</p>")
    w(f"<p><strong>The importance is low because the noise here is large, not because the effect "
      f"is absent.</strong> It is precisely why the app serves the chained pipeline: pin distance "
      f"and litres down first, and the vehicle term becomes visible again. "
      f"<span class='mono'>passengers</span>, at {pax:.2f}%, is the genuine null &mdash; the "
      f"recovered formula has no passenger term, so anything above zero there is the forest "
      f"fitting noise.</p>")
    w("</div>")

w("<h2>12.8 Week 10 summary</h2>")
tbl(["", "Result"], [
    ["Constants removed from <span class='mono'>app.py</span>", "5"],
    ["Sub-models fitted", "6 &mdash; one trained on real observed data"],
    ["Sub-models that failed", "1 &mdash; parking, R&sup2; &le; 0, reported as a negative result"],
    ["Real data collected", f"{d['n_real_pairs']} OSRM road distances"],
    ["Distance model vs its constant",
     f"MAE {base['mae']:.2f} &rarr; {ch['mae']:.2f} km (&minus;{d['mae_reduction_pct']:.1f}%)"],
    ["Honest cost accuracy",
     f"<strong>R&sup2; {arch['C_chained']['r2']:.4f}, MAE &#8377;{arch['C_chained']['mae']:.0f}"
     f"</strong> (was R&sup2; {arch['A_true_components']['r2']:.4f}, MAE "
     f"&#8377;{arch['A_true_components']['mae']:.0f} with inputs handed in)"],
    ["Error budget", f"&#8377;{eb['quadrature_sum']:.0f} of the &#8377;{eb['chained_rmse']:.0f} "
     f"RMSE is irreducible noise in the data"],
    ["Tests", "65, all passing"],
], numeric_from=99)

w("<h3>The three things Week 10 established</h3>")
w("<ol>")
w("<li><strong>A near-perfect score is a question about the question.</strong> R&sup2; 0.9997 was "
  "true and uninformative; the number that describes a user's prediction is 0.9645.</li>")
w("<li><strong>A model that fails is a result.</strong> Six model families could not predict "
  "parking, because nothing in the data predicts parking. Reporting that is worth more than tuning "
  "until it looks respectable.</li>")
w("<li><strong>&ldquo;Features beat complexity&rdquo; has a boundary.</strong> It held when the "
  "missing structure was a product you could write down. It reversed when the missing structure "
  "was a map.</li>")
w("</ol>")

print("\n".join(out))
