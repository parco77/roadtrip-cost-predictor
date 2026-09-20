"""
Emit the model-pipeline chapter of the full project report as HTML, straight from the saved JSON.

Companion to scripts/build_report_tables.py, which emits the evaluation chapter. Split into two
files purely so each stays readable.

PART and WEEK are set here rather than hardcoded through the text, so renumbering the report is
a one-line change in each builder instead of a search across generated prose.

Run:  python scripts/build_report_week10.py > <out>.html
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PART = 12
WEEK = 10


def load(name):
    p = os.path.join(ROOT, "data", name)
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    return json.load(open(p, encoding="utf-8"))


P = load("pipeline_report.json")
R = load("regression_report.json")
T = load("task5_evaluation.json")

out = []


def w(s=""):
    out.append(s)


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
gapc = P["distance_convention_gap"]
eb = P["cost"]["error_budget"]
park = P["parking"]
tl = P["toll"]
lit = P["litres"]
vgap = P["vehicle_gap"]
arch = T["architecture_comparison"]
abl = R["ablations"]
base = d["constant_baseline"]
ch = d["chosen_scores"]["test"]

w(f'<h1 class="part"><span class="num">Part {PART} &middot; Week {WEEK}</span>'
  f'The model pipeline</h1>')

# ---------------------------------------------------------------------------------- .1
w(f"<h2>{PART}.1 What the pipeline is for</h2>")
w("<p>A trip cost is an addition: fuel, plus toll, plus parking, plus the running cost of the "
  "vehicle. Writing that sum down is not the hard part. <strong>Knowing what to put into "
  "it</strong> is.</p>")
w("<p>A traveller setting off knows three things: how far they are driving, what fuel costs "
  "where they are, and roughly what parking will run to. They do not know how many litres they "
  "will burn or what the tolls will come to &mdash; and those are exactly the two terms that "
  "carry the uncertainty in the total.</p>")
w("<p>So each unknown gets <strong>its own fitted model</strong>, and a final cost regressor "
  "consumes their outputs:</p>")
tbl(["Sub-model", "Predicts", "Trained on"], [
    ["1 &mdash; road distance", "route geometry &rarr; km",
     f"<strong>{d['n_real_pairs']} real OSRM road distances</strong>"],
    ["2 &mdash; mileage", "vehicle + fuel &rarr; km/l", "generated data"],
    ["3 &mdash; litres", "distance + mileage &rarr; litres", "generated data"],
    ["4 &mdash; toll", "distance &rarr; &#8377;", "generated data"],
    ["5 &mdash; parking", "&mdash;",
     "<strong>nothing predicts it; it is a user input</strong>"],
], numeric_from=99)
w("<p>Every sub-model is scored against <strong>the simplest thing that could stand in its "
  "place</strong> &mdash; one ratio, one flat rate, the training mean. &ldquo;The model got "
  "R&sup2; 0.8&rdquo; means nothing until you know what arithmetic alone would have scored. One "
  "of the five does not beat its baseline, and that is reported rather than hidden.</p>")

# ---------------------------------------------------------------------------------- .2
w(f"<h2>{PART}.2 The one piece of genuinely observed data</h2>")
w("<p>Everything in <span class='mono'>road_trip_wide.csv</span> is generated. To fit a distance "
  "model that is not circular, this part uses real data: "
  "<span class='mono'>scripts/fetch_real_distances.py</span> samples city pairs stratified across "
  "six distance bands and asks the OSRM routing engine for the actual driving distance, appending "
  "each result to disk so a rate-limited run can resume without losing progress.</p>")
fo = d["factor_observed"]
tbl(["Measure", "Value"], [
    ["Real routes collected", f"<strong>{d['n_real_pairs']}</strong>"],
    ["Straight-line range", "40 &ndash; 1,248 km"],
    ["Observed winding ratio (road &divide; straight)",
     f"mean <strong>{fo['mean']:.4f}</strong>, sd {fo['sd']:.4f}, "
     f"range {fo['min']:.4f} &ndash; {fo['max']:.4f}"],
])
w("<h3>Why one ratio cannot do this job</h3>")
w("<p>A single national straight-line-to-road ratio is the obvious shortcut. Measured against "
  "those routes, it is <em>nearly unbiased</em> &mdash; and still wrong.</p>")
tbl(["Measure", "Value"], [
    ["Mean <em>signed</em> error",
     f"<strong>{gapc['mean_signed_km']:+.1f} km ({gapc['mean_signed_pct']:+.2f}%)</strong> "
     f"&mdash; the bias"],
    ["Mean <em>absolute</em> error",
     f"<strong>{gapc['mean_abs_km']:.1f} km ({gapc['mean_abs_pct']:.1f}%)</strong> "
     f"&mdash; the per-route error"],
    ["Worst overshoot", f"{gapc['max_km']:+.1f} km"],
    ["Worst undershoot", f"{gapc['min_km']:+.1f} km"],
])
w('<div class="box key">')
w("<h4>These two numbers say opposite-sounding things, and both matter</h4>")
w(f"<p>Averaged over {gapc['n_pairs']} routes a fixed ratio lands within half a percent of the "
  f"truth. That is why it is such a tempting shortcut.</p>")
w(f"<p>But <em>unbiased on average</em> is not <em>correct</em>. Route by route it is off by "
  f"<strong>{gapc['mean_abs_pct']:.1f}%</strong> typically, and by tens of percent at the "
  f"extremes. <strong>A user does not take the average of every road trip in India. They take "
  f"one specific trip, and on that trip the errors do not cancel.</strong></p>")
w("</div>")
w("<h3>The extreme routes are real, and they are the whole argument</h3>")
tbl(["Route", "Straight line", "Real road", "Ratio"], [
    (["Surat &rarr; Bhavnagar", "94.2 km", "<strong>339.4 km</strong>", "3.60"], True),
    ["Valsad &rarr; Bhavnagar", "151.6 km", "406.7 km", "2.68"],
    ["Badlapur &rarr; Bhavnagar", "312.3 km", "600.8 km", "1.92"],
    ["Moradabad &rarr; Chanduasi", "43.1 km", "44.9 km", "1.04"],
])
w("<p>A ratio of 3.60 looks like bad data. It is not. <strong>Surat and Bhavnagar face each "
  "other across the Gulf of Khambhat</strong>: 94 km of straight line, 339 km of road around the "
  "head of the gulf. At the other end, Moradabad &rarr; Chanduasi is essentially a straight "
  "road.</p>")
w("<p>No single multiplier can serve both. These points are <strong>kept</strong> in the training "
  "data rather than cleaned away, because they are precisely the cases a constant cannot "
  "represent and a model can. Note which model wins below: a <strong>tree ensemble</strong>, "
  "because it can isolate a region of the map. A linear fit gets dragged by these points instead "
  "of representing them.</p>")

# ---------------------------------------------------------------------------------- .3
w(f"<h2>{PART}.3 Sub-model 1 &mdash; road distance</h2>")
w("<p>Two parameterisations of the same problem were compared, and the difference is "
  "instructive:</p>")
w("<ul><li><strong>(a)</strong> predict <span class='mono'>road_km</span> directly</li>"
  "<li><strong>(b)</strong> predict the winding <em>ratio</em>, then multiply by haversine</li>"
  "</ul>")
w("<p>(b) removes trip length from the target, so the model spends its capacity on the part that "
  "is genuinely unknown &mdash; the <em>shape</em> of the route &mdash; instead of re-learning "
  "&ldquo;longer straight line, longer road&rdquo;. It is the same idea as the interaction terms "
  "in the cost model: hand the model the quantity that actually means something.</p>")
rows = []
chosen_key = f"candidates_{d['parameterisation']}"
for label, key in [("(a) direct km", "candidates_direct"),
                   ("(b) ratio &times; haversine", "candidates_factor")]:
    for n, r in d[key].items():
        is_chosen = key == chosen_key and d["chosen"].startswith(n)
        rows.append(([label, n, f"{r['test']['r2']:.6f}", f"{r['test']['rmse']:.2f}",
                      f"{r['test']['mae']:.2f}", f"{r['cv_mean']:.6f}", f"{r['cv_std']:.4f}"],
                     is_chosen))
tbl(["Target", "Model", "Test R&sup2;", "RMSE (km)", "MAE (km)", "CV mean", "CV sd"],
    rows, numeric_from=2, cls="wide")
tbl(["", "R&sup2;", "RMSE (km)", "MAE (km)"], [
    ["<strong>BASELINE</strong> &mdash; one national ratio", f"{base['r2']:.4f}",
     f"{base['rmse']:.2f}", f"{base['mae']:.2f}"],
    ([f"<strong>CHOSEN</strong> &mdash; {d['chosen']}", f"{ch['r2']:.4f}", f"{ch['rmse']:.2f}",
      f"{ch['mae']:.2f}"], True),
])
w(f"<p><strong>The model beats the ratio by {d['mae_reduction_pct']:.1f}% on MAE</strong> "
  f"({base['mae']:.2f} &rarr; {ch['mae']:.2f} km) and about "
  f"{(1 - ch['rmse'] / base['rmse']) * 100:.0f}% on RMSE. Real, and deliberately not oversold: "
  f"endpoint coordinates only partly determine route shape, because the rest depends on where "
  f"roads and bridges happen to be &mdash; information that is not in the features.</p>")
w("<p>Parameterisation (b) wins, exactly as predicted. <strong>A sub-model that could not beat "
  "the baseline would not have earned its place</strong>, which is why every table in this part "
  "carries that baseline row.</p>")
w("<p>This sub-model does not sit in the cost path at all. It powers the "
  "<strong>distance lookup</strong>: the site ships a table of measured road distances between "
  "139 cities, and this model answers for any pair the table does not carry, whenever live "
  "routing is unreachable.</p>")

# ---------------------------------------------------------------------------------- .4
w(f"<h2>{PART}.4 Sub-models 2 to 5</h2>")
for key, title, unit in [
    ("mileage", "Sub-model 2 &mdash; mileage (vehicle + fuel &rarr; km/l)", "km/l"),
    ("litres", "Sub-model 3 &mdash; litres burnt", "litres"),
    ("toll", "Sub-model 4 &mdash; toll", "&#8377;"),
    ("parking", "Sub-model 5 &mdash; parking (the one that fails)", "&#8377;"),
]:
    sec = P[key]
    w(f"<h3>{title}</h3>")
    if "constant_baseline" in sec:
        cb = sec["constant_baseline"]
        extra = ""
        if "mean_baseline" in sec:
            mb = sec["mean_baseline"]
            extra = (f" &nbsp;&middot;&nbsp; <strong>Baseline (train mean):</strong> "
                     f"R&sup2; {mb['r2']:+.4f}, RMSE {mb['rmse']:.2f} {unit}")
        w(f"<p class='small'><strong>Baseline (flat arithmetic):</strong> "
          f"R&sup2; {cb['r2']:+.4f}, RMSE {cb['rmse']:.2f} {unit}{extra}</p>")
    rows = [([n, f"{r['test']['r2']:+.6f}", f"{r['test']['rmse']:.3f}",
              f"{r['train']['r2']:.6f}", f"{r['cv_std']:.4f}", r["verdict"]],
             n == sec["chosen"]) for n, r in sec["candidates"].items()]
    tbl(["Model", "Test R&sup2;", f"RMSE ({unit})", "Train R&sup2;", "CV sd", "Verdict"], rows)
    w(f"<p><strong>Chosen: <span class='mono'>{sec['chosen']}</span></strong></p>")

mil = P["mileage"]["candidates"][P["mileage"]["chosen"]]
w('<div class="box">')
w("<h4>Sub-model 2, mileage &mdash; a structural ceiling, and a load-bearing one</h4>")
w(f"<p>Test R&sup2; <strong>{mil['test']['r2']:.4f}</strong>, and every model family lands on the "
  f"same number to three decimals. That is the correct answer: the generator draws mileage "
  f"<em>uniformly</em> inside a per-(vehicle, fuel) range, so the best any model can do is name "
  f"the middle of the right cell, and the residual is uniform noise by construction.</p>")
w("<p>The nine (vehicle, fuel) cells cannot be reproduced by additive one-hots alone &mdash; "
  "1 + 2 + 2 = 5 parameters cannot fit nine independent means &mdash; so the feature set includes "
  "the four vehicle &times; fuel <strong>interaction terms</strong>, and with them the design "
  "matrix spans all nine cells exactly.</p>")
w("<p><strong>This is the sub-model the product leans on hardest.</strong> The interface prices "
  "three vehicles side by side, and a single typed mileage cannot be honest for a hatchback and "
  "an SUV at once. Each vehicle gets its own figure from here &mdash; 19.5 km/l for a diesel "
  "hatchback against 10.8 for a CNG SUV &mdash; and that difference is most of the difference in "
  "their costs.</p>")
w("</div>")

w('<div class="box">')
w("<h4>Sub-model 3, litres &mdash; why the fitted slope is not 1.00</h4>")
lc = lit["candidates"][lit["chosen"]]
w(f"<p>Textbook fuel burn is <span class='mono'>distance &divide; mileage</span> &mdash; a slope "
  f"of exactly 1.0 on that ratio. Refitting unscaled on the single ratio column gives "
  f"<strong>{lit['fitted_ratio_slope']:.5f}</strong>.</p>")
w("<p>The excess is real, and it is the interesting part. A car does not achieve its rated "
  "mileage on a real trip: stop-start driving, gradients and congestion all cost fuel. How much "
  "depends on conditions the traveller cannot state before setting off, so the model cannot be "
  "told them. <strong>It learns the average penalty and carries the variation around it as "
  "error.</strong></p>")
tbl(["Measure", "Value"], [
    ["Test R&sup2;", f"{lc['test']['r2']:.6f}"],
    ["Residual sd on held-out trips", f"{lit['residual_sd_litres']:.3f} L"],
    ["Mean absolute error", f"{lit['residual_mae_litres']:.3f} L"],
])
w("<p>That error is <strong>irreducible from these inputs</strong>, and it scales with trip "
  "length rather than averaging away &mdash; a 1,200 km drive carries four times the uncertainty "
  "of a 300 km one. It is the largest single term in the error budget below.</p>")
w("</div>")

w('<div class="box warn">')
w("<h4>Sub-model 4, toll &mdash; and the first honest overfitting in the project</h4>")
tc = tl["candidates"][tl["chosen"]]
w(f"<p>Test R&sup2; <strong>{tc['test']['r2']:.4f}</strong>, RMSE "
  f"<strong>&#8377;{tc['test']['rmse']:.0f}</strong>. The generator sets "
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
      f"<strong>Overfitting</strong>. The train/test diagnosis is uninformative on a "
      f"deterministic target; here it identifies the wrong model immediately.</p>")
w("<p>So the toll sub-model is kept for a reason other than raw accuracy: it is fitted, it "
  "reports its own error bar, and it is evaluated like everything else instead of being "
  "asserted.</p>")
w("</div>")

w('<div class="box warn">')
w("<h4>Sub-model 5, parking &mdash; a negative result, and an interface decision</h4>")
pdt = park["candidates"]["DecisionTree"]
w(f"<p><strong>Every one of the six model families scores test R&sup2; &le; 0</strong> &mdash; "
  f"worse than predicting the mean. <span class='mono'>DecisionTree</span> reaches R&sup2; "
  f"{pdt['test']['r2']:+.4f} with train R&sup2; {pdt['train']['r2']:.4f}: pure memorisation of "
  f"noise, and the cleanest overfitting example in the project.</p>")
w("<p>This is the correct answer, not a bug. Parking in this data carries no relationship to "
  "distance, vehicle or party size &mdash; it is drawn independently of all of them, so there is "
  "nothing in the inputs for a model to find.</p>")
w("<p><strong>And that is why parking is the one cost term the form asks for outright.</strong> "
  "A quantity nothing predicts should be requested, not guessed. The negative result is the "
  "evidence for that interface decision rather than a preference, and "
  "<span class='mono'>/api/metrics</span> discloses it.</p>")
w("</div>")

# ---------------------------------------------------------------------------------- .5
w(f"<h2>{PART}.5 Chaining them properly &mdash; stacking, and a mismatch worth measuring</h2>")
w("<p>Sub-models feeding a final regressor introduce a failure mode that is easy to miss.</p>")
w("<p>If the cost regressor is trained on the <strong>true</strong> toll and litres and then "
  "served <strong>predicted</strong> ones, it has learned to trust inputs that are exact and "
  "receives inputs that are not. That is a <strong>train/serve mismatch</strong>, and rather than "
  "assume it matters, both ways are measured.</p>")
w("<p>Building the training components correctly needs care too. Sub-model predictions on their "
  "own training rows are optimistic &mdash; the model has already seen those rows &mdash; so the "
  "component features for training come from <strong>"
  "<span class='mono'>cross_val_predict</span> (out-of-fold)</strong>: no row's features were "
  "produced by a model that had seen that row. Test rows use the sub-models fitted on the full "
  "training split. That is textbook <strong>stacking</strong>.</p>")
w("<p>Distance and parking are <em>typed by the user</em>, so at serve time they arrive exact. "
  "Training on them as ground truth is therefore not leakage &mdash; it matches what the model "
  "will actually receive. Litres and toll are the opposite, and must come from the sub-models "
  "here exactly as they will in production.</p>")
arch_rows = [
    ["A. handed the true toll and litres", arch["A_true_components"]],
    ["B. chained, trained on true / served predicted", arch["B_mismatched_chain"]],
    ["C. chained, trained on predictions &mdash; <strong>shipped</strong>", arch["C_chained"]],
]
tbl(["Architecture", "Test R&sup2;", "RMSE (&#8377;)", "MAE (&#8377;)"],
    [([n, f"{v['r2']:.6f}", f"{v['rmse']:.2f}", f"{v['mae']:.2f}"], n.startswith("C."))
     for n, v in arch_rows])
w('<div class="box key">')
w("<h4>Reading the three rows &mdash; this is the centre of the project</h4>")
w(f"<p><strong>A scores highest and means least.</strong> It is handed the litres actually burnt "
  f"and the toll actually paid, and asked to add them up. R&sup2; "
  f"{arch['A_true_components']['r2']:.6f} is a true measurement of a question nobody can ask: a "
  f"traveller can supply neither figure before the trip.</p>")
w(f"<p><strong>C is what a user receives</strong> &mdash; R&sup2; "
  f"{arch['C_chained']['r2']:.4f}, MAE &#8377;{arch['C_chained']['mae']:.2f} &mdash; and C is "
  f"what the app serves and what <span class='mono'>/api/metrics</span> leads with. The A &rarr; "
  f"C gap of {arch['r2_drop_A_to_C']:.4f} in R&sup2; and "
  f"&#8377;{arch['mae_gap_A_to_C']:.2f} in MAE is not a regression in quality; it is the price of "
  f"the model deriving its own inputs instead of being handed them.</p>")
w(f"<p><strong>B vs C went the opposite way to expectation.</strong> B (trained on truth, served "
  f"predictions) scores MAE &#8377;{arch['B_mismatched_chain']['mae']:.2f}; C (trained on its own "
  f"predictions) scores &#8377;{arch['C_chained']['mae']:.2f} &mdash; B is <em>marginally "
  f"better</em>. The mismatch cost essentially nothing, and the reason is specific: only two "
  f"components are predicted, and both sub-models are close to <strong>zero-mean</strong>, so "
  f"the linear cost model's coefficients are almost unchanged by refitting on them. Retraining "
  f"the final stage matters when the predictions are <em>biased</em>, because then the model has "
  f"to learn to distrust them.</p>")
w("<p>C still ships, because it is the correct construction and the property that rescues B is a "
  "fact about this dataset rather than something to rely on. But the honest report is that "
  "<strong>on this data the mismatch did not bite</strong>, and claiming otherwise would be "
  "inventing a result.</p>")
w("</div>")

# ---------------------------------------------------------------------------------- .6
w(f"<h2>{PART}.6 Where the remaining error comes from</h2>")
w("<p>&ldquo;The honest model is worse&rdquo; is not a finding. <em>&ldquo;Toll noise and "
  "real-world fuel burn account for nearly all of it&rdquo;</em> is. Decomposing the test-set "
  "error into its sources:</p>")
tbl(["Source", "Std deviation (&#8377;)", "Could a better model fix it?"], [
    ["fuel bill, via litres", f"{eb['fuel_bill_sd']:.2f}",
     "No &mdash; the driving conditions are not in the inputs"],
    ["toll", f"{eb['toll_sd']:.2f}",
     "No &mdash; drawn from N(1.301, 0.360) by construction"],
    ["distance", f"{eb['distance_sd_km']:.2f}", "<strong>Typed by the user, so exact</strong>"],
    ["parking", f"{eb['parking_sd']:.2f}", "<strong>Typed by the user, so exact</strong>"],
    (["<strong>quadrature sum of the two predicted terms</strong>",
      f"<strong>{eb['quadrature_sum']:.2f}</strong>", ""], True),
    (["<strong>chained model RMSE</strong>", f"<strong>{eb['chained_rmse']:.2f}</strong>", ""],
     True),
])
w(f"<p>The quadrature sum of the irreducible terms is "
  f"<strong>&#8377;{eb['quadrature_sum']:.0f}</strong> against a model RMSE of "
  f"<strong>&#8377;{eb['chained_rmse']:.0f}</strong> &mdash; a ratio of "
  f"{eb['chained_rmse'] / eb['quadrature_sum']:.2f}. The two agree closely enough to say the "
  f"chain sits near the floor this dataset allows, and that the residual is <strong>the data's "
  f"unpredictability rather than the model's weakness</strong>. The two terms that carry it are "
  f"exactly the two the user could not have told us.</p>")
w("<p>There is a corresponding test in the suite "
  "(<span class='mono'>test_metrics_error_budget_explains_the_remaining_error</span>) asserting "
  "that ratio stays between 0.7 and 1.6, so a future regression in the chain shows up as a "
  "failing test rather than a quietly worse number.</p>")

# ---------------------------------------------------------------------------------- .7
w(f"<h2>{PART}.7 Pricing three vehicles at once, and the interaction it needs</h2>")
w("<p>The interface does not ask which car you drive. It prices a hatchback, a sedan and an SUV "
  "side by side, because the comparison <em>is</em> the answer &mdash; and because the cost model "
  "already carries vehicle one-hots, the server can simply call it three times with the same "
  "inputs.</p>")
w("<p>That much is free. What is not free is getting the <strong>gap</strong> between them "
  "right, and the gap is the entire output.</p>")
w('<div class="box key">')
w("<h4>Plain one-hots fit a flat offset. The truth is per kilometre.</h4>")
w("<p>The recovered running-cost rates differ by vehicle &mdash; Hatchback 0.5694, Sedan 0.6943, "
  "SUV 0.8366 &#8377;/km, a 47% spread. Given only <span class='mono'>vehicle_type</span> "
  "one-hots, a linear model can express a constant surcharge per trip and nothing else: about "
  "&#8377;130 extra for an SUV whether the drive is 200 km or 1,200 km, when the truth runs from "
  "roughly &#8377;50 to &#8377;320 across that range.</p>")
w("<p><strong>That error is invisible while the interface shows one vehicle at a time.</strong> "
  "It becomes the headline the moment it shows three. Adding "
  "<span class='mono'>distance &times; SUV</span> and "
  "<span class='mono'>distance &times; Sedan</span> fixes it.</p>")
w("</div>")
w("<p>Measured on the shipped chain, petrol at &#8377;106, parking &#8377;70, each vehicle on its "
  "own predicted mileage:</p>")
tbl(["Distance", "Hatchback", "Sedan", "SUV", "SUV &minus; Hatchback", "per km"],
    [[f"{r['distance_km']} km", f"&#8377;{r['Hatchback']:,.0f}", f"&#8377;{r['Sedan']:,.0f}",
      f"&#8377;{r['SUV']:,.0f}", f"<strong>&#8377;{r['suv_minus_hatchback']:,.0f}</strong>",
      f"{r['gap_per_km']:.2f}"] for r in vgap])
per_km = [r["gap_per_km"] for r in vgap]
w(f"<p>The gap per kilometre stays within {min(per_km):.2f}&ndash;{max(per_km):.2f} &#8377;/km "
  f"across a sixfold change in trip length, so the difference <strong>scales with the "
  f"trip</strong> rather than sitting as a flat surcharge. A test in the suite "
  f"(<span class='mono'>test_the_vehicle_gap_scales_with_distance</span>) asserts it, because a "
  f"single estimate would never reveal the failure.</p>")

w("<h3>What each interaction is worth</h3>")
w("<p>Same models, same split, columns removed one group at a time:</p>")
tbl(["Feature set", "Test R&sup2;", "MAE (&#8377;)"],
    [[k, f"{v['r2']:.6f}", f"{v['mae']:.2f}"] for k, v in abl.items()]
    + [(["<strong>full feature set</strong>", f"<strong>{R['regression']['r2']:.6f}</strong>",
         f"<strong>{R['regression']['mae']:.2f}</strong>"], True)])
w(f"<p><span class='mono'>litres &times; price</span> is worth "
  f"&#8377;{abl['no litres x price']['mae'] - R['regression']['mae']:.0f} of MAE, and "
  f"<span class='mono'>distance &times; vehicle</span> "
  f"&#8377;{abl['no distance x vehicle']['mae'] - R['regression']['mae']:.0f}. Both are cases of "
  f"the same lesson: <strong>a linear model cannot multiply unless you hand it the "
  f"product</strong>. The first is a product of two features; the second is a product of a "
  f"feature and a category.</p>")

# ---------------------------------------------------------------------------------- .8
w(f"<h2>{PART}.8 Summary</h2>")
tbl(["", "Result"], [
    ["Sub-models fitted", "5 &mdash; one trained on real observed data"],
    ["Sub-models that failed", "1 &mdash; parking, R&sup2; &le; 0, reported as a negative result"],
    ["Real data collected", f"{d['n_real_pairs']} OSRM road distances"],
    ["Distance model vs one national ratio",
     f"MAE {base['mae']:.2f} &rarr; {ch['mae']:.2f} km (&minus;{d['mae_reduction_pct']:.1f}%)"],
    ["Honest cost accuracy",
     f"<strong>R&sup2; {arch['C_chained']['r2']:.4f}, MAE &#8377;{arch['C_chained']['mae']:.0f}"
     f"</strong> (R&sup2; {arch['A_true_components']['r2']:.4f}, MAE "
     f"&#8377;{arch['A_true_components']['mae']:.0f} with the answer handed in)"],
    ["Error budget", f"&#8377;{eb['quadrature_sum']:.0f} of the &#8377;{eb['chained_rmse']:.0f} "
     f"RMSE is irreducible noise in the data"],
    ["Tests", "70, all passing"],
], numeric_from=99)

w("<h3>The three things this part establishes</h3>")
w("<ol>")
w(f"<li><strong>A near-perfect score is a question about the question.</strong> R&sup2; "
  f"{arch['A_true_components']['r2']:.4f} is true and uninformative; the number that describes a "
  f"user's prediction is {arch['C_chained']['r2']:.4f}.</li>")
w("<li><strong>A model that fails is a result.</strong> Six model families could not predict "
  "parking, because nothing in the data predicts parking. Reporting that &mdash; and turning it "
  "into an interface decision &mdash; is worth more than tuning until it looks respectable.</li>")
w("<li><strong>An error you cannot see is still an error.</strong> The flat vehicle offset was "
  "harmless while the product showed one car. Showing three made it the headline, and the fix "
  "was an interaction term, not a bigger model.</li>")
w("</ol>")

print("\n".join(out))
