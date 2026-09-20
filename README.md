# Road Trip Cost Predictor

Tell it how far you are driving and what fuel costs where you are. It prices the trip for a
**hatchback, a sedan and an SUV at once**, on all three fuels — served through an editorial
single-page site.

The litres you will burn, the tolls you will pay and the km/l each vehicle manages are each
predicted by their own fitted sub-model. Nothing in the prediction path is a hardcoded constant.

**By Param Kotadiya** · Semester 5 Machine Learning project.

```bash
pip install -r requirements.txt
uvicorn app:app          # → http://127.0.0.1:8000
```

---

## What it does

Four inputs: **distance, fuel price, parking, passengers.** Back comes a cost for every vehicle
and every fuel, with a full breakdown, an efficiency band, and the caveats that apply to that
particular estimate.

**Distance is yours to supply because it is the one input you can look up exactly.** The site
ships a reference table of **5,538 measured road distances** between 139 cities — search it,
press *Use this*, and the number lands in the estimator. Any pair it does not cover goes to
`/api/distance`, which asks OSRM and falls back to a fitted distance model when routing is
unreachable. The response always says which of the two answered.

**Everything you could not know is predicted.** A traveller cannot state their fuel burn or
their toll bill before setting off — and litres is the cost model's single strongest feature
(corr +0.96), so asking for it would be handing the model its own answer.

**Each vehicle gets its own mileage.** One typed figure cannot be honest for a hatchback and an
SUV at the same time, and the gap between the three is most of the gap between their costs. The
mileage sub-model supplies all three; override any of them if you know your own car.

---

## Results

| Model | Result | Baseline |
|---|---|---|
| **Cost, as served** | R² **0.9816** · RMSE ₹429.56 · MAE **₹291.69** | — this is what a user gets |
| **Cost, handed the true toll and litres** | R² 0.999778 · RMSE ₹47.19 · MAE ₹33.86 | ₹139.01 MAE without the interaction terms |
| **Road distance** | R² 0.9890 · MAE **30.67 km** | 34.33 km for a single national ratio |
| **Litres burnt** | R² 0.9794 · RMSE 3.52 L | slope 1.0497 on distance ÷ mileage |
| **Toll** | R² 0.7949 · RMSE ₹212 | irreducible — the spread is not predictable |
| **Mileage** | R² 0.6734 · MAE 1.56 km/l | ceiling is structural, see below |
| **Cost-band classifier** | 72.4% | 25.0% (4 balanced classes) |

**Read the first two rows together.** Both are real measurements of the same regressor, and they
answer different questions. The second is taken with the true toll and litres supplied as input
features — it asks *"given the litres burnt and the toll paid, can you total the bill?"*, which
is not a question anyone can ask before a trip. The first answers the user's actual question, and
it is the number `/api/metrics` leads with and the site quotes.

Full working: [`TASK5.md`](TASK5.md) and Weeks 9–10 of the notebook.

### The findings worth reading

**1. The dataset is synthetic — and we proved it.** Subtracting the known cost components leaves
a residual that correlates +0.85 with distance and splits cleanly by vehicle type. That recovers
the generating formula:

```
total = litres × fuel_price + toll + parking + distance × rate(vehicle)
rate:  Hatchback 0.5694 · Sedan 0.6943 · SUV 0.8366  ₹/km
```

Reconstruction accuracy: **R² = 0.999162, from arithmetic alone**. Any model scored on this data
has to be read against that number, not against zero.

**2. One engineered feature beat two ensemble models.** The real fuel bill is `litres × price`,
and an additive model cannot express a product. Supplying it directly:

| Model | Test MAE |
|---|---|
| Linear, no interaction | ₹139.01 |
| RandomForest (300 trees) | ₹69.97 |
| GradientBoosting | ₹73.50 |
| **Linear + `litres × price`** | **₹33.86** |

Domain understanding beat model complexity — and the model stays linear, so the from-scratch
gradient descent still applies to it.

**3. The second interaction is what makes three prices mean anything.** Running costs differ
between vehicles **per kilometre** — 0.5694 to 0.8366 ₹/km, a 47% spread. Give a linear model
only `vehicle_type` one-hots and it can fit a flat per-trip offset and nothing else: about ₹130
extra for an SUV whether the drive is 200 km or 1200 km, when the truth runs from roughly ₹50 to
₹320 across that range.

That error is invisible while the interface shows one vehicle at a time. It is the entire output
once it shows three. Adding `distance × vehicle` fixes it, and the measured gap now scales the
way it should:

| Trip | SUV − Hatchback |
|---|---:|
| 200 km | ₹745 |
| 1200 km | ₹4,464 |

**4. Gradient descent stopped converging, and Ridge was the wrong fix.** On 25,000 rows the
weights drift thousands apart from the closed-form solution while predictions still agree. The
cause is measurable: `corr(litres, fuel_bill) = 0.989`, giving a Hessian condition number of
**≈16,000**. Ridge fixes the conditioning exactly as theory predicts — and costs six times the
error doing it:

| α | cond(H) | GD gap @20k epochs | test MAE |
|---|---|---|---|
| 0 (OLS) | 16,155 | 4090.89 | **₹33.86** |
| 10 | 2,284 | 0.14 | ₹109.41 |
| 100 | 263 | 0.00 | ₹255.42 |

So the shipped model is ordinary least squares solved in closed form. An ill-conditioned problem
is not a broken model — it is one where iterative solvers struggle and direct solvers do not.

**5. Road distance needs a model, not a ratio.** `scripts/fetch_real_distances.py` collected
**840 actual road distances from OSRM** — the only observed data in the project. A single
national straight-line-to-road ratio turns out to be nearly *unbiased* (−0.4% averaged over all
840 routes) and still wrong **6.3% route by route**, because one multiplier cannot describe
Indian road geometry:

| Route | Straight line | Real road | Ratio |
|---|---:|---:|---:|
| Surat → Bhavnagar | 94.2 km | **339.4 km** | 3.60 |
| Moradabad → Chanduasi | 43.1 km | 44.9 km | 1.04 |

Surat and Bhavnagar face each other across the **Gulf of Khambhat**; the drive goes around the
head of the gulf. Those extremes are real data, not outliers, and they are the whole argument for
a model. A user takes one trip, not the average of all of them.

**6. A model that failed, reported as a result.** Six model families were fitted to
`parking_cost` and every one scored test R² ≤ 0 — worse than predicting the mean. That is
correct, not a bug: parking in this data carries no relationship to distance, vehicle or party
size. **This is why parking is the one cost term the form asks for outright.** A quantity nothing
predicts should be requested, not guessed, and the negative result is the evidence for that
choice rather than a preference.

**7. The fuel-burn slope is 1.05, not 1.00.** Textbook consumption is `distance ÷ mileage`. The
fitted slope sits above that, and the excess is real: stop-start driving, gradients and
congestion all cost fuel. How much depends on conditions nobody can state before setting off, so
the model learns the average penalty and carries the variation as error — 3.52 L RMSE, scaling
with trip length rather than averaging away. It is the largest single term in the error budget.

Full derivations: [`RoadTripCost.ipynb`](RoadTripCost.ipynb) Weeks 4–10, and
[`TASK5.md`](TASK5.md) for the evaluation.

---

## Running it

**Prerequisites:** Python 3.10+, Node 18+.

### Just the app

Everything needed is committed — trained models, datasets, built frontend.

```bash
pip install -r requirements.txt
uvicorn app:app
```

### Rebuilding from scratch

```bash
python scripts/build_cities.py             # 3,739 Indian cities with real lat/lon (GeoNames)
python scripts/build_fuel_prices.py        # per-state fuel prices
python scripts/build_wide_dataset.py       # 25,000 trips over real geography
python scripts/fetch_real_distances.py     # 840 REAL road distances from OSRM (~6 min)
python scripts/build_distance_matrix.py    # → frontend/public/city_distances.json (~10 s)
python scripts/train_pipeline.py           # → models/pipeline.joblib (the sub-models)
python scripts/train_models.py             # → models/model.joblib   (band + GD history)
python scripts/evaluate_models.py          # → data/task5_evaluation.json  (Task 5)
python scripts/build_task5_doc.py          # → TASK5.md
python scripts/build_report_pdf.py         # → Project_Report.pdf
python scripts/build_notebook_chapters.py  # regenerate notebook Weeks 4–8
python scripts/build_task5_chapters.py     # regenerate notebook Weeks 9–10
python scripts/execute_new_chapters.py     # run the appended cells, keep their outputs
```

> Order matters twice. `train_pipeline.py` must precede `evaluate_models.py`, which reads its
> report. And `build_notebook_chapters.py` rebuilds everything from Week 4 onwards, so it wipes
> Weeks 9–10 and they must be re-appended after it.

### Frontend development

```bash
cd frontend && npm install && npm run dev   # :5173, proxies /api → :8000
```

`npm run build` emits into `static/`, which FastAPI serves at `/`, so production is one process.

> **Note:** Vite binds to `localhost`, which Node resolves to IPv6 on Windows. Use
> `http://localhost:5173`, not `127.0.0.1`.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/cities?q=` | Autocomplete over the 537 in-domain cities (pop ≥ 100k) |
| `GET /api/distance?from=&to=` | Measured road distance between two cities, model fallback |
| `GET /api/default-mileage?fuel=` | What each of the three vehicles manages on that fuel |
| `POST /api/predict` | Cost for all 9 vehicle × fuel combinations, with breakdowns |
| `GET /api/metrics` | Test-set metrics, with honesty notes |
| `GET /api/loss-curve` | Gradient-descent loss curve and the closed-form comparison |
| `GET /api/health` | Liveness |

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"distance_km":500,"petrol_price":106.0,"parking_cost":70,"passengers":4}'
```

`/api/predict` sets `extra="forbid"`: a field this API does not read is a 422, never a silently
ignored input. Distance is bounded at 10–3000 km in the request schema rather than merely
warned about — it arrives as a typed number, so nothing else guarantees it is sane.

---

## Deploying

The `Dockerfile` builds the React app with Node and serves it from the Python image, so one
container runs both the API and the site.

### Render (free tier)

Push to GitHub, then on [render.com](https://render.com): **New → Blueprint** → pick the repo.
`render.yaml` describes the service; Render reads it and builds. Health check is `/api/health`.

### Anywhere that runs a container

```bash
docker build -t roadtrip .
docker run -p 8000:8000 roadtrip
```

Hosts that inject `$PORT` (Render, Railway, Fly, Cloud Run) are handled — the container reads it
and defaults to 8000 otherwise.

**Notes on the deployment:**

- **Single worker, deliberately.** The models and the 3,739-city table load per process, and the
  OSRM distance cache is in-process. Extra workers multiply memory and cold routing calls for
  no gain at this traffic level.
- **The image is lean.** `.dockerignore` excludes the notebook, raw CSVs, scripts, tests and
  frontend source — only `app.py`, `roadtrip_features.py`, two small data files, the two model
  bundles and the built site ship.
- **Free tiers sleep.** Render's free instance spins down when idle; the first request after
  that takes ~30s. Fine for a demo link, not for anything time-sensitive.
- **OSRM is a public demo server.** It is rate-limited and makes no uptime promise. The shipped
  distance table needs no network at all; only `/api/distance` touches OSRM, and it falls back
  to the fitted model and says so in `source`.

> ⚠️ The Docker build has **not** been run — Docker isn't installed in the environment this was
> developed in. The `--outDir` override that the image depends on *was* verified directly, but
> expect to iterate once on the first real build.

---

## Project structure

```
RoadTripCost.ipynb       Weeks 1–10. The graded artifact — start here.
Project_Report.pdf       Complete report, Weeks 1–10 + viva Q&A (generated)
TASK5.md                 Task 5 answered, tables filled (generated — see scripts/)
road_trip_data.csv       Original 1,140 trips, 9 city pairs. Untouched.
app.py                   FastAPI backend
roadtrip_features.py     Feature builders shared by the scripts and the API
data/
  india_cities.csv       3,739 cities, real coordinates (GeoNames, CC BY 4.0)
  fuel_prices.csv        35 states — `source` column marks verified vs estimated
  road_trip_wide.csv     25,000 generated trips, 534 cities, 50–1500 km
  real_distances.csv     840 REAL OSRM road distances — the only observed data
  curated_distances.csv  Fetch cache for the reference table (rebuildable)
  *_report.json          Saved metrics; TASK5.md and the notebook read these
models/
  model.joblib           Cost-band classifier + gradient-descent history
  pipeline.joblib        The sub-models + the chained cost regressor (served)
report/                  HTML sources for Project_Report.pdf
scripts/
  evaluation.py          The Task 5 checklist, implemented once
  train_pipeline.py      Fits and scores the sub-models
  build_distance_matrix.py  Builds the shipped distance reference table
  build_report_pdf.py    Assembles report/ + generated tables → Project_Report.pdf
  ...                    Data build, training, notebook and doc generation
frontend/                React 19 + Vite 7 + Tailwind v4
  public/city_distances.json   5,538 measured routes (25.6 KB gzipped)
static/                  Built frontend (generated — do not hand-edit)
tests/                   pytest suite for the API (70 tests)
Dockerfile               Two-stage build: Node compiles, Python serves
render.yaml              Render blueprint
```

`models/pipeline_eval.joblib` is **not** committed — it holds the comparison-only regressor, is
reproducible by re-running `train_pipeline.py`, and nothing needs it at serve time.

Run the tests with:

```bash
pytest -q
```

Design and process notes: [`FRONTEND_PLAN.md`](FRONTEND_PLAN.md), [`FINDINGS.md`](FINDINGS.md).
Task 5 write-up: [`TASK5.md`](TASK5.md).

---

## Limitations

Stated plainly, because several of them affect how the numbers should be read.

1. **The wide dataset is generated, not observed.** It is physically consistent with the original
   data — same recovered formula, real geography, real coordinates — but it is not independent
   evidence. The exception is `data/real_distances.csv`: 840 genuinely observed OSRM road
   distances, which is what the distance sub-model is trained on.
2. **The cost model learned on a synthetic distance convention and is served real kilometres.**
   The wide dataset sets `distance_km = haversine × N(1.2355, 0.082)`; the reference table holds
   measured road distances. Against the 840 real routes that convention is nearly unbiased in the
   mean (−0.4%) but off by 6.3% on a typical single route. The systematic part is small enough
   that the ₹/km relationships carry over intact; the per-route spread is the part that does not
   cancel. `/api/metrics` reports both figures as `distance_convention_gap`.
3. **Fuel prices are yours to enter.** Leave a field blank and the API uses a national average
   and labels it `national_average`, with a warning attached. `data/fuel_prices.csv` (6 states
   verified, 29 estimated) is used only to prefill the distance page's suggestions, never inside
   a prediction you have priced yourself.
4. **Predictions beyond 1,500 km extrapolate.** The model was trained on 50–1500 km; the API
   returns an explicit warning outside that range. The shipped distance table is filtered to that
   band so the reference page does not steer you into it.
5. **Routing depends on a public demo server.** OSRM's free instance is rate-limited. The shipped
   table needs no network; `/api/distance` caches, and falls back to the fitted distance model
   (R² 0.9890, MAE 30.67 km on held-out real routes) when OSRM is unavailable.
6. **Parking is not predicted, because it cannot be.** Six model families all scored test
   R² ≤ 0, which is why it is a user input.
7. **The error bar is large and that is the honest number.** MAE ₹291.69 on a typical estimate.
   The budget: toll ±₹212 and the fuel bill ±₹363, which combine in quadrature to ₹420.78
   against an achieved RMSE of ₹429.56 — the chain is sitting on the floor this data allows.
   Distance and parking contribute nothing, because you typed them.

---

## Data sources

- **Cities & coordinates** — [GeoNames](https://download.geonames.org/export/dump/)
  `cities15000`, CC BY 4.0
- **Road distance** — [OSRM](http://project-osrm.org/) public demo. The reference table is built
  offline with the `/table` endpoint (100 coordinates per request, 4,950 pairs in ~1.3 s);
  `/api/distance` uses `/route` live, with the fitted distance model as fallback.
- **Road distance training data** — 840 real OSRM routes collected by
  `scripts/fetch_real_distances.py` → `data/real_distances.csv`
- **Fuel prices** — metro anchors from public reporting (May 2026); remaining states estimated
