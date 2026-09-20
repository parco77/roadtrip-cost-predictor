# Road Trip Cost Predictor

Predict the cost of a road trip between any two of **537 Indian cities** — served through an
editorial single-page site.

Two city names, a vehicle, a fuel type and a departure time go in. Road distance, fuel
efficiency, litres burnt, toll and traffic level are each predicted by their own fitted
sub-model, and a chained cost regressor totals them. Nothing in the prediction path is a
hardcoded constant.

**By Param Kotadiya** · Semester 5 Machine Learning project.

```bash
pip install -r requirements.txt
uvicorn app:app          # → http://127.0.0.1:8000
```

---

## What it does

Pick two cities, a vehicle and a departure time. The model returns a cost estimate with a full
breakdown, an efficiency rating, and — importantly — the caveats that apply to that particular
prediction.

You are only asked for things you could actually know. **Distance, toll, litres burnt and even
your car's mileage are predicted server-side by their own sub-models**, because the cost
regressor's single strongest feature (`fuel_consumption_litres`, corr +0.96) is something no
traveller could ever type in. Live OSRM routing is still preferred for distance when it is
reachable — a real answer beats a predicted one — and the response says which was used.

**Fuel price is prefilled but editable.** The per-state table is verified for only 6 of 35
states, and anyone filling their own tank knows the real figure better than an estimate does.
Pick a city and the field populates from that state's table (following the fuel type as you
change it); type over it and the estimate becomes yours — the caveat that accompanies a guessed
price then correctly disappears.

---

## Results

| Model | Result | Baseline |
|---|---|---|
| **Cost, as served** | R² **0.9645** · RMSE ₹595.88 · MAE **₹391.16** | — this is what a user gets |
| **Cost, fed true toll/parking/litres** | R² 0.999707 · RMSE ₹54.16 · MAE ₹39.66 | RMSE ₹228.31 · MAE ₹139.75 without the interaction term |
| **Road distance** | R² 0.9889 · MAE **30.67 km** | 34.33 km for the constant it replaced |
| **Traffic classifier** | 57.0% | 37.7% majority → **+19.3 points** |
| **Cost-band classifier** | 85.6% | 25.7% (4 balanced classes) |

**Read the first two rows together.** R² 0.999707 is a real measurement, and this project
advertised it for months — but it was taken with the true toll, parking and litres supplied as
input features, and a traveller supplies none of those. It answers *"given the litres burnt and
the toll paid, can you total the bill?"* The first row answers the user's actual question, and
it is the number `/api/metrics` now leads with.

Full working: [`TASK5.md`](TASK5.md) and Weeks 9–10 of the notebook.

### The findings worth reading

**1. The dataset is synthetic — and we proved it.** Subtracting the known cost components leaves
a residual that correlates +0.85 with distance and splits cleanly by vehicle type. That recovers
the generating formula:

```
total = litres × fuel_price + toll + parking + distance × rate(vehicle)
rate:  Hatchback 0.5694 · Sedan 0.6943 · SUV 0.8366  ₹/km
```

Reconstruction accuracy: **R² = 0.999162, from arithmetic alone**. This reframes the original
project's R² = 0.9928 — that was measuring formula recovery, not predictive skill.

**1b. The traffic classifier earns its place in the interface.** After an estimate, the site
charts expected cost across all 24 departure hours. Bars are the classifier's probabilities
weighted against the cost at each traffic level — *not* the cost at the single most likely level,
which would collapse 24 hours onto 3 values and discard everything the model knows about its own
confidence.

The chart takes a **departure window** — the hours you could realistically leave in, wrapping
past midnight for overnight drives. Hours outside it dim but stay visible as context, and the
recommendation is the best hour *you can actually use*: "leave at 1am" is not advice most people
can act on. On Jaipur → Agra the full day spreads **₹177**; restricted to a 6–11am departure it
is still **₹152**.

**2. One engineered feature beat two ensemble models.** The real fuel bill is `litres × price`,
and an additive model cannot express a product. Supplying it directly:

| Model | Test MAE |
|---|---|
| Linear, no interaction | ₹143.85 |
| RandomForest (300 trees) | ₹69.97 |
| GradientBoosting | ₹73.50 |
| **Linear + `litres × price`** | **₹39.66** |

Domain understanding beat model complexity — and the model stays linear, so the from-scratch
gradient descent still applies.

> These four rows use the Week 6 feature set (no vehicle one-hots), which is why they differ
> from the table in [`TASK5.md`](TASK5.md) — that one scores the shipped feature set. Both are
> measured on the same split; they are answering different questions.

**3. Gradient descent stopped converging, and Ridge was the wrong fix.** On 25,000 rows the
weights drift thousands apart from the closed-form solution while predictions still agree. The
cause is measurable: `corr(litres, fuel_bill) = 0.989`, giving a Hessian condition number of
**≈16,000**. Ridge fixes the conditioning exactly as theory predicts — and costs six times the
error doing it:

| α | cond(H) | GD gap @20k epochs | test MAE |
|---|---|---|---|
| 0 (OLS) | 16,155 | 4090.89 | **₹39.66** |
| 10 | 2,284 | 0.14 | ₹109.41 |
| 100 | 263 | 0.00 | ₹255.42 |

So the shipped model is ordinary least squares solved in closed form. An ill-conditioned problem
is not a broken model — it is one where iterative solvers struggle and direct solvers do not.

**4. Five hardcoded constants were doing the model's job.** The prediction path used to be
`haversine × 1.2355` for distance, `× 1.301` for toll, a fixed ₹70 for parking, a lookup for
fuel burn — and only the final addition was a model. Worse, that arithmetic produced the
regressor's strongest feature (`fuel_consumption_litres`, corr +0.96), so the model was handed
three of the four terms of the cost formula.

Each constant is now a fitted sub-model. The distance one needed real data, so
`scripts/fetch_real_distances.py` collected **840 actual road distances from OSRM** — the only
observed data in the project. The constant turns out to be nearly *unbiased* (−0.4% averaged
over 840 routes) but wrong **6.3% route by route**, because a single multiplier cannot describe
Indian road geometry:

| Route | Straight line | Real road | Factor |
|---|---:|---:|---:|
| Surat → Bhavnagar | 94.2 km | **339.4 km** | 3.60 |
| Moradabad → Chanduasi | 43.1 km | 44.9 km | 1.04 |

Surat and Bhavnagar face each other across the **Gulf of Khambhat**; the drive goes around the
head of the gulf. Those extremes are real data, not outliers, and they are the whole argument
for a model. A user takes one trip, not the average of all of them.

**5. A model that failed, reported as a result.** Six model families were fitted to
`parking_cost` and every one scored test R² ≤ 0 — worse than the mean. That is correct: the
generator draws parking uniformly from {0, 40, 60, 80, 120, 150} independently of everything
else, so nothing predicts it. The API serves the training mean and discloses the negative result
rather than tuning until it looked respectable.

**6. "Features beat complexity" has a boundary.** Finding 2 above holds when distance is given.
Ask a model to predict cost from two city names and it reverses — the engineered columns save
₹67 of MAE, switching to 300 trees saves ₹474. When the missing structure is a product you can
write down, hand it over; when the missing structure is a *map*, use a model that partitions
space.

Full derivations: [`RoadTripCost.ipynb`](RoadTripCost.ipynb) Weeks 4–10, and
[`TASK5.md`](TASK5.md) for the evaluation.

---

## Running it

**Prerequisites:** Python 3.10+, Node 18+.

### Just the app

Everything needed is committed — trained model, datasets, built frontend.

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
python scripts/train_models.py             # → models/model.joblib   (the Week 6 regressor)
python scripts/train_pipeline.py           # → models/pipeline.joblib (the six sub-models)
python scripts/train_end_to_end.py         # → the honesty benchmark
python scripts/evaluate_models.py          # → data/task5_evaluation.json  (Task 5)
python scripts/build_task5_doc.py          # → TASK5.md
python scripts/build_report_pdf.py         # → Project_Report.pdf (80 pages)
python scripts/build_notebook_chapters.py  # regenerate notebook Weeks 4–8
python scripts/build_task5_chapters.py     # regenerate notebook Weeks 9–10
python scripts/execute_new_chapters.py     # run the appended cells, keep their outputs
```

> Order matters for the last three: `build_notebook_chapters.py` rebuilds everything from
> Week 4 onwards, so it wipes Weeks 9–10 and they must be re-appended after it.

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
| `GET /api/default-mileage` | What the mileage sub-model predicts for a vehicle + fuel |
| `POST /api/predict` | Cost regression + traffic and cost-band classification |
| `POST /api/departure-sweep` | Expected cost for all 24 departure hours — "when should I leave?" |
| `GET /api/metrics` | Test-set metrics, with honesty notes |
| `GET /api/loss-curve` | Gradient-descent loss curve and the sklearn comparison |
| `GET /api/health` | Liveness |

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"start_city":"Jaipur","destination_city":"Agra","mileage":15.5}'
```

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

- **Single worker, deliberately.** The model and the 3,739-city table load per process, and the
  OSRM distance cache is in-process. Extra workers multiply memory and cold routing calls for
  no gain at this traffic level.
- **The image is lean.** `.dockerignore` excludes the notebook, raw CSVs, scripts, tests and
  frontend source — only `app.py`, two small data files, the model and the built site ship.
- **Free tiers sleep.** Render's free instance spins down when idle; the first request after
  that takes ~30s. Fine for a demo link, not for anything time-sensitive.
- **OSRM is a public demo server.** It is rate-limited and makes no uptime promise. Under load
  the app falls back to the calibrated haversine automatically and says so in `distance_source`.

> ⚠️ The Docker build has **not** been run — Docker isn't installed in the environment this was
> developed in. The `--outDir` override that the image depends on *was* verified directly, but
> expect to iterate once on the first real build.

---

## Project structure

```
RoadTripCost.ipynb       Weeks 1–10. The graded artifact — start here.
Project_Report.pdf       80-page complete report, Weeks 1–10 + 60 viva Q&A (generated)
TASK5.md                 Task 5 answered, tables filled (generated — see scripts/)
road_trip_data.csv       Original 1,140 trips, 9 city pairs. Untouched.
app.py                   FastAPI backend
roadtrip_features.py     Feature builders shared by the scripts and the API
data/
  india_cities.csv       3,739 cities, real coordinates (GeoNames, CC BY 4.0)
  fuel_prices.csv        35 states — `source` column marks verified vs estimated
  road_trip_wide.csv     25,000 generated trips, 534 cities, 50–1500 km
  real_distances.csv     840 REAL OSRM road distances — the only observed data
  *_report.json          Saved metrics; TASK5.md and the notebook read these
models/
  model.joblib           Week 6 regressor + cost-band classifier + GD history
  pipeline.joblib        The six sub-models + the chained cost regressor (served)
report/                  HTML sources for Project_Report.pdf
scripts/
  evaluation.py          The Task 5 checklist, implemented once
  train_pipeline.py      Fits and scores the six sub-models
  build_report_pdf.py    Assembles report/ + generated tables → Project_Report.pdf
  ...                    Data build, training, notebook and doc generation
frontend/                React 19 + Vite 7 + Tailwind v4
static/                  Built frontend (generated — do not hand-edit)
tests/                   pytest suite for the API (65 tests)
Dockerfile               Two-stage build: Node compiles, Python serves
render.yaml              Render blueprint
```

Two model files are **not** committed — they are large, reproducible, and nothing needs them at
serve time: `models/pipeline_eval.joblib` (the evaluation-only models) and
`models/end_to_end.joblib` (the Week 10 benchmark). Re-run their scripts to rebuild.

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
2. **The generated geography is off by 6.3% per route.** The wide dataset sets
   `distance_km = haversine × N(1.2355, 0.082)`. Measured against the 840 real routes that is
   nearly unbiased in the mean (−0.4%) but wrong by 6.3% on a typical single route. Costs
   measured *on* that dataset stay self-consistent — its own distance column generated its own
   costs — but the distance model is deliberately trained on the real data instead, so the app
   serves real geography even though the cost model was fitted on generated trips.
2. **Fuel prices: 6 states verified, 29 estimated.** No free, current, machine-readable
   state-wise feed exists for India; three candidate sources were checked and rejected (one
   stale, one paid, one JS-only). `data/fuel_prices.csv` records which is which per row, and the
   API warns when a prediction uses an estimate. **The user can override it** — supplying the
   price at their own pump replaces the estimate entirely and is reported as `source: "user"`.
4. **The rush-hour traffic profile was injected deliberately.** In the original data traffic is
   statistically independent of departure hour, making it unpredictable — on that data all five
   classifiers land on exactly the 50.88% majority baseline (see [`TASK5.md`](TASK5.md)
   Appendix A). The generator adds a realistic profile, so the traffic classifier's 57.0%
   measures that design choice, not Indian roads.
5. **Predictions beyond 1,500 km extrapolate.** The model was trained on 50–1500 km; the API
   returns an explicit warning outside that range rather than silently guessing.
6. **Routing depends on a public demo server.** OSRM's free instance is rate-limited. Distances
   are cached, and fall back to the fitted distance model (R² 0.9889, MAE 30.67 km on held-out
   real routes) when it is unavailable. `distance_source` in the response says which was used.
7. **Parking is not predicted, because it cannot be.** Six model families all scored test
   R² ≤ 0, so the training mean (₹75.09) is served and `/api/metrics` discloses it. Roughly ₹50
   of the reported error bar is this term alone.
8. **The error bar is large and that is the honest number.** MAE ₹391 on a typical estimate,
   of which about ₹529 of ₹596 RMSE is irreducible noise in the generated data (toll ±₹219,
   parking ±₹50, and the fuel bill inheriting the distance error). Only the fuel-bill term could
   be improved by a better model.

---

## Data sources

- **Cities & coordinates** — [GeoNames](https://download.geonames.org/export/dump/)
  `cities15000`, CC BY 4.0
- **Road distance** — [OSRM](http://project-osrm.org/) public demo, live at request time, with
  the fitted distance model as fallback
- **Road distance training data** — 840 real OSRM routes collected by
  `scripts/fetch_real_distances.py` → `data/real_distances.csv`
- **Fuel prices** — metro anchors from public reporting (May 2026); remaining states estimated
