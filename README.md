# Road Trip Cost Predictor

Tell it how far you are driving and what fuel costs where you are. It prices the trip for a
**hatchback, a sedan and an SUV at once**, on all three fuels — served through an editorial
single-page site.

The litres you will burn, the tolls you will pay and the km/l each vehicle manages are each
predicted by their own fitted sub-model. Nothing in the prediction path is a hardcoded constant.

**By Param Kotadiya** · Semester 5 Machine Learning project.

| | |
|---|---|
| **Live site** | https://roadtrip-cost.vercel.app |
| **API** | https://roadtrip-cost-api.onrender.com · [health](https://roadtrip-cost-api.onrender.com/api/health) · [docs](https://roadtrip-cost-api.onrender.com/docs) |

> The API is on a free instance that sleeps after ~15 minutes idle, so the first estimate after
> a quiet spell can take up to a minute. The distance lookup page is unaffected — that table
> ships with the UI rather than coming from the API.

```bash
cd backend && pip install -r requirements.txt && uvicorn app:app   # API  :8000
cd frontend && npm ci && npm run dev                               # UI   :5173
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

Full working: [`Project_Report.pdf`](ml/Project_Report.pdf), Part 12.

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

Full derivations: [`Project_Report.pdf`](ml/Project_Report.pdf) — Part 10 for every model
and what it does, Part 12 for the evaluation.

---

## Running it

**Prerequisites:** Python 3.11+, Node 18+.

### The app

Two processes, because there are two in production. Nothing needs training first — both model
bundles and the two data files the API reads at import are committed in `backend/`.

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app                  # → http://127.0.0.1:8000
```

```bash
cd frontend
npm ci
npm run dev                      # → http://localhost:5173
```

Vite proxies `/api` → `:8000`, so no backend URL is configured for local work and development
exercises the same relative paths the deployed build does.

> **Note:** Vite binds to `localhost`, which Node resolves to IPv6 on Windows. Use
> `http://localhost:5173`, not `127.0.0.1`.

Tests:

```bash
cd backend && pytest -q          # 70 tests
```

### Rebuilding the models from scratch

Run from `ml/`. Note where the outputs land — a script that produces something the server loads
writes it **into `backend/`**, and the distance table is written into `frontend/`. There is no
copy step to forget, because there is no copy.

```bash
python scripts/build_cities.py           # 3,739 cities, real lat/lon  → backend/data/
python scripts/build_fuel_prices.py      # per-state fuel prices       → backend/data/
python scripts/build_wide_dataset.py     # 25,000 trips over real geography
python scripts/fetch_real_distances.py   # 840 REAL road distances from OSRM   (~6 min)
python scripts/build_distance_matrix.py  # → frontend/public/city_distances.json

python scripts/train_pipeline.py         # → backend/models/pipeline.joblib  (sub-models)
python scripts/train_models.py           # → backend/models/model.joblib     (band + curve)

python scripts/evaluate_models.py        # → ml/data/task5_evaluation.json    (~5 min)
python scripts/build_report_pdf.py       # → ml/Project_Report.pdf
```

> Order is load-bearing in two places. `train_pipeline.py` and `train_models.py` are independent
> of each other, but `evaluate_models.py` reads `pipeline_report.json`, so it must follow the
> first of them — and `build_report_pdf.py` reads all three JSON reports, so it goes last.

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

The two halves go to two hosts, because they are not the same kind of thing: the built UI is
static files that want a CDN, the API is a Python process that has to hold two model bundles in
memory. `backend/` and `frontend/` each contain exactly what their host uploads; `ml/` is
uploaded nowhere.

**Do the backend first — the frontend build needs its URL.**

### 1 · Push to GitHub

```bash
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin master
```

### 2 · Backend → Render

On [render.com](https://render.com): **New → Blueprint → pick the repo → Apply.** `render.yaml`
supplies every setting, so there is nothing to type.

This uses Render's **native Python runtime — no Docker.** `runtime: python` in the blueprint is
what selects it (`docker` is a separate value of that same field), and there is no Dockerfile in
the repo, so the manual flow below cannot auto-detect one either.

Prefer to click through it instead? **New → Web Service**, then:

| Field | Value |
|---|---|
| Root Directory | `backend` |
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/api/health` |
| Environment variable | `PYTHON_VERSION` = `3.13.5` |

First build takes roughly 3–5 minutes, most of it the scikit-learn wheel. When it goes live,
copy the URL and check it:

```bash
curl https://<your-service>.onrender.com/api/health
```

### 3 · Frontend → Vercel

On [vercel.com](https://vercel.com): **Add New → Project → import the repo**, then:

| Field | Value |
|---|---|
| **Root Directory** | **`frontend`** ← the one setting that will break the build if wrong |
| Framework preset | Vite (auto-detected) |
| Environment variable | `VITE_API_URL` = your Render URL, **no trailing slash** |

Deploy. Vercel reads `frontend/vercel.json` for the rest.

### 4 · Close the loop

Back on Render → your service → **Environment** → add:

```bash
ALLOWED_ORIGINS = https://<your-project>.vercel.app
```

Render restarts on save. Until you set this the API falls back to `*`, so the site works either
way — this narrows it to your origin.

### Things that actually go wrong

- **`VITE_API_URL` is inlined at build time**, not read at runtime. Changing it needs a
  **redeploy**, not a restart. If the deployed site is still calling the wrong host, this is why.
- **A trailing slash on `VITE_API_URL`** produces `//api/predict`. Leave it off.
- **Vercel's Root Directory must be `frontend`**, or the build never finds `package.json`.
- **Free Render instances sleep** after ~15 minutes idle, and the next request pays a cold start
  of up to a minute — model loading included. The distance table is a static asset served by
  Vercel, so that page stays usable while the API wakes up.
- **Versions are pinned on purpose.** The model files are pickles; scikit-learn does not promise
  that one minor version reads another's. The failure mode is not a crash but a warning and a
  model that scores differently, so `requirements.txt` is exact where it has to be.
- **Render's default Python is 3.14.3**, which has no wheels for the pinned numpy, pandas and
  scikit-learn — the build would try to compile them from source, or just fail. `PYTHON_VERSION`
  is set to `3.13.5` in `render.yaml` for exactly this reason. It must be *fully qualified* there;
  Render rejects a bare `3.13` in the environment variable, though `backend/.python-version`
  accepts one. Both are set, and the environment variable takes precedence.
- **Single worker, deliberately.** Both model bundles and the 3,739-city table load per process,
  and the OSRM distance cache is in-process, so a second worker doubles memory and re-fetches
  every route it has not seen.
- **OSRM is a public demo server** — rate-limited, no uptime promise. Only `/api/distance` touches
  it, and it falls back to the fitted distance model and says which answered in `source`.

---

## Project structure

Three folders, split by where they end up.

```
backend/                     DEPLOYED to Render.  Holds only what answers a request.
  app.py                     FastAPI — the chained prediction pipeline
  roadtrip_features.py       The feature contract, imported by the API AND by training
  requirements.txt           Exact versions: the model files are pickles
  pytest.ini                 Puts backend/ on sys.path so `pytest` finds app.py
  .python-version            Pins 3.13.5; Render's own default is 3.14.3
  data/
    india_cities.csv         3,739 cities, real coordinates (GeoNames, CC BY 4.0)
    fuel_prices.csv          35 states — `source` marks verified vs estimated
  models/
    model.joblib             Cost-band classifier + gradient-descent history
    pipeline.joblib          The sub-models + the chained cost regressor (served)
  tests/test_api.py          70 tests

frontend/                    DEPLOYED to Vercel.  React 19, Vite 7, Tailwind v4, GSAP.
  src/api.js                 Every URL in the project, in one file
  public/city_distances.json 5,538 measured routes (25.6 KB gzipped) — shipped WITH
                             the UI, so the lookup survives the API being asleep
  vercel.json                Build settings
  .env.example               Template for VITE_API_URL

ml/                          DEPLOYED NOWHERE.  Everything that produced the models.
  road_trip_data.csv         Original 1,140 trips, 9 city pairs. Untouched.
  Project_Report.pdf         The full report (generated)
  report/                    Its HTML sources; two chapters are generated, not typed
  data/
    road_trip_wide.csv       25,000 generated trips, 534 cities, 50–1500 km
    real_distances.csv       840 REAL OSRM road distances — the only observed data
    curated_distances.csv    Fetch cache for the reference table (rebuildable)
    *_report.json            Saved metrics; the report's generated chapters read these
  scripts/
    evaluation.py            The Task 5 checklist, implemented once
    train_pipeline.py        Fits and scores the sub-models   → backend/models/
    build_distance_matrix.py Builds the shipped distance reference table
    build_report_pdf.py      Assembles report/ + generated tables → Project_Report.pdf
    ...                      Data build, training, evaluation

render.yaml                  Render blueprint — names backend/ as the service root
```

`ml/models/pipeline_eval.joblib` is **not** committed — it holds the comparison-only regressor,
is reproducible by re-running `train_pipeline.py`, and nothing needs it at serve time.

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
