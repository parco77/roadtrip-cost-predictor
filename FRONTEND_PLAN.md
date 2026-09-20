# Frontend Plan — Road Trip Cost Predictor

**Design reference:** Trouvaille-style editorial travel landing page (torn-paper hero, serif display type, desaturated photography, alternating text/image bands)
**Design system:** `design-system/roadtrip-cost-predictor/MASTER.md`
**Backend:** ✅ built and verified — `app.py`
**Frontend:** ✅ built and verified in-browser — `frontend/`
**Status:** complete. Run `uvicorn app:app` and open http://127.0.0.1:8000

---

## 1. What we're building

A single-page editorial site that looks like a travel magazine and answers one question:

> **"What will this road trip actually cost me?"**

**Now covering 537 Indian cities** — any pair, not the original 9 routes.

| Fact | Value |
|---|---|
| Cities selectable | **537** (population ≥ 100k; 3739 available in `data/india_cities.csv`) |
| Training trips | **25,000** across ~5,000 real routes |
| Distance range | **50 – 1500 km** (median 412) |
| Cost regression | **R² 0.999707 · RMSE ₹54.16 · MAE ₹39.66** |
| Cost-band classifier | **85.8%** accuracy (baseline 25%) |
| Traffic classifier | **53.9%** vs. 38.8% majority baseline (**+15.1 pts**) |

### What changed from the original project

The original `road_trip_data.csv` has 1140 rows over 9 fixed city pairs. It stays untouched — Weeks 1–3 of the notebook still run on it. The site is served by a regenerated wide dataset built from the **cost formula reverse-engineered from that original data at R² = 0.999162** (see `FINDINGS.md`), applied to real geography:

```
total_trip_cost = fuel_consumption_litres × fuel_price
                + toll_cost + parking_cost
                + distance_km × maintenance_rate(vehicle_type)
```

Be honest about this in §6 of the page: the numbers are physically consistent with the original data, but the wide dataset is generated, not observed.

---

## 2. Stack

| Layer | Choice | Status |
|---|---|---|
| Backend | **FastAPI** (`app.py`) | ✅ done |
| Frontend | **React 19 + Vite 7** | ✅ scaffolded |
| Styling | **Tailwind CSS v4** (`@tailwindcss/vite`) | ✅ tokens written |
| Motion | **GSAP + ScrollTrigger** via `@gsap/react` | ✅ done |
| Charts | **Recharts** (lazy-loaded) | ✅ done |
| Icons | **lucide-react** | ✅ done |
| Fonts | Google Fonts | ✅ linked |

### Why these, specifically

- **Tailwind v4, not v3.** v4 is CSS-first: no `tailwind.config.js`, design tokens live in an `@theme` block in `src/index.css`. That maps one-to-one onto the token table in §3 — the design system *is* the config.
- **`@gsap/react`, not bare GSAP.** Its `useGSAP()` hook auto-reverts every animation and kills every ScrollTrigger on unmount. Raw `gsap.from()` inside `useEffect` leaks ScrollTriggers across React 19 StrictMode double-mounts and the scroll positions silently drift. This is the single most common way GSAP-in-React goes wrong.
- **Recharts, not Chart.js.** Chart.js is imperative and needs a ref + manual teardown; Recharts is declarative components, which is the whole reason to be in React.

### Running it

Two processes in development:

```bash
uvicorn app:app --reload
```
```bash
cd frontend && npm install && npm run dev
```

Vite serves the UI on `:5173` and **proxies `/api` to `:8000`**, so there is no CORS problem and no hardcoded backend URL in the app code.

For production, one process — `npm run build` emits straight into `../static/`, which FastAPI already mounts at `/`:

```bash
cd frontend && npm run build
```

Then `uvicorn app:app` alone serves both API and UI on `:8000`.

---

## 3. Design system

The `ui-ux-pro-max` database returned *Minimalist Monochrome* as the style but an *adventure orange + teal* palette. Those fight each other. **Resolution: monochrome base, orange rationed to a single accent** — the reference image is almost entirely desaturated, so colour appears only where it must earn attention.

In Tailwind v4 these live in an `@theme` block in `src/index.css` — no `tailwind.config.js`. Each
token automatically generates its utilities (`bg-paper`, `text-ink`, `font-display`, …):

```css
@import "tailwindcss";

@theme {
  --color-page:      #FFFFFF;
  --color-paper:     #FFF7ED;   /* alternating bands */
  --color-ink:       #0F172A;
  --color-muted:     #525252;
  --color-hairline:  #E5E5E5;
  --color-accent:    #C2410C;   /* CTA + the cost figure ONLY — see the contrast note below */
  --color-series:    #0891B2;   /* chart second series ONLY */

  /* cost-band badges - the one licensed exception to the monochrome rule */
  --color-band-excellent: #15803D;
  --color-band-good:      #0F172A;
  --color-band-average:   #B45309;
  --color-band-poor:      #B91C1C;

  --font-display: "Playfair Display", Georgia, serif;
  --font-body:    "Source Serif 4", Georgia, serif;
  --font-mono:    "JetBrains Mono", ui-monospace, monospace;

  --radius-none: 0px;           /* everything except the two pill buttons */
}
```

**Accent discipline:** `--color-accent` appears in exactly three places — the "Estimate my trip" CTA, the predicted ₹ figure, and one chart series. Nowhere else. That rationing is what stops this reading as a template.

**Why orange-700 and not the orange-600 the palette specified.** `#EA580C` measures **3.56:1** on
white. That is fine for the huge cost figure — large text only needs 3:1 — but it is a WCAG AA
**failure** for the 11px white label on the primary CTA and for the 14px accent text in the
ablation table. `#C2410C` is **5.18:1** in both directions, passes everywhere, and reads as a
deeper burnt orange that suits the desaturated photography better anyway.

| Role | Font | Spec |
|---|---|---|
| Display | **Playfair Display** 700/900 | `clamp(2.75rem, 9vw, 8rem)`, `tracking-tight`, `leading-[0.95]` |
| Body | **Source Serif 4** 300/400 | 17px, `leading-relaxed`, max **62ch** |
| Labels / data | **JetBrains Mono** 400/500 | 12px, `uppercase`, `tracking-[0.18em]` |

Every number, unit, feature name and metric is mono. That single choice makes an editorial layout read as *technical* rather than decorative.

```html
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

**Shape:** `border-radius: 0` everywhere except the two pill buttons (`rounded-full`, matching the reference's `LET'S GO` / `READ MORE`). No box-shadows — depth comes from 1px hairlines, 4px black rules, and colour inversion. Paper noise at `opacity: 0.03`. Spacing scale `16/24/32/48/64`, sections `py-24 md:py-32`.

---

## 4. Reference image → our sections

| # | Reference | Ours |
|---|---|---|
| 1 | Script logo + 4-item nav | `RoadTrip` wordmark (Playfair italic) + **Estimate · Drivers · Model · Data** |
| 2 | Mountain photo, **GO TRAVEL**, `LET'S GO` pill, torn edge | Highway photo, **KNOW THE COST**, `ESTIMATE MY TRIP` pill, same torn edge |
| 3 | *A Get Away For Creatives* + 3 icon columns | *Three Things Set The Price* — **Fuel · Tolls · Wear** with real coefficients |
| 4 | Text left / photo right, `READ MORE` | ★ **THE PREDICTOR** — form left, live result right |
| 5 | Photo left / text right | *Two Roads, One Answer* — sklearn vs. gradient descent, loss curve |
| 6 | *Our Mission Is Inspiration*, centred | *The Report Card* — R² / MAE / classifier accuracies |
| 7 | Newsletter over wave photo | Dataset download, notebook link, honest provenance note |

The torn-paper edge is the signature move. Reproduce it as an **inline SVG path mask** (`<svg viewBox="0 0 1440 60" preserveAspectRatio="none">`), not a raster PNG — it must scale cleanly to 375px.

---

## 5. Section specs

### §2 Hero
- `h-[100svh]` — `svh` not `vh`, or mobile browser chrome clips it.
- Photo `filter: grayscale(0.85) contrast(1.05)`, with an `rgba(0,0,0,0.25)` scrim behind the headline. Do **not** rely on `mix-blend-mode` alone — it fails contrast over light sky.
- Mono subline: `537 CITIES · 25,000 TRIPS · R² 0.9997`
- Torn-paper SVG at the bottom, filled `--color-bg`.

### §3 Three drivers
Centred 62ch paragraph, then a 3-column grid (stacks below 768px). **Lucide SVG icons** (`fuel`, `receipt-indian-rupee`, `wrench`) — never emoji — plus mono labels showing the real recovered rates:

```
FUEL BURNT           TOLLS               WEAR & TEAR
litres × ₹/litre     ₹1.30 / km          ₹0.57–0.84 / km by vehicle
```

### §4 The predictor — the core section

**The problem, and how the backend already solves it.** The model's strongest feature is `fuel_consumption_litres` (corr +0.96) — and no user planning a trip can possibly type that in. It's **derived server-side**, never asked:

```
litres = (distance ÷ mileage) × traffic_multiplier
         Low 0.944 · Medium 1.045 · High 1.165
```

The API returns `derived.fuel_consumption_explained` as a ready-made string —
`"304 km / 15.5 km/l x 1.165 (high traffic) = 22.856 L"`. **Render it verbatim in mono.** Showing the arithmetic instead of hiding it is the difference between a demo and a black box.

**Form fields:**

| Field | Control | Notes |
|---|---|---|
| From / To | **autocomplete** → `GET /api/cities?q=` | Replaces the old 9-route dropdown. Debounce 200ms. Show `City, State`. |
| Vehicle | segmented control | Hatchback / Sedan / SUV |
| Fuel | segmented control | Petrol / Diesel / CNG |
| Mileage | slider + number, 8–23 | default 15.5 |
| Departure hour | slider 0–23 | drives traffic inference |
| Month | select | drives weather realism |
| Traffic | segmented, **with an "Auto" option** | Auto ⇒ omit the field; the classifier infers it |
| Parking | number, 0–150 | default 70 |
| Passengers | stepper 1–8 | powers cost-per-head |

**Do not ask for:** distance, toll, or litres. All three come back from the API.

**Fuel price is the exception** — prefilled, editable, and it matters that it works this way:

- The city payload from `/api/cities` carries all three of that state's prices, so selecting a
  city prefills the field with **no extra request**.
- While untouched, the field follows the fuel type (Petrol → Diesel swaps the price).
- Once edited it stops following, tracked by a `priceEdited` flag. Without that, changing fuel
  type or origin would silently discard what the user typed.
- **The prefilled value is not sent back.** `fuel_price` is omitted from the payload unless the
  user actually edited it — otherwise a table estimate would return as `source: "user"` and
  suppress the warning that should accompany it.

**Result card:**
```
ESTIMATED TOTAL
₹ 3,013                      ← accent, Playfair, clamp(3rem, 7vw, 5.5rem)
± ₹43 typical error          ← mono, muted (this is the real MAE)
₹753 per passenger           ← mono

[ AVERAGE ]                  ← cost-band badge, see below

FUEL         ₹2,334
TOLL         ₹  396
WEAR & TEAR  ₹  214          ← horizontal black bars
PARKING      ₹   70

304 km · OSRM routing · high traffic (inferred)
```

**Cost-band badge colours** — the one place a second colour is allowed, because the label is meaningless in monochrome:
`Excellent` green · `Good` neutral-dark · `Average` amber · `Poor` red. Never colour alone — always show the word too.

**Warnings.** The API returns a `warnings[]` array. Render each as a mono line under the result, muted, with a small `alert-triangle` icon. They fire for out-of-training-range distance, haversine fallback when routing is down, and estimated (non-verified) fuel prices. **Do not suppress these** — they're the honest part of the product.

**Form UX rules (all High severity in the `ux` domain):**
- Visible `<label>` on every input — placeholder-only is a fail.
- Validate on **blur**, not submit. Error text under its own field.
- Submit shows **loading → result**, never a dead click.
- Autocomplete must be keyboard-navigable: ↑/↓ to move, Enter to select, Esc to close, `aria-activedescendant` on the input.
- Touch targets ≥ 44×44px; slider thumbs 44px.

### §5 Two roads, one answer
Chart.js line chart of the notebook's `loss_history` (log y-axis) with sklearn's solution as a flat dashed reference. ~60 words explaining the convex-bowl argument. Mono table: largest weight gap **0.0425**, prediction gap **≤ ₹0.011**.

**Add the ablation result** — it's the strongest technical finding in the project:

| Model | Test MAE |
|---|---|
| Linear, no interaction | ₹143.85 |
| RandomForest | ₹69.97 |
| GradientBoosting | ₹73.50 |
| **Linear + `litres × price`** | **₹39.66** |

One engineered feature beats both ensembles — and stays linear, so the from-scratch gradient descent still applies.

### §6 Report card
Giant mono figures on paper background, `border-t-4 border-black`:
```
R²              MAE             COST BAND       TRAFFIC
0.9996          ₹39.66          85.8%           53.9%
                                (base 25%)      (base 38.8%)
```
Always print the baseline next to each classifier accuracy. Caption, one line, honest:
*"Traffic is only partly determined by departure hour — 53.9% against a 38.8% baseline is a real signal, not a strong one."*

### §7 Footer
Same full-bleed photo + scrim treatment. Download `road_trip_wide.csv`, link the notebook, and state provenance plainly: cities and coordinates are real (GeoNames), distances are real (OSRM), fuel prices are 6 states verified / 29 estimated, and trip records are generated from the recovered formula.

---

## 6. API contract (as built — verified working)

```
GET /api/cities?q=jai&limit=12
```
```json
{ "count": 3, "results": [
  { "city": "Jaipur", "state": "Rajasthan", "lat": 26.91962, "lon": 75.78781, "population": 2711758 }
]}
```

```
POST /api/predict
```
```json
{ "start_city": "Delhi", "destination_city": "Jaipur",
  "vehicle_type": "Sedan", "fuel_type": "Petrol", "mileage": 15.5,
  "departure_hour": 9, "month": 6, "passengers": 4 }
```
Optional: `start_state`, `destination_state` (disambiguate duplicate names), `traffic_level` (omit for Auto), `parking_cost`.

```json
{
  "route":      { "from": {...}, "to": {...}, "distance_km": 304.09, "distance_source": "osrm" },
  "inputs":     { "fuel_price": 102.12, "fuel_price_source": "verified", ... },
  "derived":    { "traffic_level": "High", "traffic_inferred": true,
                  "fuel_consumption_litres": 22.856,
                  "fuel_consumption_explained": "304 km / 15.5 km/l x 1.165 (high traffic) = 22.856 L",
                  "toll_cost": 395.62, "estimated_time_minutes": 384.1 },
  "prediction": { "total_trip_cost": 3013.67, "typical_error": 42.6,
                  "cost_per_km": 9.91, "cost_per_passenger": 753.42,
                  "cost_band": "Average",
                  "breakdown": { "fuel": 2334.05, "toll": 395.62,
                                 "parking": 70.0, "wear_and_tear": 214.0 } },
  "warnings":   []
}
```

```
POST /api/departure-sweep   → expected cost for all 24 departure hours (same payload as predict)
GET  /api/metrics           → test-set numbers + honesty notes for §6
GET  /api/loss-curve        → gradient-descent curve + sklearn comparison for §5
GET  /api/health            → { "status": "ok", "cities": 537 }
```

**The departure sweep** fires alongside `/api/predict` (not after — the distance is already
cached, so it costs one request and no extra routing calls) and renders full-width below the
two columns of §4 as `DepartureChart`. Two decisions worth keeping:

- **Expected cost, not argmax cost.** Weighting each hour's traffic *probabilities* against the
  cost at each level gives 24 distinct values. Taking the single most likely level collapsed the
  whole chart onto 3 steps and threw away the classifier's confidence.
- **The y-axis does not start at zero,** because the full range is ~8% of the trip cost and a
  zero baseline renders 24 identical-looking bars. The axis is labelled in exact rupees and the
  caption states the range, so the zoom is never hidden. Ticks are plain rupees rather than
  `₹2.4k` — at a ~₹200 range, one decimal of thousands prints the same label twice.

**Errors to handle in the UI:** `400` same city · `404` city outside the 537-city domain · `422` field out of range. All return `{"detail": ...}`.

---

## 7. Motion (GSAP, tier: Standard 7/10)

| Element | Animation |
|---|---|
| Hero headline | `SplitText` per word, `y: 40 → 0`, `stagger: 0.08`, `power3.out`, 0.9s |
| Torn-paper edge | `scaleY 0 → 1` from bottom |
| 3 driver columns | `opacity 0→1, y 16, scale 0.92, stagger 0.06, back.out(1.4)` on scroll |
| Section photos | Parallax `yPercent: -12`, scrubbed |
| Result figure | Count-up 0 → predicted, 700ms, `power2.out` |
| Breakdown bars | `scaleX 0 → 1`, `transform-origin: left`, stagger 0.05 |

**Always scope animations with `useGSAP`.** The `scope` ref confines selectors to that component's
subtree, and the hook reverts everything (ScrollTriggers included) on unmount:

```jsx
import { useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { useGSAP } from '@gsap/react'

gsap.registerPlugin(ScrollTrigger, useGSAP)   // once, in main.jsx

export default function Drivers() {
  const root = useRef(null)

  useGSAP(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    gsap.from('.driver-col', {
      opacity: 0, scale: 0.92, y: 16, duration: 0.4,
      stagger: { each: 0.06, from: 'start', grid: 'auto' },
      ease: 'back.out(1.4)',
      scrollTrigger: { trigger: root.current, start: 'top 80%',
                       toggleActions: 'play none none reverse' },
    })
  }, { scope: root })

  return <section ref={root}>…</section>
}
```

**Rules:**
- Reduced motion is checked **inside** `useGSAP` and returns early — never animate then speed up.
- `toggleActions: 'play none none reverse'` stops re-triggering on every scroll direction change.
- Animate only `transform` and `opacity` — never `width`, `height`, or `top`.
- Parallax goes on decorative layers only, `yPercent` delta 5–15. Never parallax body copy.
- The count-up on the result figure uses GSAP on a plain object + `onUpdate` → `setState`, or it will
  fight React's render cycle.

---

## 8. Files

```
MLnewproject/
├── RoadTripCost.ipynb          weeks 1-3, unchanged, still on the original CSV
├── road_trip_data.csv          original 1140 rows, untouched
├── FINDINGS.md                 recovered formula + all evidence
├── app.py                      ✅ FastAPI backend
├── requirements.txt            ✅
├── data/
│   ├── india_cities.csv        ✅ 3739 cities, real lat/lon (GeoNames)
│   ├── fuel_prices.csv         ✅ 35 states, 6 verified / 29 estimated
│   └── road_trip_wide.csv      ✅ 25,000 trips, 534 cities
├── models/model.joblib         ✅ regressor + 2 classifiers + scalers + metrics
├── scripts/                    ✅ build_cities · build_fuel_prices · build_wide_dataset · train_models
├── static/                     ✅ generated by `npm run build` — do not hand-edit
└── frontend/
    ├── package.json            ✅   vite.config.js ✅   index.html ✅
    └── src/
        ├── main.jsx            ✅ registers GSAP plugins
        ├── App.jsx             ✅ composes the 6 rendered sections
        ├── index.css           ✅ @theme tokens + paper noise
        ├── api.js              ✅ every fetch lives here
        ├── lib/motion.js       ✅ shouldAnimate() guard — see the note below
        ├── hooks/
        │   ├── useCitySearch.js   ✅ debounced + AbortController
        │   └── usePrediction.js   ✅ idle/loading/error/ready state machine
        └── components/
            ├── Nav.jsx            ✅   Hero.jsx           ✅   TornEdge.jsx         ✅
            ├── Drivers.jsx        ✅   Predictor.jsx      ✅   CityAutocomplete.jsx ✅
            ├── ResultCard.jsx     ✅   CostBandBadge.jsx  ✅   Breakdown.jsx        ✅
            ├── Warnings.jsx       ✅   ModelStory.jsx     ✅   LossCurve.jsx        ✅
            └── ReportCard.jsx     ✅   Footer.jsx         ✅
```

**Bundle:** main 342 kB (118 kB gzip) + `LossCurve` 362 kB lazy chunk. Recharts is code-split
via `React.lazy` because the chart sits well below the fold — without that the single bundle was
704 kB and tripped Vite's size warning.

**The one non-obvious bug found during verification.** `gsap.from(el, {opacity: 0})` writes its
START state synchronously but advances on `requestAnimationFrame`, which browsers do not fire in a
tab that is not compositing. Loading the page in a background tab left the hero headline, driver
columns and torn edge **permanently invisible**, and the cost figure frozen at ₹0. Fixed with
`src/lib/motion.js`:

```js
export function shouldAnimate() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return false
  if (document.hidden) return false     // rAF never fires -> gsap.from() would freeze at opacity 0
  return true
}
```

When it returns false, components render the finished state directly instead of animating toward
it. Any new entrance animation must go through this guard.

`shouldAnimate()` alone would skip animation *permanently* for anyone who opened the page in a
background tab. `hooks/usePageVisible.js` fixes that: it reports false while hidden, flips true
the first time the page is shown, and is passed into every `useGSAP` dependency array so the
animations initialise at that moment.

**The second bug — the hero headline never animating.** `photoOk` was in the entrance timeline's
dependency array. When `/hero.webp` was missing, the `<img>` `onError` flipped it, the deps
changed, and `useGSAP` re-ran the timeline *mid-flight* — `useGSAP` does not revert on dependency
change, so a second `gsap.from()` landed on words that were already animating, the two tweens
fought, and the headline parked off-screen. Three fixes, all of which matter:

1. **Split the effects.** The entrance depends only on `pageVisible`; the backdrop parallax keeps
   `photoOk`. Swapping the backdrop has nothing to do with the copy.
2. **`fromTo`, not `from`.** Explicit start *and* end states make a re-run idempotent instead of
   capturing whatever mid-tween position happened to be current.
3. **`contextSafe`.** The timeline is deferred behind `document.fonts.ready` (so the masks are
   measured with Playfair, not the fallback face). `gsap.context()` only captures animations
   created during the *synchronous* run of the callback, so a deferred tween escapes the context
   and never gets reverted on unmount. `useGSAP((context, contextSafe) => …)` wraps it back in.

**Component notes:**
- **`CityAutocomplete`** is the only genuinely tricky component: debounce 200ms, `AbortController`
  on every keystroke so a slow response can't overwrite a newer one, full keyboard support, and
  `role="combobox"` + `aria-activedescendant` wiring. Build it first and build it properly.
- **`usePrediction`** should be an explicit state machine (`idle | loading | error | ready`), not
  three loose booleans — the UI has a distinct look in each state and booleans let them contradict.
- **`api.js`** holds every `fetch`. No component should ever contain a URL string.
- Form inputs are **controlled** (`value` + `onChange`), per React convention.

Images: desaturated highway/ghat photos from Unsplash or Pexels, dropped in `frontend/public/`.
**WebP with `srcset`**, explicit `width`/`height` on every `<img>` to keep CLS < 0.1.

---

## 9. Pre-delivery checklist

**React / Vite specific**
- [x] Every GSAP animation is inside `useGSAP` with a `scope` ref — no bare `useEffect` animations
- [x] No ScrollTrigger leaks across StrictMode double-mount (scroll to bottom, hot-reload, re-check)
- [x] `AbortController` cancels in-flight autocomplete requests
- [x] All form inputs controlled; no `ref`-read uncontrolled inputs
- [x] No `/api` URL strings outside `src/api.js`
- [x] `npm run build` succeeds and `uvicorn app:app` alone serves the built UI at `/`

**General**
- [x] SVG icons only (`lucide-react`) — **no emoji as icons**
- [x] Every input has a visible `<label htmlFor="...">`
- [x] Autocomplete keyboard-navigable (↑ ↓ Enter Esc, `aria-activedescendant`)
- [x] Cost-band badge never uses colour alone — the word is always shown
- [x] `warnings[]` rendered, never suppressed
- [x] Classifier accuracies always shown next to their baselines
- [x] Body text ≥ 4.5:1; hero scrim verified over the actual photo
- [x] Focus rings visible everywhere — never bare `outline: none`
- [x] Touch targets ≥ 44×44px, ≥ 8px apart
- [x] `prefers-reduced-motion` respected
- [x] Tested at **375 / 768 / 1024 / 1440**; no horizontal scroll at 375px
- [x] Viewport meta present, zoom **not** disabled
- [x] Loading state on predict; graceful error if the API is down
- [x] No raster images at all — the hero is an inline SVG ridgeline, so there is nothing to
      lazy-load and no CLS risk. If you swap in a real photograph, this reverts to: WebP +
      `srcset` + explicit `width`/`height`.

---

## 10. Honest notes for the write-up

1. **R² = 0.9996 is not a brag.** The dataset is generated from a formula, so the target is nearly deterministic given the features. Say so. The interesting result is the *ablation* — that one interaction term beats two ensemble models.
2. **Fuel prices are 6 verified / 29 estimated.** No free current state-wise feed exists for India; three candidate sources were checked and rejected (one stale, one paid, one JS-only). `data/fuel_prices.csv` carries a `source` column per state.
3. **Traffic realism was deliberately added.** In the original CSV, traffic is independent of departure hour, so a classifier could not beat the 51% majority baseline. The wide dataset gives traffic a real rush-hour profile — a modelling choice, documented in `scripts/build_wide_dataset.py`.
4. **The cost-band classifier scores high by construction**, since it's a quartile split of ₹/km that the regressor already predicts. It earns its place because the UI needs the label, not because it's a hard problem. The traffic classifier is the genuinely non-trivial one.
5. **OSRM is a public demo server** with rate limits. Distances are cached (`lru_cache`, 4096 entries) and fall back to calibrated haversine. If you deploy this for real traffic, self-host OSRM or use a keyed provider.
