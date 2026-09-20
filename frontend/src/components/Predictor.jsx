import { useEffect, useRef, useState } from "react"
import gsap from "gsap"
import { useGSAP } from "@gsap/react"
import { ArrowRight } from "lucide-react"

import ResultCard from "./ResultCard.jsx"
import RevealText from "./RevealText.jsx"
import { fetchDefaultMileage } from "../api.js"
import { usePrediction } from "../hooks/usePrediction.js"
import { revealWords, shouldAnimate } from "../lib/motion.js"
import { usePageVisible } from "../hooks/usePageVisible.js"

const VEHICLES = ["Hatchback", "Sedan", "SUV"]
const FUELS = ["Petrol", "Diesel", "CNG"]
const PRICE_FIELD = { Petrol: "petrol_price", Diesel: "diesel_price", CNG: "cng_price" }
const UNIT = { Petrol: "l", Diesel: "l", CNG: "kg" }

function Field({ label, hint, children, htmlFor }) {
  return (
    <div>
      <label htmlFor={htmlFor} className="label-mono block mb-2">
        {label}
      </label>
      {children}
      {hint && <p className="font-mono text-[13px] text-muted mt-1.5">{hint}</p>}
    </div>
  )
}

export default function Predictor({ picked }) {
  const [distance, setDistance] = useState("")
  const [prices, setPrices] = useState({ Petrol: "", Diesel: "", CNG: "" })
  // Once a price has been typed we never overwrite it. The three fields start empty and
  // fall through to the national average server-side, which the result labels as such.
  const [priceEdited, setPriceEdited] = useState({})
  const [parking, setParking] = useState(70)
  const [passengers, setPassengers] = useState(4)

  // One km/l per vehicle, seeded from the mileage sub-model. A single figure cannot be
  // honest for a hatchback and an SUV at the same time, and the gap between the three is
  // most of the gap between their costs — so this is the input that makes the comparison
  // mean anything rather than a convenience prefill.
  const [mileage, setMileage] = useState({})
  const [mileageEdited, setMileageEdited] = useState({})
  const [mileageFuel, setMileageFuel] = useState("Petrol")
  // Mirrors mileageEdited for the effect below to read. Without it, mileageEdited would have
  // to be a dependency, and every keystroke in a mileage box would fire another request.
  const editedRef = useRef({})

  const [touched, setTouched] = useState(false)
  const root = useRef(null)
  const distanceRef = useRef(null)
  const pageVisible = usePageVisible()

  const { status, result, error, predict } = usePrediction()

  // Seed the three mileages from the model. Re-runs when the reference fuel changes;
  // a vehicle the user has edited keeps their number.
  useEffect(() => {
    const controller = new AbortController()
    fetchDefaultMileage(mileageFuel, { signal: controller.signal })
      .then((r) =>
        setMileage((current) => {
          const next = { ...current }
          for (const v of VEHICLES) if (!editedRef.current[v]) next[v] = r.mileage[v]
          return next
        }),
      )
      .catch(() => {
        /* offline or aborted - keep whatever is already on screen */
      })
    return () => controller.abort()
  }, [mileageFuel])

  // "Use this" in the distance table lands here. `picked` is a fresh object every time,
  // so choosing the same distance twice still re-runs this and pulls focus back.
  useEffect(() => {
    if (!picked) return
    setDistance(String(picked.km))
    distanceRef.current?.focus({ preventScroll: true })
    distanceRef.current?.scrollIntoView({ behavior: "smooth", block: "center" })
  }, [picked])

  useGSAP(
    () => {
      if (!shouldAnimate()) return
      // Only the heading animates. The form deliberately does NOT: a stagger that gets
      // interrupted leaves its last child at opacity 0, and the last child is the submit
      // button. Never gate an interactive control on an animation completing.
      revealWords(gsap, root.current)
    },
    { dependencies: [pageVisible], scope: root },
  )

  const distanceValue = Number(distance)
  const distanceError =
    !touched || distance === ""
      ? distance === "" && touched
        ? "Enter a distance, or pick one from the table above"
        : null
      : distanceValue <= 10
        ? "Too short to estimate — enter more than 10 km"
        : distanceValue > 3000
          ? "Longer than any road trip this model can speak to — 3000 km is the ceiling"
          : null

  function onSubmit(event) {
    event.preventDefault()
    setTouched(true)
    if (distance === "" || distanceValue <= 10 || distanceValue > 3000) return

    const overrides = {}
    for (const v of VEHICLES) {
      if (mileageEdited[v] && mileage[v]) overrides[v] = Number(mileage[v])
    }

    const payload = {
      distance_km: distanceValue,
      parking_cost: Number(parking),
      passengers: Number(passengers),
      // Omit an untouched price so the backend uses the national average and SAYS it did.
      // Sending the placeholder back would relabel an average as the user's own figure and
      // silently suppress the warning that should sit beside it.
      ...Object.fromEntries(
        FUELS.filter((f) => priceEdited[f] && prices[f] !== "").map((f) => [
          PRICE_FIELD[f],
          Number(prices[f]),
        ]),
      ),
      ...(Object.keys(overrides).length ? { mileage: overrides } : {}),
    }

    predict(payload)
  }

  return (
    <section
      id="estimate"
      ref={root}
      data-section="04"
      data-section-name="The Estimator"
      className="rule-heavy py-24 md:py-32 px-6"
    >
      <div className="mx-auto max-w-6xl">
        <header className="max-w-[62ch]">
          <p className="label-mono">Section 04 — The Estimator</p>
          <RevealText
            as="h2"
            text="What Will This Trip Cost?"
            className="mt-4 block text-[clamp(2rem,5vw,3.5rem)]"
          />
          <p className="mt-5 text-muted">
            Four things you know. A price for a hatchback, a sedan and an SUV — on all three
            fuels — comes back. The litres, the tolls and each vehicle&rsquo;s mileage are
            worked out for you.
          </p>
        </header>

        <div className="mt-14 grid gap-12 lg:grid-cols-2 lg:gap-16 items-start">
          <form onSubmit={onSubmit} noValidate className="space-y-7">
            <Field
              label="Distance"
              htmlFor="distance"
              hint={
                <>
                  kilometres —{" "}
                  <a href="#distances" className="underline hover:text-ink">
                    look it up
                  </a>{" "}
                  if you do not know it
                </>
              }
            >
              <div className="flex items-baseline gap-3">
                <input
                  id="distance"
                  ref={distanceRef}
                  type="number"
                  inputMode="numeric"
                  min="11"
                  max="3000"
                  step="1"
                  value={distance}
                  onChange={(e) => setDistance(e.target.value)}
                  placeholder="500"
                  className="w-44 h-14 px-3 border-2 border-ink bg-page font-mono text-2xl tabular-nums"
                />
                <span className="font-mono text-lg text-muted">km</span>
              </div>
            </Field>

            {distanceError && (
              <p role="alert" className="font-mono text-[13px] text-band-poor -mt-3">
                {distanceError}
              </p>
            )}

            <fieldset>
              <legend className="label-mono mb-2">Fuel price ₹</legend>
              <p className="font-mono text-[13px] text-muted mb-3">
                Fill in the one you drive. Leave the others and they use the national average,
                which the result marks as an estimate.
              </p>
              <div className="grid grid-cols-3 gap-3">
                {FUELS.map((fuel) => (
                  <div key={fuel}>
                    <label
                      htmlFor={`price-${fuel}`}
                      className="font-mono text-[12px] uppercase tracking-[0.14em] block mb-1.5"
                    >
                      {fuel} <span className="text-muted">/{UNIT[fuel]}</span>
                    </label>
                    <input
                      id={`price-${fuel}`}
                      type="number"
                      min="30"
                      max="300"
                      step="0.01"
                      value={prices[fuel]}
                      placeholder="—"
                      onChange={(e) => {
                        setPrices((p) => ({ ...p, [fuel]: e.target.value }))
                        setPriceEdited((p) => ({ ...p, [fuel]: true }))
                      }}
                      className="w-full h-11 px-3 border border-ink bg-page font-mono text-base"
                    />
                  </div>
                ))}
              </div>
            </fieldset>

            <div className="grid sm:grid-cols-2 gap-6">
              <Field label="Parking ₹" htmlFor="parking" hint="for the whole trip">
                <input
                  id="parking"
                  type="number"
                  min="0"
                  max="2000"
                  value={parking}
                  onChange={(e) => setParking(e.target.value)}
                  className="w-full h-11 px-3 border border-ink bg-page font-mono text-base"
                />
              </Field>

              <Field label="Passengers" htmlFor="passengers" hint="1 to 8">
                <input
                  id="passengers"
                  type="number"
                  min="1"
                  max="8"
                  value={passengers}
                  onChange={(e) => setPassengers(e.target.value)}
                  className="w-full h-11 px-3 border border-ink bg-page font-mono text-base"
                />
              </Field>
            </div>

            <details className="border-t border-hairline pt-5">
              <summary className="label-mono cursor-pointer min-h-11 flex items-center">
                Mileage — predicted for each vehicle
              </summary>
              <p className="font-mono text-[13px] text-muted mt-3">
                From the mileage model, shown on{" "}
                <select
                  value={mileageFuel}
                  onChange={(e) => setMileageFuel(e.target.value)}
                  aria-label="Fuel to predict mileage for"
                  className="border border-ink bg-page font-mono text-[13px] px-1.5 py-0.5"
                >
                  {FUELS.map((f) => (
                    <option key={f}>{f}</option>
                  ))}
                </select>
                . Override any of them if you know your own car.
              </p>
              <div className="grid grid-cols-3 gap-3 mt-4">
                {VEHICLES.map((vehicle) => (
                  <div key={vehicle}>
                    <label
                      htmlFor={`mileage-${vehicle}`}
                      className="font-mono text-[12px] uppercase tracking-[0.14em] block mb-1.5"
                    >
                      {vehicle}
                    </label>
                    <input
                      id={`mileage-${vehicle}`}
                      type="number"
                      min="4"
                      max="60"
                      step="0.1"
                      value={mileage[vehicle] ?? ""}
                      onChange={(e) => {
                        editedRef.current[vehicle] = true
                        setMileage((m) => ({ ...m, [vehicle]: e.target.value }))
                        setMileageEdited((m) => ({ ...m, [vehicle]: true }))
                      }}
                      className="w-full h-11 px-3 border border-ink bg-page font-mono text-base"
                    />
                    <p className="font-mono text-[11px] text-muted mt-1">
                      {mileageEdited[vehicle] ? "yours" : "km/l, predicted"}
                    </p>
                  </div>
                ))}
              </div>
            </details>

            <button
              type="submit"
              disabled={status === "loading"}
              className="group w-full sm:w-auto rounded-full bg-accent text-white px-10 h-12
                         font-mono text-[13px] uppercase tracking-[0.18em] cursor-pointer
                         inline-flex items-center justify-center gap-2.5
                         transition-opacity duration-200 hover:opacity-90
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {status === "loading" ? "Estimating…" : "Price every vehicle"}
              <ArrowRight
                aria-hidden="true"
                className="w-3.5 h-3.5 transition-transform duration-200 group-hover:translate-x-1"
              />
            </button>
          </form>

          <div className="lg:sticky lg:top-10 min-w-0">
            <ResultCard status={status} result={result} error={error} />
          </div>
        </div>
      </div>
    </section>
  )
}
