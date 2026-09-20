import { lazy, Suspense, useEffect, useRef, useState } from "react"
import gsap from "gsap"
import { useGSAP } from "@gsap/react"
import { ArrowRight } from "lucide-react"

import CityAutocomplete from "./CityAutocomplete.jsx"
import ResultCard from "./ResultCard.jsx"
import RevealText from "./RevealText.jsx"
import { fetchDepartureSweep, fetchDefaultMileage } from "../api.js"
import { usePrediction } from "../hooks/usePrediction.js"

// Shares the Recharts chunk with LossCurve, so this adds no meaningful bundle weight.
const DepartureChart = lazy(() => import("./DepartureChart.jsx"))
import { revealWords, shouldAnimate } from "../lib/motion.js"
import { usePageVisible } from "../hooks/usePageVisible.js"

const VEHICLES = ["Hatchback", "Sedan", "SUV"]
const FUELS = ["Petrol", "Diesel", "CNG"]
const TRAFFIC = ["Auto", "Low", "Medium", "High"]
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

/** Segmented control. 44px min height for touch, and the selected item inverts
 *  rather than tinting - consistent with the monochrome style. */
function Segmented({ label, options, value, onChange, name }) {
  return (
    <fieldset>
      <legend className="label-mono mb-2">{label}</legend>
      <div className="flex border border-ink divide-x divide-ink">
        {options.map((option) => {
          const selected = value === option
          // The radio itself is sr-only, so focus lands on a 1px invisible input and the
          // visible control shows nothing — a keyboard user tabbing through Vehicle / Fuel /
          // Traffic had no idea where they were. has-[:focus-visible] moves the ring onto the
          // label that is actually on screen.
          return (
            <label
              key={option}
              className={`segmented-option flex-1 h-11 flex items-center justify-center
                          cursor-pointer font-mono text-[13px] uppercase tracking-[0.14em]
                          transition-colors duration-150
                          ${selected ? "bg-ink text-page" : "bg-page hover:bg-paper"}`}
            >
              <input
                type="radio"
                name={name}
                value={option}
                checked={selected}
                onChange={() => onChange(option)}
                className="sr-only"
              />
              {option}
            </label>
          )
        })}
      </div>
    </fieldset>
  )
}

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

export default function Predictor() {
  const [from, setFrom] = useState(null)
  const [to, setTo] = useState(null)
  const [vehicle, setVehicle] = useState("Sedan")
  const [fuel, setFuel] = useState("Petrol")
  const [traffic, setTraffic] = useState("Auto")
  // Seeded from the mileage sub-model for the chosen vehicle + fuel, not a fixed 15.5 for
  // every car. `mileageEdited` stops the model overwriting a figure the user typed - the same
  // pattern the fuel-price field uses below.
  const [mileage, setMileage] = useState(15.5)
  const [mileageEdited, setMileageEdited] = useState(false)
  const [hour, setHour] = useState(9)
  const [month, setMonth] = useState(6)
  const [parking, setParking] = useState(70)
  const [passengers, setPassengers] = useState(4)
  const [fuelPrice, setFuelPrice] = useState("")
  // Once the user types their own price we stop overwriting it. Without this, changing the
  // fuel type or the origin city would silently discard what they entered.
  const [priceEdited, setPriceEdited] = useState(false)
  const [touched, setTouched] = useState(false)
  const [sweep, setSweep] = useState(null)
  const root = useRef(null)
  const sweepController = useRef(null)
  const pageVisible = usePageVisible()

  useEffect(() => () => sweepController.current?.abort(), [])

  // Ask the mileage model what this vehicle + fuel typically manages. Runs on mount and on
  // every vehicle/fuel change until the user moves the slider themselves.
  useEffect(() => {
    if (mileageEdited) return
    const controller = new AbortController()
    fetchDefaultMileage(vehicle, fuel, { signal: controller.signal })
      .then((r) => setMileage(r.mileage))
      .catch(() => {
        /* offline or aborted - keep whatever the slider already shows */
      })
    return () => controller.abort()
  }, [vehicle, fuel, mileageEdited])

  // Prefill the price from the origin state's table whenever the route or fuel type changes —
  // the city payload already carries all three prices, so this costs no request. Skipped once
  // the field has been edited by hand.
  const suggestedPrice = from?.fuel_prices?.[fuel.toLowerCase()] ?? null
  const priceSource = from?.fuel_prices?.source ?? null
  useEffect(() => {
    if (!priceEdited && suggestedPrice != null) setFuelPrice(String(suggestedPrice))
  }, [suggestedPrice, priceEdited])

  const { status, result, error, predict } = usePrediction()

  useGSAP(
    () => {
      if (!shouldAnimate()) return
      // Only the heading animates. The form deliberately does NOT.
      //
      // It used to: gsap.from("form > *", { stagger: 0.05 }). The submit button is the last
      // child, so it animated last — and when the stagger was interrupted it was left at
      // opacity 0, i.e. an invisible, unclickable "Estimate my trip" button. Verified: every
      // other form child at opacity 1, the button at 0.
      //
      // The rule this cost us: never gate an interactive control on an animation completing.
      // A form is a tool, not a reveal. Trading usability for a fade is always the wrong side
      // of that deal.
      revealWords(gsap, root.current)
    },
    { dependencies: [pageVisible], scope: root },
  )

  const sameCity =
    from && to && from.city === to.city && from.state === to.state
  const routeError = touched && !from ? "Choose a starting city"
    : touched && !to ? "Choose a destination"
    : sameCity ? "Start and destination must be different"
    : null

  function onSubmit(event) {
    event.preventDefault()
    setTouched(true)
    if (!from || !to || sameCity) return

    const payload = {
      start_city: from.city,
      start_state: from.state,
      destination_city: to.city,
      destination_state: to.state,
      vehicle_type: vehicle,
      fuel_type: fuel,
      mileage: Number(mileage),
      departure_hour: Number(hour),
      month: Number(month),
      passengers: Number(passengers),
      parking_cost: Number(parking),
      // "Auto" means omit the field entirely so the classifier infers traffic.
      ...(traffic === "Auto" ? {} : { traffic_level: traffic }),
      // Omit when untouched so the backend uses its own table and reports the real source;
      // sending the prefilled value back would relabel a table estimate as a user figure and
      // silently suppress the warning that should accompany it.
      ...(priceEdited && fuelPrice !== "" ? { fuel_price: Number(fuelPrice) } : {}),
    }

    predict(payload)

    // The 24-hour sweep runs alongside the main prediction rather than after it — the distance
    // lookup is already cached by then, so it costs one request and no extra routing calls.
    sweepController.current?.abort()
    const controller = new AbortController()
    sweepController.current = controller
    setSweep(null)
    fetchDepartureSweep(payload, { signal: controller.signal })
      .then(setSweep)
      .catch((err) => {
        if (err.name !== "AbortError") setSweep(null)
      })
  }

  const hour12 = `${((hour + 11) % 12) + 1}${hour < 12 ? "am" : "pm"}`

  return (
    <section
      id="estimate"
      ref={root}
      data-section="04"
      data-section-name="The Predictor"
      className="rule-heavy py-24 md:py-32 px-6"
    >
      <div className="mx-auto max-w-6xl">
        <header className="max-w-[62ch]">
          <p className="label-mono">Section 04 — The Predictor</p>
          <RevealText
            as="h2"
            text="What Will This Trip Cost?"
            className="mt-4 block text-[clamp(2rem,5vw,3.5rem)]"
          />
          <p className="mt-5 text-muted">
            Distance, fuel price, toll and litres burnt are all worked out for you. You only
            answer what you actually know.
          </p>
        </header>

        <div className="mt-14 grid gap-12 lg:grid-cols-2 lg:gap-16 items-start">
          <form onSubmit={onSubmit} noValidate className="space-y-7">
            <div className="grid sm:grid-cols-2 gap-6">
              <CityAutocomplete
                label="From"
                value={from}
                onSelect={setFrom}
                placeholder="Search 537 cities…"
              />
              <CityAutocomplete
                label="To"
                value={to}
                onSelect={setTo}
                placeholder="Search 537 cities…"
              />
            </div>

            {routeError && (
              <p role="alert" className="font-mono text-[13px] text-band-poor -mt-3">
                {routeError}
              </p>
            )}

            <Segmented
              label="Vehicle"
              name="vehicle"
              options={VEHICLES}
              value={vehicle}
              onChange={setVehicle}
            />
            <Segmented
              label="Fuel"
              name="fuel"
              options={FUELS}
              value={fuel}
              onChange={setFuel}
            />
            <Segmented
              label="Traffic"
              name="traffic"
              options={TRAFFIC}
              value={traffic}
              onChange={setTraffic}
            />
            {traffic === "Auto" && (
              <p className="font-mono text-[13px] text-muted -mt-4">
                Auto — the classifier infers traffic from your departure hour and month
              </p>
            )}

            <Field
              label={`Mileage — ${mileage} km/l`}
              htmlFor="mileage"
              hint={mileageEdited ? "your figure" : `predicted for a ${fuel} ${vehicle}`}
            >
              <input
                id="mileage"
                type="range"
                min="8"
                max="23"
                step="0.1"
                value={mileage}
                onChange={(e) => {
                  setMileage(e.target.value)
                  setMileageEdited(true)
                }}
                className="w-full h-11 accent-[var(--color-accent)] cursor-pointer"
              />
            </Field>

            <div className="grid sm:grid-cols-2 gap-6">
              <Field label={`Departure — ${hour12}`} htmlFor="hour" hint="0 to 23">
                <input
                  id="hour"
                  type="range"
                  min="0"
                  max="23"
                  step="1"
                  value={hour}
                  onChange={(e) => setHour(e.target.value)}
                  className="w-full h-11 accent-[var(--color-accent)] cursor-pointer"
                />
              </Field>

              <Field label="Month" htmlFor="month">
                <select
                  id="month"
                  value={month}
                  onChange={(e) => setMonth(e.target.value)}
                  className="w-full h-11 px-3 border border-ink bg-page font-mono text-base cursor-pointer"
                >
                  {MONTHS.map((name, i) => (
                    <option key={name} value={i + 1}>
                      {name}
                    </option>
                  ))}
                </select>
              </Field>
            </div>

            <div className="grid sm:grid-cols-3 gap-6">
              <Field
                label={`Fuel price ₹/${fuel === "CNG" ? "kg" : "l"}`}
                htmlFor="fuelPrice"
                hint={
                  !from ? "pick a city first"
                  : priceEdited ? "your figure"
                  : priceSource === "verified" ? `${from.state} — verified`
                  : `${from.state} — estimated`
                }
              >
                <input
                  id="fuelPrice"
                  type="number"
                  min="30"
                  max="300"
                  step="0.01"
                  value={fuelPrice}
                  placeholder={suggestedPrice != null ? String(suggestedPrice) : "—"}
                  onChange={(e) => {
                    setFuelPrice(e.target.value)
                    setPriceEdited(true)
                  }}
                  className="w-full h-11 px-3 border border-ink bg-page font-mono text-base"
                />
              </Field>

              <Field label="Parking ₹" htmlFor="parking" hint="0 to 150">
                <input
                  id="parking"
                  type="number"
                  min="0"
                  max="150"
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

            <button
              type="submit"
              disabled={status === "loading"}
              className="group w-full sm:w-auto rounded-full bg-accent text-white px-10 h-12
                         font-mono text-[13px] uppercase tracking-[0.18em] cursor-pointer
                         inline-flex items-center justify-center gap-2.5
                         transition-opacity duration-200 hover:opacity-90
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {status === "loading" ? "Estimating…" : "Estimate my trip"}
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

        {/* Full width below the two columns — 24 bars need the room. Only appears once there
            is a result to compare against. */}
        {status === "ready" && sweep && (
          <Suspense fallback={null}>
            <DepartureChart data={sweep} />
          </Suspense>
        )}
      </div>
    </section>
  )
}
