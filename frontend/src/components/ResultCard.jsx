import { useEffect, useRef, useState } from "react"
import gsap from "gsap"
import { useGSAP } from "@gsap/react"
import { Loader2 } from "lucide-react"

import CostBandBadge from "./CostBandBadge.jsx"
import Breakdown from "./Breakdown.jsx"
import Warnings from "./Warnings.jsx"
import { shouldAnimate } from "../lib/motion.js"

const FUELS = ["Petrol", "Diesel", "CNG"]
const UNIT = { Petrol: "l", Diesel: "l", CNG: "kg" }
const inr = (n) => Math.round(n).toLocaleString("en-IN")

function Shell({ children, tone = "hairline", live, role }) {
  const border = tone === "poor" ? "border-2 border-band-poor" : "border border-hairline"
  return (
    <div
      className={`${border} p-8 min-h-[26rem] flex flex-col items-center justify-center gap-3 text-center`}
      aria-live={live}
      role={role}
    >
      {children}
    </div>
  )
}

/** One vehicle. The headline counts up; the rest is static. */
function VehicleCard({ row, scale, leader }) {
  const [shown, setShown] = useState(0)
  const total = row.total_trip_cost

  useGSAP(
    () => {
      if (!shouldAnimate()) {
        setShown(total)
        return
      }
      const counter = { value: 0 }
      gsap.to(counter, {
        value: total,
        duration: 0.7,
        ease: "power2.out",
        onUpdate: () => setShown(counter.value),
        onComplete: () => setShown(total), // guarantee the exact value, not 3013.6699
      })
    },
    { dependencies: [total] },
  )

  return (
    <div className={`p-6 ${leader ? "border-2 border-ink" : "border border-hairline"}`}>
      <div className="flex items-baseline justify-between gap-3 flex-wrap">
        <p className="font-mono text-[13px] uppercase tracking-[0.18em]">{row.vehicle_type}</p>
        {leader && (
          <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-accent">
            cheapest
          </p>
        )}
      </div>

      <p className="font-display font-black text-accent leading-none mt-2
                    text-[clamp(2rem,5vw,3rem)] tabular-nums">
        ₹{inr(shown)}
      </p>

      <p className="font-mono text-[13px] text-muted mt-2">
        ₹{row.cost_per_km}/km · ₹{inr(row.cost_per_passenger)} per passenger
      </p>
      <p className="font-mono text-[13px] text-muted mt-1">
        {row.mileage} km/l{" "}
        <span className="text-muted/70">
          ({row.mileage_source === "user" ? "yours" : "predicted"})
        </span>
      </p>

      <div className="mt-4">
        <CostBandBadge band={row.cost_band} />
      </div>

      <Breakdown breakdown={row.breakdown} scale={scale} />

      {/* Showing the arithmetic instead of hiding it is the difference between a
          demo and a black box. The backend returns this string ready to render. */}
      <p className="font-mono text-[12px] text-muted mt-4 pt-3 border-t border-hairline">
        {row.fuel_consumption_explained}
      </p>
    </div>
  )
}

export default function ResultCard({ status, result, error }) {
  const [fuel, setFuel] = useState("Petrol")
  const root = useRef(null)

  // When a new estimate lands, jump to whichever fuel actually came out cheapest rather
  // than leaving the user on whatever they were looking at last time.
  useEffect(() => {
    if (status === "ready" && result?.cheapest?.fuel_type) setFuel(result.cheapest.fuel_type)
  }, [status, result])

  if (status === "idle") {
    return (
      <div ref={root}>
        <Shell>
          <p className="label-mono max-w-[26ch]">
            Enter a distance and press the button — three prices appear here
          </p>
        </Shell>
      </div>
    )
  }

  if (status === "loading") {
    return (
      <div ref={root}>
        <Shell live="polite">
          <Loader2 aria-hidden="true" className="w-5 h-5 animate-spin text-muted" />
          <p className="label-mono">Pricing every vehicle…</p>
        </Shell>
      </div>
    )
  }

  if (status === "error") {
    return (
      <div ref={root}>
        <Shell tone="poor" role="alert">
          <p className="label-mono !text-band-poor">Could not estimate</p>
          <p className="font-body text-base max-w-[34ch]">{error}</p>
        </Shell>
      </div>
    )
  }

  const rows = result.results[fuel] ?? []
  // One scale across all three cards so the bars compare. Computed from the fuel on
  // screen, so switching fuel rescales together rather than per card.
  const scale = Math.max(
    ...rows.flatMap((r) => Object.values(r.breakdown).map((v) => Number(v) || 0)),
    1,
  )
  const cheapestHere = rows.length
    ? rows.reduce((a, b) => (a.total_trip_cost <= b.total_trip_cost ? a : b))
    : null
  const priceSource = result.inputs.fuel_price_sources?.[fuel]

  return (
    <div ref={root} aria-live="polite">
      {/* Fuel toggle. All nine combinations arrived in one response, so this is a
          pure client-side switch — no refetch, no chance of a stale answer landing. */}
      <div role="tablist" aria-label="Fuel type"
           className="flex border border-ink divide-x divide-ink max-w-lg">
        {FUELS.map((f) => {
          const selected = f === fuel
          return (
            <button
              key={f}
              type="button"
              role="tab"
              aria-selected={selected}
              onClick={() => setFuel(f)}
              className={`flex-1 h-11 font-mono text-[13px] uppercase tracking-[0.14em]
                          transition-colors duration-150
                          ${selected ? "bg-ink text-page" : "bg-page hover:bg-paper"}`}
            >
              {f}
            </button>
          )
        })}
      </div>

      <p className="font-mono text-[13px] text-muted mt-3">
        ₹{result.inputs.fuel_prices[fuel]}/{UNIT[fuel]}
        {" · "}
        {priceSource === "user" ? "your figure" : "national average"}
        {" · "}± ₹{inr(result.prediction.typical_error)} typical error
      </p>

      {/* A ROW, not a stack. Three prices exist to be compared, and comparison needs them on
          screen together - stacked in a column they ran 1,759px tall and you could never see
          the second and third at once. */}
      <div className="mt-6 grid gap-5 md:grid-cols-3 items-start">
        {rows.map((row) => (
          <VehicleCard
            key={row.vehicle_type}
            row={row}
            scale={scale}
            leader={cheapestHere?.vehicle_type === row.vehicle_type}
          />
        ))}
      </div>

      <div className="mt-6 pt-4 border-t border-hairline space-y-1.5">
        <p className="font-mono text-[13px] text-muted">
          {inr(result.inputs.distance_km)} km · ₹{inr(result.inputs.parking_cost)} parking ·{" "}
          {result.inputs.passengers} passengers
        </p>
        <p className="font-mono text-[13px] text-muted">
          Cheapest overall: {result.cheapest.vehicle_type} on {result.cheapest.fuel_type} at ₹
          {inr(result.cheapest.total_trip_cost)} — ₹{inr(result.max_saving)} below the dearest
          ({result.dearest.vehicle_type} on {result.dearest.fuel_type}).
        </p>
      </div>

      <Warnings warnings={result.warnings} />
    </div>
  )
}
