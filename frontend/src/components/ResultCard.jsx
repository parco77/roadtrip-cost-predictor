import { useRef, useState } from "react"
import gsap from "gsap"
import { useGSAP } from "@gsap/react"
import { Loader2 } from "lucide-react"

import CostBandBadge from "./CostBandBadge.jsx"
import Breakdown from "./Breakdown.jsx"
import Warnings from "./Warnings.jsx"
import { shouldAnimate } from "../lib/motion.js"

const inr = (n) => Math.round(n).toLocaleString("en-IN")

export default function ResultCard({ status, result, error }) {
  const root = useRef(null)
  const [shown, setShown] = useState(0)
  const total = result?.prediction?.total_trip_cost ?? 0

  // Count-up driven through React state rather than by writing to the DOM directly,
  // so GSAP never fights the render cycle.
  useGSAP(
    () => {
      if (status !== "ready") return
      // If we cannot animate (reduced motion, or a hidden tab whose rAF never fires),
      // show the real figure immediately rather than counting up from zero forever.
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
    { dependencies: [status, total], scope: root },
  )

  if (status === "idle") {
    return (
      <div
        ref={root}
        className="border border-hairline p-8 min-h-[26rem] flex items-center justify-center text-center"
      >
        <p className="label-mono max-w-[24ch]">
          Pick two cities and press estimate — the result appears here
        </p>
      </div>
    )
  }

  if (status === "loading") {
    return (
      <div
        ref={root}
        className="border border-hairline p-8 min-h-[26rem] flex flex-col items-center justify-center gap-4"
        aria-live="polite"
      >
        <Loader2 aria-hidden="true" className="w-5 h-5 animate-spin text-muted" />
        <p className="label-mono">Routing and predicting…</p>
      </div>
    )
  }

  if (status === "error") {
    return (
      <div
        ref={root}
        className="border-2 border-band-poor p-8 min-h-[26rem] flex flex-col items-center justify-center gap-3 text-center"
        role="alert"
      >
        <p className="label-mono !text-band-poor">Could not estimate</p>
        <p className="font-body text-base max-w-[34ch]">{error}</p>
      </div>
    )
  }

  const { route, derived, prediction } = result

  return (
    <div ref={root} className="border-2 border-ink p-8" aria-live="polite">
      <p className="label-mono">Estimated total</p>

      <p className="font-display font-black text-accent leading-none mt-2 text-[clamp(3rem,7vw,5.5rem)] tabular-nums">
        ₹{inr(shown)}
      </p>

      <p className="font-mono text-sm text-muted mt-3">
        ± ₹{inr(prediction.typical_error)} typical error &nbsp;·&nbsp; ₹
        {inr(prediction.cost_per_passenger)} per passenger &nbsp;·&nbsp; ₹
        {prediction.cost_per_km}/km
      </p>

      <div className="mt-6">
        <CostBandBadge band={prediction.cost_band} />
      </div>

      <Breakdown breakdown={prediction.breakdown} />

      <div className="mt-8 pt-4 border-t border-hairline space-y-1.5">
        <p className="font-mono text-[13px] text-muted">
          {route.from.city} → {route.to.city} &nbsp;·&nbsp; {Math.round(route.distance_km)} km
          &nbsp;·&nbsp;{" "}
          {route.distance_source === "osrm" ? "OSRM routing" : "estimated distance"}
        </p>
        <p className="font-mono text-[13px] text-muted">
          {derived.traffic_level.toLowerCase()} traffic
          {derived.traffic_inferred ? " (inferred)" : " (you chose)"} &nbsp;·&nbsp;{" "}
          {Math.round(derived.estimated_time_minutes / 60)}h{" "}
          {Math.round(derived.estimated_time_minutes % 60)}m
        </p>
        {/* Showing the arithmetic instead of hiding it is the difference between a
            demo and a black box. The backend returns this string ready to render. */}
        <p className="font-mono text-[13px] text-muted">
          {derived.fuel_consumption_explained}
        </p>
        <p className="font-mono text-[13px] text-muted">
          {result.inputs.fuel_type.toLowerCase()} ₹{result.inputs.fuel_price}/
          {result.inputs.fuel_type === "CNG" ? "kg" : "l"}
          {" · "}
          {{
            user: "your figure",
            verified: "verified price",
            estimated: "estimated price",
            fallback: "national fallback",
          }[result.inputs.fuel_price_source] ?? result.inputs.fuel_price_source}
        </p>
      </div>

      <Warnings warnings={result.warnings} />
    </div>
  )
}
