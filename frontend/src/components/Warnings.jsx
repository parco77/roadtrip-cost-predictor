import { AlertTriangle } from "lucide-react"

/**
 * The backend returns warnings for extrapolation beyond the training range, haversine
 * fallback when routing is down, and estimated (non-verified) fuel prices.
 *
 * Never suppress these. They are the honest part of the product - a confident number
 * with no caveat is worse than a number with its limits stated.
 */
export default function Warnings({ warnings }) {
  if (!warnings?.length) return null

  return (
    <ul className="mt-6 space-y-2 border-t border-hairline pt-4">
      {warnings.map((warning) => (
        <li key={warning} className="flex gap-2.5 items-start">
          <AlertTriangle
            aria-hidden="true"
            className="w-3.5 h-3.5 mt-1 shrink-0 text-band-average"
          />
          <span className="font-mono text-[13px] leading-relaxed text-muted">{warning}</span>
        </li>
      ))}
    </ul>
  )
}
