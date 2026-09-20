const ROWS = [
  { key: "fuel", label: "Fuel" },
  { key: "toll", label: "Toll" },
  { key: "running_cost", label: "Running cost" },
  { key: "parking", label: "Parking" },
]

const rupees = (n) => `₹${Math.round(n).toLocaleString("en-IN")}`

/**
 * Horizontal bars. Pure black — the accent is reserved for the headline figure only.
 *
 * `scale` is the value a full-width bar represents, and it matters: three of these sit
 * side by side comparing three vehicles. Left to normalise against its own largest
 * component, every card would draw its fuel bar full width and the SUV would look
 * identical to the hatchback. The caller passes one shared maximum so the bars are
 * comparable across cards, which is the entire point of showing three.
 */
export default function Breakdown({ breakdown, scale }) {
  const max = scale ?? Math.max(...ROWS.map((r) => breakdown[r.key] ?? 0), 1)

  return (
    <dl className="mt-6 space-y-2.5">
      {ROWS.map(({ key, label }) => {
        const value = breakdown[key] ?? 0
        return (
          <div key={key} className="grid grid-cols-[6.5rem_1fr_4.5rem] items-center gap-2.5">
            <dt className="label-mono">{label}</dt>
            <div className="h-2 bg-hairline" aria-hidden="true">
              <div
                className="h-full bg-ink origin-left transition-[width] duration-500 ease-out"
                style={{ width: `${Math.min(100, (value / max) * 100)}%` }}
              />
            </div>
            <dd className="font-mono text-sm text-right tabular-nums">{rupees(value)}</dd>
          </div>
        )
      })}
    </dl>
  )
}
