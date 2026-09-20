const ROWS = [
  { key: "fuel", label: "Fuel" },
  { key: "toll", label: "Toll" },
  { key: "wear_and_tear", label: "Wear & tear" },
  { key: "parking", label: "Parking" },
]

const rupees = (n) => `₹${Math.round(n).toLocaleString("en-IN")}`

/** Horizontal bars, widths relative to the largest component. Pure black - the accent
 *  is reserved for the headline figure only. */
export default function Breakdown({ breakdown }) {
  const max = Math.max(...ROWS.map((r) => breakdown[r.key] ?? 0), 1)

  return (
    <dl className="mt-8 space-y-3">
      {ROWS.map(({ key, label }) => {
        const value = breakdown[key] ?? 0
        return (
          <div key={key} className="grid grid-cols-[7.5rem_1fr_5rem] items-center gap-3">
            <dt className="label-mono">{label}</dt>
            <div className="h-2 bg-hairline" aria-hidden="true">
              <div
                className="h-full bg-ink origin-left transition-[width] duration-500 ease-out"
                style={{ width: `${(value / max) * 100}%` }}
              />
            </div>
            <dd className="font-mono text-base text-right tabular-nums">{rupees(value)}</dd>
          </div>
        )
      })}
    </dl>
  )
}
