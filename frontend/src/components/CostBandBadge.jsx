/**
 * Cost-efficiency badge. This is the one licensed exception to the monochrome rule,
 * because the label is meaningless without colour - but the WORD is always rendered
 * too, so colour is never the sole carrier of meaning (WCAG 1.4.1).
 */
const BANDS = {
  Excellent: { bg: "bg-band-excellent", note: "cheap for its distance" },
  Good: { bg: "bg-band-good", note: "slightly better than typical" },
  Average: { bg: "bg-band-average", note: "typical for its distance" },
  Poor: { bg: "bg-band-poor", note: "expensive for its distance" },
}

export default function CostBandBadge({ band }) {
  const style = BANDS[band] ?? BANDS.Average
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <span
        className={`${style.bg} text-white font-mono text-[13px] uppercase tracking-[0.18em] px-3 py-1.5`}
      >
        {band}
      </span>
      <span className="font-mono text-[13px] uppercase tracking-[0.14em] text-muted">
        {style.note}
      </span>
    </div>
  )
}
