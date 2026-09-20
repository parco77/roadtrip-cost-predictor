import { useEffect, useRef, useState } from "react"
import gsap from "gsap"
import { useGSAP } from "@gsap/react"
import { fetchMetrics } from "../api.js"
import RevealText from "./RevealText.jsx"
import { revealWords, shouldAnimate } from "../lib/motion.js"
import { usePageVisible } from "../hooks/usePageVisible.js"

/**
 * Every classifier accuracy is printed next to its baseline. An accuracy with no
 * baseline is not a result - 53.9% sounds weak and 85.8% sounds strong, but the first
 * is a real signal over 38.8% and the second is high largely by construction.
 */
export default function ReportCard() {
  const [metrics, setMetrics] = useState(null)
  const root = useRef(null)
  const pageVisible = usePageVisible()

  useGSAP(
    () => {
      if (!shouldAnimate() || !metrics) return
      revealWords(gsap, root.current)
      gsap.from(".stat-cell", {
        opacity: 0,
        yPercent: 40,
        duration: 0.6,
        stagger: 0.08,
        ease: "power3.out",
        scrollTrigger: { trigger: ".stat-grid", start: "top 85%", once: true },
      })
    },
    { dependencies: [metrics, pageVisible], scope: root },
  )

  useEffect(() => {
    const controller = new AbortController()
    fetchMetrics({ signal: controller.signal })
      .then(setMetrics)
      .catch(() => {})
    return () => controller.abort()
  }, [])

  const stats = metrics
    ? [
        { label: "R²", value: metrics.regression.r2.toFixed(4), sub: "cost regression" },
        { label: "MAE", value: `₹${metrics.regression.mae.toFixed(2)}`, sub: "typical error" },
        {
          label: "Cost band",
          value: `${(metrics.band_accuracy * 100).toFixed(1)}%`,
          sub: "baseline 25.0%",
        },
        {
          label: "Traffic",
          value: `${(metrics.traffic_accuracy * 100).toFixed(1)}%`,
          sub: `baseline ${(metrics.traffic_baseline * 100).toFixed(1)}%`,
        },
      ]
    : []

  return (
    <section
      ref={root}
      data-section="06"
      data-section-name="Report Card"
      className="paper-noise rule-heavy py-24 md:py-32 px-6"
    >
      <div className="relative z-10 mx-auto max-w-5xl text-center">
        <p className="label-mono">Section 06 — The Report Card</p>
        <RevealText
          as="h2"
          text="How Well Does It Actually Do?"
          className="mt-4 block text-[clamp(2rem,5vw,3.5rem)]"
        />

        {metrics ? (
          <>
            <dl className="stat-grid mt-14 grid grid-cols-2 md:grid-cols-4 gap-y-10 gap-x-6">
              {stats.map(({ label, value, sub }) => (
                <div key={label} className="stat-cell">
                  <dt className="label-mono">{label}</dt>
                  <dd className="font-mono text-[clamp(1.5rem,4vw,2.5rem)] mt-2 tabular-nums">
                    {value}
                  </dd>
                  <p className="font-mono text-[13px] uppercase tracking-[0.14em] text-muted mt-1">
                    {sub}
                  </p>
                </div>
              ))}
            </dl>

            <p className="mt-14 mx-auto max-w-[62ch] text-base text-muted">
              Trained on {metrics.n_rows.toLocaleString("en-IN")} trips across{" "}
              {metrics.n_cities} cities, {Math.round(metrics.distance_range[0])}–
              {Math.round(metrics.distance_range[1])} km.
            </p>

            <div className="mt-8 mx-auto max-w-[62ch] space-y-3 text-left">
              {Object.entries(metrics.notes).map(([key, note]) => (
                <p key={key} className="text-sm text-muted border-l-2 border-hairline pl-4">
                  <span className="label-mono !text-ink mr-2">{key}</span>
                  {note}
                </p>
              ))}
            </div>
          </>
        ) : (
          <p className="label-mono mt-14">Loading metrics…</p>
        )}
      </div>
    </section>
  )
}
