import { lazy, Suspense, useEffect, useRef, useState } from "react"
import gsap from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"
import { useGSAP } from "@gsap/react"

import { fetchLossCurve } from "../api.js"
import RevealText from "./RevealText.jsx"
import { shouldAnimate } from "../lib/motion.js"
import { usePageVisible } from "../hooks/usePageVisible.js"

const LossCurve = lazy(() => import("./LossCurve.jsx"))

/** How many discrete progress steps the scrubbed curve is quantised to. See onUpdate below. */
const PROGRESS_STEPS = 60

/** Three beats, advanced by scroll position while the section is pinned. */
const CHAPTERS = [
  {
    title: "sklearn jumps straight to the bottom",
    body: "The normal equations solve for the weights algebraically, in one shot. No iteration, no learning rate, nothing to tune. It simply arrives.",
  },
  {
    title: "Gradient descent walks there",
    body: "Our own loss, our own gradients, our own update rule — 60,000 steps down the slope. Watch the curve: almost all of the progress happens in the first few hundred epochs.",
  },
  {
    title: "They agree on the answer, not the route",
    body: "Predictions match to within ₹82.81 and R² is identical to five decimals. But the weights never converged, and that turns out to be the interesting part.",
  },
]

const ABLATION = [
  { model: "Linear, no interaction", mae: "₹143.85", winner: false },
  { model: "RandomForest", mae: "₹69.97", winner: false },
  { model: "GradientBoosting", mae: "₹73.50", winner: false },
  { model: "Linear + litres × price", mae: "₹39.66", winner: true },
]

export default function ModelStory() {
  const root = useRef(null)
  const pageVisible = usePageVisible()
  const [data, setData] = useState(null)
  const [failed, setFailed] = useState(false)
  const [progress, setProgress] = useState(1)
  const [chapter, setChapter] = useState(0)

  // Derived rather than relying on the matchMedia cleanup to reset state. Cleanup ordering is
  // easy to get wrong — a missed reset left the curve permanently half-drawn on mobile after a
  // resize. Deriving it from the media query cannot go out of sync.
  const [isDesktop, setIsDesktop] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(min-width: 1024px)").matches,
  )
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)")
    const onChange = (e) => setIsDesktop(e.matches)
    mq.addEventListener("change", onChange)
    return () => mq.removeEventListener("change", onChange)
  }, [])

  const effectiveProgress = isDesktop ? progress : 1
  const effectiveChapter = isDesktop ? chapter : CHAPTERS.length - 1

  useEffect(() => {
    const controller = new AbortController()
    fetchLossCurve({ signal: controller.signal })
      .then(setData)
      .catch((err) => err.name !== "AbortError" && setFailed(true))
    return () => controller.abort()
  }, [])

  useGSAP(
    () => {
      if (!data || !shouldAnimate()) return

      // The "pin" is CSS position:sticky on the left column (see lg:sticky in the JSX), NOT
      // ScrollTrigger's pin option. Three reasons:
      //   1. ScrollTrigger's pin injects a .pin-spacer element. Crossing the lg breakpoint
      //      without a reload left that spacer behind even with kill(true) + refresh(), which
      //      pushed scrollWidth to 1425px at a 375px viewport — real horizontal scroll. Verified.
      //   2. This section's content is taller than the viewport, which true pinning handles badly.
      //      Sticky-left / scrolling-right sidesteps that entirely.
      //   3. Sticky is responsive by media query alone, so there is no teardown to get wrong.
      // ScrollTrigger is now only a progress *reporter* — it moves no DOM and injects nothing.
      const mm = gsap.matchMedia()

      mm.add("(min-width: 1024px)", () => {
        setProgress(0)
        const st = ScrollTrigger.create({
          trigger: root.current.querySelector(".model-narrative"),
          start: "top 75%",
          end: "bottom 90%",
          scrub: 0.5,
          invalidateOnRefresh: true,
          onUpdate: (self) => {
            // onUpdate fires on every ticker frame while scrubbing. Feeding raw progress into
            // React state would re-render this section - Recharts included - at 60fps, which
            // janks badly on mid-range hardware. Quantising to 60 discrete steps means the
            // chart redraws ~60 times across the whole pin instead of 60 times per second,
            // and the returned-identical value makes React bail out of the render entirely.
            const step = Math.round(self.progress * PROGRESS_STEPS) / PROGRESS_STEPS
            setProgress((prev) => (prev === step ? prev : step))
            const next = Math.min(
              CHAPTERS.length - 1,
              Math.floor(self.progress * CHAPTERS.length),
            )
            setChapter((prev) => (prev === next ? prev : next))
          },
        })
        // Dropping below lg: kill the reporter and restore the finished state, so the curve is
        // never left half-drawn. Nothing to un-inject, because nothing was injected.
        return () => {
          st.kill()
          setProgress(1)
          setChapter(CHAPTERS.length - 1)
        }
      })

      return () => mm.revert()
    },
    { dependencies: [data, pageVisible], scope: root },
  )

  return (
    <section
      id="model"
      ref={root}
      data-section="05"
      data-section-name="Two Roads"
      className="rule-heavy"
    >
      <div className="py-24 md:py-32 px-6">
        <div className="mx-auto max-w-6xl grid gap-14 lg:grid-cols-2 lg:gap-20 items-start">
          {/* Sticky on desktop so the chart stays in view while the narrative scrolls past it.
              Below lg it is a normal block and the curve renders complete.

              min-w-0 is load-bearing: grid children default to min-width:auto, so this column
              cannot shrink below its content's intrinsic width. Recharts' ResponsiveContainer
              then keeps its widest measurement forever and the page gains horizontal scroll
              after a desktop -> mobile resize. Verified: .recharts-wrapper stuck at 536px in a
              375px viewport. */}
          <div className="lg:sticky lg:top-24 lg:self-start min-w-0">
            {data && (
              <Suspense
                fallback={
                  <div className="h-[300px] border border-hairline flex items-center justify-center">
                    <p className="label-mono">Drawing chart…</p>
                  </div>
                }
              >
                <LossCurve curve={data.curve} progress={effectiveProgress} />
              </Suspense>
            )}
            {failed && (
              <p className="label-mono border border-hairline p-8 text-center">
                Loss curve unavailable — is the backend running?
              </p>
            )}
            {!data && !failed && (
              <div className="h-[300px] border border-hairline flex items-center justify-center">
                <p className="label-mono">Loading loss curve…</p>
              </div>
            )}

            {data && (
              <dl className="mt-8 grid grid-cols-2 gap-x-8 gap-y-4">
                <div>
                  <dt className="label-mono">R² sklearn</dt>
                  <dd className="font-mono text-xl tabular-nums">{data.r2_sklearn.toFixed(6)}</dd>
                </div>
                <div>
                  <dt className="label-mono">R² grad. descent</dt>
                  <dd className="font-mono text-xl tabular-nums">
                    {data.r2_gradient_descent.toFixed(6)}
                  </dd>
                </div>
                <div>
                  <dt className="label-mono">Largest prediction gap</dt>
                  <dd className="font-mono text-xl tabular-nums">
                    ₹{data.max_prediction_gap.toFixed(2)}
                  </dd>
                </div>
                <div>
                  <dt className="label-mono">Epochs used</dt>
                  <dd className="font-mono text-xl tabular-nums">
                    {Math.round(data.epochs_used * (effectiveProgress || 1)).toLocaleString("en-IN")}
                  </dd>
                </div>
              </dl>
            )}
          </div>

          <div className="model-narrative max-w-[62ch] min-w-0">
            <p className="label-mono">Section 05 — Two Roads, One Answer</p>
            <RevealText
              as="h2"
              text="Two Roads, One Answer"
              className="mt-4 block text-[clamp(2rem,5vw,3.5rem)]"
            />

            {/* Chapters cross-fade as the pinned section advances. On mobile (no pin) the
                chapter index never moves, so this reads as a single static paragraph. */}
            <div className="mt-7 min-h-[9rem] relative">
              {CHAPTERS.map((c, i) => (
                <div
                  key={c.title}
                  className="transition-opacity duration-500"
                  style={{
                    opacity: i === effectiveChapter ? 1 : 0,
                    position: i === effectiveChapter ? "relative" : "absolute",
                    inset: i === effectiveChapter ? "auto" : 0,
                    pointerEvents: i === effectiveChapter ? "auto" : "none",
                  }}
                  aria-hidden={i !== effectiveChapter}
                >
                  <h3 className="font-display text-3xl">{c.title}</h3>
                  <p className="mt-2 text-muted text-base">{c.body}</p>
                </div>
              ))}
            </div>

            <div className="mt-4 flex gap-1.5" aria-hidden="true">
              {CHAPTERS.map((c, i) => (
                <span
                  key={c.title}
                  className={`h-0.5 flex-1 transition-colors duration-300 ${
                    i <= effectiveChapter ? "bg-ink" : "bg-hairline"
                  }`}
                />
              ))}
            </div>

            {data && (
              <p className="mt-8 border-l-2 border-ink pl-5 text-base text-muted">{data.verdict}</p>
            )}

            <h3 className="mt-12 font-display text-3xl">One feature beat both ensembles</h3>
            <p className="mt-3 text-muted text-base">
              The real fuel bill is litres <em>times</em> price, and an additive model cannot
              express a product. Adding that single term halves the error — and keeps the model
              linear, so the from-scratch gradient descent still applies.
            </p>

            <table className="mt-6 w-full border-t-2 border-ink">
              <caption className="sr-only">Test-set mean absolute error by model</caption>
              <thead>
                <tr className="border-b border-hairline">
                  <th scope="col" className="label-mono text-left py-2">Model</th>
                  <th scope="col" className="label-mono text-right py-2">Test MAE</th>
                </tr>
              </thead>
              <tbody>
                {ABLATION.map(({ model, mae, winner }) => (
                  <tr key={model} className="border-b border-hairline">
                    <td className={`py-2.5 text-base ${winner ? "font-semibold" : ""}`}>{model}</td>
                    <td
                      className={`py-2.5 font-mono text-base text-right tabular-nums ${
                        winner ? "text-accent font-medium" : ""
                      }`}
                    >
                      {mae}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  )
}
