import { useRef } from "react"
import gsap from "gsap"
import { useGSAP } from "@gsap/react"

import TornEdge from "./TornEdge.jsx"
import RevealText from "./RevealText.jsx"
import RevealImage from "./RevealImage.jsx"
import { revealWords, shouldAnimate } from "../lib/motion.js"
import { usePageVisible } from "../hooks/usePageVisible.js"

const PROVENANCE = [
  ["Cities & coordinates", "Real — GeoNames cities15000, 3739 Indian cities, CC BY 4.0"],
  ["Road distance", "Real — OSRM routing, falling back to a model trained on 840 real road distances"],
  ["Fuel prices", "Partly real — 6 states verified, 29 estimated (see the source column)"],
  ["Trip records", "Generated — from the cost formula recovered from the original data at R² 0.9992"],
]

export default function Footer() {
  const root = useRef(null)
  const pageVisible = usePageVisible()

  useGSAP(
    () => {
      if (!shouldAnimate()) return
      revealWords(gsap, root.current)
      gsap.from(".prov-row", {
        opacity: 0,
        x: -24,
        duration: 0.5,
        stagger: 0.07,
        ease: "power2.out",
        scrollTrigger: { trigger: ".prov-list", start: "top 88%", once: true },
      })
    },
    { dependencies: [pageVisible], scope: root },
  )

  return (
    <footer
      id="data"
      ref={root}
      data-section="07"
      data-section-name="Data & Credits"
      className="relative bg-ink text-page overflow-hidden"
    >
      <TornEdge fill="var(--color-page)" flip className="absolute top-0 inset-x-0 rotate-180" />

      <div className="relative mx-auto max-w-4xl px-6 pt-32 pb-24 text-center">
        <p className="font-mono text-[13px] uppercase tracking-[0.18em] text-page/60">
          Section 07 — Data &amp; Credits
        </p>
        <RevealText
          as="h2"
          text="Where These Numbers Come From"
          className="mt-4 block font-display text-[clamp(1.75rem,4vw,3rem)]"
        />

        <dl className="prov-list mt-12 text-left space-y-5">
          {PROVENANCE.map(([term, detail]) => (
            <div key={term} className="prov-row border-t border-page/20 pt-4">
              <dt className="font-mono text-[13px] uppercase tracking-[0.18em] text-page/60">
                {term}
              </dt>
              <dd className="mt-1.5 text-base text-page/90">{detail}</dd>
            </div>
          ))}
        </dl>

        <RevealImage
          src="/footer.webp"
          alt="A coastal highway at dusk seen from a headland"
          aspect="aspect-[21/9]"
          direction="up"
          className="mt-14"
        />

        <p className="mt-12 mx-auto max-w-[58ch] text-base text-page/70">
          The original 1,140-row dataset is untouched — weeks 1–3 of the notebook still run on
          it. This site is served by a wider dataset regenerated over real Indian geography.
        </p>

        <div className="mt-14 pt-8 border-t border-page/20">
          <p className="font-mono text-[13px] uppercase tracking-[0.18em] text-page/50">
            Built by
          </p>
          <p className="mt-2 font-display italic text-3xl md:text-4xl">Param Kotadiya</p>
          <p className="mt-3 font-mono text-[13px] uppercase tracking-[0.18em] text-page/50">
           ---- Machine Learning · Road Trip Cost Prediction ----
          </p>
        </div>
      </div>
    </footer>
  )
}
