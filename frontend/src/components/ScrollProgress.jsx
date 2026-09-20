import { useEffect, useRef, useState } from "react"
import gsap from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"
import { useGSAP } from "@gsap/react"
import { shouldAnimate } from "../lib/motion.js"
import { usePageVisible } from "../hooks/usePageVisible.js"

/**
 * Fixed progress rail with a section counter.
 *
 * Reads [data-section] / [data-section-name] off the sections themselves rather than taking a
 * hardcoded list, so adding or reordering a section needs no change here.
 *
 * Hidden below lg: on a phone this would eat horizontal space next to body copy, and the
 * page is short enough there that orientation is not a problem.
 */
export default function ScrollProgress() {
  const root = useRef(null)
  const pageVisible = usePageVisible()
  const barRef = useRef(null)
  const [sections, setSections] = useState([])
  const [activeIndex, setActiveIndex] = useState(0)

  useEffect(() => {
    setSections(
      [...document.querySelectorAll("[data-section]")].map((el) => ({
        id: el.id,
        number: el.dataset.section,
        name: el.dataset.sectionName,
      })),
    )
  }, [])

  useGSAP(
    () => {
      const nodes = [...document.querySelectorAll("[data-section]")]
      if (!nodes.length) return

      // The active-section tracking is not decoration, so it runs even when animation is
      // suppressed — only the scrubbed bar fill is animation.
      nodes.forEach((el, i) => {
        ScrollTrigger.create({
          trigger: el,
          start: "top 45%",
          end: "bottom 45%",
          onToggle: (self) => self.isActive && setActiveIndex(i),
        })
      })

      if (!shouldAnimate() || !barRef.current) return
      gsap.fromTo(
        barRef.current,
        { scaleY: 0 },
        {
          scaleY: 1,
          ease: "none",
          transformOrigin: "top",
          scrollTrigger: { start: 0, end: "max", scrub: 0.3 },
        },
      )
    },
    { dependencies: [sections.length, pageVisible], scope: root },
  )

  if (!sections.length) return null
  const active = sections[activeIndex]

  return (
    <aside
      ref={root}
      aria-hidden="true"
      className="hidden lg:flex fixed right-8 top-1/2 -translate-y-1/2 z-40 flex-col items-center gap-4 mix-blend-difference"
    >
      <span className="font-mono text-[13px] tracking-[0.18em] text-white tabular-nums">
        {active?.number}
      </span>

      <div className="relative w-px h-40 bg-white/25">
        <div ref={barRef} className="absolute inset-x-0 top-0 h-full bg-white origin-top" />
      </div>

      <span
        className="font-mono text-[13px] tracking-[0.18em] text-white whitespace-nowrap"
        style={{ writingMode: "vertical-rl" }}
      >
        {active?.name}
      </span>
    </aside>
  )
}
