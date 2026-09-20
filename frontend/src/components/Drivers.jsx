import { useRef } from "react"
import gsap from "gsap"
import { useGSAP } from "@gsap/react"
import { Fuel, ReceiptIndianRupee, Wrench } from "lucide-react"
import RevealText from "./RevealText.jsx"
import RevealImage from "./RevealImage.jsx"
import { revealWords, shouldAnimate } from "../lib/motion.js"
import { usePageVisible } from "../hooks/usePageVisible.js"

/** Real recovered coefficients, not marketing copy. This is where the page earns
 *  credibility - every number here came out of the dataset. */
const DRIVERS = [
  {
    Icon: Fuel,
    title: "Fuel burnt",
    rate: "litres × ₹/litre",
    note: "The dominant term, and the only one the model has to multiply rather than add.",
  },
  {
    Icon: ReceiptIndianRupee,
    title: "Tolls",
    rate: "₹1.30 / km",
    note: "Flat across vehicle classes in this data — a Hatchback pays what an SUV pays.",
  },
  {
    Icon: Wrench,
    title: "Wear & tear",
    rate: "₹0.57 – 0.84 / km",
    note: "Hatchback 0.57, Sedan 0.69, SUV 0.84. The one cost that scales with the car.",
  },
]

export default function Drivers() {
  const root = useRef(null)
  const pageVisible = usePageVisible()

  useGSAP(
    () => {
      if (!shouldAnimate()) return
      revealWords(gsap, root.current)

      gsap.from(".driver-col", {
        opacity: 0,
        scale: 0.9,
        y: 28,
        duration: 0.55,
        stagger: { each: 0.09, from: "start", grid: "auto" },
        ease: "back.out(1.5)",
        scrollTrigger: {
          trigger: ".driver-grid",
          start: "top 82%",
          toggleActions: "play none none reverse",
        },
      })

      // Slow drift on the band itself so the section has depth against the pinned one below it.
      gsap.fromTo(
        ".driver-inner",
        { yPercent: 4 },
        {
          yPercent: -4,
          ease: "none",
          scrollTrigger: { trigger: root.current, start: "top bottom", end: "bottom top", scrub: 0.7 },
        },
      )
    },
    { dependencies: [pageVisible], scope: root },
  )

  return (
    <section
      id="drivers"
      ref={root}
      data-section="03"
      data-section-name="Cost Drivers"
      className="paper-noise py-24 md:py-32 px-6 overflow-hidden"
    >
      <div className="driver-inner relative z-10 mx-auto max-w-5xl text-center">
        <p className="label-mono">Section 03 — Cost Drivers</p>
        <RevealText
          as="h2"
          text="Three Things Set The Price"
          className="mt-4 block text-[clamp(2rem,5vw,3.5rem)]"
        />
        <p className="mt-6 mx-auto max-w-[62ch] text-muted">
          Every rupee of a road trip lands in one of four buckets. Three of them scale with
          distance; only the fuel bill also moves with the price at the pump on the day you
          travel.
        </p>

        {/* Renders nothing until /drivers.webp exists. */}
        <RevealImage
          src="/drivers.webp"
          alt="A fuel pump nozzle resting in a car's filler neck"
          aspect="aspect-[21/9]"
          direction="up"
          className="mt-14"
        />

        <div className="driver-grid mt-16 grid gap-12 sm:grid-cols-3">
          {DRIVERS.map(({ Icon, title, rate, note }) => (
            <div key={title} className="driver-col flex flex-col items-center">
              <Icon aria-hidden="true" className="w-7 h-7" strokeWidth={1.25} />
              <h3 className="mt-5 font-display text-2xl">{title}</h3>
              <p className="label-mono mt-2 !text-ink">{rate}</p>
              <p className="mt-3 text-base text-muted max-w-[30ch]">{note}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
