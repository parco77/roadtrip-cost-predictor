import { useRef, useState } from "react"
import gsap from "gsap"
import { useGSAP } from "@gsap/react"

import Nav from "./Nav.jsx"
import TornEdge from "./TornEdge.jsx"
import { shouldAnimate } from "../lib/motion.js"
import { usePageVisible } from "../hooks/usePageVisible.js"

/**
 * Fallback backdrop, used until /hero.webp exists (and if it ever fails to load).
 * Layered monochrome ridgelines — self-contained, weightless, sharp at any size.
 */
function Ridges() {
  return (
    <svg
      viewBox="0 0 1440 900"
      preserveAspectRatio="xMidYMax slice"
      aria-hidden="true"
      className="absolute inset-0 w-full h-full"
    >
      <defs>
        <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#6B7280" />
          <stop offset="55%" stopColor="#9CA3AF" />
          <stop offset="100%" stopColor="#D1D5DB" />
        </linearGradient>
        <linearGradient id="fog" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#FFFFFF" stopOpacity="0" />
          <stop offset="100%" stopColor="#FFFFFF" stopOpacity="0.85" />
        </linearGradient>
      </defs>
      <rect width="1440" height="900" fill="url(#sky)" />
      <path className="ridge" fill="#4B5563" fillOpacity="0.55"
        d="M0,520 L150,430 L280,500 L420,380 L560,470 L700,360 L860,455 L1010,375 L1160,470 L1300,400 L1440,478 L1440,900 L0,900 Z" />
      <path className="ridge" fill="#374151" fillOpacity="0.75"
        d="M0,640 L130,560 L260,625 L400,520 L540,605 L690,505 L840,590 L1000,515 L1150,600 L1300,535 L1440,610 L1440,900 L0,900 Z" />
      <path className="ridge" fill="#1F2937"
        d="M0,760 L160,690 L300,745 L450,660 L600,730 L760,650 L920,720 L1080,655 L1240,725 L1360,675 L1440,715 L1440,900 L0,900 Z" />
      <path fill="#111827" d="M690,660 L750,660 L980,900 L430,900 Z" />
      <path stroke="#F9FAFB" strokeOpacity="0.55" strokeWidth="4" strokeDasharray="26 30"
        fill="none" d="M720,665 L706,900" />
      <rect y="500" width="1440" height="400" fill="url(#fog)" />
    </svg>
  )
}

export default function Hero() {
  const root = useRef(null)
  const pageVisible = usePageVisible()
  // Starts true and flips on error: the photo is the intended design, the SVG is the safety net.
  const [photoOk, setPhotoOk] = useState(true)

  // ENTRANCE — deliberately does NOT depend on photoOk.
  //
  // It used to. When /hero.webp was missing, the <img> onError flipped photoOk, which changed
  // the dependency array and re-ran this effect *mid-animation*. useGSAP does not revert on
  // dependency change by default, so a second gsap.from() landed on words that were already
  // in flight, the two tweens fought, and the headline ended up parked off-screen — the
  // "text motion is not working" symptom. The backdrop swap has nothing to do with the copy,
  // so the two concerns are now separate effects.
  //
  // fromTo rather than from: explicit start AND end states make a re-run idempotent instead of
  // capturing whatever mid-tween position happened to be current.
  useGSAP(
    (context, contextSafe) => {
      if (!shouldAnimate()) return

      // contextSafe is required, not optional. gsap.context() only captures animations created
      // during the SYNCHRONOUS run of this callback. Because play() is deferred behind
      // document.fonts.ready, its tweens would be created outside the context and would never
      // be reverted on unmount — a leak that survives every remount.
      const play = contextSafe(() => {
        const tl = gsap.timeline()
        tl.fromTo(".hero-word",
          { yPercent: 115 },
          { yPercent: 0, duration: 1, stagger: 0.09, ease: "power4.out" })
          .fromTo(".hero-sub, .hero-cta",
            { opacity: 0, y: 20 },
            { opacity: 1, y: 0, duration: 0.6, stagger: 0.1, ease: "power2.out" }, "-=0.45")
          .fromTo(".torn",
            { scaleY: 0 },
            { scaleY: 1, transformOrigin: "bottom", duration: 0.8, ease: "power2.out" }, "-=0.7")
      })

      // Wait for Playfair Display before measuring. Animating during the fallback font means
      // the overflow-hidden masks are sized for the wrong glyphs and the words visibly jump
      // when the real face swaps in.
      if (document.fonts?.status === "loaded") play()
      else document.fonts?.ready.then(play) ?? play()
    },
    { dependencies: [pageVisible], scope: root },
  )

  // SCROLL — the backdrop pushes in while the copy lifts and fades, so the hero recedes
  // rather than simply scrolling off. This one does depend on photoOk, because .ridge only
  // exists while the SVG fallback is rendered.
  useGSAP(
    () => {
      if (!shouldAnimate()) return

      gsap.to(".hero-backdrop", {
        scale: 1.18,
        yPercent: 12,
        ease: "none",
        scrollTrigger: { trigger: root.current, start: "top top", end: "bottom top", scrub: 0.4 },
      })
      gsap.to(".hero-copy", {
        yPercent: -34,
        opacity: 0,
        ease: "none",
        scrollTrigger: { trigger: root.current, start: "top top", end: "70% top", scrub: 0.4 },
      })
      if (!photoOk) {
        gsap.to(".ridge", {
          yPercent: (i) => -5 - i * 4,
          ease: "none",
          scrollTrigger: { trigger: root.current, start: "top top", end: "bottom top", scrub: 0.5 },
        })
      }
    },
    { dependencies: [photoOk, pageVisible], scope: root },
  )

  return (
    <header
      id="top"
      ref={root}
      data-section="01"
      data-section-name="Know The Cost"
      className="relative h-[100svh] min-h-[560px] overflow-hidden flex flex-col justify-center items-center text-center px-6"
    >
      <div className="hero-backdrop absolute inset-0 will-change-transform">
        {photoOk ? (
          <picture>
            <source media="(max-width: 640px)" srcSet="/hero-mobile.webp" type="image/webp" />
            <img
              src="/hero.webp"
              alt="An empty highway climbing through mountains in heavy fog"
              width="1376"
              height="768"
              fetchPriority="high"
              decoding="async"
              onError={() => setPhotoOk(false)}
              className="absolute inset-0 w-full h-full object-cover grayscale-[0.85] contrast-[1.05]"
            />
          </picture>
        ) : (
          <Ridges />
        )}
      </div>

      {/* The headline needs this to clear 4.5:1 over the brightest part of any sky.
          A blend mode alone is not enough. Keep it whichever backdrop is in use. */}
      <div aria-hidden="true" className="absolute inset-0 bg-black/30" />

      <Nav />

      <div className="hero-copy relative z-10">
        <h1 className="font-display font-black text-white leading-[0.9] text-[clamp(2.75rem,12vw,9rem)] drop-shadow-[0_2px_18px_rgba(0,0,0,0.5)]">
          {["KNOW", "THE", "COST"].map((word) => (
            <span
              key={word}
              className="inline-block overflow-hidden align-bottom mr-[0.22em] last:mr-0"
              style={{ paddingBottom: "0.08em", marginBottom: "-0.08em" }}
            >
              <span className="hero-word inline-block">{word}</span>
            </span>
          ))}
        </h1>

        <p className="hero-sub font-mono text-[13px] md:text-sm uppercase tracking-[0.22em] text-white/90 mt-7 drop-shadow-[0_1px_8px_rgba(0,0,0,0.7)]">
          537 cities · 25,000 trips · R² 0.9997
        </p>

        <a
          href="#estimate"
          className="hero-cta mt-10 inline-flex items-center justify-center rounded-full
                     border border-white/80 text-white px-10 h-12 font-mono text-[13px]
                     uppercase tracking-[0.18em] backdrop-blur-[2px]
                     hover:bg-white hover:text-ink transition-colors duration-200"
        >
          Estimate my trip
        </a>
      </div>

      <TornEdge className="torn absolute bottom-0 inset-x-0 z-10" />
    </header>
  )
}
