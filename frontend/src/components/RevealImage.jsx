import { useRef, useState } from "react"
import gsap from "gsap"
import { useGSAP } from "@gsap/react"
import { shouldAnimate } from "../lib/motion.js"
import { usePageVisible } from "../hooks/usePageVisible.js"

/**
 * Image revealed by a clip-path wipe on scroll, with a slow counter-parallax on the image
 * itself so the frame and its contents move at different rates.
 *
 * Renders nothing at all if the file is missing (onError), so the page is complete before any
 * image has been generated and simply gains them as files land in /public.
 */
export default function RevealImage({
  src,
  mobileSrc,
  alt,
  className = "",
  aspect = "aspect-[4/5]",
  direction = "up",
  width = 1376,
  height = 768,
}) {
  const root = useRef(null)
  const pageVisible = usePageVisible()
  const [failed, setFailed] = useState(false)

  const from =
    direction === "up" ? "inset(100% 0% 0% 0%)"
    : direction === "left" ? "inset(0% 100% 0% 0%)"
    : "inset(0% 0% 100% 0%)"

  useGSAP(
    () => {
      if (!shouldAnimate() || failed) return

      gsap.fromTo(
        root.current.querySelector(".ri-frame"),
        { clipPath: from },
        {
          clipPath: "inset(0% 0% 0% 0%)",
          duration: 1.15,
          ease: "power3.inOut",
          scrollTrigger: { trigger: root.current, start: "top 82%", once: true },
        },
      )

      // Counter-parallax: the image drifts slower than the frame it sits in.
      gsap.fromTo(
        root.current.querySelector("img"),
        { yPercent: -8, scale: 1.14 },
        {
          yPercent: 8,
          ease: "none",
          scrollTrigger: { trigger: root.current, start: "top bottom", end: "bottom top", scrub: 0.6 },
        },
      )
    },
    { dependencies: [failed, pageVisible], scope: root },
  )

  if (failed) return null

  return (
    <div ref={root} className={className}>
      <div className={`ri-frame relative overflow-hidden ${aspect}`}>
        <picture>
          {mobileSrc && <source media="(max-width: 640px)" srcSet={mobileSrc} type="image/webp" />}
          <img
            src={src}
            alt={alt}
            loading="lazy"
            decoding="async"
            width={width}
            height={height}
            onError={() => setFailed(true)}
            className="absolute inset-0 w-full h-full object-cover grayscale-[0.85] contrast-[1.05] will-change-transform"
          />
        </picture>
      </div>
    </div>
  )
}
