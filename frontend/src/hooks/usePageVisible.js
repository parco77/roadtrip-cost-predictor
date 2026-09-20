import { useEffect, useState } from "react"

/**
 * "Has this page ever been visible?"
 *
 * Why this exists: GSAP advances on requestAnimationFrame, which browsers do not fire in a tab
 * that is not compositing. Any gsap.from()/set() that writes a hidden start state will freeze
 * there. The original fix was simply to skip animation when document.hidden — but that made the
 * skip PERMANENT: open the site in a background tab, switch to it, and you get a completely
 * static page for the rest of the session.
 *
 * This hook instead reports false while hidden and flips to true the first time the page becomes
 * visible. Pass it into a useGSAP dependency array and the animations initialise at that moment.
 * It never flips back — re-running entrance animations every time someone tabs away and back
 * would be worse than not animating at all.
 */
export function usePageVisible() {
  const [visible, setVisible] = useState(() =>
    typeof document === "undefined" ? true : !document.hidden,
  )

  useEffect(() => {
    if (visible) return
    const onChange = () => {
      if (!document.hidden) setVisible(true)
    }
    document.addEventListener("visibilitychange", onChange)
    return () => document.removeEventListener("visibilitychange", onChange)
  }, [visible])

  return visible
}
