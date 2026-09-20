/**
 * Guard for every entrance animation.
 *
 * Two failure modes this prevents:
 *
 * 1. prefers-reduced-motion - the obvious one.
 * 2. Background tabs. gsap.from() writes its START state (opacity: 0) immediately, but the
 *    ticker runs on requestAnimationFrame, which browsers do not fire in a tab that is not
 *    compositing. Result: the hero headline is set to opacity 0 and stays there. Verified in
 *    this project - hero, driver columns and torn edge were all frozen invisible when the page
 *    loaded unfocused, and the cost figure was stuck at Rs 0.
 *
 * Callers must ALSO pass usePageVisible() into their useGSAP dependency array, so animations
 * initialise when a background tab is first brought to the front rather than being skipped for
 * the whole session.
 *
 * When this returns false, render the finished state directly instead of animating toward it.
 */
export function shouldAnimate() {
  if (typeof window === "undefined") return false
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return false
  if (typeof document !== "undefined" && document.hidden) return false
  return true
}

/**
 * Animate every <RevealText> word inside `scope` up from behind its mask.
 *
 * Call from inside a component's own useGSAP so the tween is registered in that component's
 * context and torn down with it.
 *
 * Uses set-then-to rather than gsap.from() on purpose: from() writes the hidden start state at
 * mount, which would leave headings far down the page invisible until their trigger fires — and
 * permanently invisible if the ticker never runs. Setting the start state is safe here only
 * because the caller has already checked shouldAnimate().
 */
export function revealWords(gsap, scopeEl, { start = "top 85%", stagger = 0.055 } = {}) {
  const words = gsap.utils.toArray(".rt-word", scopeEl)
  if (!words.length) return
  gsap.set(words, { yPercent: 110, opacity: 0 })
  gsap.to(words, {
    yPercent: 0,
    opacity: 1,
    duration: 0.75,
    stagger,
    ease: "power3.out",
    scrollTrigger: { trigger: scopeEl, start, once: true },
  })
}
