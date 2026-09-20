import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import gsap from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"
import { useGSAP } from "@gsap/react"

import App from "./App.jsx"
import "./index.css"

// Register once, here - not per component. useGSAP is registered as a plugin so that
// gsap.context() cleanup is wired up correctly under React 19 StrictMode, which mounts
// every component twice in development. Without this, ScrollTriggers accumulate on each
// hot reload and scroll positions silently drift.
gsap.registerPlugin(ScrollTrigger, useGSAP)

// ScrollTrigger measures every trigger's position once, at creation. Anything that changes
// layout AFTER that leaves those positions stale — and a reveal whose start point has moved
// out from under it may simply never fire, stranding its content at opacity 0.
//
// Two things reliably shift layout late here: the Google Fonts swap (Playfair is much wider
// than the fallback serif) and lazy-loaded section images. Both get a refresh.
if (typeof document !== "undefined") {
  document.fonts?.ready.then(() => ScrollTrigger.refresh())
  window.addEventListener("load", () => ScrollTrigger.refresh(), { once: true })
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
