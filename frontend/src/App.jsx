import { useState } from "react"

import Hero from "./components/Hero.jsx"
import Drivers from "./components/Drivers.jsx"
import DistanceTable from "./components/DistanceTable.jsx"
import Predictor from "./components/Predictor.jsx"
import ModelStory from "./components/ModelStory.jsx"
import ReportCard from "./components/ReportCard.jsx"
import Footer from "./components/Footer.jsx"
import ScrollProgress from "./components/ScrollProgress.jsx"

/** The seven sections, numbered 01-07 to match the scroll rail. */
export default function App() {
  // The one piece of state two sections share: "Use this" in the distance table fills the
  // estimator's distance field. Wrapped in an object so picking the same number twice still
  // produces a new value and re-triggers the effect that focuses the field.
  const [picked, setPicked] = useState(null)

  return (
    <>
      <ScrollProgress />
      <a
        href="#estimate"
        className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:top-3 focus:left-3
                   focus:bg-ink focus:text-page focus:px-4 focus:py-2 focus:font-mono focus:text-sm"
      >
        Skip to the estimator
      </a>

      <Hero />
      <Drivers />
      <DistanceTable onPick={(km) => setPicked({ km })} />
      <Predictor picked={picked} />
      <ModelStory />
      <ReportCard />
      <Footer />
    </>
  )
}
