import Hero from "./components/Hero.jsx"
import Drivers from "./components/Drivers.jsx"
import Predictor from "./components/Predictor.jsx"
import ModelStory from "./components/ModelStory.jsx"
import ReportCard from "./components/ReportCard.jsx"
import Footer from "./components/Footer.jsx"
import ScrollProgress from "./components/ScrollProgress.jsx"

/** The seven sections from FRONTEND_PLAN.md section 4, in order. */
export default function App() {
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
      <Predictor />
      <ModelStory />
      <ReportCard />
      <Footer />
    </>
  )
}
