import { useEffect, useMemo, useRef, useState } from "react"
import gsap from "gsap"
import { useGSAP } from "@gsap/react"
import { ArrowRight, Loader2, Search } from "lucide-react"

import CityAutocomplete from "./CityAutocomplete.jsx"
import RevealText from "./RevealText.jsx"
import { fetchDistance, loadDistanceTable } from "../api.js"
import { revealWords, shouldAnimate } from "../lib/motion.js"
import { usePageVisible } from "../hooks/usePageVisible.js"

const PAGE = 14

/**
 * The reference table the estimator sends you to.
 *
 * Two layers, on purpose. The shipped JSON covers the routes people actually drive and
 * needs no network and no model — those are measured road distances. Anything it does
 * not cover falls through to the live lookup below it, which is the one place the
 * fitted distance model does its work.
 */
export default function DistanceTable({ onPick }) {
  const [table, setTable] = useState(null)
  const [query, setQuery] = useState("")
  const [shown, setShown] = useState(PAGE)
  const root = useRef(null)
  const pageVisible = usePageVisible()

  // Live-lookup state, for pairs the table does not carry.
  const [from, setFrom] = useState(null)
  const [to, setTo] = useState(null)
  const [lookup, setLookup] = useState(null)
  const [lookupState, setLookupState] = useState("idle")
  const lookupController = useRef(null)

  useEffect(() => {
    const controller = new AbortController()
    loadDistanceTable({ signal: controller.signal })
      .then(setTable)
      .catch(() => setTable({ cities: [], routes: [], failed: true }))
    return () => controller.abort()
  }, [])

  useEffect(() => () => lookupController.current?.abort(), [])

  useGSAP(
    () => {
      if (!shouldAnimate()) return
      revealWords(gsap, root.current)
    },
    { dependencies: [pageVisible], scope: root },
  )

  // Rebuilt only when the query or the table changes, not on every render — 5,538 rows
  // is cheap to filter once and wasteful to filter on a keystroke that changed nothing.
  const matches = useMemo(() => {
    if (!table?.routes?.length) return []
    const term = query.trim().toLowerCase()
    const { cities, routes } = table
    const rows = []
    for (const [a, b, km] of routes) {
      const ca = cities[a]
      const cb = cities[b]
      if (!ca || !cb) continue
      if (term) {
        const hit =
          ca.city.toLowerCase().includes(term) ||
          cb.city.toLowerCase().includes(term) ||
          ca.state.toLowerCase().includes(term) ||
          cb.state.toLowerCase().includes(term)
        if (!hit) continue
      }
      rows.push({ from: ca, to: cb, km })
    }
    rows.sort((x, y) => x.km - y.km)
    return rows
  }, [table, query])

  useEffect(() => setShown(PAGE), [query])

  function runLookup(event) {
    event.preventDefault()
    if (!from || !to || (from.city === to.city && from.state === to.state)) return
    lookupController.current?.abort()
    const controller = new AbortController()
    lookupController.current = controller
    setLookupState("loading")
    setLookup(null)
    fetchDistance(from, to, { signal: controller.signal })
      .then((data) => {
        setLookup(data)
        setLookupState("ready")
      })
      .catch((err) => {
        if (err.name === "AbortError") return
        setLookup({ error: err.message })
        setLookupState("error")
      })
  }

  const sameCity = from && to && from.city === to.city && from.state === to.state

  return (
    <section
      id="distances"
      ref={root}
      data-section="03"
      data-section-name="Distances"
      className="rule-heavy py-24 md:py-32 px-6"
    >
      <div className="mx-auto max-w-6xl">
        <header className="max-w-[62ch]">
          <p className="label-mono">Section 03 — Distances</p>
          <RevealText
            as="h2"
            text="How Far Is It?"
            className="mt-4 block text-[clamp(2rem,5vw,3.5rem)]"
          />
          <p className="mt-5 text-muted">
            Measured road distances, not straight lines. Find your route, take the number
            across to the estimator.
            {table?.cities?.length ? (
              <>
                {" "}
                {table.routes.length.toLocaleString("en-IN")} routes between{" "}
                {table.cities.length} cities, every one between {table.range_km?.[0] ?? 50} and{" "}
                {table.range_km?.[1] ?? 1500} km.
              </>
            ) : null}
          </p>
        </header>

        {/* ---------------------------------------------------------------- the table */}
        <div className="mt-12">
          <label htmlFor="route-search" className="label-mono block mb-2">
            Search by city or state
          </label>
          <div className="relative max-w-md">
            <Search
              aria-hidden="true"
              className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted"
            />
            <input
              id="route-search"
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Jaipur, Kerala, Pune…"
              className="w-full h-11 pl-9 pr-3 border border-ink bg-page font-mono text-base"
            />
          </div>

          {table === null && (
            <p className="label-mono mt-6 flex items-center gap-2">
              <Loader2 aria-hidden="true" className="w-4 h-4 animate-spin" />
              Loading the table…
            </p>
          )}

          {table?.failed && (
            <p className="font-mono text-[13px] text-band-poor mt-6">
              The reference table could not be loaded. Use the lookup below instead.
            </p>
          )}

          {table && !table.failed && (
            <>
              <p className="label-mono mt-6" aria-live="polite">
                {matches.length.toLocaleString("en-IN")}{" "}
                {matches.length === 1 ? "route" : "routes"}
                {query.trim() ? ` matching “${query.trim()}”` : ""}
              </p>

              {matches.length === 0 ? (
                <p className="font-body text-muted mt-4 max-w-[52ch]">
                  Nothing in the table for that. The lookup below covers any pair of cities,
                  including routes shorter or longer than the table carries.
                </p>
              ) : (
                <>
                  <ul className="mt-4 border-t border-hairline">
                    {matches.slice(0, shown).map((row) => (
                      <li
                        key={`${row.from.city}-${row.to.city}`}
                        className="border-b border-hairline py-3 grid gap-3 items-center
                                   grid-cols-[1fr_auto] sm:grid-cols-[1fr_7rem_9rem]"
                      >
                        <div className="min-w-0">
                          <p className="font-body text-base truncate">
                            {row.from.city} <span className="text-muted">→</span> {row.to.city}
                          </p>
                          <p className="font-mono text-[12px] text-muted truncate">
                            {row.from.state} to {row.to.state}
                          </p>
                        </div>
                        <p className="font-mono text-base tabular-nums sm:text-right">
                          {row.km.toLocaleString("en-IN")} km
                        </p>
                        <button
                          type="button"
                          onClick={() => onPick(row.km)}
                          className="col-span-2 sm:col-span-1 justify-self-start sm:justify-self-end
                                     inline-flex items-center gap-2 min-h-11 px-4 border border-ink
                                     font-mono text-[12px] uppercase tracking-[0.14em]
                                     hover:bg-ink hover:text-page transition-colors duration-150"
                        >
                          Use this
                          <ArrowRight aria-hidden="true" className="w-3 h-3" />
                        </button>
                      </li>
                    ))}
                  </ul>

                  {shown < matches.length && (
                    <button
                      type="button"
                      onClick={() => setShown((n) => n + PAGE * 2)}
                      className="mt-6 min-h-11 px-6 border border-ink font-mono text-[13px]
                                 uppercase tracking-[0.14em] hover:bg-ink hover:text-page
                                 transition-colors duration-150"
                    >
                      Show more ({(matches.length - shown).toLocaleString("en-IN")} left)
                    </button>
                  )}
                </>
              )}
            </>
          )}
        </div>

        {/* ------------------------------------------------------------ the live lookup */}
        <div className="mt-16 pt-10 border-t-2 border-ink">
          <h3 className="font-display text-[clamp(1.4rem,3vw,2rem)]">Not in the table?</h3>
          <p className="mt-3 text-muted max-w-[58ch]">
            Any two cities. This asks the routing service for the real road distance, and if
            that is unreachable it falls back to a model trained on 840 measured routes — the
            answer says which one you got.
          </p>

          <form onSubmit={runLookup} noValidate className="mt-8 max-w-3xl">
            <div className="grid sm:grid-cols-2 gap-6">
              <CityAutocomplete label="From" value={from} onSelect={setFrom} placeholder="Search cities…" />
              <CityAutocomplete label="To" value={to} onSelect={setTo} placeholder="Search cities…" />
            </div>

            {sameCity && (
              <p role="alert" className="font-mono text-[13px] text-band-poor mt-3">
                Pick two different cities
              </p>
            )}

            <button
              type="submit"
              disabled={!from || !to || sameCity || lookupState === "loading"}
              className="mt-6 inline-flex items-center justify-center gap-2.5 min-h-11 px-8
                         border-2 border-ink font-mono text-[13px] uppercase tracking-[0.18em]
                         hover:bg-ink hover:text-page transition-colors duration-150
                         disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-page
                         disabled:hover:text-ink"
            >
              {lookupState === "loading" ? "Looking up…" : "Get the distance"}
            </button>
          </form>

          {lookupState === "error" && (
            <p role="alert" className="font-mono text-[13px] text-band-poor mt-6">
              {lookup?.error}
            </p>
          )}

          {lookupState === "ready" && lookup && (
            <div className="mt-8 border-2 border-ink p-6 max-w-3xl" aria-live="polite">
              <p className="label-mono">
                {lookup.from.city} → {lookup.to.city}
              </p>
              <p className="font-display font-black text-accent leading-none mt-2
                            text-[clamp(2.2rem,5vw,3.5rem)] tabular-nums">
                {Math.round(lookup.distance_km).toLocaleString("en-IN")} km
              </p>
              <p className="font-mono text-[13px] text-muted mt-3">
                {Math.round(lookup.straight_line_km).toLocaleString("en-IN")} km in a straight
                line — the road is {lookup.winding_ratio}× that
                {" · "}
                {lookup.source === "measured" ? "measured route" : "predicted by the model"}
              </p>
              {!lookup.in_trained_range && (
                <p className="font-mono text-[13px] text-band-average mt-2">
                  Outside the 50–1500 km range the cost model was trained on — the estimate
                  will say so too.
                </p>
              )}
              <button
                type="button"
                onClick={() => onPick(Math.round(lookup.distance_km))}
                className="mt-5 inline-flex items-center gap-2 min-h-11 px-6 bg-ink text-page
                           font-mono text-[12px] uppercase tracking-[0.14em]
                           hover:opacity-90 transition-opacity duration-150"
              >
                Use this
                <ArrowRight aria-hidden="true" className="w-3 h-3" />
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
