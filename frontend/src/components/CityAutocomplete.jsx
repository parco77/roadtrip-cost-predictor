import { useEffect, useId, useRef, useState } from "react"
import { MapPin } from "lucide-react"
import { useCitySearch } from "../hooks/useCitySearch.js"

/**
 * Accessible combobox over the backend's 537 in-domain cities.
 *
 * Keyboard contract: ArrowDown/ArrowUp move, Enter selects, Escape closes, Tab commits.
 * aria-activedescendant points at the highlighted option so screen readers announce it
 * without moving DOM focus off the input.
 */
export default function CityAutocomplete({ label, value, onSelect, placeholder }) {
  const listId = useId()
  const inputId = useId()
  const [query, setQuery] = useState("")
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const wrapRef = useRef(null)

  const { results, loading } = useCitySearch(open ? query : "")

  // Show the committed selection when the field is not being edited.
  const display = open ? query : value ? `${value.city}, ${value.state}` : ""

  useEffect(() => setActive(0), [results])

  useEffect(() => {
    function onDocClick(event) {
      if (wrapRef.current && !wrapRef.current.contains(event.target)) setOpen(false)
    }
    document.addEventListener("mousedown", onDocClick)
    return () => document.removeEventListener("mousedown", onDocClick)
  }, [])

  function commit(city) {
    onSelect(city)
    setOpen(false)
    setQuery("")
  }

  function onKeyDown(event) {
    if (!open && (event.key === "ArrowDown" || event.key === "ArrowUp")) {
      setOpen(true)
      return
    }
    if (!open) return

    if (event.key === "ArrowDown") {
      event.preventDefault()
      setActive((i) => (results.length ? (i + 1) % results.length : 0))
    } else if (event.key === "ArrowUp") {
      event.preventDefault()
      setActive((i) => (results.length ? (i - 1 + results.length) % results.length : 0))
    } else if (event.key === "Enter") {
      if (results[active]) {
        event.preventDefault()
        commit(results[active])
      }
    } else if (event.key === "Escape") {
      setOpen(false)
      setQuery("")
    }
  }

  return (
    <div ref={wrapRef} className="relative">
      <label htmlFor={inputId} className="label-mono block mb-2">
        {label}
      </label>

      <div className="relative">
        <MapPin
          aria-hidden="true"
          className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted pointer-events-none"
        />
        <input
          id={inputId}
          type="text"
          role="combobox"
          aria-expanded={open}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={open && results[active] ? `${listId}-${active}` : undefined}
          autoComplete="off"
          value={display}
          placeholder={placeholder}
          onChange={(e) => {
            setQuery(e.target.value)
            setOpen(true)
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
          className="w-full h-12 pl-10 pr-3 border border-ink bg-page font-mono text-base
                     placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent"
        />
      </div>

      {/* A floating panel has to read as floating. A 1px hairline on a white panel over a white
          page gave almost no separation — the suggestions did not look like a menu. The design
          system bans soft drop shadows, so depth comes from its own vocabulary instead: a 2px
          black edge plus a hard, zero-blur offset block. */}
      {open && (
        <ul
          id={listId}
          role="listbox"
          className="absolute z-30 w-full mt-1 max-h-64 overflow-y-auto border-2 border-ink
                     bg-page divide-y divide-hairline
                     shadow-[5px_5px_0_0_var(--color-ink)]"
        >
          {loading && results.length === 0 && (
            <li className="px-3 py-3 label-mono">Searching…</li>
          )}
          {!loading && query.trim() && results.length === 0 && (
            <li className="px-3 py-3 label-mono">No city found in the 537-city domain</li>
          )}
          {results.map((city, i) => (
            <li
              key={`${city.city}-${city.state}`}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => commit(city)}
              className={`px-3 py-2.5 cursor-pointer flex items-baseline justify-between gap-3
                          ${i === active ? "bg-ink text-page" : "bg-page text-ink hover:bg-paper"}`}
            >
              <span className="font-body leading-tight">{city.city}</span>
              <span
                className={`font-mono text-[13px] uppercase tracking-[0.14em] shrink-0
                            ${i === active ? "text-page/75" : "text-muted"}`}
              >
                {city.state}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
