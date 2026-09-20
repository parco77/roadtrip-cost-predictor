import { useCallback, useEffect, useRef, useState } from "react"
import { searchCities } from "../api.js"

/**
 * Debounced city autocomplete.
 *
 * The AbortController is the important part: without it, a slow response to "de" can
 * land after a fast response to "delh" and silently replace the correct results with
 * stale ones. Every new keystroke aborts the request in flight.
 */
export function useCitySearch(query, { limit = 8, minLength = 1, delay = 200 } = {}) {
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const controllerRef = useRef(null)

  useEffect(() => {
    const term = query.trim()

    if (term.length < minLength) {
      controllerRef.current?.abort()
      setResults([])
      setLoading(false)
      setError(null)
      return
    }

    setLoading(true)
    const timer = setTimeout(() => {
      controllerRef.current?.abort()
      const controller = new AbortController()
      controllerRef.current = controller

      searchCities(term, { limit, signal: controller.signal })
        .then((data) => {
          setResults(data.results)
          setError(null)
        })
        .catch((err) => {
          if (err.name === "AbortError") return // superseded by a newer keystroke
          setError(err.message)
          setResults([])
        })
        .finally(() => {
          if (!controller.signal.aborted) setLoading(false)
        })
    }, delay)

    return () => clearTimeout(timer)
  }, [query, limit, minLength, delay])

  // Abort any request still in flight when the component unmounts.
  useEffect(() => () => controllerRef.current?.abort(), [])

  const reset = useCallback(() => {
    controllerRef.current?.abort()
    setResults([])
    setLoading(false)
    setError(null)
  }, [])

  return { results, loading, error, reset }
}
