import { useCallback, useEffect, useRef, useState } from "react"
import { predictTrip } from "../api.js"

/**
 * Explicit state machine: "idle" | "loading" | "error" | "ready".
 *
 * Deliberately not three separate booleans - the UI looks different in each state and
 * booleans allow contradictory combinations (loading && error) that then have to be
 * defended against at every render site.
 */
export function usePrediction() {
  const [status, setStatus] = useState("idle")
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const controllerRef = useRef(null)

  const predict = useCallback(async (payload) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller

    setStatus("loading")
    setError(null)

    try {
      const data = await predictTrip(payload, { signal: controller.signal })
      if (controller.signal.aborted) return
      setResult(data)
      setStatus("ready")
    } catch (err) {
      if (err.name === "AbortError") return
      setError(err.message)
      setStatus("error")
    }
  }, [])

  const reset = useCallback(() => {
    controllerRef.current?.abort()
    setStatus("idle")
    setResult(null)
    setError(null)
  }, [])

  useEffect(() => () => controllerRef.current?.abort(), [])

  return { status, result, error, predict, reset }
}
