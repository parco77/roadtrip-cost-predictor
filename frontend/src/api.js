/**
 * Every call to the FastAPI backend lives here. No component should contain a URL string.
 *
 * The UI and the API are deployed to different hosts, so the API origin is a build-time
 * variable rather than something the code assumes. VITE_API_URL is set on the host that
 * builds the UI; left unset, API is "" and the paths stay relative, which is exactly what
 * Vite's dev proxy expects. One expression covers both, with no branch on environment.
 */
const API = import.meta.env.VITE_API_URL ?? ""

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  })

  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body = await response.json()
      if (typeof body.detail === "string") {
        detail = body.detail
      } else if (Array.isArray(body.detail) && body.detail.length) {
        // pydantic 422: surface the first field error in plain language
        const first = body.detail[0]
        const field = Array.isArray(first.loc) ? first.loc.at(-1) : "input"
        detail = `${field}: ${first.msg}`
      }
    } catch {
      /* non-JSON error body - keep the generic message */
    }
    const error = new Error(detail)
    error.status = response.status
    throw error
  }

  return response.json()
}

/** Autocomplete over the in-domain cities. Pass an AbortSignal so a slow
 *  response from an earlier keystroke cannot overwrite a newer one. */
export function searchCities(query, { limit = 8, signal } = {}) {
  const params = new URLSearchParams({ q: query, limit: String(limit) })
  return request(`${API}/api/cities?${params}`, { signal })
}

/** Cost for every vehicle on every fuel, in ONE response.
 *
 *  Deliberately not three calls. usePrediction aborts the request in flight on every
 *  new call, so three concurrent ones would leave only the last — silently, because
 *  AbortError is swallowed. One request cannot race itself. */
export function predictTrip(payload, { signal } = {}) {
  return request(`${API}/api/predict`, {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  })
}

/** Road distance between two cities, for routes the shipped table does not cover.
 *  Falls back to the fitted distance model when routing is unavailable; the response
 *  says which of the two answered. */
export function fetchDistance(from, to, { signal } = {}) {
  const params = new URLSearchParams({
    from: from.city, from_state: from.state,
    to: to.city, to_state: to.state,
  })
  return request(`${API}/api/distance?${params}`, { signal })
}

/** What each of the three vehicles manages on a given fuel, from the mileage sub-model.
 *  All three at once, because the result prices all three at once. */
export function fetchDefaultMileage(fuel, { signal } = {}) {
  const params = new URLSearchParams({ fuel })
  return request(`${API}/api/default-mileage?${params}`, { signal })
}

/** The shipped reference table of measured road distances.
 *
 *  A static file in public/, not an endpoint: it never changes between deploys, the
 *  browser caches it, and the lookup keeps working if the API is down. ~25 KB gzipped.
 *
 *  Note the missing API: this one is served by the host that serves the UI, not by the
 *  API, so prefixing it would send the browser looking for it on the wrong origin. */
export function loadDistanceTable({ signal } = {}) {
  return request("/city_distances.json", { signal })
}

/** Gradient-descent loss curve plus the closed-form comparison. */
export function fetchLossCurve({ signal } = {}) {
  return request(`${API}/api/loss-curve`, { signal })
}

/** Test-set metrics for the report-card section, with the honesty notes attached. */
export function fetchMetrics({ signal } = {}) {
  return request(`${API}/api/metrics`, { signal })
}

export function fetchHealth({ signal } = {}) {
  return request(`${API}/api/health`, { signal })
}
