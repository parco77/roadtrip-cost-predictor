/**
 * Every call to the FastAPI backend lives here. No component should contain a URL
 * string. Paths are relative: Vite proxies /api to :8000 in dev, and in production
 * FastAPI serves the built UI from the same origin, so this works unchanged in both.
 */

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

/** Autocomplete over the 537 in-domain cities. Pass an AbortSignal so a slow
 *  response from an earlier keystroke cannot overwrite a newer one. */
export function searchCities(query, { limit = 8, signal } = {}) {
  const params = new URLSearchParams({ q: query, limit: String(limit) })
  return request(`/api/cities?${params}`, { signal })
}

/** Cost regression + band and traffic classification.
 *  Omit `traffic_level` to have the backend infer it from departure hour and month. */
export function predictTrip(payload, { signal } = {}) {
  return request("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  })
}

/** Predicted cost across all 24 departure hours for the same route — the "when should I
 *  leave?" chart. Takes the same payload as predictTrip; traffic_level is ignored because the
 *  point is to let the classifier infer it per hour. */
export function fetchDepartureSweep(payload, { signal } = {}) {
  return request("/api/departure-sweep", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  })
}

/** What the mileage sub-model predicts for a vehicle + fuel combination.
 *  Used to prefill the mileage field, so the number the user starts from came from a model
 *  rather than being the same 15.5 for an SUV as for a hatchback. */
export function fetchDefaultMileage(vehicle, fuel, { signal } = {}) {
  const params = new URLSearchParams({ vehicle, fuel })
  return request(`/api/default-mileage?${params}`, { signal })
}

/** Gradient-descent loss curve plus the sklearn-vs-scratch comparison for section 5. */
export function fetchLossCurve({ signal } = {}) {
  return request("/api/loss-curve", { signal })
}

/** Test-set metrics for the report-card section, with the honesty notes attached. */
export function fetchMetrics({ signal } = {}) {
  return request("/api/metrics", { signal })
}

export function fetchHealth({ signal } = {}) {
  return request("/api/health", { signal })
}
