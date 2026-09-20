import { useMemo, useState } from "react"
import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

const axis = {
  stroke: "var(--color-muted)",
  fontSize: 13,
  fontFamily: "var(--font-mono)",
}

const label = (h) => `${((h + 11) % 12) + 1}${h < 12 ? "am" : "pm"}`
const inr = (n) => `₹${Math.round(n).toLocaleString("en-IN")}`

/** Inclusive, and wraps past midnight so an overnight window (22 → 6) works. */
function inWindow(hour, from, to) {
  return from <= to ? hour >= from && hour <= to : hour >= from || hour <= to
}

/**
 * Cost across all 24 departure hours, with a window the user can actually travel in.
 *
 * Bars show EXPECTED cost — each hour's traffic probabilities weighted against the cost at each
 * traffic level — rather than the cost at the single most likely level. Using argmax collapsed
 * all 24 hours onto three values and threw away everything the classifier knew about its own
 * confidence.
 *
 * The window is computed here rather than server-side: all 24 hours are already in the payload,
 * so narrowing it is instant and costs no request.
 *
 * The y-axis deliberately does NOT start at zero. The whole range is ~8% of the total cost, so a
 * zero baseline would render 24 near-identical bars and communicate nothing. The axis is labelled
 * in exact rupees and the caption states the range, so the zoom is never hidden.
 */
export default function DepartureChart({ data }) {
  const [fromHour, setFromHour] = useState(0)
  const [toHour, setToHour] = useState(23)

  const { rows, domain, best, worst, saving, windowed } = useMemo(() => {
    const hours = data?.hours ?? []
    if (!hours.length) return { rows: [], domain: [0, 1] }

    const marked = hours.map((h) => ({ ...h, inside: inWindow(h.hour, fromHour, toHour) }))
    const pick = marked.filter((h) => h.inside)
    const pool = pick.length ? pick : marked

    const costs = hours.map((h) => h.total_trip_cost)
    const min = Math.min(...costs)
    const max = Math.max(...costs)
    const pad = (max - min) * 0.25 || 10

    const cheapest = pool.reduce((a, b) => (b.total_trip_cost < a.total_trip_cost ? b : a))
    const dearest = pool.reduce((a, b) => (b.total_trip_cost > a.total_trip_cost ? b : a))

    return {
      rows: marked.map((h) => ({ ...h, isBest: h.hour === cheapest.hour && h.inside })),
      domain: [Math.floor(min - pad), Math.ceil(max + pad)],
      best: cheapest,
      worst: dearest,
      saving: dearest.total_trip_cost - cheapest.total_trip_cost,
      windowed: pick.length > 0 && pick.length < 24,
    }
  }, [data, fromHour, toHour])

  if (!data?.hours) return null

  return (
    <section className="mt-10 border-t-2 border-ink pt-8">
      <p className="label-mono">When should you leave?</p>

      <p className="mt-3 font-display text-3xl">
        {windowed ? "In that window, leaving at " : "Leaving at "}
        <span className="text-accent">{label(best.hour)}</span>
        {saving > 1 ? (
          <>
            {" "}saves <span className="text-accent">{inr(saving)}</span>
          </>
        ) : (
          " is as good as it gets"
        )}
      </p>
      <p className="mt-2 text-base text-muted max-w-[54ch]">
        {saving > 1 ? (
          <>
            against the worst time {windowed ? "you can leave" : ""} ({label(worst.hour)}) — that
            is {((saving / worst.total_trip_cost) * 100).toFixed(1)}% of the trip.
          </>
        ) : (
          "every hour in that window costs about the same."
        )}
      </p>

      {/* The window lives with the chart, not the form: it changes what the chart recommends,
          not what the model predicts, and its effect should be visible as you drag it. */}
      <div className="mt-6 grid sm:grid-cols-2 gap-x-8 gap-y-4 max-w-xl">
        <div>
          <label htmlFor="fromHour" className="label-mono block mb-2">
            Can leave after — {label(fromHour)}
          </label>
          <input
            id="fromHour"
            type="range"
            min="0"
            max="23"
            value={fromHour}
            onChange={(e) => setFromHour(Number(e.target.value))}
            className="w-full h-11 accent-[var(--color-accent)] cursor-pointer"
          />
        </div>
        <div>
          <label htmlFor="toHour" className="label-mono block mb-2">
            Must leave by — {label(toHour)}
          </label>
          <input
            id="toHour"
            type="range"
            min="0"
            max="23"
            value={toHour}
            onChange={(e) => setToHour(Number(e.target.value))}
            className="w-full h-11 accent-[var(--color-accent)] cursor-pointer"
          />
        </div>
      </div>

      <div className="h-[220px] w-full mt-6">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 20, left: 8 }}>
            <CartesianGrid stroke="var(--color-hairline)" vertical={false} />
            <XAxis
              dataKey="hour"
              tick={axis}
              tickLine={false}
              interval={2}
              tickFormatter={label}
            />
            <YAxis
              domain={domain}
              tick={axis}
              tickLine={false}
              width={68}
              // Plain rupees, not "₹2.4k". The whole axis spans ~₹200, so one decimal of
              // thousands rounds neighbouring ticks onto the same label — the axis rendered
              // "₹2.4k" twice and "₹2.3k" twice, which reads as a chart bug.
              tickFormatter={(v) => `₹${Math.round(v).toLocaleString("en-IN")}`}
            />
            <Tooltip
              cursor={{ fill: "var(--color-paper)" }}
              contentStyle={{
                border: "2px solid var(--color-ink)",
                borderRadius: 0,
                fontFamily: "var(--font-mono)",
                fontSize: 13,
              }}
              formatter={(value, _n, item) => [
                `${inr(value)} · ${item.payload.traffic_level.toLowerCase()} traffic ` +
                  `(${Math.round(item.payload.traffic_confidence * 100)}% confident)` +
                  (item.payload.inside ? "" : " · outside your window"),
                "expected cost",
              ]}
              labelFormatter={(h) => `depart ${label(h)}`}
            />
            <Bar dataKey="total_trip_cost" isAnimationActive={false}>
              {rows.map((row) => (
                <Cell
                  key={row.hour}
                  fill={row.isBest ? "var(--color-accent)" : "var(--color-ink)"}
                  // Dimmed rather than hidden: the hours you cannot use are still the context
                  // that makes the ones you can look good or bad.
                  fillOpacity={row.inside ? 1 : 0.18}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="font-mono text-[13px] leading-relaxed text-muted mt-3">
        {data.note} Bars are expected cost, weighted by the traffic classifier&apos;s
        probabilities. Axis is zoomed — across the full day the spread is{" "}
        {inr(data.max_saving)}.
      </p>
    </section>
  )
}
