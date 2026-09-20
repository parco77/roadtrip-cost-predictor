import { useMemo } from "react"
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

const axis = {
  stroke: "var(--color-muted)",
  fontSize: 13, // matches the page-wide 13px floor for small text
  fontFamily: "var(--font-mono)",
}

/**
 * Loss vs epoch for the from-scratch gradient descent, log-log so the steep early drop and the
 * long flat tail are both readable on one pair of axes.
 *
 * `progress` (0-1) draws the curve incrementally when the parent pins and scrubs it. The axis
 * domains are pinned to the FULL dataset regardless of progress — otherwise the axes rescale on
 * every frame and the line appears to stay still while the numbers slide around underneath it.
 */
export default function LossCurve({ curve, progress = 1 }) {
  const { shown, head, xDomain, yDomain } = useMemo(() => {
    if (!curve?.length) return { shown: [], head: null, xDomain: [1, 1], yDomain: [1, 1] }

    const count = Math.max(2, Math.round(curve.length * Math.min(1, Math.max(0, progress))))
    const slice = curve.slice(0, count)
    const epochs = curve.map((d) => d.epoch).filter((e) => e > 0)
    const losses = curve.map((d) => d.loss)

    return {
      shown: slice,
      head: slice[slice.length - 1],
      xDomain: [Math.min(...epochs), Math.max(...epochs)],
      yDomain: [Math.min(...losses), Math.max(...losses)],
    }
  }, [curve, progress])

  if (!curve?.length) return null

  return (
    <figure>
      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={shown} margin={{ top: 8, right: 16, bottom: 26, left: 8 }}>
            <CartesianGrid stroke="var(--color-hairline)" vertical={false} />
            <XAxis
              dataKey="epoch"
              scale="log"
              type="number"
              domain={xDomain}
              allowDataOverflow
              tick={axis}
              tickLine={false}
              label={{
                value: "EPOCH (LOG)",
                position: "insideBottom",
                offset: -14,
                style: { ...axis, letterSpacing: "0.18em" },
              }}
            />
            <YAxis
              scale="log"
              type="number"
              domain={yDomain}
              allowDataOverflow
              tick={axis}
              tickLine={false}
              width={64}
              tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0))}
            />
            <Tooltip
              contentStyle={{
                border: "1px solid var(--color-ink)",
                borderRadius: 0,
                fontFamily: "var(--font-mono)",
                fontSize: 13,
              }}
              formatter={(value) => [Math.round(value).toLocaleString("en-IN"), "MSE loss"]}
              labelFormatter={(label) => `epoch ${Number(label).toLocaleString("en-IN")}`}
            />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="var(--color-accent)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false} // the scrub drives it; Recharts' own tween would fight it
            />
            {head && progress < 1 && (
              <ReferenceDot
                x={head.epoch}
                y={head.loss}
                r={4}
                fill="var(--color-accent)"
                stroke="var(--color-page)"
                strokeWidth={2}
                isFront
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="label-mono mt-3">
        MSE loss vs epoch — gradient descent written from scratch, no sklearn
      </figcaption>
    </figure>
  )
}
