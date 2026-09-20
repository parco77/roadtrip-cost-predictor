/**
 * The reference layout's signature move: a torn-paper edge where the photographic hero
 * meets the page. Inline SVG with preserveAspectRatio="none" so it stretches cleanly
 * from 375px to 1440px+ — a raster PNG would band or blur at the extremes.
 */
export default function TornEdge({ className = "", fill = "var(--color-page)", flip = false }) {
  return (
    <svg
      viewBox="0 0 1440 70"
      preserveAspectRatio="none"
      aria-hidden="true"
      focusable="false"
      className={`block w-full h-[42px] md:h-[70px] ${flip ? "rotate-180" : ""} ${className}`}
    >
      <path
        fill={fill}
        d="M0,70 L0,44 L24,49 L52,38 L83,46 L119,33 L148,44 L186,30 L214,41 L252,27
           L288,39 L317,25 L356,37 L389,22 L423,35 L461,24 L497,38 L531,26 L570,40
           L604,28 L642,42 L677,30 L713,45 L749,32 L788,46 L822,34 L859,48 L895,36
           L932,50 L968,38 L1003,52 L1041,40 L1076,54 L1113,42 L1149,56 L1186,44
           L1221,58 L1258,46 L1294,60 L1330,48 L1367,62 L1404,50 L1440,64 L1440,70 Z"
      />
    </svg>
  )
}
