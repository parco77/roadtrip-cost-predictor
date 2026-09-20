const LINKS = [
  { href: "#distances", label: "Distances" },
  { href: "#estimate", label: "Estimate" },
  { href: "#drivers", label: "Drivers" },
  { href: "#model", label: "Model" },
  { href: "#data", label: "Data" },
]

export default function Nav() {
  return (
    <nav
      aria-label="Primary"
      className="absolute top-0 inset-x-0 z-20 flex flex-col items-center gap-4 pt-7 px-6"
    >
      {/* min-h-11 (44px) on every link: the text is only 11-24px tall, which fails the
          44x44 minimum touch target on mobile. Padding does the work, not font size. */}
      <a
        href="#top"
        className="inline-flex items-center min-h-11 px-2 font-display italic text-3xl
                   md:text-4xl text-white drop-shadow-[0_1px_6px_rgba(0,0,0,0.5)]"
      >
        RoadTrip
      </a>
      <ul className="flex flex-wrap justify-center gap-x-4 sm:gap-x-7">
        {LINKS.map(({ href, label }) => (
          <li key={href}>
            <a
              href={href}
              className="inline-flex items-center min-h-11 px-2 font-mono text-[13px] uppercase
                         tracking-[0.18em] text-white/90 hover:text-white
                         transition-colors duration-200 drop-shadow-[0_1px_5px_rgba(0,0,0,0.6)]"
            >
              {label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  )
}
