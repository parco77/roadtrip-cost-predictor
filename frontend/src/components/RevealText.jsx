/**
 * Word-by-word reveal, without GSAP's SplitText.
 *
 * SplitText is a Club GreenSock (paid) plugin, so this project does not use it. The effect it
 * provides here is simple enough to build directly: wrap each word in an overflow-hidden mask
 * and translate the inner span up from below. Animate `.rt-word` and the mask does the rest.
 *
 * Rendering the words as real text (rather than injecting them from JS after mount) means the
 * heading is in the DOM for screen readers and crawlers even if no animation ever runs.
 * aria-label carries the unbroken string so assistive tech does not read it word-by-word.
 */
export default function RevealText({ text, as: Tag = "span", className = "", stagger }) {
  const words = String(text).split(" ")

  return (
    <Tag className={className} aria-label={text}>
      {words.map((word, i) => (
        <span
          key={`${word}-${i}`}
          aria-hidden="true"
          className="rt-mask inline-block overflow-hidden align-bottom"
          // pb/-mb pair: descenders (g, y, p) get clipped by overflow-hidden without it.
          style={{ paddingBottom: "0.12em", marginBottom: "-0.12em" }}
        >
          <span className="rt-word inline-block" data-index={i} data-stagger={stagger}>
            {word}
            {i < words.length - 1 ? " " : ""}
          </span>
        </span>
      ))}
    </Tag>
  )
}
