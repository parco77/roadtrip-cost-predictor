"""
Build Project_Report.pdf - the complete project report, Weeks 1 to 10.

HOW THE DOCUMENT IS PUT TOGETHER
--------------------------------
The report is assembled from nine HTML sections and then printed to PDF by headless Chrome,
which is used because it is the only PDF engine available here that honours real print CSS
(@page size and margins, page-break-before, page-break-inside: avoid).

Two of the nine sections are NOT stored as files - they are generated:

    report/01_cover_and_overview.html      hand-written
    report/02_dataset_week1_week2.html     hand-written
    report/03_week3_regression.html        hand-written
    report/04_week4_to_week8.html          hand-written
    <generated>                            scripts/build_report_tables.py   -> Part 11, Week 9
    <generated>                            scripts/build_report_week10.py   -> Part 12, Week 10
    report/07_application_and_bugs.html    hand-written
    report/08_viva_questions.html          hand-written
    report/09_glossary_and_repro.html      hand-written  (closes <body>/<html>)

Parts 11 and 12 are generated because between them they quote roughly two hundred measured
numbers. Typed by hand they would start with a transcription error and go stale on the next
retrain; generated from the same JSON that TASK5.md and the notebook read, the four artefacts
cannot disagree with each other.

The stylesheet lives at the top of section 01 and therefore applies to the whole document.

Run:  python scripts/build_report_pdf.py
      python scripts/build_report_pdf.py --keep-html    (also leave the merged HTML in place)
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "report")
OUT_PDF = os.path.join(ROOT, "Project_Report.pdf")
MERGED = os.path.join(REPORT_DIR, "_merged.html")

# Where Chrome usually is on Windows. Add to this list rather than editing the lookup below.
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]

# (filename, generator script) - exactly one of the two is set per section.
SECTIONS = [
    ("01_cover_and_overview.html", None),
    ("02_dataset_week1_week2.html", None),
    ("03_week3_regression.html", None),
    ("04_week4_to_week8.html", None),
    (None, "build_report_tables.py"),     # Part 11 - Week 9 / Task 5
    (None, "build_report_week10.py"),     # Part 12 - Week 10
    ("07_application_and_bugs.html", None),
    ("08_viva_questions.html", None),
    ("09_glossary_and_repro.html", None),
]


def find_chrome():
    for path in CHROME_CANDIDATES:
        if path and os.path.exists(path):
            return path
    found = shutil.which("chrome") or shutil.which("chromium") or shutil.which("google-chrome")
    if found:
        return found
    sys.exit("Chrome/Chromium not found. Add its path to CHROME_CANDIDATES in this script.")


def build_html():
    parts = []
    for filename, generator in SECTIONS:
        if generator:
            script = os.path.join(ROOT, "scripts", generator)
            result = subprocess.run([sys.executable, script], capture_output=True, text=True,
                                    encoding="utf-8", cwd=ROOT)
            if result.returncode != 0:
                sys.exit(f"{generator} failed:\n{result.stderr}")
            parts.append(result.stdout)
            print(f"  generated  {generator}")
        else:
            path = os.path.join(REPORT_DIR, filename)
            if not os.path.exists(path):
                sys.exit(f"missing section: {path}")
            parts.append(open(path, encoding="utf-8").read())
            print(f"  read       {filename}")

    html = "\n".join(parts)
    with open(MERGED, "w", encoding="utf-8") as fh:
        fh.write(html)
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-html", action="store_true",
                    help="leave report/_merged.html on disk for inspection")
    args = ap.parse_args()

    print("assembling sections:")
    html = build_html()
    n_parts = html.count('<h1 class="part"')
    print(f"\nmerged: {len(html):,} chars, {n_parts} top-level parts")

    chrome = find_chrome()
    print(f"printing with: {chrome}")
    # Chrome needs native absolute paths here; forward slashes are accepted on Windows.
    subprocess.run([
        chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={OUT_PDF.replace(os.sep, '/')}",
        MERGED.replace(os.sep, "/"),
    ], capture_output=True, text=True)

    if not os.path.exists(OUT_PDF):
        sys.exit("Chrome did not produce a PDF. Try running it once without --headless to check.")

    size_mb = os.path.getsize(OUT_PDF) / 1e6
    pages = "?"
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(OUT_PDF).pages)
    except Exception:
        pass

    if not args.keep_html and os.path.exists(MERGED):
        os.remove(MERGED)

    print(f"\nwrote {OUT_PDF}")
    print(f"  {pages} pages, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
