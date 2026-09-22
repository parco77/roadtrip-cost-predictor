"""
Build Project_Report.pdf - the complete project report, organised by topic.

HOW THE DOCUMENT IS PUT TOGETHER
--------------------------------
The report is assembled from ten HTML sections and then printed to PDF by headless Chrome,
which is used because it is the only PDF engine available here that honours real print CSS
(@page size and margins, page-break-before, page-break-inside: avoid).

Two of the ten sections are NOT stored as files - they are generated:

    report/01_cover_and_overview.html   cover, contents, Part 1 (the model catalogue)
    report/02_dataset.html              Parts 2-4   data, exploration, preprocessing
    report/03_formula_and_data.html     Parts 5-6   formula recovery, training data + distances
    report/04_regression.html           Part 7      regression and gradient descent
    report/05_features_and_ridge.html   Parts 8-9   interactions, regularisation
    <generated>                         scripts/build_report_week10.py  -> Part 10, the models
    report/06_classification.html       Part 11     the cost band
    <generated>                         scripts/build_report_tables.py  -> Part 12, evaluation
    report/07_application.html          Parts 13-14 the application, bugs and limitations
    report/08_glossary_and_repro.html   Parts 15-16 glossary, reproduction (closes body/html)

Parts 10 and 12 are generated because between them they quote roughly two hundred measured
numbers. Typed by hand they would start with a transcription error and go stale on the next
retrain; generated from the same JSON that TASK5.md and the notebook read, the four artefacts
cannot disagree with each other.

The order is by SUBJECT, not by the order the work happened in. Each part covers one topic and
can be read on its own; Part 10 is the catalogue of every model and what it does.

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
# Document order IS this list, so a part's position in the report is changed here.
SECTIONS = [
    ("01_cover_and_overview.html", None),     # cover, contents, Part 1
    ("02_dataset.html", None),                # Parts 2-4
    ("03_formula_and_data.html", None),       # Parts 5-6
    ("04_regression.html", None),             # Part 7
    ("05_features_and_ridge.html", None),     # Parts 8-9
    (None, "build_report_week10.py"),         # Part 10 - the model pipeline
    ("06_classification.html", None),         # Part 11
    (None, "build_report_tables.py"),         # Part 12 - evaluation
    ("07_application.html", None),            # Parts 13-14
    ("08_glossary_and_repro.html", None),     # Parts 15-16
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
