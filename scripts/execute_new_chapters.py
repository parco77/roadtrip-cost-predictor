"""
Execute the appended notebook chapters and merge their outputs back in.

WHY NOT JUST RUN THE WHOLE NOTEBOOK
-----------------------------------
Weeks 1-8 include a 60,000-epoch gradient descent written from scratch in pure NumPy, plus the
Week 3 learning-rate sweep. Re-executing all of it to render two new chapters wastes several
minutes and risks perturbing outputs that are already correct and already discussed in the
markdown around them.

So this extracts the cells from a marker onwards into a temporary notebook, executes just those
against a real Jupyter kernel in the project root, and copies the executed cells (outputs and
execution counts included) back into place. The chapters end up with genuine outputs, exactly as
if the whole notebook had been run.

Run:  python scripts/execute_new_chapters.py
      python scripts/execute_new_chapters.py --marker "<!-- WEEK4PLUS -->"
"""
import argparse
import os
import sys

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "RoadTripCost.ipynb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", default="<!-- WEEK9PLUS -->",
                    help="execute from the cell containing this marker onwards")
    ap.add_argument("--timeout", type=int, default=900, help="per-cell timeout, seconds")
    args = ap.parse_args()

    nb = nbformat.read(NB, as_version=4)
    start = next((i for i, c in enumerate(nb.cells) if args.marker in c.source), None)
    if start is None:
        sys.exit(f"marker not found in the notebook: {args.marker}")

    tail = nb.cells[start:]
    n_code = sum(1 for c in tail if c.cell_type == "code")
    print(f"executing {len(tail)} cells from index {start} ({n_code} code cells)")

    sub = nbformat.v4.new_notebook(cells=tail, metadata=nb.metadata)
    client = NotebookClient(sub, timeout=args.timeout, kernel_name="python3",
                            resources={"metadata": {"path": ROOT}},
                            allow_errors=False)
    client.execute()

    # Continue the execution counter from wherever the untouched cells left off, so the notebook
    # reads as one coherent run rather than restarting at 1 half way down.
    offset = max((c.get("execution_count") or 0
                  for c in nb.cells[:start] if c.cell_type == "code"), default=0)
    counter = offset
    for cell in sub.cells:
        if cell.cell_type == "code":
            counter += 1
            cell["execution_count"] = counter
            for out in cell.get("outputs", []):
                if out.get("output_type") == "execute_result":
                    out["execution_count"] = counter

    nb.cells = nb.cells[:start] + sub.cells
    nbformat.write(nb, NB)

    with_out = sum(1 for c in sub.cells if c.cell_type == "code" and c.get("outputs"))
    print(f"executed OK - {with_out}/{n_code} code cells produced output")
    print(f"execution counts {offset + 1}..{counter}")
    print(f"wrote {NB}")


if __name__ == "__main__":
    main()
