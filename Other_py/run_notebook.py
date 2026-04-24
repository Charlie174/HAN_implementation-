"""
Cell-by-cell notebook runner.
Executes all code cells from train_careai_march.ipynb sequentially in a shared namespace.
Stops on first error and prints full traceback.

Run from project root:
    python Other_py/run_notebook.py
"""

import json
import os
import sys
import traceback

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(BASE, "notebooks", "train_careai_march.ipynb")

# Start in the notebooks/ directory — this mirrors VS Code's working directory
# when it runs the notebook. Cell 2 does os.path.abspath('..') to find project root,
# which only works correctly when cwd is notebooks/.
NOTEBOOKS_DIR = os.path.join(BASE, "notebooks")
os.chdir(NOTEBOOKS_DIR)
sys.path.insert(0, BASE)

# ── Load notebook ─────────────────────────────────────────────────────────────
with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [
    (i, cell)
    for i, cell in enumerate(nb["cells"])
    if cell["cell_type"] == "code"
]

print(f"Loaded notebook: {NOTEBOOK}")
print(f"Total code cells: {len(code_cells)}")
print("=" * 60)

# ── Shared namespace ──────────────────────────────────────────────────────────
g = {"__name__": "__main__"}

# ── Execute cell by cell ──────────────────────────────────────────────────────
for seq, (i, cell) in enumerate(code_cells):
    source = cell["source"]
    if isinstance(source, list):
        src = "".join(source)
    else:
        src = source

    src = src.strip()
    if not src:
        print(f"[Cell {i:2d}] (empty — skipped)")
        continue

    preview = src[:80].replace("\n", " ")
    print(f"\n[Cell {i:2d}] Running: {preview}...")

    try:
        exec(compile(src, f"<cell {i}>", "exec"), g)
        print(f"[Cell {i:2d}] OK")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR in Cell {i} (notebook cell #{seq+1}/{len(code_cells)})")
        print(f"{'='*60}")
        print(f"Source snippet:\n{src[:400]}")
        print(f"\nException: {type(e).__name__}: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        print(f"\nStopped at cell {i}.")
        sys.exit(1)

print("\n" + "=" * 60)
print("All cells completed successfully.")
