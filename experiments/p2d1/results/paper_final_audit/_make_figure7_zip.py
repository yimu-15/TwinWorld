"""Package the necessary Figure 7 evidence files into Figure7_Evidence.zip.

Created at the project root D:\\OnShouldersGiants\\TwinWorld_P0.
Read-only: only reads original files and writes the new zip. No source modified.
"""
import os, zipfile
from pathlib import Path

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0")
ZIP = ROOT / "Figure7_Evidence.zip"

# (absolute path, arcname relative to ROOT)
files = [
    "experiments/p2d1/results/paper_final_audit/PAPER_FIGURE7_MATERIALS.md",
    "experiments/p2d1/98_p4g1a_independent_physics_generator_feasibility.py",
    "experiments/p2d1/results/p4g1a/p4g1a_feasibility_report.json",
    "experiments/p2d1/results/p4g1b/p4g1b_fresh_report.json",
    "experiments/p2d1/results/p4g1b/p4g1b_source_manifest.json",
    "experiments/p2d1/results/p4g1c/p4g1c_closure_report.json",
    "experiments/p2d1/results/p4g2a/p4g2a_diagnostic_report.json",
    "experiments/p2d1/results/p4g2a/p4g2a_source_manifest.json",
    "experiments/p2d1/results/p4g2b/p4g2b_closure_report.json",
    "experiments/p2d1/results/p4g2b/p4g2b_source_manifest.json",
    "experiments/p2d1/results/p4g2c/p4g2c_fresh_report.json",
    "experiments/p2d1/results/p4g2c/p4g2c_source_manifest.json",
    "experiments/p2d1/results/p4g2c/p4g2c_384case_results.csv",
    "experiments/p2d1/results/p4g2d/p4g2d_closure_report.json",
]

missing = []
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel in files:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        zf.write(p, arcname=rel)
        print("added", rel)

print("ZIP:", ZIP)
print("total added:", len(files) - len(missing))
if missing:
    print("MISSING (not packaged):")
    for m in missing:
        print("  ", m)

# report contents
with zipfile.ZipFile(ZIP) as zf:
    for info in zf.infolist():
        print(f"  {info.file_size:>12}  {info.filename}")
