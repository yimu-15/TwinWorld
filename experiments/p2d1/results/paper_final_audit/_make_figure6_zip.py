"""Package Figure 6 evidence files into Figure6_Evidence.zip (project root).

Read-only. Includes #106/#110 reports, per-case CSVs (both Cap 100 & 200 for #110),
compute-cost audit, and algorithm/budget source code. No checkpoints included.
"""
import zipfile
from pathlib import Path

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0")
ZIP = ROOT / "Figure6_Evidence.zip"

files = [
    "experiments/p2d1/results/paper_final_audit/PAPER_COMPUTE_COST_AUDIT.md",
    # #106 P4-B2B
    "experiments/p2d1/results/p4b2b/p4b2b_fresh_report.json",
    "experiments/p2d1/results/p4b2b/p4b2b_source_manifest.json",
    "experiments/p2d1/results/p4b2b/p4b2b_384case_results.csv",
    "experiments/p2d1/106_p4b2b_fresh_classical_identification_comparison.py",
    "experiments/p2d1/105_p4b2a_independent_identification_baseline_pilot.py",
    # #110 P4-B3C
    "experiments/p2d1/results/p4b3c/p4b3c_fresh_report.json",
    "experiments/p2d1/results/p4b3c/p4b3c_source_manifest.json",
    "experiments/p2d1/results/p4b3c/p4b3c_384case_results.csv",
    "experiments/p2d1/110_p4b3c_fresh_equal_budget_confirmation.py",
    "experiments/p2d1/108_p4b3a_equal_model_evaluation_budget_pilot.py",
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
with zipfile.ZipFile(ZIP) as zf:
    print("contents:")
    for info in zf.infolist():
        print(f"  {info.file_size:>10}  {info.filename}")
