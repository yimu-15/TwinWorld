"""Package Figure 8 evidence files into Figure8_Evidence.zip (project root).

Covers the full comparison: model-selection criteria (#96-97), classical
estimators (#105-107), and strict budget control (#108-110), plus per-case
metrics, algorithm source, environment, and compute-cost audit. No checkpoints.
"""
import zipfile
from pathlib import Path

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0")
ZIP = ROOT / "Figure8_Evidence.zip"

files = [
    # common: compute-cost audit + environment
    "experiments/p2d1/results/paper_final_audit/PAPER_COMPUTE_COST_AUDIT.md",
    "experiments/p2d1/environment.txt",
    # #96-97 P4-B1A/B1B (model-selection criteria PEM/AIC/BIC)
    "experiments/p2d1/96_p4b1a_frozen_classical_model_selection_baselines.py",
    "experiments/p2d1/97_p4b1b_fresh_classical_baseline_confirmation.py",
    "experiments/p2d1/97_p4b1b_fresh_classical_baseline_confirmation_v2.py",
    "experiments/p2d1/results/p4b1a/p4b1a_pilot_report.json",
    "experiments/p2d1/results/p4b1b/p4b1b_source_manifest.json",
    "experiments/p2d1/results/p4b1b_v2/p4b1b_v2_fresh_report.json",
    "experiments/p2d1/results/p4b1b_v2/p4b1b_v2_source_manifest.json",
    # #105-107 P4-B2A/B2B/B2C (EQERR/TRF/SLSQP)
    "experiments/p2d1/105_p4b2a_independent_identification_baseline_pilot.py",
    "experiments/p2d1/106_p4b2b_fresh_classical_identification_comparison.py",
    "experiments/p2d1/107_p4b2c_fresh_baseline_closure_audit.py",
    "experiments/p2d1/results/p4b2a/p4b2a_pilot_report.json",
    "experiments/p2d1/results/p4b2b/p4b2b_fresh_report.json",
    "experiments/p2d1/results/p4b2b/p4b2b_source_manifest.json",
    "experiments/p2d1/results/p4b2b/p4b2b_384case_results.csv",
    "experiments/p2d1/results/p4b2c/p4b2c_closure_report.json",
    # #108-110 P4-B3A/B3B/B3C (Cap 100/200)
    "experiments/p2d1/108_p4b3a_equal_model_evaluation_budget_pilot.py",
    "experiments/p2d1/109_p4b3b_equal_budget_closure_audit.py",
    "experiments/p2d1/110_p4b3c_fresh_equal_budget_confirmation.py",
    "experiments/p2d1/results/p4b3a/p4b3a_pilot_report.json",
    "experiments/p2d1/results/p4b3b/p4b3b_closure_report.json",
    "experiments/p2d1/results/p4b3c/p4b3c_fresh_report.json",
    "experiments/p2d1/results/p4b3c/p4b3c_source_manifest.json",
    "experiments/p2d1/results/p4b3c/p4b3c_384case_results.csv",
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
    print("MISSING:")
    for m in missing:
        print("  ", m)
with zipfile.ZipFile(ZIP) as zf:
    print("contents:")
    for info in zf.infolist():
        print(f"  {info.file_size:>10}  {info.filename}")
