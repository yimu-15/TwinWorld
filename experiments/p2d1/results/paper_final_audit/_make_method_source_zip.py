"""Package method-source verification + per-case data into a root-level zip.

Batch 1: source code for physics / candidate competition / budget verification.
Batch 2: supplementary sources + per-case CSVs (incl. full competition-margin CSV
for #103 and budget-100/200 per-case CSV for #110). Read-only; no checkpoints.
"""
import zipfile
from pathlib import Path

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0")
ZIP = ROOT / "MethodSource_Evidence.zip"

files = [
    # Batch 1 — source verification (physics / candidate competition / budget)
    "scripts/twincore.py",
    "experiments/p2d1/16_p2d1b_probe_diversity.py",
    "experiments/p2d1/22_p3a1b_structural_revision.py",
    "experiments/p2d1/60_p3a7e1_action_family_pilot.py",
    "experiments/p2d1/80_p4m2c0_prior_mismatch_stress.py",
    "experiments/p2d1/103_p4g2c_fresh_frozen_vs_refit_confirmation.py",
    "experiments/p2d1/108_p4b3a_equal_model_evaluation_budget_pilot.py",
    # Batch 2 — formal experiment / Discussion supplements
    "experiments/p2d1/88_p4m2d1_fresh_action_prior_interaction.py",
    "experiments/p2d1/92_p4s1b_full_solver_budget_audit.py",
    "experiments/p2d1/97_p4b1b_fresh_classical_baseline_confirmation_v2.py",
    "experiments/p2d1/results/p4b1b_v2/p4b1b_v2_fresh_report.json",
    "experiments/p2d1/results/p4g1b/p4g1b_fresh_report.json",
    "experiments/p2d1/105_p4b2a_independent_identification_baseline_pilot.py",
    "experiments/p2d1/results/p4g2c/p4g2c_384case_results.csv",
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
