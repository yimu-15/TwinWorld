"""Organize figure evidence files into figure1..figure8 folders (project root).

Read-only COPY (shutil.copy2). Also generates the deterministic Figure 3 PCA
2D coordinates from P2-C0 B-only features. Reports any missing source files.
"""
import os, shutil
from pathlib import Path

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0")

FIG = {
    "figure1": [
        "data/train_public.npz", "data/val_public.npz", "data/test_public.npz",
        "data/train_private_labels.npz", "data/val_private_labels.npz", "data/test_private_labels.npz",
        "data/generation_config.json",
        "scripts/twincore.py", "scripts/01_generate.py", "scripts/02_validate.py",
        "scripts/03_baseline.py", "scripts/04_plot.py",
        "outputs/p0_baselines.json", "outputs/p0_twinworld_comparison.png",
        "outputs/p1b/p1b_report.json", "outputs/p1b/p1b_audit_only.npz", "outputs/p1b/p1b_comparison.png",
        "outputs/p1c/p1c_report.json", "outputs/p1c/p1c_raw_trajectory.npz",
        "outputs/p1c/p1c_raw_trajectory.csv", "outputs/p1c/p1c_matched_motion.png",
        "outputs/p2c0/p2c0_baselines.json",
        "experiments/p2d1/results/p4m2b2b/p4m2b2b_fresh_report.json",
        "experiments/p2d1/results/p4m2c2e/p4m2c2e_fresh_report.json",
        "experiments/p2d1/results/p4m2d1/p4m2d1_fresh_report.json",
        "experiments/p2d1/results/p4m2d2b/p4m2d2b_v2_fresh_report.json",
        "experiments/p2d1/results/paper_final_audit/PAPER_SOURCE_MANIFEST.json",
    ],
    "figure2": [
        "docs/PROTOCOL_V1.md",
        "scripts/twincore.py", "scripts/05_mujoco_smoke.py", "scripts/06_twinworld_mujoco.py",
        "scripts/07_p1c_matched_motion.py", "scripts/08_p1c2_robustness.py", "scripts/09_p1c3_unknown_parameters.py",
        "experiments/p2d1/16_p2d1b_probe_diversity.py",
        "experiments/p2d1/22_p3a1b_structural_revision.py",
        "experiments/p2d1/60_p3a7e1_action_family_pilot.py",
        "experiments/p2d1/98_p4g1a_independent_physics_generator_feasibility.py",
        "experiments/p2d1/74_p4m2a_generator_inference_fidelity.py",
        "experiments/p2d1/75_p4m2b0_physical_domain_fit_pilot.py",
        "experiments/p2d1/77_p4m2b2a_matched_domain_fit.py",
        "experiments/p2d1/79_p4m2b2b_independent_scene_confirmation.py",
        "experiments/p2d1/environment.txt", "requirements.txt",
        "data/generation_config.json",
    ],
    "figure3": [
        "outputs/p2c0/p2c0_baselines.json",
        "outputs/p2c0/p2c0_per_scene_predictions.csv",
        "outputs/p2c0/p2c0_B_only_minus_Passive_only_paired_bootstrap.json",
        "scripts/12_p2c_baselines.py",
        "outputs/p2b/test_observations.npz", "outputs/p2b/test_private.npz",
    ],
    "figure4": [
        "experiments/p2d1/results/p4m2b2b/p4m2b2b_fresh_report.json",
        "experiments/p2d1/results/p4m2c2e/p4m2c2e_fresh_report.json",
        "experiments/p2d1/77_p4m2b2a_matched_domain_fit.py",
        "experiments/p2d1/80_p4m2c0_prior_mismatch_stress.py",
        "experiments/p2d1/81_p4m2c1_prior_conflict_audit.py",
        "experiments/p2d1/results/p4m2c0/p4m2c0_dev_report.json",
        "experiments/p2d1/results/p4m2c1/p4m2c1_dev_report.json",
        "experiments/p2d1/results/p4m2c2a/p4m2c2a_dev_report.json",
        "experiments/p2d1/results/p4m2c2c/p4m2c2c_dev_report.json",
        "experiments/p2d1/results/p4m2c2d/p4m2c2d_dev_report.json",
        "experiments/p2d1/results/p4s1a/p4s1a_diagnostic_report.json",
        "experiments/p2d1/results/p4s1b/p4s1b_fresh_report.json",
        "experiments/p2d1/results/p4s1c/p4s1c_solver_closure_report.json",
    ],
    "figure5": [
        "experiments/p2d1/results/p4m2d2b/p4m2d2b_v2_fresh_report.json",
        "experiments/p2d1/results/p4m2d1/p4m2d1_fresh_report.json",
        "experiments/p2d1/90_p4m2d2b_amplitude_dose_response_v2.py",
        "experiments/p2d1/88_p4m2d1_fresh_action_prior_interaction.py",
    ],
    "figure6": [
        "experiments/p2d1/results/p4b2b/p4b2b_fresh_report.json",
        "experiments/p2d1/results/p4b3c/p4b3c_fresh_report.json",
        "experiments/p2d1/results/p4b2b/p4b2b_384case_results.csv",
        "experiments/p2d1/results/p4b3c/p4b3c_384case_results.csv",
        "experiments/p2d1/results/paper_final_audit/PAPER_COMPUTE_COST_AUDIT.md",
        "experiments/p2d1/106_p4b2b_fresh_classical_identification_comparison.py",
        "experiments/p2d1/110_p4b3c_fresh_equal_budget_confirmation.py",
        "experiments/p2d1/105_p4b2a_independent_identification_baseline_pilot.py",
        "experiments/p2d1/108_p4b3a_equal_model_evaluation_budget_pilot.py",
    ],
    "figure7": [
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
        "experiments/p2d1/98_p4g1a_independent_physics_generator_feasibility.py",
        "experiments/p2d1/results/paper_final_audit/PAPER_FIGURE7_MATERIALS.md",
    ],
    "figure8": [
        "experiments/p2d1/results/p4b1b_v2/p4b1b_v2_fresh_report.json",
        "experiments/p2d1/results/p4b1a/p4b1a_pilot_report.json",
        "experiments/p2d1/results/p4b2b/p4b2b_fresh_report.json",
        "experiments/p2d1/results/p4b2a/p4b2a_pilot_report.json",
        "experiments/p2d1/results/p4b2c/p4b2c_closure_report.json",
        "experiments/p2d1/results/p4b3c/p4b3c_fresh_report.json",
        "experiments/p2d1/results/p4b3c/p4b3c_384case_results.csv",
        "experiments/p2d1/results/p4b3a/p4b3a_pilot_report.json",
        "experiments/p2d1/results/p4b3b/p4b3b_closure_report.json",
        "experiments/p2d1/96_p4b1a_frozen_classical_model_selection_baselines.py",
        "experiments/p2d1/97_p4b1b_fresh_classical_baseline_confirmation_v2.py",
        "experiments/p2d1/105_p4b2a_independent_identification_baseline_pilot.py",
        "experiments/p2d1/106_p4b2b_fresh_classical_identification_comparison.py",
        "experiments/p2d1/107_p4b2c_fresh_baseline_closure_audit.py",
        "experiments/p2d1/108_p4b3a_equal_model_evaluation_budget_pilot.py",
        "experiments/p2d1/109_p4b3b_equal_budget_closure_audit.py",
        "experiments/p2d1/110_p4b3c_fresh_equal_budget_confirmation.py",
        "experiments/p2d1/results/paper_final_audit/PAPER_COMPUTE_COST_AUDIT.md",
    ],
}

missing = []
for folder, files in FIG.items():
    d = ROOT / folder
    d.mkdir(parents=True, exist_ok=True)
    for rel in files:
        src = ROOT / rel
        if not src.exists():
            missing.append(f"{folder} <- {rel}")
            continue
        shutil.copy2(src, d / Path(rel).name)
    print(f"{folder}: {len(files)} listed")

print("\n=== MISSING (not copied) ===")
for m in missing:
    print("  ", m)
print("total missing:", len(missing))
