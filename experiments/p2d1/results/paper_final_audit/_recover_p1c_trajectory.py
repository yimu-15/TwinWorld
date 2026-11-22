"""Recover P1-C matched-motion per-timestep trajectories (deterministic replay).

The historical script 07_p1c_matched_motion.py did NOT persist raw trajectories
(only summary JSON + PNG). This script re-runs the SAME deterministic MuJoCo
rollout (fixed XML, V0=0.15, TARGET_ACCEL=0.4, KP/KD) and writes the raw arrays
to NEW files, then verifies the recovered summary values against the saved
p1c_report.json (must match exactly; no randomness involved).

This is data recovery, not a new experiment. Original files are not modified.
"""
import json
import runpy
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0")
SRC = ROOT / "scripts" / "07_p1c_matched_motion.py"
OUT = ROOT / "outputs" / "p1c"
REPORT = json.loads((OUT / "p1c_report.json").read_text(encoding="utf-8"))

mod = runpy.run_path(str(SRC))
simulate = mod["simulate"]
TARGET_ACCEL = mod["TARGET_ACCEL"]

# coupled=False -> independent world; coupled=True -> coupled world
pi, ai, fi = simulate(False, TARGET_ACCEL)
pc, ac, fc = simulate(True, TARGET_ACCEL)

# Verify recovered summary matches saved report exactly
r = REPORT["results"]
def close(a, b, tol=1e-12):
    return bool(abs(a - b) < tol)

checks = {
    "passive_gap_max_m": close(np.max(np.abs(pi[:, 1:] - pc[:, 1:])), r["passive_gap_max_m"]),
    "active_A_trajectory_gap_max_m": close(np.max(np.abs(ai[:, 1] - ac[:, 1])), r["active_A_trajectory_gap_max_m"]),
    "active_B_final_gap_m": close(abs(ai[-1, 2] - ac[-1, 2]), r["active_B_final_gap_m"]),
    "control_force_gap_max_N": close(np.max(np.abs(fi - fc)), r["control_force_gap_max_N"]),
}
print("VERIFY", json.dumps(checks, indent=2))
assert all(checks.values()), "recovered data does NOT match saved report"

# Save raw arrays (NEW files, no overwrite)
np.savez_compressed(
    OUT / "p1c_raw_trajectory.npz",
    passive_independent=pi, active_independent=ai, force_independent=fi,
    passive_coupled=pc, active_coupled=ac, force_coupled=fc,
)

# Long-form CSV: world, phase, step, time_s, x_A_m, x_B_m, force_N
import csv
rows = []
for world, (p, a, f) in [("independent", (pi, ai, fi)), ("coupled", (pc, ac, fc))]:
    for i in range(p.shape[0]):
        rows.append([world, "passive", i, p[i, 0], p[i, 1], p[i, 2], 0.0])
    for i in range(a.shape[0]):
        rows.append([world, "active", i, a[i, 0], a[i, 1], a[i, 2], f[i]])
with open(OUT / "p1c_raw_trajectory.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["world", "phase", "step", "time_s", "x_A_m", "x_B_m", "force_N"])
    w.writerows(rows)

print("SAVED npz + csv ->", OUT)
print("rows:", len(rows))
print("shapes: pi", pi.shape, "ai", ai.shape, "fi", fi.shape,
      "pc", pc.shape, "ac", ac.shape, "fc", fc.shape)
