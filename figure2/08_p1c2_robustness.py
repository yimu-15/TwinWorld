
from pathlib import Path
from runpy import run_path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p1c2"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse the audited P1-C1 simulation
module = run_path(
    str(ROOT / "scripts" / "07_p1c_matched_motion.py")
)
simulate = module["simulate"]

ACCELS = [0.0, 0.1, 0.2, 0.4, 0.6]
NOISE_STD = [0.0, 0.001, 0.005, 0.01]
N_NOISE_REPEATS = 200
SEED = 20261113

rng = np.random.default_rng(SEED)


def prototype_accuracy(mu0, mu1, sigma):
    """
    Audit-only binary diagnostic using two known
    noiseless class prototypes.

    This is NOT a trained model or an OOD test.
    """
    if abs(mu0 - mu1) < 1e-12:
        return 0.5

    n = N_NOISE_REPEATS

    obs0 = mu0 + rng.normal(0, sigma, n)
    obs1 = mu1 + rng.normal(0, sigma, n)

    # Nearest noiseless class prototype
    pred0 = (
        np.abs(obs0 - mu1)
        < np.abs(obs0 - mu0)
    ).astype(int)

    pred1 = (
        np.abs(obs1 - mu1)
        < np.abs(obs1 - mu0)
    ).astype(int)

    return float(
        (np.sum(pred0 == 0) +
         np.sum(pred1 == 1)) / (2 * n)
    )


records = []

for accel in ACCELS:
    pi, ai, fi = simulate(False, accel)
    pc, ac, fc = simulate(True, accel)

    passive_gap = float(np.max(np.abs(
        pi[:, 1:] - pc[:, 1:]
    )))

    a_gap_max = float(np.max(np.abs(
        ai[:, 1] - ac[:, 1]
    )))

    b_gap_final = float(abs(
        ai[-1, 2] - ac[-1, 2]
    ))

    # Compare only the final observed position.
    # Do not expose force or constraint state.
    for sigma in NOISE_STD:
        acc_a = prototype_accuracy(
            ai[-1, 1], ac[-1, 1], sigma
        )
        acc_b = prototype_accuracy(
            ai[-1, 2], ac[-1, 2], sigma
        )

        records.append({
            "acceleration": accel,
            "noise_std_m": sigma,
            "passive_gap_max_m": passive_gap,
            "A_gap_max_m": a_gap_max,
            "B_gap_final_m": b_gap_final,
            "A_only_accuracy": acc_a,
            "B_only_accuracy": acc_b
        })

print(
    "accel | noise | A_gap | B_gap | "
    "A-only | B-only"
)

for r in records:
    print(
        f"{r['acceleration']:5.2f} | "
        f"{r['noise_std_m']:5.3f} | "
        f"{r['A_gap_max_m']:6.4f} | "
        f"{r['B_gap_final_m']:6.4f} | "
        f"{r['A_only_accuracy']:6.3f} | "
        f"{r['B_only_accuracy']:6.3f}"
    )

report = {
    "protocol": "P1-C2 controlled noise audit",
    "seed": SEED,
    "noise_repeats_per_class": N_NOISE_REPEATS,
    "important": (
        "Known-prototype diagnostic only; "
        "not learned classification performance. "
        "Noise repetitions are not independent "
        "physical environments."
    ),
    "records": records
}

(OUT / "p1c2_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

assert all(
    r["passive_gap_max_m"] < 1e-8
    for r in records
)

assert all(
    r["A_gap_max_m"] < 0.01
    for r in records
)

assert all(
    r["B_gap_final_m"] > 0.10
    for r in records
    if r["acceleration"] >= 0.4
)

print("\nP1-C2 diagnostic checks PASS")
print("Saved:", OUT / "p1c2_report.json")
