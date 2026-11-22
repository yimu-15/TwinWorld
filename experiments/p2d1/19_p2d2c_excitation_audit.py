"""P2-D2C: Physics-identifiability / excitation feasibility audit.

Purpose: Compare fixed-parameter known/unknown physical laws under stronger or
weaker pulse excitation, without training or testing a learning agent.

Requires existing 16_p2d1b_probe_diversity.py in the same directory.
Run with Giant Python from the TwinWorld_P0 repository root.
"""
from pathlib import Path
from runpy import run_path
import json

import matplotlib.pyplot as plt
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results" / "p2d2c"
OUT.mkdir(parents=True, exist_ok=True)

sim = run_path(str(HERE / "16_p2d1b_probe_diversity.py"))
sample_scene = sim["sample_scene"]
make_model = sim["make_model"]
simulate = sim["simulate"]
base_pulse = sim["make_profiles"]()["pulse"]
MODES = tuple(sim["MODES"])

assert MODES == ("independent", "rigid", "elastic", "deadzone", "nonlinear")

SEED = 20261116
N_SCENES = 30
AMP_SCALES = (0.25, 0.5, 1.0, 1.5, 2.0)
NOISE_STD_M = 0.005
SAMPLES_PER_TRACE = 51
KNOWN_INDEX = (0, 1, 2)
UNKNOWN_INDEX = (3, 4)


def rms(y):
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.mean(y * y)))


def noise_reference(rng):
    # Reference: two independent 5-mm-noise recordings of the SAME trajectory.
    # This is NOT a classification threshold or an identifiability guarantee.
    n = 3000
    noise_difference = rng.normal(0, np.sqrt(2)*NOISE_STD_M,
                                  size=(n, SAMPLES_PER_TRACE))
    values = np.sqrt(np.mean(noise_difference**2, axis=1))
    return float(np.quantile(values, 0.95))


def main():
    rng = np.random.default_rng(SEED)
    floor = noise_reference(np.random.default_rng(SEED + 1))
    records = []
    scene_params = []
    max_passive_gap = 0.0
    max_a_gap = 0.0
    total_rollouts = 0

    for scene_id in range(N_SCENES):
        scene = sample_scene(rng)
        scene_params.append(scene)
        model = make_model(scene)

        for amp in AMP_SCALES:
            command = base_pulse * amp
            nominal_budget = float(np.sum(np.abs(command)) * sim["DT"])
            rollouts = [simulate(model, scene, mode, command) for mode in MODES]
            total_rollouts += len(MODES)

            passive = np.stack([r[0] for r in rollouts])
            active = np.stack([r[1] for r in rollouts])
            works = np.asarray([r[2] for r in rollouts], dtype=float)
            impulses = np.asarray([r[3] for r in rollouts], dtype=float)

            max_passive_gap = max(max_passive_gap,
                                  float(np.max(np.abs(passive - passive[:1]))))
            max_a_gap = max(max_a_gap, float(np.max(
                np.ptp(active[:, :, 0], axis=0))))
            if not np.isfinite(active).all():
                raise AssertionError(f"Nonfinite trajectory at scene={scene_id}, amp={amp}")

            b = active[:, :, 1]
            # Same-scene, same-parameter differences: privileged diagnostic ONLY.
            deadzone_independent = rms(b[3] - b[0])
            deadzone_elastic = rms(b[3] - b[2])
            nonlinear_elastic = rms(b[4] - b[2])
            nearest_deadzone = min(rms(b[3] - b[k]) for k in KNOWN_INDEX)
            nearest_nonlinear = min(rms(b[4] - b[k]) for k in KNOWN_INDEX)

            records.append({
                "scene_id": int(scene_id),
                "amplitude_scale": float(amp),
                "command_abs_accel_integral_m_per_s": nominal_budget,
                "B_rms_deadzone_vs_independent_m": deadzone_independent,
                "B_rms_deadzone_vs_elastic_m": deadzone_elastic,
                "B_rms_nonlinear_vs_elastic_m": nonlinear_elastic,
                "B_rms_deadzone_nearest_known_m": nearest_deadzone,
                "B_rms_nonlinear_nearest_known_m": nearest_nonlinear,
                "mean_abs_work_proxy_J": float(np.mean(works)),
                "mean_abs_impulse_Ns": float(np.mean(impulses)),
                "A_cross_mechanism_max_gap_m": float(np.max(np.ptp(active[:, :, 0], axis=0))),
            })

    if max_passive_gap >= 1e-8:
        raise AssertionError("Passive-equivalence audit failed")
    assert total_rollouts == N_SCENES * len(AMP_SCALES) * len(MODES)

    metrics = [
        "B_rms_deadzone_vs_independent_m",
        "B_rms_deadzone_vs_elastic_m",
        "B_rms_nonlinear_vs_elastic_m",
        "B_rms_deadzone_nearest_known_m",
        "B_rms_nonlinear_nearest_known_m",
    ]
    summaries = {}
    bootstrap_rng = np.random.default_rng(SEED + 2)
    for amp in AMP_SCALES:
        current = [r for r in records if r["amplitude_scale"] == amp]
        sums = {
            "n_scenes": len(current),
            "nominal_budget_m_per_s": current[0]["command_abs_accel_integral_m_per_s"],
            "mean_abs_work_proxy_J": float(np.mean([r["mean_abs_work_proxy_J"] for r in current])),
            "mean_abs_impulse_Ns": float(np.mean([r["mean_abs_impulse_Ns"] for r in current])),
            "mean_A_cross_mechanism_max_gap_m": float(np.mean([r["A_cross_mechanism_max_gap_m"] for r in current])),
        }
        for name in metrics:
            values = np.asarray([r[name] for r in current])
            boot = values[bootstrap_rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
            sums[name] = {
                "mean_m": float(np.mean(values)),
                "mean_scene_bootstrap_ci95_m": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
                "median_m": float(np.median(values)),
                "min_m": float(np.min(values)),
                "fraction_above_independent_noise_reference": float(np.mean(values > floor)),
            }
        summaries[str(amp)] = sums

    report = {
        "protocol": "P2-D2C physics-identifiability excitation audit (DEVELOPMENT ONLY)",
        "mujoco_version": mujoco.__version__,
        "random_seed": SEED,
        "n_independent_scenes": N_SCENES,
        "amplitude_scales": list(AMP_SCALES),
        "n_rollouts": total_rollouts,
        "max_passive_gap_m": max_passive_gap,
        "max_A_cross_mechanism_gap_m": max_a_gap,
        "sensor_noise_std_assumed_m": NOISE_STD_M,
        "noise_reference": {
            "definition": "95th percentile RMS difference between two independent noisy 51-frame recordings of the same B trajectory",
            "rms_difference_95th_percentile_m": floor,
            "not_a_rejection_threshold": True,
        },
        "summaries": summaries,
        "notes": [
            "All candidate outcomes and exact scene parameters are PRIVILEGED audit data.",
            "Models are compared at matching parameters, WITHOUT best-fit parameter refitting; the nearest-known gap is not a structural identifiability proof.",
            "The reference assumes independent 5-mm Gaussian positional noise; it ignores temporal correlation and model uncertainty.",
            "Stronger excitation changes actual work/impulse and may enlarge A-only leakage.",
            "No learned agent, no RGB video, no model revision, and no final confirmatory testing.",
        ],
    }

    (OUT / "p2d2c_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "p2d2c_scene_parameters_PRIVATE.json").write_text(
        json.dumps(scene_params, indent=2), encoding="utf-8")
    (OUT / "p2d2c_scene_metrics_audit_only.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")

    x = np.array(AMP_SCALES, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    for ax, metric, title in [
        (axes[0], "B_rms_deadzone_nearest_known_m", "Deadzone vs nearest known"),
        (axes[1], "B_rms_nonlinear_nearest_known_m", "Nonlinear vs nearest known"),
    ]:
        med = [summaries[str(a)][metric]["median_m"] * 1000 for a in AMP_SCALES]
        ax.plot(x, med, "o-", label="Median fixed-parameter B RMS")
        ax.axhline(floor * 1000, linestyle="--", color="gray", label="Independent-noise reference")
        ax.set_xlabel("Pulse amplitude multiplier")
        ax.set_ylabel("B trajectory RMS difference (mm)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "p2d2c_excitation_audit.png", bbox_inches="tight")
    plt.close(fig)

    print("P2-D2C physics excitation audit")
    print("scenes:", N_SCENES, "rollouts:", total_rollouts,
          "passive_max_gap:", max_passive_gap)
    print(f"independent-noise 95% RMS reference: {floor*1000:.2f} mm")
    print("amp  deadzone-nearest-med(mm)  nonlinear-nearest-med(mm)  mean_work(J)")
    for a in AMP_SCALES:
        item = summaries[str(a)]
        d = item["B_rms_deadzone_nearest_known_m"]["median_m"] * 1000
        n = item["B_rms_nonlinear_nearest_known_m"]["median_m"] * 1000
        print(f"{a:>3.2f} {d:>24.3f} {n:>26.3f} {item['mean_abs_work_proxy_J']:>13.4f}")
    print("P2-D2C ENGINEERING CHECKS PASS (NOT concept discovery)")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
