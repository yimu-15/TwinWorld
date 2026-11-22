
"""
P3-A5A: Oracle identifiability signal audit.

Read-only re-simulation of P3-A4D scene banks.

Compare noiseless B trajectories generated under:
  H0: independent
  H1: deadzone

The counterfactual true mechanisms are used ONLY
for retrospective physical feasibility analysis.

Not an online action-selection algorithm.
No refitting, no new scenes, no threshold tuning.
No output files written.
"""

from pathlib import Path
from runpy import run_path
from collections import Counter
import argparse
import json

import numpy as np


HERE = Path(__file__).resolve().parent

AUDIT_PATH = (
    HERE / "results" / "p3a4d"
    / "p3a4d_stress_audit_PRIVATE.json"
)

REPORT_PATH = (
    HERE / "results" / "p3a4d"
    / "p3a4d_stress_report.json"
)

NOISE_SIGMA_M = 0.005

BANDS = (
    "original",
    "high_in_range",
    "above_original_range",
)

E3_ACTIONS = (
    "weak",
    "strong",
    "sweep",
)


def make_commands(profiles):
    return {
        "e1": profiles["pulse"] * 0.25,
        "e2": profiles["pulse"] * 0.50,

        "weak": profiles["pulse"] * 1.50,
        "strong": profiles["pulse"] * 2.25,
        "sweep": profiles["sweep"] * 2.25,

        "e4": profiles["pulse"] * 2.75,
    }


def paired_rollout(sim, model, scene, command):
    """
    PRIVATE oracle counterfactual simulation.

    Same initial scene and same commanded action.
    Different hidden physical force laws.

    B separation is valid for the independent-vs-
    deadzone comparison because independent B
    has no A-to-B interaction force.
    """
    _, independent, _, _ = sim["simulate"](
        model,
        scene,
        "independent",
        command,
    )

    _, deadzone, work, impulse = sim["simulate"](
        model,
        scene,
        "deadzone",
        command,
    )

    independent = np.asarray(
        independent, dtype=float
    )

    deadzone = np.asarray(
        deadzone, dtype=float
    )

    assert independent.shape == deadzone.shape
    assert independent.ndim == 2
    assert independent.shape[1] == 2

    difference_b = (
        deadzone[:, 1]
        - independent[:, 1]
    )

    difference_a = (
        deadzone[:, 0]
        - independent[:, 0]
    )

    b_rms = float(np.sqrt(
        np.mean(difference_b ** 2)
    ))

    a_rms = float(np.sqrt(
        np.mean(difference_a ** 2)
    ))

    return {
        "B_signal_rms_m": b_rms,
        "B_signal_over_noise":
            b_rms / NOISE_SIGMA_M,
        "A_signal_rms_m": a_rms,
        "B_peak_difference_m": float(
            np.max(np.abs(difference_b))
        ),
        "B_difference_vector": difference_b,
        "true_work_proxy_J": float(work),
        "true_impulse_Ns": float(impulse),
    }


def combine_early(e1, e2):
    x = np.concatenate([
        e1["B_difference_vector"],
        e2["B_difference_vector"],
    ])

    rms = float(np.sqrt(np.mean(x ** 2)))

    return {
        "B_signal_rms_m": rms,
        "B_signal_over_noise":
            rms / NOISE_SIGMA_M,
    }


def median_metric(rows, key):
    if not rows:
        return None

    return float(np.median([
        r[key] for r in rows
    ]))


def show_group(name, rows):
    if not rows:
        return

    print("\nGROUP:", name, "| N =", len(rows))

    for action in (
        "early",
        "weak",
        "strong",
        "sweep",
        "e4",
    ):
        signals = [
            r["signals"][action][
                "B_signal_rms_m"
            ] * 1000.0
            for r in rows
        ]

        normalized = [
            r["signals"][action][
                "B_signal_over_noise"
            ]
            for r in rows
        ]

        print(
            f"  {action:7s}",
            f"B median={np.median(signals):8.3f}mm",
            f"S median={np.median(normalized):7.3f}",
            f"S>1: {sum(v > 1 for v in normalized)}/{len(rows)}",
        )

    for action in E3_ACTIONS:
        costs = [
            r["signals"][action][
                "true_work_proxy_J"
            ]
            for r in rows
        ]

        print(
            f"  E3 {action:6s} "
            f"oracle mean work={np.mean(costs):.4f}J"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    audit = json.loads(
        AUDIT_PATH.read_text(encoding="utf-8")
    )

    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    fullstep = run_path(str(
        HERE / "33_p3a4e_fullstep_activation_audit.py"
    ))

    profiles = sim["make_profiles"]()
    commands = make_commands(profiles)

    n = (
        2 if args.smoke
        else int(report["scenes_per_band"])
    )

    reference_scenes = audit[
        "original"
    ]["scene_parameters_PRIVATE"]

    for band in BANDS:

        content = audit[band]

        scenes = content[
            "scene_parameters_PRIVATE"
        ][:n]

        deadzone_rows = {
            int(r["scene_id"]): r
            for r in content["trajectories"]
            if r["truth_PRIVATE_EVAL_ONLY"]
            == "deadzone"
        }

        activation = content[
            "activation_by_scene_PRIVATE"
        ]

        records = []

        for scene_id, scene in enumerate(scenes):

            # Preserve paired scene identity.
            reference = reference_scenes[scene_id]

            for key in reference:
                if key != "gap":
                    assert scene[key] == reference[key]

            model = sim["make_model"](scene)

            signals = {
                action: paired_rollout(
                    sim, model, scene, command
                )
                for action, command
                in commands.items()
            }

            signals["early"] = combine_early(
                signals["e1"],
                signals["e2"],
            )

            # Full-step truth force audit.
            e1_force = fullstep[
                "truth_force_rollout"
            ](
                sim,
                model,
                scene,
                commands["e1"],
            )

            e2_force = fullstep[
                "truth_force_rollout"
            ](
                sim,
                model,
                scene,
                commands["e2"],
            )

            # Cross-check the existing saved
            # sampled-activation measurements.
            saved = activation[str(scene_id)][
                "max_relative_m_PRIVATE"
            ]

            assert np.isclose(
                e1_force["max_sampled_relative_m"],
                saved["e1"],
                atol=1e-8,
                rtol=0,
            )

            assert np.isclose(
                e2_force["max_sampled_relative_m"],
                saved["e2"],
                atol=1e-8,
                rtol=0,
            )

            active_steps = (
                e1_force["force_active_steps"]
                +
                e2_force["force_active_steps"]
            )

            force_integral = (
                e1_force["abs_force_integral_Ns"]
                +
                e2_force["abs_force_integral_Ns"]
            )

            previous = deadzone_rows[scene_id]

            old_stop = (
                previous["policies"]
                ["old_stop_or_strong"]["action"]
                == "none"
            )

            aware_stop = (
                previous["policies"]
                ["hypothesis_aware"]["action"]
                == "none"
            )

            records.append({
                "scene_id": scene_id,
                "gap_m": float(scene["gap"]),
                "early_force_steps": active_steps,
                "early_force_Ns": force_integral,
                "old_stopped": old_stop,
                "aware_stopped": aware_stop,
                "early_bic": float(
                    previous["early_evidence"]
                    ["bic_advantage"]
                ),
                "signals": signals,
            })

        print("\n" + "=" * 72)
        print("P3-A5A BAND:", band)
        print("SCENES:", len(records))
        print("=" * 72)

        old_stopped = [
            r for r in records
            if r["old_stopped"]
        ]

        old_stopped_force = [
            r for r in old_stopped
            if r["early_force_steps"] > 0
        ]

        old_stopped_zero = [
            r for r in old_stopped
            if r["early_force_steps"] == 0
        ]

        show_group("all deadzone", records)
        show_group(
            "old stopped deadzone",
            old_stopped,
        )
        show_group(
            "old stopped: force present",
            old_stopped_force,
        )
        show_group(
            "old stopped: force absent",
            old_stopped_zero,
        )

        best_actions = Counter()

        for r in old_stopped:
            best = max(
                E3_ACTIONS,
                key=lambda action: (
                    r["signals"][action]
                    ["B_signal_over_noise"]
                ),
            )
            best_actions[best] += 1

        print(
            "\nORACLE most separating E3 actions:",
            dict(best_actions),
        )

        print(
            "Hypothesis-Aware deadzone stops:",
            sum(
                r["aware_stopped"]
                for r in records
            ),
        )

        print(
            "\nPhysical activation versus B signal "
            "(old stopped scenes):"
        )

        for r in old_stopped:
            print(
                f"  scene={r['scene_id']:02d}",
                f"gap={r['gap_m']:.4f}m",
                f"force_steps={r['early_force_steps']:3d}",
                f"force_Ns={r['early_force_Ns']:.5f}",
                f"early_B={1000*r['signals']['early']['B_signal_rms_m']:.3f}mm",
                f"weak_B={1000*r['signals']['weak']['B_signal_rms_m']:.3f}mm",
                f"strong_B={1000*r['signals']['strong']['B_signal_rms_m']:.3f}mm",
                f"sweep_B={1000*r['signals']['sweep']['B_signal_rms_m']:.3f}mm",
            )

    print(
        "\nP3-A5A ORACLE SIGNAL AUDIT COMPLETE"
    )

    print(
        "No model refitting, no policy tuning, "
        "no new scene generation."
    )

    print(
        "Private truth used for retrospective "
        "feasibility only, not action selection."
    )


if __name__ == "__main__":
    main()
