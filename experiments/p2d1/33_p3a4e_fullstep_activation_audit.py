
"""
P3-A4E: Full-step activation audit (read-only).

Re-simulates only physical Deadzone truth for the
existing P3-A4D paired scenes.

Checks:
- Did physical force actually become nonzero?
- Does sampled activation agree with full-step activation?
- Why did the old policy stop despite physical activation?

No refitting, policy changes, or threshold tuning.
Physical truth is evaluation-only.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from runpy import run_path

import numpy as np
import mujoco


HERE = Path(__file__).resolve().parent

SOURCE = (
    HERE / "results" / "p3a4d"
)

AUDIT = (
    SOURCE /
    "p3a4d_stress_audit_PRIVATE.json"
)

REPORT = (
    SOURCE /
    "p3a4d_stress_report.json"
)

BANDS = (
    "original",
    "high_in_range",
    "above_original_range",
)

ACTIONS = (
    "e1",
    "e2",
    "e4_pulse",
)


def truth_force_rollout(
    sim,
    model,
    scene,
    command,
):
    """
    Repeat source 16 deadzone simulation.

    Unlike saved 51-point observations, this
    function measures internal forces at every
    2 ms MuJoCo integration step.
    """
    data = mujoco.MjData(model)

    data.eq_active[0] = 0
    data.qvel[:] = scene["velocity"]

    mujoco.mj_forward(model, data)

    # Identical passive preparation.
    for _ in range(sim["STEPS"]):

        data.ctrl[0] = 0.0
        data.qfrc_applied[:] = 0.0

        mujoco.mj_step(model, data)

    target_x = float(data.qpos[0])
    target_v = float(scene["velocity"])

    initial_r = float(
        data.qpos[1] - data.qpos[0]
    )

    sampled_r = [initial_r]

    max_abs_r_full = abs(initial_r)
    force_abs_integral = 0.0
    max_abs_force = 0.0
    max_excess = 0.0
    force_active_steps = 0

    for step, acceleration in enumerate(
        command
    ):

        target_v += (
            float(acceleration) * sim["DT"]
        )

        target_x += (
            target_v * sim["DT"]
        )

        u = float(
            sim["KP"] * (
                target_x - data.qpos[0]
            )
            +
            sim["KD"] * (
                target_v - data.qvel[0]
            )
        )

        r = float(
            data.qpos[1] - data.qpos[0]
        )

        vr = float(
            data.qvel[1] - data.qvel[0]
        )

        force = float(
            sim["internal_force"](
                "deadzone",
                r,
                vr,
                scene,
            )
        )

        max_abs_r_full = max(
            max_abs_r_full,
            abs(r),
        )

        max_excess = max(
            max_excess,
            max(
                0.0,
                abs(r) - scene["gap"],
            ),
        )

        max_abs_force = max(
            max_abs_force,
            abs(force),
        )

        force_abs_integral += (
            abs(force) * sim["DT"]
        )

        if abs(force) > 1e-12:
            force_active_steps += 1

        data.qfrc_applied[:] = (
            -force,
            force,
        )

        data.ctrl[0] = u

        mujoco.mj_step(model, data)

        max_abs_r_full = max(
            max_abs_r_full,
            abs(float(
                data.qpos[1] - data.qpos[0]
            )),
        )

        if (
            (step + 1)
            % sim["SAMPLE_EVERY"]
            == 0
        ):
            sampled_r.append(
                float(
                    data.qpos[1]
                    - data.qpos[0]
                )
            )

    assert len(sampled_r) == 51

    return {
        "max_sampled_relative_m": float(
            np.max(np.abs(sampled_r))
        ),

        "max_full_relative_m": float(
            max_abs_r_full
        ),

        "maximum_excess_over_gap_m":
            float(max_excess),

        "abs_force_integral_Ns": float(
            force_abs_integral
        ),

        "peak_abs_force_N": float(
            max_abs_force
        ),

        "force_active_steps": int(
            force_active_steps
        ),
    }


def mean_e4_delta_mm(row, method):
    """
    E4 is retrospective evaluation only.

    Positive means worse than Always-Strong.
    """
    tests = (
        "pulse_2p75",
        "oscillation_2p25",
    )

    selected = row[
        "policies"
    ][method]["E4_m"]

    strong = row[
        "policies"
    ]["always_strong"]["E4_m"]

    return 500.0 * sum(
        float(selected[t])
        -
        float(strong[t])
        for t in tests
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    if not AUDIT.exists():
        raise FileNotFoundError(AUDIT)

    if not REPORT.exists():
        raise FileNotFoundError(REPORT)

    audit = json.loads(
        AUDIT.read_text(encoding="utf-8")
    )

    report = json.loads(
        REPORT.read_text(encoding="utf-8")
    )

    sim = run_path(str(
        HERE /
        "16_p2d1b_probe_diversity.py"
    ))

    profiles = sim["make_profiles"]()

    commands = {
        "e1":
            profiles["pulse"] * 0.25,

        "e2":
            profiles["pulse"] * 0.50,

        "e4_pulse":
            profiles["pulse"] * 2.75,
    }

    n_scenes = (
        2 if args.smoke
        else report["scenes_per_band"]
    )

    original_scenes = audit[
        "original"
    ]["scene_parameters_PRIVATE"]

    for band in BANDS:

        print("\n" + "=" * 76)
        print("BAND:", band)
        print("SCENES:", n_scenes)
        print("=" * 76)

        group = audit[band]

        scenes = group[
            "scene_parameters_PRIVATE"
        ][:n_scenes]

        rows = {
            int(r["scene_id"]): r
            for r in group["trajectories"]
            if r["truth_PRIVATE_EVAL_ONLY"]
            == "deadzone"
        }

        saved_activation = group[
            "activation_by_scene_PRIVATE"
        ]

        coarse_no_cross = 0
        fine_no_force = 0

        stopped_deadzone = 0
        stopped_with_force = 0
        stopped_without_force = 0

        aware_stopped_deadzone = 0
        e4_active = 0

        force_integrals = []

        for sid, scene in enumerate(scenes):

            # Paired-scene integrity.
            original = original_scenes[sid]

            for key, value in original.items():
                if key == "gap":
                    continue

                assert scene[key] == value, (
                    band,
                    sid,
                    key,
                )

            model = sim["make_model"](scene)

            traces = {
                action: truth_force_rollout(
                    sim,
                    model,
                    scene,
                    commands[action],
                )
                for action in ACTIONS
            }

            saved = saved_activation[
                str(sid)
            ]

            # Verify exact reproduction of
            # the earlier sampled trajectory.
            for action in ACTIONS:

                expected = saved[
                    "max_relative_m_PRIVATE"
                ][action]

                actual = traces[action][
                    "max_sampled_relative_m"
                ]

                if not np.isclose(
                    expected,
                    actual,
                    rtol=0.0,
                    atol=1e-8,
                ):
                    raise AssertionError(
                        "Physical replay mismatch: "
                        f"{band}, scene={sid}, "
                        f"action={action}"
                    )

            gap = float(scene["gap"])

            sampled_cross = any(
                traces[action][
                    "max_sampled_relative_m"
                ] > gap + 1e-8
                for action in ("e1", "e2")
            )

            early_steps = sum(
                traces[action][
                    "force_active_steps"
                ]
                for action in ("e1", "e2")
            )

            early_force = sum(
                traces[action][
                    "abs_force_integral_Ns"
                ]
                for action in ("e1", "e2")
            )

            e4_has_force = (
                traces["e4_pulse"][
                    "force_active_steps"
                ] > 0
            )

            force_integrals.append(
                early_force
            )

            coarse_no_cross += int(
                not sampled_cross
            )

            fine_no_force += int(
                early_steps == 0
            )

            e4_active += int(
                e4_has_force
            )

            row = rows[sid]

            old = row["policies"][
                "old_stop_or_strong"
            ]

            aware = row["policies"][
                "hypothesis_aware"
            ]

            if aware["action"] == "none":
                aware_stopped_deadzone += 1

            if old["action"] == "none":

                stopped_deadzone += 1

                if early_steps > 0:
                    stopped_with_force += 1
                else:
                    stopped_without_force += 1

                e4_delta = mean_e4_delta_mm(
                    row,
                    "old_stop_or_strong",
                )

                evidence = row[
                    "early_evidence"
                ]

                print(
                    f"  STOP scene={sid:02d} "
                    f"gap={gap:.4f}m "
                    f"sampled_cross={sampled_cross} "
                    f"force_steps={early_steps:4d} "
                    f"early_force_Ns={early_force:.6f} "
                    f"BIC={evidence['bic_advantage']:.2f} "
                    f"fit_RMS="
                    f"{1000*evidence['known_fit_rms_m']:.2f}mm "
                    f"E4_delta={e4_delta:+.2f}mm"
                )

        print(
            "\nSampled E1/E2 no threshold crossing:",
            coarse_no_cross,
            "/",
            n_scenes,
        )

        print(
            "Full-step E1/E2 zero applied force:",
            fine_no_force,
            "/",
            n_scenes,
        )

        print(
            "Full-step E4 force present:",
            e4_active,
            "/",
            n_scenes,
        )

        print(
            "Old policy wrong stops on Deadzone:",
            stopped_deadzone,
            "/",
            n_scenes,
        )

        print(
            "  Stopped with early physical force:",
            stopped_with_force,
        )

        print(
            "  Stopped without early physical force:",
            stopped_without_force,
        )

        print(
            "Hypothesis-Aware Deadzone stops:",
            aware_stopped_deadzone,
        )

        print(
            "E1/E2 absolute force integral (N*s):",
            {
                "min": float(
                    np.min(force_integrals)
                ),
                "median": float(
                    np.median(force_integrals)
                ),
                "max": float(
                    np.max(force_integrals)
                ),
            },
        )

    print(
        "\nP3-A4E FULL-STEP PHYSICS AUDIT PASS"
    )

    print(
        "No fitting, policy modification, "
        "threshold tuning, or output files."
    )

    print(
        "Physical activation is "
        "evaluation-only."
    )


if __name__ == "__main__":
    main()
