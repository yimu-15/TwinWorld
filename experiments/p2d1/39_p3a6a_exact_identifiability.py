
"""
P3-A6A: Exact Structural Identifiability Audit

Research question:
Can an Independent law and a Deadzone law generate
identical E1/E2 observations, yet differ under E3?

Method:
1. Replay Independent at every MuJoCo physics step.
2. Compute max relative displacement for each action.
3. Construct a non-activating Deadzone candidate.
4. Verify E1/E2 trajectories agree numerically.
5. Measure which subsequent actions break equivalence.

Uses P3-A4D's existing 24 paired base scenes.

IMPORTANT:
- Scene physical parameters are oracle information.
- This is a feasibility / identifiability audit only.
- It is NOT a learned or deployed action policy.
- No model refitting or threshold optimization.
- No historical files are modified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import mujoco
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

BANDS = (
    "original",
    "high_in_range",
    "above_original_range",
)

# These are analysis ranges, not estimated
# parameter confidence intervals.
GAP_SUPPORTS = {
    "original_support": (0.015, 0.120),
    "wide_support": (0.120, 0.180),
    "combined_support": (0.015, 0.180),
}

EARLY = ("e1", "e2")

INTERVENTIONS = (
    "weak",
    "strong",
    "sweep",
)

ALL_ACTIONS = EARLY + INTERVENTIONS + ("e4",)

NOISE_SIGMA_M = 0.005

EXACT_TOL = 1e-8
GAP_MARGIN = 1e-6


def make_commands(sim):
    profiles = sim["make_profiles"]()

    return {
        "e1": profiles["pulse"] * 0.25,
        "e2": profiles["pulse"] * 0.50,
        "weak": profiles["pulse"] * 1.50,
        "strong": profiles["pulse"] * 2.25,
        "sweep": profiles["sweep"] * 2.25,
        "e4": profiles["pulse"] * 2.75,
    }


def independent_fullstep(
    sim, model, scene, command
):
    """
    Reproduce source-16 independent simulation.

    Records the relative coordinate at every
    physical integration step, not merely the
    51 saved observation points.

    IMPORTANT:
    The stored observed B coordinate includes
    a +1.2 offset, whereas the physical relative
    coordinate is qpos[1] - qpos[0].
    """
    data = mujoco.MjData(model)

    data.eq_active[0] = 0
    data.qvel[:] = scene["velocity"]

    mujoco.mj_forward(model, data)

    # Same passive preparation as P3-A4E.
    for _ in range(sim["STEPS"]):
        data.ctrl[0] = 0.0
        data.qfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)

    target_x = float(data.qpos[0])
    target_v = float(scene["velocity"])

    max_relative = abs(
        float(data.qpos[1] - data.qpos[0])
    )

    sampled = [[
        float(data.qpos[0]),
        1.2 + float(data.qpos[1]),
    ]]

    for i, acceleration in enumerate(command):

        target_v += (
            float(acceleration) * sim["DT"]
        )

        target_x += (
            target_v * sim["DT"]
        )

        control = (
            sim["KP"] *
            (target_x - data.qpos[0])
            +
            sim["KD"] *
            (target_v - data.qvel[0])
        )

        data.ctrl[0] = float(control)
        data.qfrc_applied[:] = 0.0

        mujoco.mj_step(model, data)

        relative = abs(
            float(data.qpos[1] - data.qpos[0])
        )

        max_relative = max(
            max_relative,
            relative,
        )

        if (
            (i + 1)
            % sim["SAMPLE_EVERY"]
            == 0
        ):
            sampled.append([
                float(data.qpos[0]),
                1.2 + float(data.qpos[1]),
            ])

    sampled = np.asarray(sampled)

    if sampled.shape != (51, 2):
        raise AssertionError(
            f"Unexpected output shape: "
            f"{sampled.shape}"
        )

    return max_relative, sampled


def validate_independent_replay(
    sim, model, scene, command,
    sampled
):
    """
    Validate full-step implementation against
    the existing source-16 simulator.

    Stop immediately if physical replay differs.
    """
    _, active, _, _ = sim["simulate"](
        model,
        scene,
        "independent",
        command,
    )

    error = float(np.max(np.abs(
        sampled - np.asarray(active)
    )))

    if error > EXACT_TOL:
        raise AssertionError(
            "Independent replay mismatch: "
            f"{error:.12g} m"
        )

    return error


def compute_envelope(
    sim, model, scene, commands
):
    """
    M(a) = maximum physical relative displacement
    under Independent for each action.
    """
    envelope = {}

    for action in ALL_ACTIONS:
        maximum, sampled = independent_fullstep(
            sim,
            model,
            scene,
            commands[action],
        )

        validate_independent_replay(
            sim,
            model,
            scene,
            commands[action],
            sampled,
        )

        envelope[action] = float(maximum)

    return envelope


def equivalent_gap_interval(
    envelope, low, high
):
    """
    Sufficient condition for exact E1/E2
    observational equivalence:

        gap > max(M(e1), M(e2))

    An explicit positive margin avoids using
    the numerical equality boundary.
    """
    early_max = max(
        envelope["e1"],
        envelope["e2"],
    )

    lower = max(
        float(low),
        early_max + GAP_MARGIN,
    )

    upper = float(high)

    if lower >= upper:
        return None

    return (lower, upper)


def compare_laws(
    sim, model, scene, command
):
    """
    Oracle counterfactual comparison under the
    same initial scene and applied command.

    B is in world observation coordinates.
    """
    _, independent, _, _ = sim["simulate"](
        model,
        scene,
        "independent",
        command,
    )

    _, deadzone, _, _ = sim["simulate"](
        model,
        scene,
        "deadzone",
        command,
    )

    independent = np.asarray(independent)
    deadzone = np.asarray(deadzone)

    delta = deadzone - independent

    return {
        "joint_max_m": float(
            np.max(np.abs(delta))
        ),

        "A_RMS_m": float(np.sqrt(
            np.mean(delta[:, 0] ** 2)
        )),

        "B_RMS_m": float(np.sqrt(
            np.mean(delta[:, 1] ** 2)
        )),
    }


def verify_constructed_candidate(
    sim, scene, interval, commands
):
    """
    Construct ONE illustrative Deadzone candidate
    from the E1/E2-equivalent gap interval.

    The midpoint is fixed independently of E3
    performance.

    This is an oracle construction, not a fit
    from noisy E1/E2 observations.
    """
    low, high = interval

    candidate = dict(scene)
    candidate["gap"] = 0.5 * (
        low + high
    )

    model = sim["make_model"](candidate)

    differences = {
        name: compare_laws(
            sim,
            model,
            candidate,
            command,
        )
        for name, command in commands.items()
    }

    for action in EARLY:
        difference = differences[action]

        if (
            difference["joint_max_m"]
            > EXACT_TOL
        ):
            raise AssertionError(
                "Exact equivalence violated: "
                f"action={action}, "
                f"gap={candidate['gap']}, "
                f"difference="
                f"{difference['joint_max_m']}"
            )

    return {
        "candidate_gap_m":
            float(candidate["gap"]),

        "differences": differences,
    }


def action_break_possible(
    envelope, interval, action
):
    """
    Necessary opportunity for activation.

    If M(action) > some candidate gap from the
    early-equivalent interval, the new action
    can potentially break equivalence.

    Activation is NOT yet statistical detection.
    """
    if interval is None:
        return False

    lower, _ = interval

    return bool(
        envelope[action] > lower
    )


def inspect_true_gap(
    scene, envelope, saved, true_row
):
    """
    Private truth evaluation only.

    Use full-step Independent envelope to find
    whether the actual Deadzone is provably
    inactive on E1/E2.

    Compare against the archived activation audit.
    """
    gap = float(scene["gap"])

    early_max = max(
        envelope["e1"],
        envelope["e2"],
    )

    exact_inactive = bool(
        gap > early_max + GAP_MARGIN
    )

    archived_early_activation = bool(
        saved["early_activated_PRIVATE"]
    )

    old_stopped = bool(
        true_row["policies"]
        ["old_stop_or_strong"]["action"]
        == "none"
    )

    aware_stopped = bool(
        true_row["policies"]
        ["hypothesis_aware"]["action"]
        == "none"
    )

    return {
        "gap_m_PRIVATE": gap,
        "early_M_m": early_max,
        "provably_inactive_early":
            exact_inactive,
        "archived_activated":
            archived_early_activation,
        "old_stopped": old_stopped,
        "aware_stopped": aware_stopped,
    }


def median_or_none(values):
    if not values:
        return None

    return float(np.median(values))


def print_support_summary(name, rows):
    n = len(rows)

    if n == 0:
        print(
            f"  {name}: no early-equivalent "
            "candidate in this gap support"
        )
        return

    counts = {
        action: sum(
            r["can_break"][action]
            for r in rows
        )
        for action in INTERVENTIONS
    }

    print(
        f"  {name}: exact E1/E2-equivalent "
        f"candidate exists in {n} scenes"
    )

    print(
        "    Potential activation-breaking "
        "actions:",
        counts,
    )

    for action in INTERVENTIONS:
        signals = [
            1000.0
            * r["verified"]["differences"]
            [action]["B_RMS_m"]
            for r in rows
        ]

        above_noise = sum(
            value > NOISE_SIGMA_M * 1000.0
            for value in signals
        )

        print(
            f"    {action:7s} "
            f"verified B median="
            f"{np.median(signals):.4f}mm, "
            f"B_RMS>5mm={above_noise}/{n}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            AUDIT_PATH
        )

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            REPORT_PATH
        )

    audit = json.loads(
        AUDIT_PATH.read_text(
            encoding="utf-8"
        )
    )

    report = json.loads(
        REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    sim = run_path(str(
        HERE /
        "16_p2d1b_probe_diversity.py"
    ))

    commands = make_commands(sim)

    original = audit["original"][
        "scene_parameters_PRIVATE"
    ]

    n = (
        2 if args.smoke
        else int(report["scenes_per_band"])
    )

    support_records = {
        name: []
        for name in GAP_SUPPORTS
    }

    truth_records = {
        band: []
        for band in BANDS
    }

    print(
        "P3-A6A EXACT STRUCTURAL "
        "IDENTIFIABILITY AUDIT"
    )

    print(
        "Base scenes:", n,
        "| Candidate law: Deadzone",
        "| Null law: Independent"
    )

    for sid in range(n):
        base = original[sid]

        model = sim["make_model"](base)

        envelope = compute_envelope(
            sim,
            model,
            base,
            commands,
        )

        # Test which candidate gap intervals
        # remain exactly equivalent after E1/E2.
        for name, (low, high) in (
            GAP_SUPPORTS.items()
        ):
            interval = equivalent_gap_interval(
                envelope,
                low,
                high,
            )

            if interval is None:
                continue

            verified = verify_constructed_candidate(
                sim,
                base,
                interval,
                commands,
            )

            support_records[name].append({
                "scene_id": sid,
                "interval": interval,
                "envelope": envelope,

                "can_break": {
                    action: action_break_possible(
                        envelope,
                        interval,
                        action,
                    )
                    for action in INTERVENTIONS
                },

                "verified": verified,
            })

        # Existing true-gap conditions are
        # evaluated separately from construction.
        for band in BANDS:
            group = audit[band]

            scene = group[
                "scene_parameters_PRIVATE"
            ][sid]

            # Paired scene integrity.
            for key in base:
                if key != "gap":
                    if scene[key] != base[key]:
                        raise AssertionError(
                            "Scene pairing mismatch: "
                            f"{band} scene={sid}"
                        )

            deadzone_row = next(
                r
                for r in group["trajectories"]
                if (
                    int(r["scene_id"]) == sid
                    and
                    r["truth_PRIVATE_EVAL_ONLY"]
                    == "deadzone"
                )
            )

            saved = group[
                "activation_by_scene_PRIVATE"
            ][str(sid)]

            truth_records[band].append(
                inspect_true_gap(
                    scene,
                    envelope,
                    saved,
                    deadzone_row,
                )
            )

    print("\n" + "=" * 70)
    print("ORACLE EXACT EQUIVALENCE")
    print("=" * 70)

    for name, rows in (
        support_records.items()
    ):
        print_support_summary(
            name, rows
        )

    print("\n" + "=" * 70)
    print("ARCHIVED DEADZONE TRUTH CHECK")
    print("=" * 70)

    for band, rows in (
        truth_records.items()
    ):
        inactive = [
            r for r in rows
            if r["provably_inactive_early"]
        ]

        old_wrong = [
            r for r in rows
            if r["old_stopped"]
        ]

        old_wrong_inactive = [
            r for r in old_wrong
            if r["provably_inactive_early"]
        ]

        archived_no_activation = sum(
            not r["archived_activated"]
            for r in rows
        )

        disagreements = sum(
            r["provably_inactive_early"]
            ==
            r["archived_activated"]
            for r in rows
        )

        print("\nBAND:", band)

        print(
            "  Provably E1/E2-inactive:",
            f"{len(inactive)}/{len(rows)}"
        )

        print(
            "  Archived no activation:",
            f"{archived_no_activation}/{len(rows)}"
        )

        print(
            "  Envelope/archive classification "
            "disagreements:",
            disagreements
        )

        print(
            "  Old stops on actual Deadzone:",
            f"{len(old_wrong)}/{len(rows)}"
        )

        print(
            "  Of old stops, provably inactive:",
            len(old_wrong_inactive)
        )

        print(
            "  Of old stops, interaction may "
            "have been activated:",
            (
                len(old_wrong)
                - len(old_wrong_inactive)
            )
        )

        print(
            "  Hypothesis-Aware stops:",
            sum(
                r["aware_stopped"]
                for r in rows
            )
        )

    print(
        "\nP3-A6A EXACT IDENTIFIABILITY "
        "AUDIT PASS"
    )

    print(
        "Full-step replay verified against "
        "source-16 independent trajectories."
    )

    print(
        "E1/E2 equivalence candidates verified "
        "with both A and B trajectories."
    )

    print(
        "No refitting, no threshold tuning, "
        "no new scene generation, "
        "no files modified."
    )

    print(
        "This is an ORACLE feasibility audit, "
        "not an online candidate-set estimator."
    )


if __name__ == "__main__":
    main()
