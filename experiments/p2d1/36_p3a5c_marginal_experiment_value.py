
"""
P3-A5C: Marginal Experiment Value Audit.

Question:
Does pre-action model disagreement predict the
actual incremental E4 value of Strong versus Weak?

Inputs:
- P3-A4D frozen stress scenes / audit
- E1/E2 fitted model disagreement from P3-A5B
- Retrospective Weak / Strong evaluation outcomes

No threshold optimization.
No new scene generation.
No historical files modified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent

SOURCE = HERE / "results" / "p3a4d"

AUDIT_PATH = (
    SOURCE / "p3a4d_stress_audit_PRIVATE.json"
)

REPORT_PATH = (
    SOURCE / "p3a4d_stress_report.json"
)

BANDS = (
    "original",
    "high_in_range",
    "above_original_range",
)

TESTS = (
    "pulse_2p75",
    "oscillation_2p25",
)

WEAK = "pulse_1p5"
STRONG = "pulse_2p25"

SEED = 20261128

# Descriptive threshold only.
# Not used to learn an action policy.
MATERIAL_BENEFIT_MM = 1.0


def paired_e4_mm(row, policy):
    errors = row["policies"][policy]["E4_m"]

    return 1000.0 * float(np.mean([
        errors[t] for t in TESTS
    ]))


def nominal_effort(command, dt):
    """
    Public command-based cost proxy.

    This is NOT realized physical work.
    """
    command = np.asarray(
        command, dtype=float
    )

    return float(
        np.sum(command ** 2) * dt
    )


def safe_spearman(x, y):
    if len(x) < 4:
        return None

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if (
        np.std(x) < 1e-12
        or np.std(y) < 1e-12
    ):
        return None

    result = spearmanr(x, y)

    value = float(result.statistic)

    return value if np.isfinite(value) else None


def describe(values):
    if not values:
        return None

    x = np.asarray(values, dtype=float)

    return {
        "median": round(
            float(np.median(x)), 4
        ),
        "q25": round(
            float(np.quantile(x, 0.25)), 4
        ),
        "q75": round(
            float(np.quantile(x, 0.75)), 4
        ),
        "min": round(
            float(np.min(x)), 4
        ),
        "max": round(
            float(np.max(x)), 4
        ),
    }


def reconstruct_band(
    band, scenes, audit, sim, api, source35,
    commands,
):
    expected = {
        (
            int(row["scene_id"]),
            row["truth_PRIVATE_EVAL_ONLY"],
        ): row
        for row in audit[band]["trajectories"]
    }

    # Exact original noise stream.
    rng = np.random.default_rng(
        SEED + 222 + 100000
    )

    records = []

    weak_effort = nominal_effort(
        commands[WEAK],
        sim["DT"],
    )

    strong_effort = nominal_effort(
        commands[STRONG],
        sim["DT"],
    )

    for sid, scene in enumerate(scenes):

        model = sim["make_model"](scene)

        # This function already validates that
        # reconstructed E1/E2 BIC and RMS match
        # the frozen P3-A4D audit.
        early_results = source35[
            "recover_early"
        ](
            scene,
            model,
            sim,
            api,
            commands,
            rng,
            expected,
            sid,
        )

        for early in early_results:

            truth = early["mode"]
            old = expected[(sid, truth)]

            pred_weak = (
                1000.0
                * early["predicted"][WEAK]
            )

            pred_strong = (
                1000.0
                * early["predicted"][STRONG]
            )

            e4_weak = paired_e4_mm(
                old, "always_weak"
            )

            e4_strong = paired_e4_mm(
                old, "always_strong"
            )

            work_weak = float(
                old["policies"]["always_weak"]
                ["work_J"]
            )

            work_strong = float(
                old["policies"]["always_strong"]
                ["work_J"]
            )

            # What was knowable before E3:
            predicted_increment = (
                pred_strong - pred_weak
            )

            # What was measurable only after
            # the counterfactual evaluation:
            realized_benefit = (
                e4_weak - e4_strong
            )

            weak_decision = old[
                "policies"
            ]["always_weak"]

            strong_decision = old[
                "policies"
            ]["always_strong"]

            correct_weak = (
                truth == "deadzone"
                and weak_decision["decision"]
                == "edit"
                and weak_decision["kind"]
                == "deadzone"
            )

            correct_strong = (
                truth == "deadzone"
                and strong_decision["decision"]
                == "edit"
                and strong_decision["kind"]
                == "deadzone"
            )

            # Two naive baselines for diagnosis.
            # Neither is claimed as learned.
            max_disagreement = (
                STRONG
                if pred_strong >= pred_weak
                else WEAK
            )

            per_effort = (
                STRONG
                if (
                    pred_strong
                    / max(strong_effort, 1e-12)
                ) > (
                    pred_weak
                    / max(weak_effort, 1e-12)
                )
                else WEAK
            )

            records.append({
                "scene_id": sid,
                "truth_PRIVATE": truth,

                "early_known_kind": old[
                    "early_evidence"
                ]["known_kind"],

                "predicted_weak_mm":
                    pred_weak,

                "predicted_strong_mm":
                    pred_strong,

                "predicted_increment_mm":
                    predicted_increment,

                "predicted_weak_per_effort":
                    pred_weak
                    / max(weak_effort, 1e-12),

                "predicted_strong_per_effort":
                    pred_strong
                    / max(strong_effort, 1e-12),

                "realized_E4_benefit_mm":
                    realized_benefit,

                "extra_realized_work_J":
                    work_strong - work_weak,

                "material_strong_benefit":
                    realized_benefit
                    > MATERIAL_BENEFIT_MM,

                "strong_corrects_weak":
                    (
                        not correct_weak
                        and correct_strong
                    ),

                "strong_worsens_weak":
                    (
                        correct_weak
                        and not correct_strong
                    ),

                "max_disagreement_action":
                    max_disagreement,

                "per_effort_action":
                    per_effort,

                "weak_correct": correct_weak,
                "strong_correct": correct_strong,
            })

    return records


def print_group(label, rows):
    if not rows:
        return

    print("\nGROUP:", label, "N =", len(rows))

    signal_gain = [
        r["predicted_increment_mm"]
        for r in rows
    ]

    actual_gain = [
        r["realized_E4_benefit_mm"]
        for r in rows
    ]

    cost_gain = [
        r["extra_realized_work_J"]
        for r in rows
    ]

    print(
        "Predicted extra disagreement (mm):",
        describe(signal_gain),
    )

    print(
        "Actual Strong E4 benefit (mm):",
        describe(actual_gain),
    )

    print(
        "Extra realized work (J):",
        describe(cost_gain),
    )

    print(
        "Spearman(predicted gain, E4 benefit):",
        safe_spearman(
            signal_gain, actual_gain
        ),
    )

    print(
        "Strong materially better (>1 mm):",
        sum(
            r["material_strong_benefit"]
            for r in rows
        ),
        "/",
        len(rows),
    )

    print(
        "Max predicted disagreement actions:",
        dict(Counter(
            r["max_disagreement_action"]
            for r in rows
        )),
    )

    print(
        "Predicted disagreement / "
        "nominal effort actions:",
        dict(Counter(
            r["per_effort_action"]
            for r in rows
        )),
    )

    deadzone = [
        r for r in rows
        if r["truth_PRIVATE"] == "deadzone"
    ]

    if deadzone:
        print(
            "Deadzone: Strong corrects Weak:",
            sum(
                r["strong_corrects_weak"]
                for r in deadzone
            ),
            "/",
            len(deadzone),
        )

        print(
            "Deadzone: Strong worsens Weak:",
            sum(
                r["strong_worsens_weak"]
                for r in deadzone
            ),
            "/",
            len(deadzone),
        )

        print(
            "Deadzone: Weak correct:",
            sum(
                r["weak_correct"]
                for r in deadzone
            ),
            "/",
            len(deadzone),
        )

        print(
            "Deadzone: Strong correct:",
            sum(
                r["strong_correct"]
                for r in deadzone
            ),
            "/",
            len(deadzone),
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(AUDIT_PATH)

    if not REPORT_PATH.exists():
        raise FileNotFoundError(REPORT_PATH)

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
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    source35 = run_path(str(
        HERE / "35_p3a5b_preaction_model_disagreement.py"
    ))

    commands = source35[
        "command_bank"
    ](
        sim["make_profiles"]()
    )

    n = (
        2 if args.smoke
        else int(report["scenes_per_band"])
    )

    print(
        "P3-A5C retrospective marginal-value audit"
    )

    for band in BANDS:

        scenes = audit[band][
            "scene_parameters_PRIVATE"
        ][:n]

        rows = reconstruct_band(
            band,
            scenes,
            audit,
            sim,
            api,
            source35,
            commands,
        )

        print("\n" + "=" * 72)
        print(
            "BAND:", band,
            "| SCENES:", n,
            "| EPISODES:", len(rows)
        )
        print("=" * 72)

        independent = [
            r for r in rows
            if r["truth_PRIVATE"]
            == "independent"
        ]

        deadzone = [
            r for r in rows
            if r["truth_PRIVATE"]
            == "deadzone"
        ]

        old_stop_ids = {
            int(row["scene_id"])
            for row in audit[band]["trajectories"]
            if (
                row["truth_PRIVATE_EVAL_ONLY"]
                == "deadzone"
                and row["policies"]
                ["old_stop_or_strong"]["action"]
                == "none"
            )
        }

        stopped_deadzone = [
            r for r in deadzone
            if r["scene_id"] in old_stop_ids
        ]

        print_group(
            "Independent",
            independent
        )

        print_group(
            "Deadzone",
            deadzone
        )

        print_group(
            "Deadzone missed by old stopping",
            stopped_deadzone
        )

    print(
        "\nP3-A5C MARGINAL VALUE AUDIT PASS"
    )

    print(
        "Retrospective analysis only; "
        "no new policy validated."
    )

    print(
        "Future E4 was used to evaluate "
        "incremental benefit, never to "
        "calculate pre-action disagreement."
    )


if __name__ == "__main__":
    main()
