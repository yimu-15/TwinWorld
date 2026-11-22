
"""
P3-A8C1: Matched-Command Temporal Action Feasibility

Compare four predefined actions:
  high
  reverse_high
  biphasic_high
  shifted_high

All actions have equal:
  sum(abs(command))
  max(abs(command))

This does NOT imply equal mechanical work.

Frozen model fitting:
  E1/E2/Weak/Strong
Heldout validation:
  Mid and ONE challenge action

Independent calibration/test scenes.
No E4, no test tuning.
No future observation used in action selection.
No adaptive policy claimed.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a8c1"

HISTORY_REPORT = (
    HERE / "results" / "p3a8b2"
    / "p3a8b2_fresh_report.json"
)

FAIR_REPORT = (
    HERE / "results" / "p3a8c0"
    / "p3a8c0_fair_action_value_report.json"
)

CAL_SEED = 20280115
TEST_SEED = 20280116
CAL_NOISE = 2028011501
TEST_NOISE = 2028011601

N_CAL = 79
N_TEST = 36

ALPHA_MID = 0.05
ALPHA_ACTION = 0.025

BANDS = (
    "original",
    "above_original_range",
)

KNOWN = (
    "independent", "rigid",
    "elastic", "deadzone", "nonlinear",
)

MODES = KNOWN + ("saturating",)

TRUE_LAW = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

ARMS = (
    "high",
    "reverse_high",
    "biphasic_high",
    "shifted_high",
)

METHODS = ("mid_only",) + ARMS


def read(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def initialize():
    core = run_path(str(
        HERE /
        "60_p3a7e1_action_family_pilot.py"
    ))

    src, sim, api, commands = (
        core["initialize"]()
    )

    commands = dict(commands)

    high = np.asarray(
        commands["high"], dtype=float
    ).copy()

    if (
        high.ndim != 1
        or len(high) < 8
        or not np.isfinite(high).all()
        or np.sum(np.abs(high)) <= 0
    ):
        raise RuntimeError(
            "Invalid original High command"
        )

    n = len(high)

    # Reverse the command at the point
    # where cumulative L1 command budget
    # reaches half its total.
    cumulative = np.cumsum(np.abs(high))
    split = int(np.searchsorted(
        cumulative,
        cumulative[-1] * 0.5
    ))

    biphasic = high.copy()
    biphasic[split + 1:] *= -1.0

    # Circular time shift: budget and
    # peak amplitude remain identical.
    shifted = np.roll(high, n // 3)

    commands["reverse_high"] = -high
    commands["biphasic_high"] = biphasic
    commands["shifted_high"] = shifted

    reference_l1 = float(
        np.sum(np.abs(high))
    )

    reference_peak = float(
        np.max(np.abs(high))
    )

    budgets = {}

    for name in ARMS:
        cmd = np.asarray(
            commands[name], dtype=float
        )

        l1 = float(
            np.sum(np.abs(cmd))
        )

        peak = float(
            np.max(np.abs(cmd))
        )

        if cmd.shape != high.shape:
            raise RuntimeError(
                "Command shape mismatch: " + name
            )

        if not np.isclose(
            l1, reference_l1,
            rtol=1e-10, atol=1e-12
        ):
            raise RuntimeError(
                "L1 budget mismatch: " + name
            )

        if not np.isclose(
            peak, reference_peak,
            rtol=1e-10, atol=1e-12
        ):
            raise RuntimeError(
                "Peak mismatch: " + name
            )

        budgets[name] = {
            "command_L1": l1,
            "command_peak": peak,
            "command_sha256": hashlib.sha256(
                cmd.tobytes()
            ).hexdigest(),
        }

    if (
        np.allclose(biphasic, high)
        or np.allclose(biphasic, -high)
        or np.allclose(shifted, high)
    ):
        raise RuntimeError(
            "New temporal actions are not distinct"
        )

    return core, src, sim, api, commands, budgets


def collect_case(
    core, src, sim, api,
    commands, scene,
    sid, mode, band, noise_seed
):
    # Identical four-probe fitting and
    # Vmid observation as P3-A7E1.
    state = core["collect_first"](
        src, sim, api, commands,
        scene, sid, mode, band,
        noise_seed,
    )

    scores = {
        "mid": state["mid_scores"]
    }

    works = {}

    # Fixed challenge order; each action is
    # an independently simulated probe.
    for index, name in enumerate(
        ARMS, start=5
    ):
        observation, work = core["probe"](
            src, sim, api,
            state["model"],
            scene, mode,
            commands[name],
            state["seedbase"] + index,
        )

        works[name] = float(work)

        scores[name] = {
            law: float(
                api["prediction_rms"](
                    observation,
                    state["fits"][law],
                )
            )
            for law in src["LAWS"]
        }

        if not np.isfinite(
            list(scores[name].values())
        ).all():
            raise RuntimeError(
                "Nonfinite challenge scores"
            )

    return {
        "scores_m": scores,
        "work_J": works,
        "initial_work_J": float(
            state["base_work"]
        ),
    }


def calibrate(
    core, src, sim, api, commands,
    scene_sets, noise_seed, smoke
):
    actions = ("mid",) + ARMS

    raw = {
        band: {
            law: {
                action: []
                for action in actions
            }
            for law in src["LAWS"]
        }
        for band in BANDS
    }

    for sid, scene_set in enumerate(
        scene_sets
    ):
        for band in BANDS:
            for mode in KNOWN:
                result = collect_case(
                    core, src, sim, api,
                    commands,
                    scene_set[band],
                    sid, mode, band,
                    noise_seed,
                )

                true_law = TRUE_LAW[mode]

                for action in actions:
                    raw[band][true_law][action].append(
                        result["scores_m"][
                            action
                        ][true_law]
                    )

    thresholds = {}

    for band in BANDS:
        thresholds[band] = {
            action: {}
            for action in actions
        }

        print("\nCALIBRATION:", band)

        for action in actions:
            alpha = (
                ALPHA_MID
                if action == "mid"
                else ALPHA_ACTION
            )

            for law in src["LAWS"]:
                q = core["quantile"](
                    raw[band][law][action],
                    alpha,
                    smoke,
                )

                thresholds[band][action][law] = q

                print(
                    f" {action:15s}",
                    f"{law:12s}",
                    f"q={1000*q:.4f} mm",
                )

    return thresholds


def evaluate_case(
    core, src, sim, api, commands,
    scene, sid, mode, band,
    noise_seed, thresholds
):
    result = collect_case(
        core, src, sim, api, commands,
        scene, sid, mode, band,
        noise_seed,
    )

    first = core["allowed"](
        result["scores_m"]["mid"],
        thresholds["mid"],
    )

    final_sets = {
        "mid_only": first
    }

    costs = {
        "mid_only":
            result["initial_work_J"]
    }

    selected_actions = {
        "mid_only": None
    }

    for action in ARMS:
        # Fair stopping rule: there is no
        # reason to perform an extra probe
        # if the hypothesis set is empty
        # under pure set intersection.
        if len(first) == 0:
            final_sets[action] = []
            costs[action] = (
                result["initial_work_J"]
            )
            selected_actions[action] = None

        else:
            compatible = core["allowed"](
                result["scores_m"][action],
                thresholds[action],
            )

            final_sets[action] = sorted(
                set(first).intersection(
                    compatible
                )
            )

            costs[action] = (
                result["initial_work_J"]
                + result["work_J"][action]
            )

            selected_actions[action] = action

    outcomes = {
        method: core["classify"](
            final_sets[method], mode
        )
        for method in METHODS
    }

    return {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "first_set": first,
        "selected_actions": selected_actions,
        "final_sets": final_sets,
        "outcomes_PRIVATE": outcomes,
        "costs_J": costs,
        "challenge_work_J_PRIVATE":
            result["work_J"],
        "scores_m_PRIVATE":
            result["scores_m"],
    }


def summarize(rows):
    known = [
        r for r in rows
        if r["truth_PRIVATE"] in KNOWN
    ]

    unknown = [
        r for r in rows
        if r["truth_PRIVATE"] == "saturating"
    ]

    summary = {}

    for method in METHODS:
        cost = float(sum(
            r["costs_J"][method]
            for r in rows
        ))

        correct = sum(
            r["outcomes_PRIVATE"][method]
            == "correct_singleton"
            for r in known
        )

        item = {
            "known_correct_singleton": correct,

            "known_wrong_singleton": sum(
                r["outcomes_PRIVATE"][method]
                == "wrong_singleton"
                for r in known
            ),

            "known_contains_true": sum(
                TRUE_LAW[r["truth_PRIVATE"]]
                in r["final_sets"][method]
                for r in known
            ),

            "known_ambiguous": sum(
                len(r["final_sets"][method]) > 1
                for r in known
            ),

            "unknown_empty": sum(
                r["outcomes_PRIVATE"][method]
                == "unknown_empty"
                for r in unknown
            ),

            "unknown_false_singleton": sum(
                r["outcomes_PRIVATE"][method]
                == "unknown_false_singleton"
                for r in unknown
            ),

            "total_cost_J": cost,

            "actions_executed": sum(
                r["selected_actions"][method]
                is not None
                for r in rows
            ),

            "outcomes": dict(Counter(
                r["outcomes_PRIVATE"][method]
                for r in rows
            )),
        }

        summary[method] = item

    high = summary["high"]

    for method in METHODS:
        item = summary[method]

        item["cost_delta_vs_high_J"] = (
            item["total_cost_J"]
            - high["total_cost_J"]
        )

        if method == "mid_only":
            continue

        # Direct paired comparison of a
        # proposed action and High.
        item["known_action_only_correct"] = sum(
            r["outcomes_PRIVATE"][method]
            == "correct_singleton"
            and r["outcomes_PRIVATE"]["high"]
            != "correct_singleton"
            for r in known
        )

        item["known_high_only_correct"] = sum(
            r["outcomes_PRIVATE"]["high"]
            == "correct_singleton"
            and r["outcomes_PRIVATE"][method]
            != "correct_singleton"
            for r in known
        )

        item["unknown_action_only_empty"] = sum(
            r["outcomes_PRIVATE"][method]
            == "unknown_empty"
            and r["outcomes_PRIVATE"]["high"]
            != "unknown_empty"
            for r in unknown
        )

        item["unknown_high_only_empty"] = sum(
            r["outcomes_PRIVATE"]["high"]
            == "unknown_empty"
            and r["outcomes_PRIVATE"][method]
            != "unknown_empty"
            for r in unknown
        )

    for method, item in summary.items():
        print(
            f"  {method:16s}",
            "correct=",
            item["known_correct_singleton"],
            "wrong=",
            item["known_wrong_singleton"],
            "contains=",
            item["known_contains_true"],
            "unknown empty=",
            item["unknown_empty"],
            "unknown false=",
            item["unknown_false_singleton"],
            "cost_J=",
            round(item["total_cost_J"], 3),
        )

        if method not in (
            "mid_only", "high"
        ):
            print(
                "    additional known correct vs High:",
                item["known_action_only_correct"],
                "| High-only:",
                item["known_high_only_correct"],
                "| additional unknown empty:",
                item["unknown_action_only_empty"],
            )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true"
    )
    args = parser.parse_args()

    old = read(HISTORY_REPORT)
    fair = read(FAIR_REPORT)

    if old["smoke"]:
        raise RuntimeError(
            "Expected formal historical report"
        )

    # Historical physics and fitting
    # sources must remain unchanged.
    for filename, expected in (
        old["source_sha256"].items()
    ):
        if sha(HERE / filename) != expected:
            raise RuntimeError(
                "Historical source changed: "
                + filename
            )

    for filename, expected in (
        fair["source_sha256"].items()
    ):
        if sha(
            HERE / "results" /
            "p3a8b2" / filename
        ) != expected:
            raise RuntimeError(
                "Historical report changed: "
                + filename
            )

    (
        core, src, sim,
        api, commands, budgets
    ) = initialize()

    offset = 100000 if args.smoke else 0

    n_cal = 4 if args.smoke else N_CAL
    n_test = 2 if args.smoke else N_TEST

    print(
        "\nP3-A8C1 MATCHED-COMMAND "
        "TEMPORAL ACTION PILOT"
    )

    print(
        "Calibration:", n_cal,
        "| Test:", n_test
    )

    print(
        "Smoke only: no coverage interpretation"
        if args.smoke
        else "Formal per-policy alpha budget: "
             "0.05 + 0.025"
    )

    print("\nCOMMAND BUDGETS")

    for name, values in budgets.items():
        print(
            name,
            "L1=",
            round(
                values["command_L1"], 8
            ),
            "peak=",
            round(
                values["command_peak"], 8
            ),
        )

    cal_scenes = src["make_scenes"](
        sim, CAL_SEED + offset, n_cal
    )

    test_scenes = src["make_scenes"](
        sim, TEST_SEED + offset, n_test
    )

    thresholds = calibrate(
        core, src, sim, api,
        commands, cal_scenes,
        CAL_NOISE + offset,
        args.smoke,
    )

    summaries = {}
    audits = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(
            test_scenes
        ):
            for mode in MODES:
                row = evaluate_case(
                    core, src, sim, api,
                    commands,
                    scene_set[band],
                    sid, mode, band,
                    TEST_NOISE + offset,
                    thresholds[band],
                )

                rows.append(row)

        if len(rows) != n_test * len(MODES):
            raise RuntimeError(
                "Incomplete test matrix"
            )

        print("\nBAND:", band)

        summaries[band] = summarize(rows)
        audits[band] = rows

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report = {
        "protocol":
            "P3-A8C1 matched-command "
            "temporal action feasibility",

        "smoke": args.smoke,
        "n_cal": n_cal,
        "n_test": n_test,
        "cal_seed": CAL_SEED + offset,
        "test_seed": TEST_SEED + offset,

        "alpha_mid": ALPHA_MID,
        "alpha_action": ALPHA_ACTION,

        "command_budgets": budgets,
        "thresholds_m": thresholds,
        "summary": summaries,

        "selection_policy_learned": False,
        "actions_predeclared": True,

        "online_uses_truth": False,
        "test_threshold_tuning": False,
        "E4_accessed": False,

        "limitations": [
            "Fixed predefined action family",
            "Only command L1 and peak are matched",
            "Mechanical work is not matched",
            "Circular phase shift may wrap waveform",
            "Actuator clipping is not independently audited",
            "No learned or adaptive action policy",
            "Four challenge arms share scenes",
            "Per-arm coverage is not joint coverage",
            "Calibrated guarantees need exchangeability",
            "No validity claim for smoke thresholds",
            "Only one held-out unknown mechanism",
            "Gap bands share base scenes",
            "Conditional B prediction uses observed A",
            "No pre-action forecasting for new shapes",
            "MuJoCo only, no RGB or E4",
        ],

        "source_sha256": {
            name: sha(HERE / name)
            for name in (
                "60_p3a7e1_action_family_pilot.py",
                "68_p3a8c1_temporal_action_feasibility.py",
            )
        },
    }

    report_path = OUT / (
        f"p3a8c1_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a8c1_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            audits, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\nP3-A8C1 ACTION AUDIT COMPLETE")
    print("No E4. No test threshold tuning.")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
