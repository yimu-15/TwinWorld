
"""
P3-A6I: Calibrated Hypothesis Revision

Compare on identical fresh test scenes:
A. Original minimum-Strong-RMS rule
B. BIC-anchored, calibrated switch rule

Calibration:
  Known five represented mechanisms only.
  Separate scene seed from evaluation.
  Q95 absolute RMS and Q95 switch evidence.

Training: E1/E2 + Weak.
Confirmation: Strong.
No E4, no adaptive action choice.

True labels enter offline calibration and
private evaluation, never online scoring.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a6i"

CAL_SEED = 20261227
TEST_SEED = 20261228
CAL_NOISE = 2026122701
TEST_NOISE = 2026122801

CAL_N = 36
TEST_N = 48
Q = 0.95

BANDS = (
    "original",
    "above_original_range",
)

MODES = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear",
    "saturating",
)

TRUTH_TO_LAW = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def calibrate(rows, smoke=False):
    """
    True-law labels used for OFFLINE calibration.

    Correctly anchored cases estimate how much
    a competing law can beat a correct BIC anchor
    on a new Strong observation.

    This is empirical calibration, not a formal
    finite-sample coverage guarantee.
    """
    absolute = np.asarray([
        r["minimum_RMS_m"]
        for r in rows
    ], dtype=float)

    correct_anchor = [
        r for r in rows
        if (
            r["selected_by_BIC_before_strong"]
            == TRUTH_TO_LAW[
                r["truth_PRIVATE"]
            ]
        )
    ]

    required = 1 if smoke else 30

    if len(correct_anchor) < required:
        raise RuntimeError(
            "Insufficient correct-anchor "
            "calibration episodes"
        )

    gains = np.asarray([
        max(
            0.0,
            r["BIC_chosen_RMS_m"]
            - r["minimum_RMS_m"],
        )
        for r in correct_anchor
    ])

    tau_absolute = float(
        np.quantile(absolute, Q)
    )

    tau_switch = float(
        np.quantile(gains, Q)
    )

    if not (
        np.isfinite(tau_absolute)
        and np.isfinite(tau_switch)
    ):
        raise RuntimeError(
            "Invalid calibration thresholds"
        )

    return {
        "tau_absolute_m": tau_absolute,
        "tau_switch_m": tau_switch,
        "calibration_episodes": len(rows),
        "correct_anchor_episodes":
            len(correct_anchor),
    }


def decide(row, thresholds):
    """
    No truth labels enter this function.

    All model fits are frozen before Strong.
    Strong is used only for confirmation.
    """
    h0 = row[
        "selected_by_BIC_before_strong"
    ]

    h1 = row["selected_by_strong"]

    r0 = float(
        row["BIC_chosen_RMS_m"]
    )

    r1 = float(
        row["minimum_RMS_m"]
    )

    actual_min = min(
        row["all_law_RMS_m"].values()
    )

    if not np.isclose(
        r1, actual_min,
        atol=1e-10,
        rtol=0,
    ):
        raise RuntimeError(
            "Minimum RMS integrity failure"
        )

    tau_abs = thresholds[
        "tau_absolute_m"
    ]

    tau_switch = thresholds[
        "tau_switch_m"
    ]

    if r1 > tau_abs:
        return (
            "unresolved",
            "all_models_inadequate",
        )

    if h0 == h1:
        return h0, "agreement"

    gain = r0 - r1

    if gain > tau_switch:
        return (
            h1,
            "sufficient_new_evidence",
        )

    if r0 > tau_abs:
        return (
            "unresolved",
            "anchor_inadequate",
        )

    return (
        h0,
        "retain_anchor",
    )


def baseline(row, thresholds):
    if (
        row["minimum_RMS_m"]
        > thresholds["tau_absolute_m"]
    ):
        return "unresolved"

    return row["selected_by_strong"]


def verdict(truth, decision):
    if truth == "saturating":
        return (
            "correct_reject"
            if decision == "unresolved"
            else "unknown_false_accept"
        )

    if decision == "unresolved":
        return "represented_false_reject"

    return (
        "correct"
        if decision == TRUTH_TO_LAW[truth]
        else "wrong_law"
    )


def evaluate(rows, thresholds):
    output = []

    for row in rows:
        candidate, reason = decide(
            row, thresholds
        )

        original = baseline(
            row, thresholds
        )

        truth = row["truth_PRIVATE"]

        output.append({
            "scene_id": int(
                row["scene_id"]
            ),

            "truth_PRIVATE": truth,

            "BIC_anchor":
                row[
                    "selected_by_BIC_before_strong"
                ],

            "heldout_winner":
                row["selected_by_strong"],

            "minimum_RMS_mm":
                1000 * row["minimum_RMS_m"],

            "switch_gain_mm":
                1000 * (
                    row["BIC_chosen_RMS_m"]
                    - row["minimum_RMS_m"]
                ),

            "baseline_decision": original,
            "candidate_decision": candidate,

            "reason": reason,

            "baseline_outcome_PRIVATE":
                verdict(truth, original),

            "candidate_outcome_PRIVATE":
                verdict(truth, candidate),
        })

    return output


def summarize(band, rows):
    print("\n" + "=" * 72)
    print("BAND:", band)
    print("EPISODES:", len(rows))
    print("=" * 72)

    summary = {}

    for mode in MODES:
        subset = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        before = Counter(
            r["baseline_outcome_PRIVATE"]
            for r in subset
        )

        after = Counter(
            r["candidate_outcome_PRIVATE"]
            for r in subset
        )

        reasons = Counter(
            r["reason"]
            for r in subset
        )

        summary[mode] = {
            "n": len(subset),
            "baseline": dict(before),
            "candidate": dict(after),
            "decision_reasons": dict(reasons),
        }

        print("\nMODE:", mode)
        print("  Baseline :", dict(before))
        print("  Candidate:", dict(after))
        print("  Reasons  :", dict(reasons))

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    # Source definitions for simulator and
    # existing five-law evaluation.
    src = run_path(str(
        HERE /
        "47_p3a6h_six_mechanism_openworld_audit.py"
    ))

    # Check source integrity against P3-A6H.
    old_report = json.loads(
        (
            HERE / "results" / "p3a6h"
            / "p3a6h_fresh_report.json"
        ).read_text(encoding="utf-8")
    )

    for filename, expected in (
        old_report["source_sha256"].items()
    ):
        if (
            sha256(HERE / filename)
            != expected
        ):
            raise RuntimeError(
                "Historical source modified: "
                + filename
            )

    sim = run_path(str(
        HERE /
        "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE /
        "22_p3a1b_structural_revision.py"
    ))

    helper27 = run_path(str(
        HERE /
        "27_p3a3b_evidence_vs_structure.py"
    ))

    helper27[
        "install_saturating"
    ](sim)

    commands = src[
        "make_commands"
    ](sim)

    offset = (
        100000 if args.smoke else 0
    )

    n_cal, n_test = (
        (2, 2)
        if args.smoke
        else (CAL_N, TEST_N)
    )

    # Separate calibration scenes.
    calibration_scenes = src[
        "make_scenes"
    ](
        sim,
        CAL_SEED + offset,
        n_cal,
    )

    calibration_rows = src[
        "collect"
    ](
        sim,
        api,
        calibration_scenes,
        commands,
        CAL_NOISE + offset,
        src["REPRESENTED"],
        ("original",),
    )["original"]

    thresholds = calibrate(
        calibration_rows,
        smoke=args.smoke,
    )

    print(
        "\nP3-A6I CALIBRATED "
        "HYPOTHESIS REVISION"
    )

    print(
        "Calibration episodes:",
        thresholds["calibration_episodes"],
    )

    print(
        "Correct calibration anchors:",
        thresholds["correct_anchor_episodes"],
    )

    print(
        "Absolute cutoff:",
        round(
            1000 *
            thresholds["tau_absolute_m"],
            4,
        ),
        "mm",
    )

    print(
        "Switch evidence cutoff:",
        round(
            1000 *
            thresholds["tau_switch_m"],
            4,
        ),
        "mm",
    )

    # Test scenes are generated only AFTER
    # the thresholds have been frozen.
    test_scenes = src[
        "make_scenes"
    ](
        sim,
        TEST_SEED + offset,
        n_test,
    )

    test_rows = src["collect"](
        sim,
        api,
        test_scenes,
        commands,
        TEST_NOISE + offset,
        src["MODES"],
        BANDS,
    )

    results = {
        "protocol":
            "P3-A6I calibrated revision",

        "smoke": args.smoke,

        "calibration_seed":
            CAL_SEED + offset,

        "test_seed":
            TEST_SEED + offset,

        "thresholds": thresholds,

        "E4_accessed": False,

        "adaptive_action_selection": False,

        "calibration_uses_private_true_laws":
            True,

        "limitations": [
            "Synthetic MuJoCo scenes",
            "Only one unknown physical law",
            "Known-law labels used offline for calibration",
            "No formal finite-sample coverage guarantee",
            "B conditioned on observed A",
            "Weak and Strong are fixed actions",
            "No E4 evaluation",
            "No physical cost-saving policy tested",
            "Smoke thresholds are diagnostic only",
        ],

        "summary": {},
        "audit_PRIVATE": {},
    }

    for band in BANDS:
        evaluated = evaluate(
            test_rows[band],
            thresholds,
        )

        results["audit_PRIVATE"][band] = (
            evaluated
        )

        results["summary"][band] = summarize(
            band,
            evaluated,
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    private_path = OUT / (
        f"p3a6i_{suffix}_audit_PRIVATE.json"
    )

    report_path = OUT / (
        f"p3a6i_{suffix}_report.json"
    )

    private_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    public_report = {
        k: v
        for k, v in results.items()
        if k != "audit_PRIVATE"
    }

    report_path.write_text(
        json.dumps(
            public_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A6I CALIBRATED "
        "REVISION AUDIT PASS"
    )

    print(
        "No E4, no action adaptation, "
        "no test threshold tuning."
    )

    print("Saved:", report_path)
    print("Saved:", private_path)


if __name__ == "__main__":
    main()
