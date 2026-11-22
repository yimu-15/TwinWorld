
"""
P3-A6K: Frozen-Policy Excitation Shift Audit.

Purpose:
    Test A6I calibrated revision under weaker
    physical excitation on entirely new scenes.

Frozen:
    Five candidate laws
    Absolute RMS threshold
    Switching evidence threshold
    Decision logic

Three fixed regimes:
    reference: Weak 1.50, Strong 2.25
    moderate:  Weak 1.25, Strong 1.75
    low:       Weak 0.75, Strong 1.10

No threshold refitting.
No E4.
No adaptive action policy.
No tuning on the new test scenes.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "p3a6k"

FROZEN_REPORT = (
    RESULTS / "p3a6i"
    / "p3a6i_fresh_report.json"
)

SCENE_SEED = 20270115
NOISE_SEED = 2027011501

N_FULL = 48

REGIMES = {
    "reference": (1.50, 2.25),
    "moderate": (1.25, 1.75),
    "low": (0.75, 1.10),
}

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


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def read(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def evaluate(records, thresholds, policy):
    output = []

    for record in records:
        old = policy["baseline"](
            record, thresholds
        )

        new, reason = policy["decide"](
            record, thresholds
        )

        truth = record["truth_PRIVATE"]

        output.append({
            "scene_id": record["scene_id"],
            "truth_PRIVATE": truth,

            "baseline_decision": old,
            "candidate_decision": new,
            "decision_reason": reason,

            "baseline_outcome_PRIVATE":
                policy["verdict"](
                    truth, old
                ),

            "candidate_outcome_PRIVATE":
                policy["verdict"](
                    truth, new
                ),

            "minimum_RMS_mm":
                1000.0
                * record["minimum_RMS_m"],

            "switch_gain_mm":
                1000.0 * (
                    record["BIC_chosen_RMS_m"]
                    - record["minimum_RMS_m"]
                ),

            "weak_work_J":
                record["weak_work_proxy_J"],

            "strong_work_J":
                record["strong_work_proxy_J"],
        })

    return output


def summarize(records):
    summary = {}

    for mode in MODES:
        rows = [
            r for r in records
            if r["truth_PRIVATE"] == mode
        ]

        old = Counter(
            r["baseline_outcome_PRIVATE"]
            for r in rows
        )

        new = Counter(
            r["candidate_outcome_PRIVATE"]
            for r in rows
        )

        item = {
            "n": len(rows),

            "baseline_outcomes":
                dict(old),

            "candidate_outcomes":
                dict(new),

            "median_Rmin_mm":
                float(np.median([
                    r["minimum_RMS_mm"]
                    for r in rows
                ])),

            "median_weak_work_J":
                float(np.median([
                    r["weak_work_J"]
                    for r in rows
                ])),

            "median_strong_work_J":
                float(np.median([
                    r["strong_work_J"]
                    for r in rows
                ])),
        }

        summary[mode] = item

        print(
            "\n", mode,
            "| N =", len(rows)
        )

        print(
            "  Baseline:",
            dict(old)
        )

        print(
            "  Candidate:",
            dict(new)
        )

        print(
            "  Rmin median:",
            round(
                item["median_Rmin_mm"], 3
            ),
            "mm"
        )

    known_modes = MODES[:-1]

    known = [
        r for r in records
        if r["truth_PRIVATE"]
        in known_modes
    ]

    old_correct = sum(
        r["baseline_outcome_PRIVATE"]
        == "correct"
        for r in known
    )

    new_correct = sum(
        r["candidate_outcome_PRIVATE"]
        == "correct"
        for r in known
    )

    summary["_overall"] = {
        "known_n": len(known),
        "baseline_correct": old_correct,
        "candidate_correct": new_correct,

        "accuracy_gain_pp":
            100.0 * (
                new_correct - old_correct
            ) / len(known),
    }

    print(
        "\nKNOWN LAW ACCURACY:",
        old_correct, "->", new_correct,
        "/", len(known)
    )

    print(
        "GAIN:",
        round(
            summary["_overall"][
                "accuracy_gain_pp"
            ],
            4,
        ),
        "percentage points"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    frozen = read(FROZEN_REPORT)

    if (
        frozen["smoke"]
        or frozen["test_seed"]
        != 20261228
    ):
        raise RuntimeError(
            "Expected frozen A6I formal policy"
        )

    thresholds = frozen["thresholds"]

    print(
        "P3-A6K EXCITATION SHIFT AUDIT"
    )

    print(
        "Frozen absolute cutoff:",
        round(
            thresholds["tau_absolute_m"]
            * 1000.0, 4
        ),
        "mm"
    )

    print(
        "Frozen switch cutoff:",
        round(
            thresholds["tau_switch_m"]
            * 1000.0, 4
        ),
        "mm"
    )

    src = run_path(str(
        HERE /
        "47_p3a6h_six_mechanism_openworld_audit.py"
    ))

    policy = run_path(str(
        HERE /
        "48_p3a6i_calibrated_revision.py"
    ))

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

    # Preserve the established Saturating law.
    helper27[
        "install_saturating"
    ](sim)

    smoke_offset = (
        100000 if args.smoke else 0
    )

    n = 2 if args.smoke else N_FULL

    scenes = src["make_scenes"](
        sim,
        SCENE_SEED + smoke_offset,
        n,
    )

    profiles = sim["make_profiles"]()

    base_commands = src[
        "make_commands"
    ](sim)

    results = {
        "protocol":
            "P3-A6K excitation-shift audit",

        "smoke": args.smoke,

        "scene_seed":
            SCENE_SEED + smoke_offset,

        "noise_seed":
            NOISE_SEED + smoke_offset,

        "n_base_scenes": n,

        "frozen_thresholds":
            thresholds,

        "regimes":
            REGIMES,

        "E4_accessed": False,

        "policy_retrained": False,

        "thresholds_retuned": False,

        "limitations": [
            "Same MuJoCo generator",
            "One unknown Saturating law",
            "Weak and Strong are fixed actions",
            "Calibration thresholds belong to reference excitation",
            "Other regimes are distribution shifts",
            "No statistical coverage guarantee under shift",
            "B prediction conditions on observed A",
            "Work values are action-specific proxies",
            "No online action selection is tested",
        ],

        "summary": {},
        "audit_PRIVATE": {},
    }

    for regime, gains in REGIMES.items():
        weak_gain, strong_gain = gains

        commands = dict(
            base_commands
        )

        commands["weak"] = (
            profiles["pulse"]
            * weak_gain
        )

        commands["strong"] = (
            profiles["pulse"]
            * strong_gain
        )

        # All regimes share the same scene set,
        # noise seed, and law implementations.
        raw = src["collect"](
            sim,
            api,
            scenes,
            commands,
            NOISE_SEED + smoke_offset,
            src["MODES"],
            BANDS,
        )

        results["summary"][regime] = {}
        results[
            "audit_PRIVATE"
        ][regime] = {}

        for band in BANDS:
            records = evaluate(
                raw[band],
                thresholds,
                policy,
            )

            expected = n * len(MODES)

            if len(records) != expected:
                raise RuntimeError(
                    "Unexpected episode count"
                )

            print("\n" + "=" * 72)
            print(
                "REGIME:", regime,
                "| BAND:", band,
                "| N:", n
            )
            print("=" * 72)

            results[
                "summary"
            ][regime][band] = summarize(
                records
            )

            results[
                "audit_PRIVATE"
            ][regime][band] = records

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report_path = OUT / (
        f"p3a6k_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a6k_{suffix}_audit_PRIVATE.json"
    )

    report = {
        key: value
        for key, value in results.items()
        if key != "audit_PRIVATE"
    }

    report["source_sha256"] = {
        name: sha256(HERE / name)
        for name in (
            "16_p2d1b_probe_diversity.py",
            "22_p3a1b_structural_revision.py",
            "27_p3a3b_evidence_vs_structure.py",
            "47_p3a6h_six_mechanism_openworld_audit.py",
            "48_p3a6i_calibrated_revision.py",
        )
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            results["audit_PRIVATE"],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A6K EXCITATION SHIFT "
        "AUDIT COMPLETE"
    )

    print(
        "Frozen policy, no E4, "
        "no threshold tuning."
    )

    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
