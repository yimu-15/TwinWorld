
from pathlib import Path
from collections import Counter, defaultdict
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT / "experiments" / "p2d1"
    / "results" / "p3a1b"
)

SEED = 20261113
BOOTSTRAP = 3000

KNOWN = {"independent", "rigid", "elastic"}
UNKNOWN = {
    "deadzone": "deadzone",
    "nonlinear": "cubic"
}

METHODS = (
    "known_12",
    "known_123",
    "unrestricted_12",
    "gated_revision"
)


def load_json(name):
    path = SOURCE / name

    if not path.exists():
        raise FileNotFoundError(
            f"Missing experiment file: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_mm(values):
    return 1000.0 * float(np.mean(values))


def bootstrap_scene_delta(rows, method, baseline):
    """
    Cluster bootstrap over independent scene IDs.

    Negative delta means method has lower E4 error.
    """
    groups = defaultdict(list)

    for row in rows:
        diff = (
            row["E4_RMS_m"][method]
            - row["E4_RMS_m"][baseline]
        )

        groups[row["scene_id"]].append(diff)

    scene_means = np.array([
        np.mean(values)
        for values in groups.values()
    ])

    rng = np.random.default_rng(SEED)

    samples = rng.choice(
        scene_means,
        size=(BOOTSTRAP, len(scene_means)),
        replace=True
    )

    draws = samples.mean(axis=1)

    return {
        "mean_delta_mm": mean_mm(scene_means),
        "ci95_mm": (
            np.quantile(draws, [0.025, 0.975])
            * 1000
        ).tolist()
    }


def analyze_split(name, rows, threshold):
    print("\n" + "=" * 65)
    print("SPLIT:", name)
    print("EPISODES:", len(rows))
    print("=" * 65)

    decisions = Counter(
        row["decision"] for row in rows
    )

    print("Decision counts:", dict(decisions))

    known_rows = [
        r for r in rows
        if r["truth_PRIVATE_EVAL_ONLY"] in KNOWN
    ]

    unknown_rows = [
        r for r in rows
        if r["truth_PRIVATE_EVAL_ONLY"] in UNKNOWN
    ]

    if known_rows:
        false_edits = sum(
            r["decision"] == "edit"
            for r in known_rows
        )

        inconclusive = sum(
            r["decision"] == "inconclusive"
            for r in known_rows
        )

        print(
            "Known false-edit rate:",
            f"{false_edits}/{len(known_rows)}"
        )

        print(
            "Known inconclusive rate:",
            f"{inconclusive}/{len(known_rows)}"
        )

    if unknown_rows:
        correct = sum(
            r["decision"] == "edit"
            and r["final_kind"]
            == UNKNOWN[r["truth_PRIVATE_EVAL_ONLY"]]
            for r in unknown_rows
        )

        missed = [
            r for r in unknown_rows
            if r["decision"] != "edit"
        ]

        print(
            "Unknown correct edits:",
            f"{correct}/{len(unknown_rows)}"
        )

        print(
            "Unknown not edited:",
            len(missed)
        )

        # Descriptive evidence audit only.
        # Do NOT change the threshold here.
        old_model_acceptable = [
            r for r in missed
            if r["known_e3_m"] <= threshold
        ]

        print(
            "Missed edits with known E3 "
            "residual below threshold:",
            len(old_model_acceptable)
        )

        print(
            "Missed edit proposals:",
            dict(Counter(
                r["edit_proposal_kind"]
                for r in missed
            ))
        )

        for truth in UNKNOWN:
            subset = [
                r for r in unknown_rows
                if r["truth_PRIVATE_EVAL_ONLY"] == truth
            ]

            correct_n = sum(
                r["decision"] == "edit"
                and r["final_kind"] == UNKNOWN[truth]
                for r in subset
            )

            print(
                f"{truth} correct edits:",
                f"{correct_n}/{len(subset)}"
            )

    print("\nE4 mean prediction RMS (mm)")

    for method in METHODS:
        values = [
            row["E4_RMS_m"][method]
            for row in rows
        ]

        print(
            f"{method:<20}",
            f"{mean_mm(values):.4f}"
        )

    print("\nE4 by true mechanism (mm)")

    for truth in (
        "independent",
        "rigid",
        "elastic",
        "deadzone",
        "nonlinear"
    ):
        subset = [
            r for r in rows
            if r["truth_PRIVATE_EVAL_ONLY"] == truth
        ]

        if not subset:
            continue

        values = {
            method: mean_mm([
                r["E4_RMS_m"][method]
                for r in subset
            ])
            for method in METHODS
        }

        print(truth, {
            k: round(v, 4)
            for k, v in values.items()
        })

    print("\nScene-cluster bootstrap comparisons")

    for baseline in (
        "known_12",
        "known_123",
        "unrestricted_12"
    ):
        comparison = bootstrap_scene_delta(
            rows,
            "gated_revision",
            baseline
        )

        print(
            f"Gated minus {baseline}:",
            comparison
        )


def main():
    report = load_json(
        "p3a1b_development_report.json"
    )

    audit = load_json(
        "p3a1b_development_audit_PRIVATE.json"
    )

    threshold = report[
        "calibration_thresholds"
    ]["known_e3_q95_m"]

    print("=" * 65)
    print("P3-A1C READ-ONLY GATE DIAGNOSTIC")
    print("=" * 65)

    print(
        "Known E3 threshold (mm):",
        round(threshold * 1000, 4)
    )

    print(
        "Edit E3 threshold (mm):",
        round(
            report["calibration_thresholds"][
                "edit_e3_q95_m"
            ] * 1000,
            4
        )
    )

    for split_name in ("development", "mass_shift"):
        rows = audit[split_name]

        # Check original decision counts.
        original_counts = report[
            "split_stats"
        ][split_name]["decision_counts"]

        actual_counts = dict(Counter(
            r["decision"] for r in rows
        ))

        assert actual_counts == original_counts, (
            f"Decision count mismatch: {split_name}"
        )

        analyze_split(
            split_name,
            rows,
            threshold
        )

    print("\nP3-A1C READ-ONLY AUDIT PASS")
    print("No model retraining.")
    print("No threshold modification.")
    print("No original data modification.")
    print("E4 used only for retrospective evaluation.")


if __name__ == "__main__":
    main()
