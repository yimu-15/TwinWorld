
"""
P3-A6L: Low-Excitation Failure Diagnosis
Stage 1: Paired Evidence Audit

Read-only historical analysis.
No E4 access.
No model fitting.
No threshold tuning.
No historical result modification.
"""

from pathlib import Path
from collections import Counter
import csv
import json
import statistics


HERE = Path(__file__).resolve().parent

ROOT = HERE / "results"

REPORT_PATH = (
    ROOT / "p3a6k"
    / "p3a6k_fresh_report.json"
)

AUDIT_PATH = (
    ROOT / "p3a6k"
    / "p3a6k_fresh_audit_PRIVATE.json"
)

OUT = ROOT / "p3a6l"

REGIMES = (
    "reference",
    "moderate",
    "low",
)

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


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def make_index(rows):
    result = {}

    for r in rows:
        key = (
            int(r["scene_id"]),
            r["truth_PRIVATE"],
        )

        if key in result:
            raise RuntimeError(
                f"Duplicate record: {key}"
            )

        result[key] = r

    return result


def median(values):
    return statistics.median(values)


def build_paired(audit, band):
    indexes = {
        regime: make_index(
            audit[regime][band]
        )
        for regime in REGIMES
    }

    keys = set(indexes["reference"])

    for regime in REGIMES:
        if set(indexes[regime]) != keys:
            raise RuntimeError(
                "Unpaired experimental records"
            )

    if len(keys) != 48 * len(MODES):
        raise RuntimeError(
            "Unexpected experiment size"
        )

    paired = []

    for sid, truth in sorted(keys):
        row = {
            "scene_id": sid,
            "truth_PRIVATE": truth,
            "band": band,
        }

        for regime in REGIMES:
            r = indexes[regime][
                (sid, truth)
            ]

            prefix = regime + "_"

            row[
                prefix + "decision"
            ] = r["candidate_decision"]

            row[
                prefix + "outcome"
            ] = r[
                "candidate_outcome_PRIVATE"
            ]

            row[
                prefix + "Rmin_mm"
            ] = float(
                r["minimum_RMS_mm"]
            )

            row[
                prefix + "switch_gain_mm"
            ] = float(
                r["switch_gain_mm"]
            )

            row[
                prefix + "weak_work_J"
            ] = float(
                r["weak_work_J"]
            )

            row[
                prefix + "strong_work_J"
            ] = float(
                r["strong_work_J"]
            )

        row["Rmin_change_low_minus_ref_mm"] = (
            row["low_Rmin_mm"]
            - row["reference_Rmin_mm"]
        )

        row["low_total_work_J"] = (
            row["low_weak_work_J"]
            + row["low_strong_work_J"]
        )

        row["reference_total_work_J"] = (
            row["reference_weak_work_J"]
            + row["reference_strong_work_J"]
        )

        paired.append(row)

    return paired


def print_confusion(rows, band, regime):
    print(
        "\nCONFUSION:",
        band,
        regime,
    )

    for mode in MODES:
        subset = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        counts = Counter(
            r[regime + "_decision"]
            for r in subset
        )

        print(
            f"  {mode:12s}",
            dict(counts)
        )


def diagnose_unknown(rows, band):
    unknown = [
        r for r in rows
        if r["truth_PRIVATE"]
        == "saturating"
    ]

    failed = [
        r for r in unknown
        if r["low_outcome"]
        == "unknown_false_accept"
    ]

    print("\n" + "=" * 70)
    print("BAND:", band)
    print("UNKNOWN-MECHANISM FAILURE")
    print("=" * 70)

    print(
        "Low false accepts:",
        len(failed),
        "/", len(unknown)
    )

    print(
        "Accepted as:",
        dict(Counter(
            r["low_decision"]
            for r in failed
        ))
    )

    print(
        "Reference outcomes for these cases:",
        dict(Counter(
            r["reference_outcome"]
            for r in failed
        ))
    )

    print(
        "Moderate outcomes for these cases:",
        dict(Counter(
            r["moderate_outcome"]
            for r in failed
        ))
    )

    if failed:
        for regime in REGIMES:
            errors = [
                r[regime + "_Rmin_mm"]
                for r in failed
            ]

            print(
                regime,
                "Rmin median:",
                round(median(errors), 4),
                "mm",
                "range:",
                (
                    round(min(errors), 4),
                    round(max(errors), 4),
                ),
            )

        differences = [
            r["Rmin_change_low_minus_ref_mm"]
            for r in failed
        ]

        print(
            "Low-Reference Rmin change median:",
            round(median(differences), 4),
            "mm"
        )

    return failed


def diagnose_known(rows, band):
    print("\nKNOWN-MECHANISM DEGRADATION")

    summary = {}

    for mode in MODES[:-1]:
        subset = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        ref_correct = sum(
            r["reference_outcome"] == "correct"
            for r in subset
        )

        low_correct = sum(
            r["low_outcome"] == "correct"
            for r in subset
        )

        worsened = sum(
            r["reference_outcome"] == "correct"
            and r["low_outcome"] != "correct"
            for r in subset
        )

        improved = sum(
            r["reference_outcome"] != "correct"
            and r["low_outcome"] == "correct"
            for r in subset
        )

        summary[mode] = {
            "reference_correct": ref_correct,
            "low_correct": low_correct,
            "worsened": worsened,
            "improved": improved,
        }

        print(
            f"{mode:12s}",
            f"correct {ref_correct}->{low_correct}",
            f"worsened={worsened}",
            f"improved={improved}",
        )

    return summary


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    report = load_json(REPORT_PATH)
    audit = load_json(AUDIT_PATH)

    if report["smoke"]:
        raise RuntimeError(
            "Expected formal P3-A6K data"
        )

    threshold = (
        report["frozen_thresholds"][
            "tau_absolute_m"
        ] * 1000
    )

    print(
        "P3-A6L PAIRED FAILURE DIAGNOSIS"
    )
    print(
        "Frozen absolute threshold:",
        round(threshold, 4),
        "mm"
    )

    print(
        "No fitting, no model tuning, no E4."
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []
    failure_rows = []
    summaries = {}

    for band in BANDS:
        rows = build_paired(audit, band)

        all_rows.extend(rows)

        for regime in REGIMES:
            print_confusion(
                rows, band, regime
            )

        failed = diagnose_unknown(
            rows, band
        )

        failure_rows.extend(failed)

        known = diagnose_known(
            rows, band
        )

        summaries[band] = {
            "unknown_false_accept_count":
                len(failed),

            "unknown_wrong_law_counts":
                dict(Counter(
                    r["low_decision"]
                    for r in failed
                )),

            "known_degradation":
                known,
        }

    write_csv(
        OUT / "p3a6l_all_paired_PRIVATE.csv",
        all_rows,
    )

    write_csv(
        OUT / "p3a6l_low_unknown_failures_PRIVATE.csv",
        failure_rows,
    )

    output = {
        "protocol":
            "P3-A6L stage-1 paired failure audit",

        "frozen_absolute_cutoff_mm":
            threshold,

        "threshold_retuned": False,

        "E4_accessed": False,

        "summary": summaries,

        "limitations": [
            "Historical observational audit only",
            "No clean-trajectory replay",
            "No force-level comparison",
            "Cannot establish exact identifiability",
            "Same paired base scenes",
            "No new independent validation",
        ],
    }

    path = OUT / "p3a6l_stage1_report.json"

    path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A6L STAGE-1 "
        "FAILURE AUDIT PASS"
    )

    print("Saved:", path)


if __name__ == "__main__":
    main()
