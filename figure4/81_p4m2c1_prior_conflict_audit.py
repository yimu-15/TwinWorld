"""
P4-M2C1 — Observable Prior-Conflict Audit

Read-only diagnostic using frozen P4-M2C0 results.

Questions:
1. Can training-only domain conflict reveal prior mismatch?
2. Does model-selection improvement coexist with worse
   true-law trajectory fitting?
3. How often do finite SLSQP fits violate the ideal
   nested-domain training-loss ordering?

No simulation.
No refitting.
No learned detector.
No threshold selection.
No E4.
No historical result modification.

This is development-only evidence, not an independent
confirmation or a deployable adaptive policy.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "results" / "p4m2c0"
OUT = HERE / "results" / "p4m2c1"

LAWS = ("elastic", "deadzone", "cubic")
DOMAINS = ("wide", "outer_box")
BANDS = ("original", "above_original_range")

MODES = ("elastic", "deadzone", "nonlinear")

TRUE_LAW = {
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

AXES = {
    "elastic": ("alpha_high",),
    "deadzone": ("alpha_high", "gap_high"),
    "nonlinear": ("alpha_high", "gamma_high"),
}

LEVELS = ("M1", "M2", "M3")

OUTER_UPPER = {
    "alpha_high": 20.0,
    "gap_high": 0.18,
    "gamma_high": 1062.5,
}

WIDE_UPPER = {
    "alpha_high": 70.0,
    "gap_high": 0.20,
    "gamma_high": 2500.0,
}

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 2029051801

NESTED_TOL = 1e-6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def finite(x):
    return x is not None and np.isfinite(float(x))


def load_inputs(smoke):
    suffix = "smoke" if smoke else "dev"

    report_path = SOURCE_DIR / (
        f"p4m2c0_{suffix}_report.json"
    )

    audit_path = SOURCE_DIR / (
        f"p4m2c0_{suffix}_audit_PRIVATE.json"
    )

    report = load_json(report_path)
    rows = load_json(audit_path)

    n = 1 if smoke else 12

    assert report["smoke"] is smoke
    assert report["n_base_scenes"] == n
    assert report["E4_accessed"] is False
    assert report["truth_used_in_fitting"] is False
    assert report["development_only"] is True

    assert tuple(report["bands"]) == BANDS
    assert tuple(report["modes"]) == MODES
    assert tuple(report["laws"]) == LAWS
    assert tuple(report["domains"]) == DOMAINS

    if report["solver"] != "SLSQP":
        raise RuntimeError("Solver mismatch")

    for name, expected in report[
        "source_sha256"
    ].items():
        path = HERE / name

        if not path.is_file():
            raise FileNotFoundError(path)

        if sha(path) != expected:
            raise RuntimeError(
                "Historical source mismatch: " + name
            )

    expected_rows = n * 2 * 18

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"got {len(rows)}"
        )

    return report, rows, report_path, audit_path


def winner_from_fits(row, domain):

    fits = row["fits"][domain]

    for law in LAWS:
        item = fits[law]

        if not item["valid"]:
            return None

        if not item["converged"]:
            return None

        if not finite(item["validation_rms_mm"]):
            return None

    return min(
        LAWS,
        key=lambda law:
            fits[law]["validation_rms_mm"],
    )


def validate_rows(rows, n):

    indexed = {}

    for row in rows:
        sid = row["scene_id"]
        band = row["band"]
        mode = row["truth_PRIVATE"]
        axis = row["stress_axis"]
        level = row["level"]

        if not isinstance(sid, int) or not 0 <= sid < n:
            raise RuntimeError("Invalid scene ID")

        if band not in BANDS or mode not in MODES:
            raise RuntimeError("Invalid band or mode")

        if level == "M0":
            if axis != "matched":
                raise RuntimeError("Invalid matched row")
        else:
            if axis not in AXES[mode]:
                raise RuntimeError("Invalid stress axis")
            if level not in LEVELS:
                raise RuntimeError("Invalid stress level")

        key = (sid, band, mode, axis, level)

        if key in indexed:
            raise RuntimeError("Duplicate audit row")

        indexed[key] = row

        true_params = np.asarray(
            row["true_effective_PRIVATE"],
            dtype=float,
        )

        if not np.isfinite(true_params).all():
            raise RuntimeError("Invalid true parameters")

        if level != "M0":
            idx = 0 if axis == "alpha_high" else 2
            value = float(true_params[idx])

            if not (
                OUTER_UPPER[axis] + 1e-8
                < value
                < WIDE_UPPER[axis] + 1e-8
            ):
                raise RuntimeError(
                    "Invalid prior-mismatch construction"
                )

        for domain in DOMAINS:
            computed = winner_from_fits(row, domain)
            recorded = row[
                "outcomes_PRIVATE"
            ][domain]["winner"]

            if computed != recorded:
                raise RuntimeError(
                    "Recorded winner mismatch"
                )

            expected_correct = (
                computed == TRUE_LAW[mode]
                if computed is not None else None
            )

            if expected_correct != row[
                "outcomes_PRIVATE"
            ][domain]["correct_PRIVATE"]:
                raise RuntimeError(
                    "Recorded correctness mismatch"
                )

    for sid in range(n):
        for band in BANDS:
            for mode in MODES:

                required = [
                    (sid, band, mode, "matched", "M0")
                ]

                for axis in AXES[mode]:
                    for level in LEVELS:
                        required.append(
                            (sid, band, mode, axis, level)
                        )

                for key in required:
                    if key not in indexed:
                        raise RuntimeError(
                            "Missing expected audit case"
                        )

    print("[PASS] Complete case matrix")
    print("[PASS] True-parameter support audit")
    print("[PASS] Winner and correctness replay")

    return indexed


def analyze_row(row):

    fits = row["fits"]

    losses = {}

    violations = []

    for law in LAWS:

        w = fits["wide"][law]
        o = fits["outer_box"][law]

        if not (
            w["valid"] and o["valid"]
            and w["converged"] and o["converged"]
        ):
            raise RuntimeError(
                "Nonconverged fit: audit separately "
                "before scoring"
            )

        lw = float(w["train_mse"])
        lo = float(o["train_mse"])

        if not (
            np.isfinite(lw) and np.isfinite(lo)
            and lw >= 0 and lo >= 0
        ):
            raise RuntimeError("Invalid training loss")

        losses[law] = {
            "wide": lw,
            "outer_box": lo,
            "outer_minus_wide": lo - lw,
        }

        if lo < lw - NESTED_TOL:
            violations.append(law)

    best_wide = min(
        losses[law]["wide"] for law in LAWS
    )

    best_outer = min(
        losses[law]["outer_box"] for law in LAWS
    )

    raw_gap = (
        np.log1p(best_outer)
        - np.log1p(best_wide)
    )

    conflict = max(0.0, raw_gap)

    truth = TRUE_LAW[row["truth_PRIVATE"]]

    true_rms_wide = float(
        fits["wide"][truth]["validation_rms_mm"]
    )

    true_rms_outer = float(
        fits["outer_box"][truth]["validation_rms_mm"]
    )

    if true_rms_wide <= 0:
        raise RuntimeError("Invalid true-law RMS")

    wide_correct = row[
        "outcomes_PRIVATE"
    ]["wide"]["correct_PRIVATE"]

    outer_correct = row[
        "outcomes_PRIVATE"
    ]["outer_box"]["correct_PRIVATE"]

    if wide_correct is None or outer_correct is None:
        raise RuntimeError(
            "Unavailable model-selection decision"
        )

    return {
        "scene_id": row["scene_id"],
        "band": row["band"],
        "mode_PRIVATE": row["truth_PRIVATE"],
        "axis_PRIVATE": row["stress_axis"],
        "level_PRIVATE": row["level"],

        "best_train_mse_wide": best_wide,
        "best_train_mse_outer": best_outer,

        "raw_log_loss_gap": float(raw_gap),
        "conflict_score": float(conflict),

        "nested_violations": violations,
        "per_law_losses": losses,

        "true_law_rms_ratio_PRIVATE":
            true_rms_outer / true_rms_wide,

        "wide_correct_PRIVATE": bool(wide_correct),
        "outer_correct_PRIVATE": bool(outer_correct),

        "rescue_PRIVATE": (
            not wide_correct and outer_correct
        ),

        "harm_PRIVATE": (
            wide_correct and not outer_correct
        ),
    }


def auc_score(negatives, positives):

    neg = np.asarray(negatives, dtype=float)
    pos = np.asarray(positives, dtype=float)

    if len(neg) == 0 or len(pos) == 0:
        return None

    diff = pos[:, None] - neg[None, :]

    return float(
        np.mean(
            (diff > 0).astype(float)
            + 0.5 * (diff == 0)
        )
    )


def group_metrics(matched, stressed):

    assert len(matched) == len(stressed)

    n = len(stressed)

    auc = auc_score(
        [x["conflict_score"] for x in matched],
        [x["conflict_score"] for x in stressed],
    )

    changes = np.asarray([
        stressed[i]["conflict_score"]
        - matched[i]["conflict_score"]
        for i in range(n)
    ])

    rms_ratios = np.asarray([
        x["true_law_rms_ratio_PRIVATE"]
        for x in stressed
    ])

    return {
        "n_pairs": n,
        "matched_median_score": float(
            np.median([
                x["conflict_score"] for x in matched
            ])
        ),
        "stress_median_score": float(
            np.median([
                x["conflict_score"] for x in stressed
            ])
        ),
        "auroc": auc,
        "paired_score_increase_rate": float(
            np.mean(changes > 1e-10)
        ),
        "median_paired_score_change": float(
            np.median(changes)
        ),
        "wide_correct": sum(
            x["wide_correct_PRIVATE"] for x in stressed
        ),
        "outer_correct": sum(
            x["outer_correct_PRIVATE"] for x in stressed
        ),
        "outer_rescues": sum(
            x["rescue_PRIVATE"] for x in stressed
        ),
        "outer_harms": sum(
            x["harm_PRIVATE"] for x in stressed
        ),
        "true_law_median_rms_ratio": float(
            np.median(rms_ratios)
        ),
        "true_law_rms_worse_rate": float(
            np.mean(rms_ratios > 1.01)
        ),
    }


def bootstrap_auc(matched, stressed, smoke, seed):

    n = len(matched)

    if smoke or n < 10:
        return None

    rng = np.random.default_rng(seed)

    aucs = []
    deltas = []

    m = np.asarray([
        x["conflict_score"] for x in matched
    ])

    s = np.asarray([
        x["conflict_score"] for x in stressed
    ])

    for _ in range(N_BOOTSTRAP):

        # The same base-scene indices are used
        # on matched and stressed observations.

        ids = rng.integers(0, n, size=n)

        mb = m[ids]
        sb = s[ids]

        aucs.append(
            auc_score(mb, sb)
        )

        deltas.append(
            float(np.median(sb - mb))
        )

    return {
        "n_clusters": n,
        "cluster_unit": "base_scene_id",
        "n_bootstrap": N_BOOTSTRAP,
        "auc_95CI": [
            float(np.quantile(aucs, 0.025)),
            float(np.quantile(aucs, 0.975)),
        ],
        "paired_score_change_95CI": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("P4-M2C1 OBSERVABLE PRIOR-CONFLICT AUDIT")
    print("=" * 72)

    print("Smoke:", args.smoke)
    print("Read-only historical inputs.")
    print("No MuJoCo simulation.")
    print("No refitting. No E4.")

    report, rows, report_path, audit_path = (
        load_inputs(args.smoke)
    )

    n = report["n_base_scenes"]

    indexed = validate_rows(rows, n)

    analyzed = {
        key: analyze_row(row)
        for key, row in indexed.items()
    }

    violations = Counter()

    for row in analyzed.values():
        for law in row["nested_violations"]:
            violations[law] += 1

    print("\nNESTED OPTIMIZATION AUDIT")
    print("Total candidate comparisons:",
          len(rows) * len(LAWS))
    print("Violations by law:", dict(violations))
    print("Total violations:", sum(violations.values()))

    summary = {}

    counter = 0

    for band in BANDS:

        summary[band] = {}

        for mode in MODES:

            summary[band][mode] = {}

            for axis in AXES[mode]:

                matched = [
                    analyzed[
                        (sid, band, mode, "matched", "M0")
                    ]
                    for sid in range(n)
                ]

                summary[band][mode][axis] = {}

                for level in LEVELS:

                    stressed = [
                        analyzed[
                            (sid, band, mode, axis, level)
                        ]
                        for sid in range(n)
                    ]

                    metrics = group_metrics(
                        matched, stressed
                    )

                    uncertainty = bootstrap_auc(
                        matched,
                        stressed,
                        args.smoke,
                        BOOTSTRAP_SEED + counter,
                    )

                    counter += 1

                    summary[band][mode][axis][level] = {
                        "metrics": metrics,
                        "bootstrap": uncertainty,
                    }

                    print(
                        f"\n[{band}] [{mode}] "
                        f"[{axis}] [{level}]"
                    )

                    print(
                        "  AUROC:",
                        round(metrics["auroc"], 4),
                        "| Paired score increase:",
                        round(
                            metrics[
                                "paired_score_increase_rate"
                            ], 4
                        ),
                    )

                    print(
                        "  Wide/Outer correct:",
                        metrics["wide_correct"],
                        "/",
                        metrics["outer_correct"],
                    )

                    print(
                        "  Rescue/Harm:",
                        metrics["outer_rescues"],
                        "/",
                        metrics["outer_harms"],
                    )

                    print(
                        "  True-law RMS ratio:",
                        round(
                            metrics[
                                "true_law_median_rms_ratio"
                            ], 4
                        ),
                    )

                    if uncertainty is not None:
                        print(
                            "  AUROC 95% CI:",
                            uncertainty["auc_95CI"],
                        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke else "dev"
    )

    output_report = OUT / (
        f"p4m2c1_{suffix}_report.json"
    )

    output_audit = OUT / (
        f"p4m2c1_{suffix}_audit_PRIVATE.json"
    )

    result = {
        "protocol":
            "P4-M2C1 observable prior-conflict audit",
        "smoke": args.smoke,
        "development_only": True,
        "n_base_scenes": n,
        "n_cases": len(rows),
        "n_candidate_fit_pairs":
            len(rows) * len(LAWS),
        "nested_optimization_violations":
            dict(violations),
        "nested_violation_count":
            sum(violations.values()),
        "score_definition":
            "max(0, log1p(min_outer_train_mse)"
            "-log1p(min_wide_train_mse))",
        "score_uses_truth": False,
        "score_uses_mid_validation": False,
        "summary": summary,
        "source_inputs": {
            "report": str(report_path),
            "report_sha256": sha(report_path),
            "audit": str(audit_path),
            "audit_sha256": sha(audit_path),
        },
        "source_sha256": {
            Path(__file__).name:
                sha(Path(__file__)),
        },
        "E4_accessed": False,
        "new_simulations": 0,
        "new_fits": 0,
        "thresholds_selected": False,
        "limitations": [
            "Development data only",
            "No independent evaluation",
            "Thirty correlated stress comparisons",
            "Small 12-cluster bootstrap",
            "Local SLSQP optimization",
            "AUROC is diagnostic, not calibrated",
            "No deployment policy",
            "No unknown-mechanism detection",
        ],
    }

    output_report.write_text(
        json.dumps(
            result, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    output_audit.write_text(
        json.dumps(
            list(analyzed.values()),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("P4-M2C1 COMPLETE")
    print("=" * 72)

    print("Saved:", output_report)
    print("Saved:", output_audit)


if __name__ == "__main__":
    main()