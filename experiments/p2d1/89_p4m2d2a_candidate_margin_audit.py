
"""P4-M2D2A — Read-only candidate-margin mechanism audit.

Input: P4-M2D1 fresh report and 384-case audit.
No simulation, fitting, training, action selection or E4.
"""

import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE / "results"

INPUT = BASE / "p4m2d1"
OUT = BASE / "p4m2d2a"

R = INPUT / "p4m2d1_fresh_report.json"
A = INPUT / "p4m2d1_fresh_audit_PRIVATE.json"

P88 = (
    HERE /
    "88_p4m2d1_fresh_action_prior_interaction.py"
)

LAWS = ("elastic", "deadzone", "cubic")

METHODS = (
    "MID",
    "HIGH_ONLY",
    "MID_PLUS_HIGH",
)

CONFIGS = ("WWW", "OOO", "WOO")

TRUTH = {
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

BOOT_SEED = 2029101801
N_BOOT = 2000


def need(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def read(path):
    need(path.is_file(), f"Missing file: {path}")

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def key(row):
    return (
        int(row["scene_id"]),
        row["band"],
        row["mode_PRIVATE"],
        row["axis_PRIVATE"],
        row["level_PRIVATE"],
    )


def law_rms(row, domain, law, method):

    m = float(
        row["mid_rms_PRIVATE"][domain][law]
    )

    h = float(
        row["high_rms_PRIVATE"][domain][law]
    )

    need(
        np.isfinite(m) and m >= 0
        and np.isfinite(h) and h >= 0,
        "Invalid RMS data",
    )

    if method == "MID":
        return m

    if method == "HIGH_ONLY":
        return h

    need(
        method == "MID_PLUS_HIGH",
        "Unknown scoring method",
    )

    return float(
        np.sqrt((m*m + h*h) / 2.0)
    )


def scores(row, config, method):

    result = {}

    for i, law in enumerate(LAWS):

        domain = (
            "outer_box"
            if config[i] == "O"
            else "wide"
        )

        result[law] = law_rms(
            row, domain, law, method
        )

    return result


def decision(row, config, method):

    values = scores(row, config, method)

    return min(
        LAWS,
        key=lambda law: values[law],
    )


def correct(row, method, config):

    return int(
        row["decisions"][method][config][
            "correct_PRIVATE"
        ]
    )


def accuracy(rows, method, config):

    need(len(rows) > 0, "Empty subset")

    return float(
        np.mean([
            correct(row, method, config)
            for row in rows
        ])
    )


def macro(rows, method, config):

    rates = []

    for mode in TRUTH:

        subset = [
            row for row in rows
            if row["mode_PRIVATE"] == mode
        ]

        need(
            bool(subset),
            f"Missing mode: {mode}",
        )

        rates.append(
            accuracy(subset, method, config)
        )

    return float(np.mean(rates))


def interaction(rows):

    joint = (
        accuracy(rows, "MID_PLUS_HIGH", "WOO")
        -
        accuracy(rows, "MID_PLUS_HIGH", "OOO")
    )

    mid = (
        accuracy(rows, "MID", "WOO")
        -
        accuracy(rows, "MID", "OOO")
    )

    return float(joint - mid)


def verify(report, rows):

    need(
        report["smoke"] is False
        and report["n_cases"] == 384
        and report["n_base_scenes"] == 24,
        "Formal P88 report required",
    )

    need(
        not report["E4_accessed"]
        and not report["thresholds_tuned"]
        and not report["high_used_for_fitting"]
        and not report["truth_used_for_fitting"],
        "Protocol or leakage flag",
    )

    need(
        P88.is_file()
        and sha(P88) ==
        report["source_sha256"][P88.name],
        "P88 source identity mismatch",
    )

    need(
        len(rows) == 384
        and len(set(map(key, rows))) == 384,
        "Record identity mismatch",
    )

    counts = Counter(
        row["scene_id"]
        for row in rows
    )

    need(
        set(counts) == set(range(24))
        and all(n == 16 for n in counts.values()),
        "Scene-cluster matrix mismatch",
    )

    nested = Counter(
        law
        for row in rows
        for law in row["nested_violations_PRIVATE"]
    )

    need(
        dict(nested) ==
        report["nested_solver_violations"],
        "Solver-gap count mismatch",
    )

    for j, row in enumerate(rows):

        for config in CONFIGS:

            for method in METHODS:

                expected = decision(
                    row, config, method
                )

                stored = row["decisions"][
                    method
                ][config]

                need(
                    stored["available"] is True
                    and stored["winner"] == expected,
                    f"Prediction mismatch: "
                    f"{j}/{config}/{method}",
                )

                truth = TRUTH[
                    row["mode_PRIVATE"]
                ]

                need(
                    stored["correct_PRIVATE"] is
                    (expected == truth),
                    f"Label mismatch: "
                    f"{j}/{config}/{method}",
                )

    for method in METHODS:

        for config in CONFIGS:

            historical = report[
                "metrics"
            ][method][config]

            total = sum(
                correct(row, method, config)
                for row in rows
            )

            need(
                total == historical["correct"],
                f"Count mismatch: {method}/{config}",
            )

            need(
                np.isclose(
                    accuracy(rows, method, config),
                    historical["accuracy"],
                    atol=1e-12,
                    rtol=0,
                ),
                "Accuracy replay mismatch",
            )

            need(
                np.isclose(
                    macro(rows, method, config),
                    historical["macro"],
                    atol=1e-12,
                    rtol=0,
                ),
                "Macro replay mismatch",
            )

    need(
        np.isclose(
            interaction(rows),
            report["comparisons"][
                "PRIMARY_action_prior_interaction_overall"
            ],
            atol=1e-12,
            rtol=0,
        ),
        "Predeclared interaction mismatch",
    )


def ec_gap(row, method, config):

    # Positive: Cubic favored.
    # Negative: Elastic favored.

    values = scores(row, config, method)

    return float(
        values["elastic"] - values["cubic"]
    )


def ec_summary(rows, config):

    def stats(values):

        v = np.asarray(values, dtype=float)

        return {
            "median_mm": float(np.median(v)),
            "q25_mm": float(np.quantile(v, .25)),
            "q75_mm": float(np.quantile(v, .75)),
            "cubic_favored": int((v > 0).sum()),
            "elastic_favored": int((v < 0).sum()),
        }

    return {
        method: stats([
            ec_gap(row, method, config)
            for row in rows
        ])
        for method in METHODS
    }


def subset_result(rows):

    return {
        "n": len(rows),

        "interaction": interaction(rows),

        "WOO_minus_OOO_MID": (
            accuracy(rows, "MID", "WOO")
            -
            accuracy(rows, "MID", "OOO")
        ),

        "WOO_minus_OOO_MID_PLUS_HIGH": (
            accuracy(rows, "MID_PLUS_HIGH", "WOO")
            -
            accuracy(rows, "MID_PLUS_HIGH", "OOO")
        ),

        "WOO_minus_OOO_HIGH_ONLY": (
            accuracy(rows, "HIGH_ONLY", "WOO")
            -
            accuracy(rows, "HIGH_ONLY", "OOO")
        ),
    }


def cluster_bootstrap(rows, seed):

    groups = defaultdict(list)

    for row in rows:
        groups[int(row["scene_id"])].append(row)

    ids = sorted(groups)

    need(
        len(ids) == 24,
        "Expected all 24 scene clusters",
    )

    rng = np.random.default_rng(seed)

    values = []

    for _ in range(N_BOOT):

        sampled = rng.choice(
            ids,
            size=len(ids),
            replace=True,
        )

        sample = [
            row
            for sid in sampled
            for row in groups[int(sid)]
        ]

        values.append(interaction(sample))

    return {
        "cluster_unit": "scene_id",
        "n_clusters": 24,
        "n_bootstrap": N_BOOT,
        "seed": seed,

        "CI_95": [
            float(np.quantile(values, .025)),
            float(np.quantile(values, .975)),
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
    print("P4-M2D2A — CANDIDATE MARGIN AUDIT")
    print("=" * 72)

    report = read(R)
    all_rows = read(A)

    print("[1] Frozen provenance and prediction parity")

    verify(report, all_rows)

    print(
        "[PASS] 384 records, 9 model-selection variants"
    )

    print(
        "[PASS] Historical report and code identity"
    )

    work = (
        [r for r in all_rows if r["scene_id"] == 0]
        if args.smoke
        else all_rows
    )

    print("\n[2] Class-level interaction decomposition")

    mode_contrib = {}

    for mode in TRUTH:

        subset = [
            row for row in work
            if row["mode_PRIVATE"] == mode
        ]

        d_mid = sum(
            correct(row, "MID", "WOO")
            -
            correct(row, "MID", "OOO")
            for row in subset
        )

        d_joint = sum(
            correct(row, "MID_PLUS_HIGH", "WOO")
            -
            correct(row, "MID_PLUS_HIGH", "OOO")
            for row in subset
        )

        contribution = d_joint - d_mid

        mode_contrib[mode] = {
            "n": len(subset),
            "mid_difference_cases": int(d_mid),
            "joint_difference_cases": int(d_joint),
            "interaction_contribution_cases":
                int(contribution),
        }

        print(
            f"{mode}: "
            f"MID={d_mid:+d}, "
            f"JOINT={d_joint:+d}, "
            f"contribution={contribution:+d}"
        )

    print("\n[3] Elastic–Cubic margin decomposition")

    ec = {}

    for mode in ("elastic", "nonlinear"):

        subset = [
            row for row in work
            if row["mode_PRIVATE"] == mode
            and row["level_PRIVATE"] == "M2"
        ]

        ec[mode + "_M2"] = {
            config: ec_summary(subset, config)
            for config in ("WOO", "OOO")
        }

        for config in ("WOO", "OOO"):

            item = ec[mode + "_M2"][config]

            print(
                f"{mode} M2 / {config}: "
                f"MID median="
                f"{item['MID']['median_mm']:+.4f} mm; "
                f"JOINT median="
                f"{item['MID_PLUS_HIGH']['median_mm']:+.4f} mm"
            )

    print("\n[4] Stratified robustness")

    stratified = {}

    groups = {
        "original": [
            r for r in work
            if r["band"] == "original"
        ],

        "above_original_range": [
            r for r in work
            if r["band"] == "above_original_range"
        ],

        "M0": [
            r for r in work
            if r["level_PRIVATE"] == "M0"
        ],

        "M2": [
            r for r in work
            if r["level_PRIVATE"] == "M2"
        ],

        "without_nested_violations": [
            r for r in work
            if not r["nested_violations_PRIVATE"]
        ],
    }

    for name, subset in groups.items():

        item = subset_result(subset)
        stratified[name] = item

        print(
            f"{name}: n={item['n']}, "
            f"interaction={item['interaction']:+.6f}"
        )

    bootstrap = None

    if not args.smoke:

        bootstrap = {
            "all": cluster_bootstrap(
                work,
                BOOT_SEED,
            ),

            "without_nested_violations":
                cluster_bootstrap(
                    groups["without_nested_violations"],
                    BOOT_SEED + 1,
                ),
        }

    print("\n[5] Overall interaction")

    print(
        f"Interaction = "
        f"{100 * interaction(work):+.3f} pp"
    )

    print("\n[6] Save read-only diagnostic")

    trace = []

    for row in work:

        trace.append({
            "key_PRIVATE": list(key(row)),

            "truth_PRIVATE":
                TRUTH[row["mode_PRIVATE"]],

            "nested_solver_violation": bool(
                row["nested_violations_PRIVATE"]
            ),

            "WOO_MID_winner":
                row["decisions"]["MID"]["WOO"][
                    "winner"
                ],

            "WOO_JOINT_winner":
                row["decisions"]["MID_PLUS_HIGH"][
                    "WOO"
                ]["winner"],

            "OOO_MID_winner":
                row["decisions"]["MID"]["OOO"][
                    "winner"
                ],

            "OOO_JOINT_winner":
                row["decisions"]["MID_PLUS_HIGH"][
                    "OOO"
                ]["winner"],

            "elastic_cubic_gap_mm_PRIVATE": {
                config: {
                    method: ec_gap(
                        row, method, config
                    )
                    for method in METHODS
                }
                for config in ("WOO", "OOO")
            },
        })

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = "smoke" if args.smoke else "fresh"

    report_path = OUT / (
        f"p4m2d2a_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2d2a_{suffix}_audit_PRIVATE.json"
    )

    result = {
        "protocol":
            "P4-M2D2A candidate-ranking mechanism audit",

        "smoke": args.smoke,

        "n_cases": len(work),

        "overall_interaction": interaction(work),

        "mode_contributions_PRIVATE": mode_contrib,

        "elastic_cubic_margin_PRIVATE": ec,

        "stratified_descriptive": stratified,

        "cluster_bootstrap": bootstrap,

        "source_sha256": {
            R.name: sha(R),
            A.name: sha(A),
            P88.name: sha(P88),
        },

        "E4_accessed": False,
        "new_fits": 0,
        "new_simulations": 0,

        "limitations": [
            "Mechanism interpretation is post hoc",
            "Same P88 physical scenes",
            "Labels only used for offline analysis",
            "No causal identifiability proof",
            "More observations in MID_PLUS_HIGH",
            "Original solver gaps remain",
            "Same MuJoCo simulator family",
        ],
    }

    report_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            trace,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("Saved:", report_path)
    print("Saved:", audit_path)

    print("\nP4-M2D2A COMPLETE")


if __name__ == "__main__":
    main()
