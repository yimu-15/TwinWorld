"""
P4-M2C2C — Solver-Corrected Decision-Risk Audit

Post-hoc sensitivity analysis of 21 Wide-domain
Deadzone solver-search gaps.

Inputs:
  P4-M2C0: 432 full fitting and validation records
  P4-M2C1: 432 observable-conflict audit records
  P4-M2C2B: 21 repaired Wide feasible solutions

Method:
  1. Verify frozen provenance and exact case alignment.
  2. Reconstruct frozen development observations.
  3. Replay original Mid predictions for parity.
  4. Evaluate repaired parameters on the same Mid data.
  5. Recompute winners, accuracy, rescue, harm and conflict.
  6. Report scene-cluster bootstrap sensitivity.

No new scenes.
No optimizer calls.
No truth-assisted fitting.
No threshold tuning.
No E4.
No historical output modification.

IMPORTANT:
The repaired Wide solution uses an Outer-derived start.
This is NOT an independent, matched-compute Wide baseline.
"""

import argparse
import copy
import hashlib
import json

from collections import defaultdict
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "p4m2c2c"

P80 = HERE / "80_p4m2c0_prior_mismatch_stress.py"
P83 = HERE / "83_p4m2c2b_warm_start_transfer.py"

FILES = {
    "c0_report":
        RESULTS / "p4m2c0" / "p4m2c0_dev_report.json",

    "c0_audit":
        RESULTS / "p4m2c0" /
        "p4m2c0_dev_audit_PRIVATE.json",

    "c1_report":
        RESULTS / "p4m2c1" / "p4m2c1_dev_report.json",

    "c1_audit":
        RESULTS / "p4m2c1" /
        "p4m2c1_dev_audit_PRIVATE.json",

    "b_report":
        RESULTS / "p4m2c2b" /
        "p4m2c2b_dev_report.json",

    "b_audit":
        RESULTS / "p4m2c2b" /
        "p4m2c2b_dev_audit_PRIVATE.json",
}

LAWS = ("elastic", "deadzone", "cubic")

TRUE_LAW = {
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

BANDS = (
    "original",
    "above_original_range",
)

KEY_TOL = 1e-7
RMS_TOL_MM = 1e-6

REPETITIONS = 2000
SEED = 2029052001


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def must(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(a, b, tol=KEY_TOL):
    return bool(
        np.isclose(
            float(a),
            float(b),
            rtol=tol,
            atol=tol,
        )
    )


def key_c0(row):
    return (
        int(row["scene_id"]),
        row["band"],
        row["truth_PRIVATE"],
        row["stress_axis"],
        row["level"],
    )


def key_c1(row):
    return (
        int(row["scene_id"]),
        row["band"],
        row["mode_PRIVATE"],
        row["axis_PRIVATE"],
        row["level_PRIVATE"],
    )


def key_b(row):
    return (
        int(row["scene_id"]),
        row["band"],
        row["mode"],
        row["axis"],
        row["level"],
    )


def index(rows, key_fn, label):
    result = {}

    for row in rows:
        key = key_fn(row)

        must(
            key not in result,
            f"Duplicate {label} key: {key}",
        )

        result[key] = row

    return result


def resolve_provenance(name):

    if name.endswith(".py"):
        return HERE / name

    if name.startswith("p4m2c2a_"):
        return RESULTS / "p4m2c2a" / name

    if name.startswith("p4m2c1_"):
        return RESULTS / "p4m2c1" / name

    if name.startswith("p4m2c0_"):
        return RESULTS / "p4m2c0" / name

    raise RuntimeError(
        f"Unexpected provenance item: {name}"
    )


def verify_and_index():

    data = {
        name: load(path)
        for name, path in FILES.items()
    }

    c0r = data["c0_report"]
    c1r = data["c1_report"]
    br = data["b_report"]

    for report in (c0r, c1r, br):

        must(
            report["smoke"] is False,
            "Formal report required",
        )

        must(
            report["development_only"] is True,
            "Development-only requirement failed",
        )

        must(
            report["E4_accessed"] is False,
            "E4 access flag detected",
        )

    must(
        c0r["truth_used_in_fitting"] is False,
        "C0 truth-leakage flag",
    )

    must(
        br["uses_truth_in_fitting"] is False,
        "C2B truth-leakage flag",
    )

    must(
        br["uses_mid_validation"] is False,
        "C2B Mid-fitting flag",
    )

    must(
        c1r["score_uses_mid_validation"] is False,
        "C1 score-leakage flag",
    )

    must(
        c0r["n_base_scenes"] == 12
        and c1r["n_base_scenes"] == 12,
        "Scene count mismatch",
    )

    must(
        c1r["n_cases"] == 432
        and br["n_cases"] == 21,
        "Expected case counts mismatch",
    )

    must(
        tuple(c0r["laws"]) == LAWS,
        "Candidate-law mismatch",
    )

    must(
        tuple(c0r["bands"]) == BANDS,
        "Band mismatch",
    )

    must(
        tuple(c0r["domains"]) ==
        ("wide", "outer_box"),
        "Domain mismatch",
    )

    must(
        c1r["nested_violation_count"] == 21,
        "Historical violation count mismatch",
    )

    must(
        c1r["source_inputs"]["report_sha256"]
        == sha(FILES["c0_report"]),
        "C1-C0 report SHA mismatch",
    )

    must(
        c1r["source_inputs"]["audit_sha256"]
        == sha(FILES["c0_audit"]),
        "C1-C0 audit SHA mismatch",
    )

    for report in (c0r, c1r, br):

        for name, digest in report.get(
            "source_sha256", {}
        ).items():

            path = resolve_provenance(name)

            must(
                path.is_file() and sha(path) == digest,
                f"Source SHA mismatch: {path}",
            )

    c0 = index(
        data["c0_audit"],
        key_c0,
        "C0",
    )

    c1 = index(
        data["c1_audit"],
        key_c1,
        "C1",
    )

    repairs = index(
        data["b_audit"],
        key_b,
        "C2B",
    )

    must(
        len(c0) == len(c1) == 432
        and set(c0) == set(c1),
        "432-row alignment failed",
    )

    must(
        len(repairs) == 21,
        "Expected 21 unique repair cases",
    )

    must(
        sum(
            len(r["nested_violations"])
            for r in c1.values()
        ) == 21,
        "Unexpected violation structure",
    )

    expected_keys = {
        key
        for key, row in c1.items()
        if "deadzone" in row["nested_violations"]
    }

    must(
        set(repairs) == expected_keys,
        "Repairs must match all 21 violations",
    )

    must(
        sorted(
            row["case"]
            for row in repairs.values()
        ) == list(range(1, 22)),
        "C2B case sequence invalid",
    )

    for key, row in c0.items():

        r1 = c1[key]

        for law in LAWS:

            for domain in ("wide", "outer_box"):

                original = row["fits"][domain][law]

                must(
                    original["valid"]
                    and original["converged"],
                    f"Unusable fit: {key}/{domain}/{law}",
                )

                recorded = r1[
                    "per_law_losses"
                ][law][domain]

                must(
                    close(
                        original["train_mse"],
                        recorded,
                    ),
                    f"Train-loss mismatch: "
                    f"{key}/{domain}/{law}",
                )

        for domain in ("wide", "outer_box"):

            field = (
                "wide_correct_PRIVATE"
                if domain == "wide"
                else "outer_correct_PRIVATE"
            )

            original_correct = (
                row["outcomes_PRIVATE"][
                    domain
                ]["correct_PRIVATE"] is True
            )

            must(
                original_correct == r1[field],
                f"Correctness mismatch: {key}/{domain}",
            )

    for key, row in repairs.items():

        source = c0[key]

        must(
            row["warm_start"]["success"]
            and row["warm_start"]["final_feasible"]
            and row["best_so_far"]["closes_historical_gap"],
            f"Repair flag mismatch: {key}",
        )

        must(
            close(
                row["historical_wide_fun"],
                source["fits"]["wide"][
                    "deadzone"
                ]["train_mse"],
            ),
            f"Historical Wide mismatch: {key}",
        )

        must(
            close(
                row["historical_outer_fun"],
                source["fits"]["outer_box"][
                    "deadzone"
                ]["train_mse"],
            ),
            f"Historical Outer mismatch: {key}",
        )

        must(
            float(row["best_so_far"]["fun"])
            <= float(row["historical_outer_fun"])
            + KEY_TOL,
            f"Repair worse than Outer: {key}",
        )

    print(
        "[PASS] Frozen-source hashes and "
        "C0/C1/C2B linkage"
    )

    print(
        "[PASS] 432 paired records; "
        "21 exactly matched repairs"
    )

    return c0, c1, repairs


def winner(fits, domain):

    for law in LAWS:

        item = fits[domain][law]

        if not (
            item["valid"]
            and item["converged"]
        ):
            return None

        value = item["validation_rms_mm"]

        if value is None or not np.isfinite(value):
            return None

    return min(
        LAWS,
        key=lambda law:
        fits[domain][law]["validation_rms_mm"],
    )


def conflict(fits):

    wide = min(
        float(fits["wide"][law]["train_mse"])
        for law in LAWS
    )

    outer = min(
        float(fits["outer_box"][law]["train_mse"])
        for law in LAWS
    )

    must(
        wide >= 0 and outer >= 0,
        "Invalid training MSE",
    )

    raw = float(
        np.log1p(outer) - np.log1p(wide)
    )

    return float(max(0.0, raw)), raw


def replay_mid(
    row,
    scenes,
    c0,
    p74,
    p75,
    p77,
    sim,
    api,
    commands,
    noise_seed,
    p80,
    make_objective,
):

    key = key_b(row)

    sid, band, mode, axis, level = key

    scene = scenes[sid][band]

    if level != "M0":

        scene, _ = p80["make_stress_scene"](
            p74,
            p77,
            scene,
            mode,
            axis,
            level,
        )

    probes = p75["create_noisy_probes"](
        sim,
        api,
        commands,
        scene,
        sid,
        band,
        mode,
        noise_seed,
    )

    training = [
        probes[name]
        for name in p77["TRAIN"]
    ]

    mid = probes[p77["VALID"]]

    source = c0[key]["fits"]

    objective = make_objective(training, p77)

    # Historical parity: verify both domains
    # before evaluating the repaired solution.

    for domain in ("wide", "outer_box"):

        old = source[domain]["deadzone"]

        old_mse = float(
            objective(old["params_PRIVATE"])
        )

        old_mid = float(
            p75["rms_mm"](
                mid.predict(
                    "deadzone",
                    old["params_PRIVATE"],
                ),
                mid.y_b,
            )
        )

        must(
            close(old_mse, old["train_mse"]),
            f"Training replay failed: {key}/{domain}",
        )

        must(
            abs(
                old_mid -
                float(old["validation_rms_mm"])
            ) <= RMS_TOL_MM,
            f"Mid replay failed: {key}/{domain}",
        )

    params = row["best_so_far"]["params"]

    must(
        p77["is_feasible"](
            "deadzone",
            params,
            "wide",
        ),
        f"Wide-infeasible repair: {key}",
    )

    new_mse = float(objective(params))

    must(
        close(
            new_mse,
            row["best_so_far"]["fun"],
        ),
        f"Repaired train-loss replay failed: {key}",
    )

    must(
        new_mse
        <= float(
            source["outer_box"]["deadzone"][
                "train_mse"
            ]
        ) + KEY_TOL,
        f"Repair exceeds Outer incumbent: {key}",
    )

    new_mid = float(
        p75["rms_mm"](
            mid.predict("deadzone", params),
            mid.y_b,
        )
    )

    must(
        np.isfinite(new_mid),
        f"Nonfinite repaired Mid RMS: {key}",
    )

    return {
        "key": key,

        "old_wide_mid_mm":
            source["wide"]["deadzone"][
                "validation_rms_mm"
            ],

        "new_wide_mid_mm": new_mid,

        "old_wide_train_mse":
            source["wide"]["deadzone"]["train_mse"],

        "new_wide_train_mse": new_mse,

        "outer_mid_mm":
            source["outer_box"]["deadzone"][
                "validation_rms_mm"
            ],
    }


def auc(scores, labels):

    positive = np.asarray(
        [
            score
            for score, label in zip(scores, labels)
            if label
        ],
        dtype=float,
    )

    negative = np.asarray(
        [
            score
            for score, label in zip(scores, labels)
            if not label
        ],
        dtype=float,
    )

    if len(positive) == 0 or len(negative) == 0:
        return None

    differences = (
        positive[:, None] - negative[None, :]
    )

    return float(
        np.mean(
            (differences > 0).astype(float)
            + 0.5 * (differences == 0)
        )
    )


def summarize(rows):

    n = len(rows)

    if not n:
        return None

    before = np.asarray([
        int(r["wide_correct_before_PRIVATE"])
        for r in rows
    ])

    after = np.asarray([
        int(r["wide_correct_after_PRIVATE"])
        for r in rows
    ])

    outer = np.asarray([
        int(r["outer_correct_PRIVATE"])
        for r in rows
    ])

    return {
        "n": n,

        "repaired_cases": sum(
            r["repaired"] for r in rows
        ),

        "wide_correct_before": int(before.sum()),
        "wide_correct_after": int(after.sum()),
        "outer_correct": int(outer.sum()),

        "rescue_before": sum(
            r["rescue_before_PRIVATE"]
            for r in rows
        ),

        "rescue_after": sum(
            r["rescue_after_PRIVATE"]
            for r in rows
        ),

        "harm_before": sum(
            r["harm_before_PRIVATE"]
            for r in rows
        ),

        "harm_after": sum(
            r["harm_after_PRIVATE"]
            for r in rows
        ),

        "delta_outer_minus_wide_before":
            float(np.mean(outer - before)),

        "delta_outer_minus_wide_after":
            float(np.mean(outer - after)),

        "auroc_conflict_to_harm_before": auc(
            [r["conflict_before"] for r in rows],
            [r["harm_before_PRIVATE"] for r in rows],
        ),

        "auroc_conflict_to_harm_after": auc(
            [r["conflict_after"] for r in rows],
            [r["harm_after_PRIVATE"] for r in rows],
        ),
    }


def clustered_bootstrap(rows):

    ids = sorted({
        r["scene_id"] for r in rows
    })

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["scene_id"]].append(row)

    rng = np.random.default_rng(SEED)

    changes = []

    for _ in range(REPETITIONS):

        sampled_ids = rng.choice(
            ids,
            size=len(ids),
            replace=True,
        )

        sample = [
            row
            for sid in sampled_ids
            for row in grouped[int(sid)]
        ]

        changes.append(
            float(
                np.mean([
                    int(
                        r["wide_correct_after_PRIVATE"]
                    ) -
                    int(
                        r["wide_correct_before_PRIVATE"]
                    )
                    for r in sample
                ])
            )
        )

    return {
        "cluster_unit": "scene_id",
        "n_clusters": len(ids),
        "n_bootstrap": REPETITIONS,

        "wide_accuracy_change_95CI": [
            float(np.quantile(changes, 0.025)),
            float(np.quantile(changes, 0.975)),
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
    print(
        "P4-M2C2C — SOLVER-CORRECTED "
        "DECISION-RISK AUDIT"
    )
    print("=" * 72)

    print("Smoke:", args.smoke)
    print("No fitting. No new scenes. E4 sealed.")

    print("\n[1] Provenance and case alignment")

    c0, c1, repairs = verify_and_index()

    print("\n[2] Loading frozen reconstruction")

    p80 = run_path(str(P80))
    p83 = run_path(str(P83))

    p80["check_provenance"]()

    (
        p74,
        p75,
        p77,
        p79,
        src,
        sim,
        api,
        commands,
    ) = p80["initialize"]()

    must(
        p77["VALID"] == "mid",
        "Validation-action mismatch",
    )

    must(
        tuple(p77["TRAIN"]) ==
        ("e1", "e2", "weak", "strong"),
        "Training-action mismatch",
    )

    scenes = src["make_scenes"](
        sim,
        p80["SCENE_SEED"],
        p80["N_DEV"],
    )

    must(
        len(scenes) == 12,
        "Frozen scene reconstruction failed",
    )

    selected = sorted(
        repairs.values(),
        key=lambda row: row["case"],
    )

    if args.smoke:
        selected = selected[:1]

    print("\n[3] Historical Mid parity + repair replay")

    replay = {}

    for i, row in enumerate(selected, 1):

        item = replay_mid(
            row,
            scenes,
            c0,
            p74,
            p75,
            p77,
            sim,
            api,
            commands,
            p80["NOISE_SEED"],
            p80,
            p83["make_objective"],
        )

        replay[key_b(row)] = item

        print(
            f"[MID PARITY PASS] {i}/{len(selected)} "
            f"| case={row['case']} "
            f"| Wide RMS "
            f"{item['old_wide_mid_mm']:.6f} -> "
            f"{item['new_wide_mid_mm']:.6f} mm"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = "smoke" if args.smoke else "dev"

    report_path = OUT / (
        f"p4m2c2c_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2c2c_{suffix}_audit_PRIVATE.json"
    )

    if args.smoke:

        result = {
            "protocol": "P4-M2C2C replay smoke",
            "smoke": True,

            "mid_evaluations": 1,
            "full_risk_comparison_performed": False,

            "source_sha256": {
                name: sha(path)
                for name, path in FILES.items()
            },

            "E4_accessed": False,
            "new_fits": 0,
        }

        audit = [
            {
                "scene_id": key[0],
                "band": key[1],
                "mode_PRIVATE": key[2],
                "axis_PRIVATE": key[3],
                "level_PRIVATE": key[4],
                **item,
            }
            for key, item in replay.items()
        ]

    else:

        print("\n[4] Recomputing mechanism selections")

        audit = []

        for key, row in c0.items():

            before = row["fits"]
            after = copy.deepcopy(before)

            if key in replay:

                after["wide"]["deadzone"][
                    "train_mse"
                ] = replay[key]["new_wide_train_mse"]

                after["wide"]["deadzone"][
                    "validation_rms_mm"
                ] = replay[key]["new_wide_mid_mm"]

            old_wide = winner(before, "wide")

            old_outer = winner(before, "outer_box")

            new_wide = winner(after, "wide")

            must(
                old_wide ==
                row["outcomes_PRIVATE"][
                    "wide"
                ]["winner"],
                f"Historical Wide winner mismatch: {key}",
            )

            must(
                old_outer ==
                row["outcomes_PRIVATE"][
                    "outer_box"
                ]["winner"],
                f"Historical Outer winner mismatch: {key}",
            )

            score_before, raw_before = conflict(before)

            score_after, raw_after = conflict(after)

            must(
                close(
                    score_before,
                    c1[key]["conflict_score"],
                ),
                f"Conflict replay mismatch: {key}",
            )

            true_law = TRUE_LAW[key[2]]

            wc0 = old_wide == true_law
            wc1 = new_wide == true_law
            oc = old_outer == true_law

            audit.append({
                "scene_id": key[0],
                "band": key[1],

                "mode_PRIVATE": key[2],
                "axis_PRIVATE": key[3],
                "level_PRIVATE": key[4],

                "repaired": key in replay,

                "winner_wide_before": old_wide,
                "winner_wide_after": new_wide,
                "winner_outer": old_outer,

                "wide_correct_before_PRIVATE": wc0,
                "wide_correct_after_PRIVATE": wc1,
                "outer_correct_PRIVATE": oc,

                "rescue_before_PRIVATE":
                    oc and not wc0,

                "rescue_after_PRIVATE":
                    oc and not wc1,

                "harm_before_PRIVATE":
                    wc0 and not oc,

                "harm_after_PRIVATE":
                    wc1 and not oc,

                "conflict_before": score_before,
                "conflict_after": score_after,

                "raw_conflict_before": raw_before,
                "raw_conflict_after": raw_after,

                "deadzone_wide_mid_before_mm":
                    before["wide"]["deadzone"][
                        "validation_rms_mm"
                    ],

                "deadzone_wide_mid_after_mm":
                    after["wide"]["deadzone"][
                        "validation_rms_mm"
                    ],

                "deadzone_wide_train_before":
                    before["wide"]["deadzone"][
                        "train_mse"
                    ],

                "deadzone_wide_train_after":
                    after["wide"]["deadzone"][
                        "train_mse"
                    ],
            })

        must(
            len(audit) == 432,
            "Incorrect total case count",
        )

        must(
            sum(
                r["repaired"] for r in audit
            ) == 21,
            "Incorrect repair coverage",
        )

        for row in audit:

            if row["repaired"]:

                must(
                    row["deadzone_wide_train_after"]
                    <= row["deadzone_wide_train_before"]
                    + KEY_TOL,
                    "Corrected Wide train loss worsened",
                )

        summary = {
            "all_432": summarize(audit),

            "mismatch_360": summarize([
                r for r in audit
                if r["level_PRIVATE"] != "M0"
            ]),

            "matched_72": summarize([
                r for r in audit
                if r["level_PRIVATE"] == "M0"
            ]),

            "repaired_21": summarize([
                r for r in audit
                if r["repaired"]
            ]),
        }

        summary["all_432"]["cluster_bootstrap"] = (
            clustered_bootstrap(audit)
        )

        summary["mismatch_360"][
            "cluster_bootstrap"
        ] = clustered_bootstrap([
            r for r in audit
            if r["level_PRIVATE"] != "M0"
        ])

        summary["by_band"] = {
            band: summarize([
                r for r in audit
                if r["band"] == band
            ])
            for band in BANDS
        }

        summary["by_mode"] = {
            mode: summarize([
                r for r in audit
                if r["mode_PRIVATE"] == mode
            ])
            for mode in TRUE_LAW
        }

        result = {
            "protocol":
                "P4-M2C2C solver-corrected "
                "decision-risk audit",

            "smoke": False,
            "development_only": True,

            "n_rows": len(audit),
            "n_target_repairs": 21,

            "summary": summary,

            "E4_accessed": False,
            "new_fits": 0,

            "mid_validation_used_only_for_posthoc_scoring":
                True,

            "source_sha256": {
                name: sha(path)
                for name, path in FILES.items()
            },

            "code_sha256": sha(Path(__file__)),

            "limitations": [
                "Retrospectively selected 21 violations",
                "Not a deployable decision policy",
                "Wide corrections reuse Outer solutions",
                "Not a matched-compute Wide baseline",
                "Only 12 correlated development scenes",
                "Mid replayed for prediction only",
                "No independent confirmation",
                "No global optimality certificate",
                "Harm AUROC is descriptive",
                "No classification threshold calibrated",
            ],
        }

        print("\n[5] Decision-risk summary")

        for label in (
            "all_432",
            "mismatch_360",
            "matched_72",
            "repaired_21",
        ):

            metrics = summary[label]

            print(f"\nSCOPE: {label}")

            print(
                "Wide correct:",
                metrics["wide_correct_before"],
                "->",
                metrics["wide_correct_after"],
            )

            print(
                "Outer correct:",
                metrics["outer_correct"],
            )

            print(
                "Outer rescue:",
                metrics["rescue_before"],
                "->",
                metrics["rescue_after"],
            )

            print(
                "Outer harm:",
                metrics["harm_before"],
                "->",
                metrics["harm_after"],
            )

            print(
                "Conflict-to-harm AUROC:",
                metrics["auroc_conflict_to_harm_before"],
                "->",
                metrics["auroc_conflict_to_harm_after"],
            )

    report_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("P4-M2C2C COMPLETE")
    print("=" * 72)

    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()