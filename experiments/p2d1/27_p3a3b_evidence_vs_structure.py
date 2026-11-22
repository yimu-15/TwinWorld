
"""
P3-A3B: Evidence vs structural revision ablation.

Compare known-only refitting against structural editing
using matched E1/E2/E3 evidence.

E4 is evaluation-only.
Development experiment, not final confirmation.
"""

from pathlib import Path
from runpy import run_path
from collections import Counter, defaultdict

import argparse
import hashlib
import json
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a3b"

SEED = 20261123

KNOWN = (
    "independent",
    "rigid",
    "elastic"
)

IN_LIBRARY = {
    "deadzone": "deadzone",
    "nonlinear": "cubic"
}

SATURATING = "saturating"

ARMS = (
    "pulse_1p5",
    "pulse_2p25",
    "sweep_2p25"
)

TESTS = (
    "pulse_2p75",
    "oscillation_2p25"
)

METHODS = (
    "early_known",
    "early_gate",
    "known_refit",
    "unrestricted_refit",
    "gated_revision"
)


def bootstrap(rows, fn, seed, n=2000):
    if not rows:
        return None

    groups = defaultdict(list)

    for row in rows:
        groups[row["scene_id"]].append(
            float(fn(row))
        )

    values = np.array([
        np.mean(groups[k])
        for k in sorted(groups)
    ])

    rng = np.random.default_rng(seed)

    samples = values[
        rng.integers(
            len(values),
            size=(n, len(values))
        )
    ].mean(axis=1)

    return {
        "mean": float(values.mean()),
        "ci95_scene": np.quantile(
            samples, [0.025, 0.975]
        ).tolist(),
        "n_scenes": len(values),
        "n_episodes": len(rows)
    }


def self_test():
    assert len(set(METHODS)) == 5

    test = [{
        "scene_id": 0,
        "delta": 0.002
    }]

    result = bootstrap(
        test,
        lambda r: r["delta"],
        SEED
    )

    assert np.isclose(
        result["mean"], 0.002
    )

    print(
        "P3-A3B PURE LOGIC SELF-TEST PASS "
        "(not MuJoCo validation)"
    )


def install_saturating(sim):
    """
    Extend only the in-memory simulator.
    Original source files remain unchanged.
    """
    namespace = sim["simulate"].__globals__

    original = namespace["internal_force"]

    def force(mode, r, vr, scene):
        if mode == SATURATING:
            value = (
                -scene["k"] * r
                -scene["c"] * vr
            )

            limit = (
                0.4
                + 0.3 * scene["mass_b"]
            )

            return float(
                np.clip(value, -limit, limit)
            )

        return original(
            mode, r, vr, scene
        )

    namespace["internal_force"] = force


def build_evaluations(rows, cuts, api, stage):
    results = []

    for row in rows:

        # E1/E2-only decision.
        early_status, early_model = (
            stage["decide"](
                row["early"],
                cuts["early"]
            )
        )

        for arm in ARMS:
            fits = row["arms"][arm]

            known_model = fits["known"]
            edited_model = fits["edit"]

            # Original evidence gate.
            gate_status, gated_model = (
                stage["decide"](
                    fits,
                    cuts[arm]
                )
            )

            # Unrestricted BIC selection.
            choose_edit = (
                edited_model["bic"]
                < known_model["bic"]
            )

            unrestricted_model = (
                edited_model
                if choose_edit
                else known_model
            )

            models = {
                "early_known": (
                    "keep",
                    row["early"]["known"]
                ),

                "early_gate": (
                    early_status,
                    early_model
                ),

                "known_refit": (
                    "keep",
                    known_model
                ),

                "unrestricted_refit": (
                    "edit" if choose_edit
                    else "keep",
                    unrestricted_model
                ),

                "gated_revision": (
                    gate_status,
                    gated_model
                )
            }

            # Decisions and fitted models are fixed
            # before any E4 evaluation.
            record = {
                "scene_id": row["scene_id"],

                "truth_PRIVATE_EVAL_ONLY":
                    row["truth_PRIVATE"],

                "arm": arm,

                "E3_work_J": float(
                    row["costs"][arm]["work_J"]
                ),

                "methods": {}
            }

            for name, (status, model) in models.items():

                errors = {}

                for test in TESTS:
                    probe = row[
                        "tests_PRIVATE"
                    ][test]

                    errors[test] = float(
                        api["prediction_rms"](
                            probe,
                            model
                        )
                    )

                record["methods"][name] = {
                    "decision": status,
                    "kind": model["kind"],
                    "E4_RMS_m": errors
                }

            results.append(record)

    return results


def summarize(rows, seed):
    summary = {}

    for arm_i, arm in enumerate(ARMS):

        arm_rows = [
            r for r in rows
            if r["arm"] == arm
        ]

        known = [
            r for r in arm_rows
            if r["truth_PRIVATE_EVAL_ONLY"]
            in KNOWN
        ]

        in_library = [
            r for r in arm_rows
            if r["truth_PRIVATE_EVAL_ONLY"]
            in IN_LIBRARY
        ]

        out_library = [
            r for r in arm_rows
            if r["truth_PRIVATE_EVAL_ONLY"]
            == SATURATING
        ]

        groups = {
            "all": arm_rows,
            "known": known,
            "in_library": in_library,
            "out_of_library": out_library
        }

        mean_cost = float(np.mean([
            r["E3_work_J"]
            for r in arm_rows
        ]))

        entry = {
            "E3_work_proxy_J": mean_cost,

            "E3_cost_by_method_J": {
                method: (
                    0.0 if method.startswith("early_")
                    else mean_cost
                )
                for method in METHODS
            },

            "methods": {},
            "paired_comparisons": {}
        }

        for method_i, method in enumerate(METHODS):

            def get(r):
                return r["methods"][method]

            performance = {
                "decision_counts": dict(
                    Counter(
                        get(r)["decision"]
                        for r in arm_rows
                    )
                ),

                "E4_RMS_mm_mean": {
                    test: 1000 * float(np.mean([
                        get(r)["E4_RMS_m"][test]
                        for r in arm_rows
                    ]))
                    for test in TESTS
                },

                "E4_by_truth_mm": {
                    truth: {
                        test: 1000 * float(np.mean([
                            get(r)["E4_RMS_m"][test]
                            for r in arm_rows
                            if r[
                                "truth_PRIVATE_EVAL_ONLY"
                            ] == truth
                        ]))
                        for test in TESTS
                    }
                    for truth in sorted(set(
                        r["truth_PRIVATE_EVAL_ONLY"]
                        for r in arm_rows
                    ))
                }
            }

            metrics = {
                "known_false_edit": (
                    known,
                    lambda r: get(r)["decision"]
                    == "edit"
                ),

                "known_inconclusive": (
                    known,
                    lambda r: get(r)["decision"]
                    == "inconclusive"
                ),

                "known_kind_correct": (
                    known,
                    lambda r: get(r)["kind"]
                    == r["truth_PRIVATE_EVAL_ONLY"]
                ),

                "in_library_correct_edit": (
                    in_library,
                    lambda r: (
                        get(r)["decision"] == "edit"
                        and get(r)["kind"]
                        == IN_LIBRARY[
                            r["truth_PRIVATE_EVAL_ONLY"]
                        ]
                    )
                ),

                "in_library_wrong_edit": (
                    in_library,
                    lambda r: (
                        get(r)["decision"] == "edit"
                        and get(r)["kind"]
                        != IN_LIBRARY[
                            r["truth_PRIVATE_EVAL_ONLY"]
                        ]
                    )
                ),

                "out_library_forced_edit": (
                    out_library,
                    lambda r: get(r)["decision"]
                    == "edit"
                ),

                "out_library_inconclusive": (
                    out_library,
                    lambda r: get(r)["decision"]
                    == "inconclusive"
                ),

                "coverage": (
                    arm_rows,
                    lambda r: get(r)["decision"]
                    != "inconclusive"
                )
            }

            for metric_i, (name, (subset, fn)) in enumerate(
                metrics.items()
            ):
                performance[name] = bootstrap(
                    subset,
                    fn,
                    seed
                    + 100 * arm_i
                    + 10 * method_i
                    + metric_i
                )

            entry["methods"][method] = performance

        # Paired comparison: same scenes, same E4 probes.
        for comparison_name, method in (
            ("gated_minus_known", "gated_revision"),
            ("unrestricted_minus_known",
             "unrestricted_refit")
        ):

            entry["paired_comparisons"][
                comparison_name
            ] = {}

            for group_name, subset in groups.items():

                entry["paired_comparisons"][
                    comparison_name
                ][group_name] = bootstrap(
                    subset,

                    lambda r: 1000 * np.mean([
                        (
                            r["methods"][method]
                            ["E4_RMS_m"][test]
                        )
                        -
                        (
                            r["methods"]["known_refit"]
                            ["E4_RMS_m"][test]
                        )
                        for test in TESTS
                    ]),

                    seed + 1000
                    + arm_i * 100
                    + len(group_name)
                )

        summary[arm] = entry

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test-only",
        action="store_true"
    )

    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args = parser.parse_args()

    self_test()

    if args.self_test_only:
        return

    stage = run_path(str(
        HERE / "26_p3a3a_weak_evidence_stress.py"
    ))

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    install_saturating(sim)

    profiles = sim["make_profiles"]()

    if args.smoke:
        ncal, ndev, nshift = 2, 2, 2
        nfev = 12
    else:
        ncal, ndev, nshift = 12, 18, 18
        nfev = 40

    make_data = stage["make_dataset"]

    cal, cal_physics = make_data(
        sim, api, profiles,
        ncal, KNOWN,
        SEED + 11, nfev
    )

    dev, dev_physics = make_data(
        sim, api, profiles,
        ndev,
        KNOWN + tuple(IN_LIBRARY) + (SATURATING,),
        SEED + 22,
        nfev,
        evaluation=True
    )

    shift, shift_physics = make_data(
        sim, api, profiles,
        nshift, KNOWN,
        SEED + 33,
        nfev,
        shifted=True,
        evaluation=True
    )

    cuts = stage["calibrate"](cal)

    dev_rows = build_evaluations(
        dev, cuts, api, stage
    )

    shift_rows = build_evaluations(
        shift, cuts, api, stage
    )

    report = {
        "protocol": (
            "P3-A3B matched evidence and "
            "structural revision ablation"
        ),

        "development_only": True,

        "seed": SEED,

        "scene_counts": {
            "calibration": ncal,
            "development": ndev,
            "mass_shift": nshift
        },

        "calibration_known_only": cuts,

        "development": summarize(
            dev_rows, SEED + 100
        ),

        "mass_shift": summarize(
            shift_rows, SEED + 200
        ),

        "max_passive_gap_m": max(
            p["passive_max_m"]
            for p in (
                cal_physics,
                dev_physics,
                shift_physics
            )
        ),

        "source_sha256": {
            name: hashlib.sha256(
                (HERE / name).read_bytes()
            ).hexdigest()
            for name in (
                "16_p2d1b_probe_diversity.py",
                "22_p3a1b_structural_revision.py",
                "26_p3a3a_weak_evidence_stress.py"
            )
        },

        "limitations": [
            "Known-refit may reselect known structures.",
            "Candidate edits are handcrafted.",
            "Saturating is excluded from the model library.",
            "E4 is used for evaluation only.",
            "Inconclusive E4 scores are fallback diagnostics.",
            "Prediction conditions on observed E4 A.",
            "No adaptive action policy is trained.",
            "BIC uses temporally correlated residuals.",
            "Development only, not final confirmation."
        ]
    }

    assert report["max_passive_gap_m"] < 1e-8

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "development"
    )

    report_path = OUT / (
        f"p3a3b_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a3b_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    audit_path.write_text(
        json.dumps({
            "development": dev_rows,
            "mass_shift": shift_rows
        }, indent=2),
        encoding="utf-8"
    )

    print(
        "P3-A3B", suffix,
        "cal/dev/shift scenes:",
        ncal, ndev, nshift
    )

    for split in ("development", "mass_shift"):
        print("\n", split)

        for arm in ARMS:
            result = report[split][arm]

            print(
                arm,
                "E3 work:",
                round(
                    result["E3_work_proxy_J"], 3
                ),
                "J"
            )

            for method in (
                "early_gate",
                "known_refit",
                "unrestricted_refit",
                "gated_revision"
            ):
                item = result["methods"][method]

                correct = item[
                    "in_library_correct_edit"
                ]

                print(
                    " ",
                    method,
                    "correct_new:",
                    None if correct is None
                    else round(correct["mean"], 3),
                    "false_edit:",
                    round(
                        item["known_false_edit"]["mean"],
                        3
                    ),
                    "E4mm:",
                    {
                        k: round(v, 2)
                        for k, v in item[
                            "E4_RMS_mm_mean"
                        ].items()
                    }
                )

    print(
        "P3-A3B EXPERIMENT COMPLETE "
        "(development only)"
    )

    print("Saved:", report_path)


if __name__ == "__main__":
    main()
