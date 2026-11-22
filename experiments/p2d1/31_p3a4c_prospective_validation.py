
"""
P3-A4C
Prospective Hypothesis-Aware Intervention Validation

Frozen policy:
  High-confidence Rigid -> Stop
  Independent-like -> Pulse 1.5x
  Otherwise -> Pulse 2.25x

New independent scenes and noise streams.

This validates a frozen heuristic policy inside the
same MuJoCo task generator, not external generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from collections import Counter, defaultdict
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a4c"

SEED = 20261207

KNOWN = (
    "independent",
    "rigid",
    "elastic",
)

NEW = {
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

ALL = KNOWN + tuple(NEW) + ("saturating",)

WEAK = "pulse_1p5"
STRONG = "pulse_2p25"

TESTS = (
    "pulse_2p75",
    "oscillation_2p25",
)

POLICIES = (
    "early_gate",
    "always_weak",
    "always_strong",
    "old_stop_or_strong",
    "hypothesis_aware",
    "known_refit_strong",
)

NONINFERIORITY_MARGIN_MM = 0.5
MIN_COST_SAVING = 0.10


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def bootstrap(rows, fn, seed, n=3000):
    """
    Paired cluster bootstrap over scenes.
    """
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

    draws = values[
        rng.integers(
            0,
            len(values),
            size=(n, len(values)),
        )
    ].mean(axis=1)

    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(
            draws, [0.025, 0.975]
        ).tolist(),
        "n_scenes": len(values),
        "n_episodes": len(rows),
    }


def early_evidence(row):
    early = row["early"]

    return {
        "known_kind": early["known"]["kind"],
        "bic_advantage": float(early["margin"]),
        "known_fit_rms_m": float(
            early["known"]["fit_rms_m"]
        ),
    }


def select_models(row, cuts, stage, old, router):
    """
    All policy actions are chosen from E1/E2
    before reading their corresponding E3 result.

    The mechanism-aware policy cannot access
    any unchosen E3 observation.
    """
    early = row["early"]
    evidence = early_evidence(row)

    early_cut = cuts["early"][
        "known_fit_rms_m"
    ]

    mechanism_action = router["choose_action"](
        evidence,
        early_cut,
    )

    old_stop = old["should_stop"](
        early,
        cuts["early"],
    )

    actions = {
        "early_gate": None,
        "always_weak": WEAK,
        "always_strong": STRONG,
        "old_stop_or_strong": (
            None if old_stop else STRONG
        ),
        "hypothesis_aware": (
            None if mechanism_action == "none"
            else mechanism_action
        ),
        "known_refit_strong": STRONG,
    }

    selected = {}

    for name, action in actions.items():

        if name == "old_stop_or_strong" and old_stop:
            status = "keep"
            model = early["known"]

        elif name == "known_refit_strong":
            status = "keep"
            model = row["arms"][STRONG]["known"]

        elif action is None:
            status, model = stage["decide"](
                early,
                cuts["early"],
            )

        else:
            # Only the selected E3 observation
            # enters this policy's model decision.
            status, model = stage["decide"](
                row["arms"][action],
                cuts[action],
            )

        selected[name] = (
            status,
            model,
            action,
        )

    return selected, evidence


def evaluate(rows, cuts, api, stage, old, router):
    output = []

    for row in rows:

        # All decisions are frozen first.
        selected, evidence = select_models(
            row, cuts, stage, old, router
        )

        # E4 probes are accessed only afterwards.
        tests = row["tests_PRIVATE"]

        record = {
            "scene_id": row["scene_id"],
            "truth_PRIVATE_EVAL_ONLY":
                row["truth_PRIVATE"],
            "early_evidence": evidence,
            "policies": {},
        }

        for name, (
            status, model, action
        ) in selected.items():

            if action is None:
                work = 0.0
                impulse = 0.0
                action_name = "none"
            else:
                work = float(
                    row["costs"][action]["work_J"]
                )
                impulse = float(
                    row["costs"][action]["impulse_Ns"]
                )
                action_name = action

            errors = {
                test: float(
                    api["prediction_rms"](
                        tests[test],
                        model,
                    )
                )
                for test in TESTS
            }

            record["policies"][name] = {
                "decision": status,
                "kind": model["kind"],
                "action": action_name,
                "work_J": work,
                "impulse_Ns": impulse,
                "E4_m": errors,
            }

        output.append(record)

    return output


def error_mm(row, method, test=None):
    scores = row["policies"][method]["E4_m"]

    if test is not None:
        return 1000.0 * float(scores[test])

    return 1000.0 * float(
        np.mean([scores[t] for t in TESTS])
    )


def summarize(rows, seed):
    known = [
        r for r in rows
        if r["truth_PRIVATE_EVAL_ONLY"] in KNOWN
    ]

    new = [
        r for r in rows
        if r["truth_PRIVATE_EVAL_ONLY"] in NEW
    ]

    outside = [
        r for r in rows
        if r["truth_PRIVATE_EVAL_ONLY"]
        == "saturating"
    ]

    result = {}

    for i, policy in enumerate(POLICIES):

        def get(r):
            return r["policies"][policy]

        accepted = [
            r for r in rows
            if get(r)["decision"] != "inconclusive"
        ]

        rejected = [
            r for r in rows
            if get(r)["decision"] == "inconclusive"
        ]

        stopped = [
            r for r in rows
            if get(r)["action"] == "none"
        ]

        wrong_stops = [
            r for r in stopped
            if (
                r["truth_PRIVATE_EVAL_ONLY"]
                != get(r)["kind"]
            )
        ]

        def fraction(subset, predicate):
            if not subset:
                return None

            return {
                "count": sum(
                    bool(predicate(r)) for r in subset
                ),
                "total": len(subset),
                "bootstrap": bootstrap(
                    subset,
                    predicate,
                    seed + i * 100 + len(subset),
                ),
            }

        result[policy] = {
            "actions": dict(Counter(
                get(r)["action"] for r in rows
            )),

            "decisions": dict(Counter(
                get(r)["decision"] for r in rows
            )),

            "mean_work_J": float(np.mean([
                get(r)["work_J"] for r in rows
            ])),

            "coverage": fraction(
                rows,
                lambda r:
                    get(r)["decision"]
                    != "inconclusive",
            ),

            "known_false_edit": fraction(
                known,
                lambda r:
                    get(r)["decision"] == "edit",
            ),

            "known_kind_correct": fraction(
                known,
                lambda r: (
                    get(r)["kind"]
                    == r["truth_PRIVATE_EVAL_ONLY"]
                ),
            ),

            "new_correct_edit": fraction(
                new,
                lambda r: (
                    get(r)["decision"] == "edit"
                    and get(r)["kind"]
                    == NEW[
                        r["truth_PRIVATE_EVAL_ONLY"]
                    ]
                ),
            ),

            "new_wrong_edit": fraction(
                new,
                lambda r: (
                    get(r)["decision"] == "edit"
                    and get(r)["kind"]
                    != NEW[
                        r["truth_PRIVATE_EVAL_ONLY"]
                    ]
                ),
            ),

            "outside_inconclusive": fraction(
                outside,
                lambda r:
                    get(r)["decision"]
                    == "inconclusive",
            ),

            "outside_forced_edit": fraction(
                outside,
                lambda r:
                    get(r)["decision"] == "edit",
            ),

            "stops": {
                "count": len(stopped),
                "wrong_count": len(wrong_stops),
                "truth_PRIVATE": dict(Counter(
                    r["truth_PRIVATE_EVAL_ONLY"]
                    for r in stopped
                )),
            },

            "E4_all_fallback_diagnostic_mm": {
                test: float(np.mean([
                    error_mm(r, policy, test)
                    for r in rows
                ]))
                for test in TESTS
            },

            "E4_accepted_only_mm": (
                float(np.mean([
                    error_mm(r, policy)
                    for r in accepted
                ]))
                if accepted else None
            ),

            "E4_rejected_fallback_mm": (
                float(np.mean([
                    error_mm(r, policy)
                    for r in rejected
                ]))
                if rejected else None
            ),
        }

    comparisons = {}

    for i, policy in enumerate(POLICIES):
        if policy == "always_strong":
            continue

        common_accepted = [
            r for r in rows
            if (
                r["policies"][policy]["decision"]
                != "inconclusive"
                and r["policies"]["always_strong"]
                ["decision"] != "inconclusive"
            )
        ]

        comparisons[policy] = {
            "delta_work_J": bootstrap(
                rows,
                lambda r: (
                    r["policies"][policy]["work_J"]
                    - r["policies"]["always_strong"]
                    ["work_J"]
                ),
                seed + 1000 + i,
            ),

            "delta_E4_both_mm": bootstrap(
                rows,
                lambda r: (
                    error_mm(r, policy)
                    - error_mm(r, "always_strong")
                ),
                seed + 1100 + i,
            ),

            "delta_E4_by_test_mm": {
                test: bootstrap(
                    rows,
                    lambda r: (
                        error_mm(r, policy, test)
                        -
                        error_mm(
                            r, "always_strong", test
                        )
                    ),
                    seed + 1200 + i * 10 + j,
                )
                for j, test in enumerate(TESTS)
            },

            "delta_E4_common_accepted_mm":
                bootstrap(
                    common_accepted,
                    lambda r: (
                        error_mm(r, policy)
                        -
                        error_mm(r, "always_strong")
                    ),
                    seed + 1300 + i,
                ),
        }

    result["paired_vs_always_strong"] = comparisons

    return result


def audit_cross_mode(sim, profiles):
    """
    Explicitly compare different physical modes
    under identical scene and command.
    """
    rng = np.random.default_rng(SEED + 9999)

    actions = {
        "weak_initial": profiles["pulse"] * 0.25,
        "strong_E3": profiles["pulse"] * 2.25,
    }

    output = {}

    for name, command in actions.items():

        passive_gap = 0.0
        a_gap = 0.0

        for _ in range(3):
            scene = sim["sample_scene"](rng)
            model = sim["make_model"](scene)

            ref_passive = None
            ref_a = None

            for mode in ALL:
                passive, active, _, _ = (
                    sim["simulate"](
                        model, scene, mode, command
                    )
                )

                if ref_passive is None:
                    ref_passive = passive.copy()
                    ref_a = active[:, 0].copy()
                else:
                    passive_gap = max(
                        passive_gap,
                        float(np.max(np.abs(
                            passive - ref_passive
                        ))),
                    )

                    a_gap = max(
                        a_gap,
                        float(np.max(np.abs(
                            active[:, 0] - ref_a
                        ))),
                    )

        output[name] = {
            "passive_gap_m": passive_gap,
            "A_gap_m": a_gap,
        }

        assert passive_gap < 1e-8

    return output


def self_test(router):
    router["check_policy"]()

    evidence = {
        "known_kind": "independent",
        "bic_advantage": -20.0,
        "known_fit_rms_m": 0.005,
    }

    assert router["choose_action"](
        evidence, 0.009
    ) == WEAK

    evidence["known_kind"] = "rigid"

    assert router["choose_action"](
        evidence, 0.009
    ) == "none"

    evidence["known_kind"] = "elastic"

    assert router["choose_action"](
        evidence, 0.009
    ) == STRONG

    print(
        "P3-A4C FROZEN POLICY SELF-TEST PASS"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test-only",
        action="store_true",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    router = run_path(str(
        HERE / "30_p3a4b_mechanism_aware_replay.py"
    ))

    assert router["RIGID_BIC_CUTOFF"] == -10.0

    self_test(router)

    if args.self_test_only:
        return

    stage = run_path(str(
        HERE / "26_p3a3a_weak_evidence_stress.py"
    ))

    old = run_path(str(
        HERE / "29_p3a4a_evidence_stopping.py"
    ))

    helper = run_path(str(
        HERE / "27_p3a3b_evidence_vs_structure.py"
    ))

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    assert old["STOP_BIC_MARGIN"] == -10.0

    helper["install_saturating"](sim)

    profiles = sim["make_profiles"]()

    if args.smoke:
        ncal = 2
        ndev = 2
        nshift = 2
        nfev = 12
    else:
        ncal = 12
        ndev = 36
        nshift = 36
        nfev = 40

    make_data = stage["make_dataset"]

    cal, _ = make_data(
        sim, api, profiles,
        ncal, KNOWN,
        SEED + 11,
        nfev,
    )

    dev, _ = make_data(
        sim, api, profiles,
        ndev, ALL,
        SEED + 22,
        nfev,
        evaluation=True,
    )

    shift, _ = make_data(
        sim, api, profiles,
        nshift, KNOWN,
        SEED + 33,
        nfev,
        shifted=True,
        evaluation=True,
    )

    # Only known mechanisms calibrate thresholds.
    cuts = stage["calibrate"](cal)

    dev_records = evaluate(
        dev, cuts, api, stage, old, router
    )

    shift_records = evaluate(
        shift, cuts, api, stage, old, router
    )

    physics = audit_cross_mode(
        sim, profiles
    )

    dev_stats = summarize(
        dev_records,
        SEED + 100,
    )

    shift_stats = summarize(
        shift_records,
        SEED + 200,
    )

    report = {
        "protocol": (
            "P3-A4C frozen-policy prospective "
            "validation on fresh MuJoCo scenes"
        ),

        "seed": SEED,

        "scene_counts": {
            "calibration": ncal,
            "development": ndev,
            "mass_shift": nshift,
        },

        "policy_frozen_from":
            "P3-A4B script 30",

        "hypothesis_aware_rule": {
            "high_confidence_rigid": "stop",
            "independent_like": "pulse_1p5",
            "otherwise": "pulse_2p25",
            "rigid_BIC_cutoff": -10.0,
        },

        "predeclared_exploratory_criteria": {
            "min_E3_cost_saving_fraction":
                MIN_COST_SAVING,

            "max_E4_delta_upper_CI_mm":
                NONINFERIORITY_MARGIN_MM,

            "wrong_stops": 0,
        },

        "calibration_known_only": cuts,

        "prospective": dev_stats,

        "mass_shift": shift_stats,

        "physics": physics,

        "source_sha256": {
            name: sha256(HERE / name)
            for name in (
                "16_p2d1b_probe_diversity.py",
                "22_p3a1b_structural_revision.py",
                "26_p3a3a_weak_evidence_stress.py",
                "27_p3a3b_evidence_vs_structure.py",
                "29_p3a4a_evidence_stopping.py",
                "30_p3a4b_mechanism_aware_replay.py",
            )
        },

        "limitations": [
            "Fresh seeds; same simulator family.",
            "Frozen handcrafted policy, not learned.",
            "All potential E3 paths generated offline.",
            "Only chosen E3 is used by each policy.",
            "No private truth is used for action choice.",
            "E4 is accessed after frozen decisions.",
            "Inconclusive E4 is fallback diagnostic.",
            "B prediction conditions on observed E4 A.",
            "Candidate structure library is handcrafted.",
            "Not a real-world or RGB vision test.",
        ],
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke
        else "prospective"
    )

    report_path = OUT / (
        f"p3a4c_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a4c_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps({
            "prospective": dev_records,
            "mass_shift": shift_records,
        }, indent=2),
        encoding="utf-8",
    )

    print(
        f"P3-A4C {suffix} | "
        f"cal/eval/shift={ncal}/{ndev}/{nshift}"
    )

    for split, stats in (
        ("prospective", dev_stats),
        ("mass_shift", shift_stats),
    ):
        print("\nSPLIT:", split)

        for policy in POLICIES:
            item = stats[policy]

            correct = item["new_correct_edit"]

            print(
                policy,
                "actions=", item["actions"],
                "work_J=",
                round(item["mean_work_J"], 4),
                "false_edit=",
                item["known_false_edit"]["count"],
                "correct_new=",
                None if correct is None
                else correct["count"],
                "wrong_stops=",
                item["stops"]["wrong_count"],
                "coverage=",
                round(
                    item["coverage"]["bootstrap"]["mean"],
                    4,
                ),
                "E4mm=",
                {
                    k: round(v, 3)
                    for k, v in item[
                        "E4_all_fallback_diagnostic_mm"
                    ].items()
                },
            )

        print(
            "HypothesisAware vs AlwaysStrong:",
            stats["paired_vs_always_strong"]
            ["hypothesis_aware"],
        )

    # Predeclared exploratory check.
    strong_cost = dev_stats[
        "always_strong"
    ]["mean_work_J"]

    policy_cost = dev_stats[
        "hypothesis_aware"
    ]["mean_work_J"]

    saving = (
        (strong_cost - policy_cost) / strong_cost
        if strong_cost > 0 else 0.0
    )

    comparison = dev_stats[
        "paired_vs_always_strong"
    ]["hypothesis_aware"][
        "delta_E4_both_mm"
    ]

    wrong_stops = dev_stats[
        "hypothesis_aware"
    ]["stops"]["wrong_count"]

    checks = {
        "cost_saving_at_least_10pct":
            saving >= MIN_COST_SAVING,

        "E4_delta_CI_upper_below_0p5mm":
            comparison["ci95"][1]
            < NONINFERIORITY_MARGIN_MM,

        "zero_wrong_stops":
            wrong_stops == 0,
    }

    print("\nExploratory criteria:", checks)

    print("\nCross-mode physics:", physics)

    print(
        "\nP3-A4C PROSPECTIVE EVALUATION COMPLETE"
    )

    print(
        "Fresh-seed evaluation, "
        "same task generator."
    )

    print(
        "Do not tune policy against these results "
        "and call them independent again."
    )

    print("Saved:", report_path)


if __name__ == "__main__":
    main()
