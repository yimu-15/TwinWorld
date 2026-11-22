
"""
P3-A4A: Evidence-Grounded Stopping.

Question:
Can an agent skip an unnecessary E3 intervention
using E1/E2 evidence only?

Policies:
1. early_gate
2. always_weak
3. always_strong
4. stop_or_strong
5. known_refit_strong
6. unrestricted_strong

Simulator and model fits are reused from P3-A3A.

Development-only evaluation.
No E4 information enters decisions.
"""

from pathlib import Path
from runpy import run_path
from collections import Counter, defaultdict
import argparse
import hashlib
import json

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a4a"

SEED = 20261126

KNOWN = (
    "independent",
    "rigid",
    "elastic",
)

IN_LIBRARY = {
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

OUT_LIBRARY = "saturating"

ALL_MODES = KNOWN + tuple(IN_LIBRARY) + (
    OUT_LIBRARY,
)

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
    "stop_or_strong",
    "known_refit_strong",
    "unrestricted_strong",
)

# Frozen exploratory rule.
# Motivated by prior development results.
# Must not be tuned on this experiment's E4.
STOP_BIC_MARGIN = -10.0


def should_stop(early, early_cut):
    """
    Reads only E1/E2 fitted evidence.

    Never reads:
      - true mechanism
      - E3 observation
      - E4 observation
      - realized E3 costs
    """
    known = early["known"]

    return bool(
        known["kind"] in (
            "independent",
            "rigid",
        )
        and early["margin"] <= STOP_BIC_MARGIN
        and known["fit_rms_m"]
        <= early_cut["known_fit_rms_m"]
    )


def self_test():
    early = {
        "margin": -30.0,
        "known": {
            "kind": "rigid",
            "fit_rms_m": 0.005,
        },
    }

    cut = {
        "known_fit_rms_m": 0.009,
    }

    assert should_stop(early, cut)

    early["margin"] = -3.0
    assert not should_stop(early, cut)

    early["margin"] = -30.0
    early["known"]["kind"] = "elastic"
    assert not should_stop(early, cut)

    early["known"]["kind"] = "independent"
    early["known"]["fit_rms_m"] = 0.015
    assert not should_stop(early, cut)

    print(
        "P3-A4A STOPPING SELF-TEST PASS "
        "(not MuJoCo validation)"
    )


def cluster_stat(rows, fn, seed, n_boot=2000):
    """
    Resample independent scenes rather than
    treating mechanisms in one scene as independent.
    """
    if not rows:
        return None

    groups = defaultdict(list)

    for row in rows:
        groups[row["scene_id"]].append(
            float(fn(row))
        )

    values = np.asarray([
        np.mean(groups[k])
        for k in sorted(groups)
    ])

    rng = np.random.default_rng(seed)

    draws = values[
        rng.integers(
            len(values),
            size=(n_boot, len(values)),
        )
    ].mean(axis=1)

    return {
        "mean": float(values.mean()),
        "ci95_scene": np.quantile(
            draws,
            [0.025, 0.975],
        ).tolist(),
        "n_scenes": len(values),
        "n_episodes": len(rows),
    }


def policy_decisions(row, cuts, stage):
    """
    All decisions are made before accessing E4.

    Alternative E3 data already exist offline,
    but each policy reads only its chosen E3.
    """
    early = row["early"]

    early_status, early_model = stage["decide"](
        early,
        cuts["early"],
    )

    weak_status, weak_model = stage["decide"](
        row["arms"][WEAK],
        cuts[WEAK],
    )

    strong_status, strong_model = stage["decide"](
        row["arms"][STRONG],
        cuts[STRONG],
    )

    strong_fits = row["arms"][STRONG]

    known_model = strong_fits["known"]
    edit_model = strong_fits["edit"]

    if edit_model["bic"] < known_model["bic"]:
        unrestricted_status = "edit"
        unrestricted_model = edit_model
    else:
        unrestricted_status = "keep"
        unrestricted_model = known_model

    stop = should_stop(
        early,
        cuts["early"],
    )

    if stop:
        adaptive_status = "keep"
        adaptive_model = early["known"]
        adaptive_action = None
    else:
        adaptive_status = strong_status
        adaptive_model = strong_model
        adaptive_action = STRONG

    # (decision, selected_model, E3_action)
    return {
        "early_gate": (
            early_status,
            early_model,
            None,
        ),
        "always_weak": (
            weak_status,
            weak_model,
            WEAK,
        ),
        "always_strong": (
            strong_status,
            strong_model,
            STRONG,
        ),
        "stop_or_strong": (
            adaptive_status,
            adaptive_model,
            adaptive_action,
        ),
        "known_refit_strong": (
            "keep",
            known_model,
            STRONG,
        ),
        "unrestricted_strong": (
            unrestricted_status,
            unrestricted_model,
            STRONG,
        ),
    }


def evaluate(rows, cuts, api, stage):
    records = []

    for row in rows:

        # Freeze all policy decisions first.
        decisions = policy_decisions(
            row,
            cuts,
            stage,
        )

        # E4 is accessed only after decisions.
        tests = row["tests_PRIVATE"]

        result = {
            "scene_id": row["scene_id"],
            "truth_PRIVATE_EVAL_ONLY":
                row["truth_PRIVATE"],

            "early_evidence": {
                "known_kind":
                    row["early"]["known"]["kind"],
                "bic_advantage":
                    float(row["early"]["margin"]),
                "known_fit_rms_m":
                    float(
                        row["early"]["known"]
                        ["fit_rms_m"]
                    ),
            },

            "policies": {},
        }

        for policy, (
            status,
            model,
            action,
        ) in decisions.items():

            if action is None:
                cost = 0.0
                impulse = 0.0
            else:
                cost = float(
                    row["costs"][action]["work_J"]
                )
                impulse = float(
                    row["costs"][action]["impulse_Ns"]
                )

            errors = {
                test: float(
                    api["prediction_rms"](
                        tests[test],
                        model,
                    )
                )
                for test in TESTS
            }

            result["policies"][policy] = {
                "decision": status,
                "kind": model["kind"],
                "E3_action": (
                    "none" if action is None
                    else action
                ),
                "E3_work_proxy_J": cost,
                "E3_impulse_Ns": impulse,
                "E4_RMS_m": errors,
            }

        records.append(result)

    return records


def error_mm(row, policy):
    errors = row["policies"][policy]["E4_RMS_m"]

    return float(np.mean([
        errors[test]
        for test in TESTS
    ])) * 1000.0


def summarize(rows, seed):
    known = [
        row for row in rows
        if row["truth_PRIVATE_EVAL_ONLY"] in KNOWN
    ]

    in_library = [
        row for row in rows
        if row["truth_PRIVATE_EVAL_ONLY"]
        in IN_LIBRARY
    ]

    out_library = [
        row for row in rows
        if row["truth_PRIVATE_EVAL_ONLY"]
        == OUT_LIBRARY
    ]

    summary = {}

    for i, policy in enumerate(POLICIES):

        def get(row):
            return row["policies"][policy]

        accepted = [
            r for r in rows
            if get(r)["decision"] != "inconclusive"
        ]

        rejected = [
            r for r in rows
            if get(r)["decision"] == "inconclusive"
        ]

        result = {
            "actions": dict(Counter(
                get(r)["E3_action"]
                for r in rows
            )),

            "decisions": dict(Counter(
                get(r)["decision"]
                for r in rows
            )),

            "mean_E3_work_J": float(np.mean([
                get(r)["E3_work_proxy_J"]
                for r in rows
            ])),

            "mean_E3_impulse_Ns": float(np.mean([
                get(r)["E3_impulse_Ns"]
                for r in rows
            ])),

            "E4_mean_RMS_mm": {
                test: 1000.0 * float(np.mean([
                    get(r)["E4_RMS_m"][test]
                    for r in rows
                ]))
                for test in TESTS
            },

            "E4_accepted_risk_mm": (
                float(np.mean([
                    error_mm(r, policy)
                    for r in accepted
                ]))
                if accepted else None
            ),

            "known_refit_risk_on_same_accepted_mm": (
                float(np.mean([
                    error_mm(
                        r,
                        "known_refit_strong"
                    )
                    for r in accepted
                ]))
                if accepted else None
            ),

            "E4_rejected_fallback_RMS_mm": (
                float(np.mean([
                    error_mm(r, policy)
                    for r in rejected
                ]))
                if rejected else None
            ),
        }

        metrics = {
            "coverage": (
                rows,
                lambda r:
                    get(r)["decision"]
                    != "inconclusive",
            ),

            "known_false_edit": (
                known,
                lambda r:
                    get(r)["decision"] == "edit",
            ),

            "known_inconclusive": (
                known,
                lambda r:
                    get(r)["decision"]
                    == "inconclusive",
            ),

            "known_kind_correct": (
                known,
                lambda r:
                    get(r)["kind"]
                    == r["truth_PRIVATE_EVAL_ONLY"],
            ),

            "in_library_correct_edit": (
                in_library,
                lambda r: (
                    get(r)["decision"] == "edit"
                    and get(r)["kind"]
                    == IN_LIBRARY[
                        r["truth_PRIVATE_EVAL_ONLY"]
                    ]
                ),
            ),

            "in_library_wrong_edit": (
                in_library,
                lambda r: (
                    get(r)["decision"] == "edit"
                    and get(r)["kind"]
                    != IN_LIBRARY[
                        r["truth_PRIVATE_EVAL_ONLY"]
                    ]
                ),
            ),

            "out_library_inconclusive": (
                out_library,
                lambda r:
                    get(r)["decision"]
                    == "inconclusive",
            ),

            "out_library_forced_edit": (
                out_library,
                lambda r:
                    get(r)["decision"] == "edit",
            ),

            "out_library_wrong_keep": (
                out_library,
                lambda r:
                    get(r)["decision"] == "keep",
            ),
        }

        for j, (name, (subset, fn)) in enumerate(
            metrics.items()
        ):
            result[name] = cluster_stat(
                subset,
                fn,
                seed + 100 * i + j,
            )

        summary[policy] = result

    # Paired policy comparisons.
    # Errors remain fallback diagnostics for
    # inconclusive predictions, not accepted risk.
    comparisons = {}

    for name, subset in (
        ("all", rows),
        ("known", known),
        ("in_library", in_library),
        ("out_library", out_library),
    ):

        comparisons[name] = {
            "stop_minus_always_strong_E4_mm":
                cluster_stat(
                    subset,
                    lambda r: (
                        error_mm(r, "stop_or_strong")
                        - error_mm(r, "always_strong")
                    ),
                    seed + 1000,
                ),

            "stop_minus_always_strong_E3_work_J":
                cluster_stat(
                    subset,
                    lambda r: (
                        r["policies"]["stop_or_strong"]
                        ["E3_work_proxy_J"]
                        -
                        r["policies"]["always_strong"]
                        ["E3_work_proxy_J"]
                    ),
                    seed + 1001,
                ),
        }

    summary["paired_comparisons"] = comparisons

    # Evaluation only: use truth labels to inspect
    # the stopping decision, never to determine it.
    stopped = [
        r for r in rows
        if r["policies"]["stop_or_strong"]
        ["E3_action"] == "none"
    ]

    summary["stopping_audit_PRIVATE"] = {
        "stopped_total": len(stopped),

        "stopped_by_true_mechanism": dict(
            Counter(
                r["truth_PRIVATE_EVAL_ONLY"]
                for r in stopped
            )
        ),

        "stopped_wrong_known_kind": sum(
            r["truth_PRIVATE_EVAL_ONLY"] in KNOWN
            and r["policies"]["stop_or_strong"]
            ["kind"] != r["truth_PRIVATE_EVAL_ONLY"]
            for r in stopped
        ),

        "stopped_in_library": sum(
            r["truth_PRIVATE_EVAL_ONLY"]
            in IN_LIBRARY
            for r in stopped
        ),

        "stopped_out_library": sum(
            r["truth_PRIVATE_EVAL_ONLY"]
            == OUT_LIBRARY
            for r in stopped
        ),
    }

    return summary


def check_physics_across_modes(sim, profiles):
    """
    Correct cross-mode audit.

    P3-A3A's make_dataset initializes its
    passive-reference dictionary inside each mode,
    so its passive gap check is not a reliable
    cross-mechanism comparison.

    This audit explicitly compares different modes
    within the SAME scene and command.
    """
    rng = np.random.default_rng(SEED + 9000)

    commands = {
        "weak_initial": profiles["pulse"] * 0.25,
        "strong_E3": profiles["pulse"] * 2.25,
    }

    result = {}

    for name, command in commands.items():
        max_passive_gap = 0.0
        max_A_gap = 0.0

        for _ in range(3):
            scene = sim["sample_scene"](rng)
            model = sim["make_model"](scene)

            reference_passive = None
            reference_A = None

            for mode in ALL_MODES:
                passive, active, _, _ = (
                    sim["simulate"](
                        model,
                        scene,
                        mode,
                        command,
                    )
                )

                if reference_passive is None:
                    reference_passive = passive.copy()
                    reference_A = active[:, 0].copy()
                    continue

                max_passive_gap = max(
                    max_passive_gap,
                    float(np.max(np.abs(
                        passive - reference_passive
                    ))),
                )

                max_A_gap = max(
                    max_A_gap,
                    float(np.max(np.abs(
                        active[:, 0] - reference_A
                    ))),
                )

        result[name] = {
            "cross_mode_passive_gap_m":
                max_passive_gap,
            "cross_mode_A_gap_m":
                max_A_gap,
        }

    return result


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

    helper = run_path(str(
        HERE / "27_p3a3b_evidence_vs_structure.py"
    ))

    # Install held-out simulation mechanism
    # in memory only.
    helper["install_saturating"](sim)

    profiles = sim["make_profiles"]()

    if args.smoke:
        ncal, ndev, nshift = 2, 2, 2
        nfev = 12
    else:
        ncal, ndev, nshift = 12, 18, 18
        nfev = 40

    make_data = stage["make_dataset"]

    cal, _ = make_data(
        sim, api, profiles,
        ncal, KNOWN,
        SEED + 11, nfev,
    )

    dev, _ = make_data(
        sim, api, profiles,
        ndev, ALL_MODES,
        SEED + 22, nfev,
        evaluation=True,
    )

    shift, _ = make_data(
        sim, api, profiles,
        nshift, KNOWN,
        SEED + 33, nfev,
        shifted=True,
        evaluation=True,
    )

    # Known-only calibration.
    cuts = stage["calibrate"](cal)

    # All policy choices are based on E1/E2,
    # and where chosen, on E3 only.
    dev_records = evaluate(
        dev, cuts, api, stage
    )

    shift_records = evaluate(
        shift, cuts, api, stage
    )

    physics = check_physics_across_modes(
        sim, profiles
    )

    report = {
        "protocol": (
            "P3-A4A stop-or-strong E3 "
            "versus fixed and early-only baselines"
        ),

        "development_only": True,

        "seed": SEED,

        "scene_counts": {
            "calibration": ncal,
            "development": ndev,
            "mass_shift": nshift,
        },

        "stopping_rule": {
            "allowed_kinds": [
                "independent",
                "rigid",
            ],
            "BIC_margin_max":
                STOP_BIC_MARGIN,
            "known_RMS_max":
                "known-only early calibration q95",
        },

        "calibration_known_only": cuts,

        "development": summarize(
            dev_records,
            SEED + 100,
        ),

        "mass_shift": summarize(
            shift_records,
            SEED + 200,
        ),

        "cross_mode_physics_audit": physics,

        "source_sha256": {
            name: hashlib.sha256(
                (HERE / name).read_bytes()
            ).hexdigest()
            for name in (
                "16_p2d1b_probe_diversity.py",
                "22_p3a1b_structural_revision.py",
                "26_p3a3a_weak_evidence_stress.py",
                "27_p3a3b_evidence_vs_structure.py",
            )
        },

        "limitations": [
            "The stopping rule is exploratory and handcrafted.",
            "The BIC cutoff was informed by previous development.",
            "Only stop versus strong-pulse is adaptive.",
            "All counterfactual E3 trajectories are generated offline.",
            "The policy never observes unchosen E3 trajectories.",
            "Saturating is excluded from the candidate library.",
            "Inconclusive is abstention, not a successful forecast.",
            "Fallback E4 RMS is only a diagnostic.",
            "B predictions condition on observed future A.",
            "No RGB visual perception is evaluated.",
            "This is not a final confirmatory test.",
        ],
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke"
        if args.smoke
        else "development"
    )

    report_path = (
        OUT / f"p3a4a_{suffix}_report.json"
    )

    audit_path = (
        OUT / f"p3a4a_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            {
                "development": dev_records,
                "mass_shift": shift_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"P3-A4A {suffix}: "
        f"cal/dev/shift="
        f"{ncal}/{ndev}/{nshift}"
    )

    for split in (
        "development",
        "mass_shift",
    ):

        print("\n", split)

        stats = report[split]

        print(
            "Stop distribution:",
            stats["stopping_audit_PRIVATE"],
        )

        for policy in POLICIES:
            item = stats[policy]

            correct = item[
                "in_library_correct_edit"
            ]

            print(
                f"{policy:>22s}",
                "actions=", item["actions"],
                "E3_J=",
                round(item["mean_E3_work_J"], 3),
                "false_edit=",
                round(
                    item["known_false_edit"]["mean"],
                    3,
                ),
                "correct_new=",
                None if correct is None
                else round(correct["mean"], 3),
                "coverage=",
                round(item["coverage"]["mean"], 3),
                "E4mm=",
                {
                    k: round(v, 3)
                    for k, v in
                    item["E4_mean_RMS_mm"].items()
                },
            )

        paired = stats["paired_comparisons"]["all"]

        print(
            "Stop minus AlwaysStrong E4:",
            paired[
                "stop_minus_always_strong_E4_mm"
            ],
        )

        print(
            "Stop minus AlwaysStrong E3 work:",
            paired[
                "stop_minus_always_strong_E3_work_J"
            ],
        )

    print("\nCross-mode physics audit:")
    print(physics)

    print(
        "\nP3-A4A DEVELOPMENT AUDIT COMPLETE"
    )
    print(
        "Stop/continue prototype only; "
        "not a learned active policy."
    )
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
