
"""
P3-A2A: Gate priority and diagnostic excitation ablation.

Development-only experiment.

E1: Pulse 1.0x
E2: Pulse 2.0x
E3: Pulse 1.5x OR Pulse 2.25x
E4: Pulse 2.75x AND Sweep 2.25x

No unknown labels used for fitting or calibration.
No E4 observations used for model selection.
No learned active action-selection policy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a2a"

SEED = 20261120

KNOWN = (
    "independent",
    "rigid",
    "elastic",
)

MODES = KNOWN + (
    "deadzone",
    "nonlinear",
)

TRUTH_TO_EDIT = {
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

E3_ARMS = (
    "weak_pulse_1p5",
    "strong_pulse_2p25",
)

METHODS = (
    "absolute_first",
    "relative_first",
)

E4_NAMES = (
    "e4_pulse_2p75",
    "e4_sweep_2p25",
)

BIC_MARGIN = 2.0


def score_rms(probe, fit, api):
    return api["prediction_rms"](probe, fit)


def calibrate(rows, arm):
    """
    Calibrate using known mechanisms only.

    No E4 observations or unknown labels are used.
    """
    scores = [
        row["arms"][arm]
        for row in rows
    ]

    return {
        "known_q95_m": float(
            np.quantile(
                [s["known_e3"] for s in scores],
                0.95,
            )
        ),
        "edit_q95_m": float(
            np.quantile(
                [s["edit_e3"] for s in scores],
                0.95,
            )
        ),
        "gain_positive_q95_m": float(
            np.quantile(
                [
                    max(0.0, s["gain_e3"])
                    for s in scores
                ],
                0.95,
            )
        ),
    }


def select(scores, threshold, method):
    """
    Compare two gate priorities using identical evidence.
    """
    sufficient_edit = (
        scores["bic_advantage"] > BIC_MARGIN
        and scores["gain_e3"]
        > threshold["gain_positive_q95_m"]
        and scores["edit_e3"]
        <= threshold["edit_q95_m"]
    )

    known_acceptable = (
        scores["known_e3"]
        <= threshold["known_q95_m"]
    )

    if method == "absolute_first":
        if known_acceptable:
            return "keep", "known"

        if sufficient_edit:
            return "edit", "edit"

        return "inconclusive", "known"

    if method == "relative_first":
        if sufficient_edit:
            return "edit", "edit"

        if known_acceptable:
            return "keep", "known"

        return "inconclusive", "known"

    raise ValueError(method)


def self_test():
    threshold = {
        "known_q95_m": 0.009,
        "edit_q95_m": 0.014,
        "gain_positive_q95_m": 0.0004,
    }

    scores = {
        "known_e3": 0.007,
        "edit_e3": 0.005,
        "gain_e3": 0.002,
        "bic_advantage": 30.0,
    }

    assert select(
        scores, threshold, "absolute_first"
    )[0] == "keep"

    assert select(
        scores, threshold, "relative_first"
    )[0] == "edit"

    scores["bic_advantage"] = -1.0

    assert select(
        scores, threshold, "relative_first"
    )[0] == "keep"

    scores["known_e3"] = 0.017

    assert select(
        scores, threshold, "relative_first"
    )[0] == "inconclusive"

    print(
        "P3-A2A DECISION SELF-TEST PASS "
        "(not MuJoCo validation)"
    )


def make_split(
    sim,
    api,
    profiles,
    n_scenes,
    modes,
    seed,
    mass_shift=False,
    evaluation=False,
    nfev=30,
):
    scene_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 50000)

    rows = []

    physics = {
        "max_passive_gap_m": 0.0,
        "max_A_gap_m": 0.0,
    }

    commands = {
        "e1": profiles["pulse"] * 1.0,
        "e2": profiles["pulse"] * 2.0,
        "weak_pulse_1p5":
            profiles["pulse"] * 1.5,
        "strong_pulse_2p25":
            profiles["pulse"] * 2.25,
        "e4_pulse_2p75":
            profiles["pulse"] * 2.75,
        "e4_sweep_2p25":
            profiles["sweep"] * 2.25,
    }

    Probe = api["RevisionProbe"]
    add_noise = api["noisy_observation"]
    fit_library = api["fit_library"]

    for scene_id in range(n_scenes):

        scene = sim["sample_scene"](scene_rng)

        if mass_shift:
            scene["mass_a"] = float(
                scene_rng.uniform(1.8, 2.3)
            )
            scene["mass_b"] = float(
                scene_rng.uniform(1.8, 2.3)
            )

        model = sim["make_model"](scene)

        baseline_passive = None
        scene_a = []

        for mode in modes:

            needed = (
                "e1",
                "e2",
                *E3_ARMS,
            )

            if evaluation:
                needed += E4_NAMES

            probes = {}
            costs = {}

            for action in needed:

                passive, active, work, impulse = (
                    sim["simulate"](
                        model,
                        scene,
                        mode,
                        commands[action],
                    )
                )

                if baseline_passive is None:
                    baseline_passive = passive

                gap = float(
                    np.max(
                        np.abs(
                            passive - baseline_passive
                        )
                    )
                )

                physics["max_passive_gap_m"] = max(
                    physics["max_passive_gap_m"],
                    gap,
                )

                if action == "e1":
                    scene_a.append(active[:, 0])

                noisy_p, noisy_a = add_noise(
                    passive,
                    active,
                    noise_rng,
                )

                probes[action] = Probe(
                    noisy_p,
                    noisy_a,
                )

                costs[action] = {
                    "work_proxy_J": float(work),
                    "impulse_Ns": float(impulse),
                    "command_budget_m_per_s": float(
                        np.sum(
                            np.abs(commands[action])
                        ) * sim["DT"]
                    ),
                }

            early_probes = [
                probes["e1"],
                probes["e2"],
            ]

            known_fit = fit_library(
                early_probes,
                KNOWN,
                nfev,
            )[0]

            edit_fit = fit_library(
                early_probes,
                ("deadzone", "cubic"),
                nfev,
            )[0]

            result = {
                "scene_id": scene_id,
                "truth_PRIVATE": mode,
                "known": known_fit,
                "edit": edit_fit,
                "arms": {},
                "costs": costs,
            }

            for arm in E3_ARMS:

                e3 = probes[arm]

                known_e3 = score_rms(
                    e3, known_fit, api
                )

                edit_e3 = score_rms(
                    e3, edit_fit, api
                )

                known_three = fit_library(
                    early_probes + [e3],
                    KNOWN,
                    nfev,
                )[0]

                result["arms"][arm] = {
                    "known_e3": known_e3,
                    "edit_e3": edit_e3,
                    "gain_e3":
                        known_e3 - edit_e3,
                    "bic_advantage": float(
                        known_fit["bic"]
                        - edit_fit["bic"]
                    ),
                    "known_three": known_three,
                }

            if evaluation:
                # Saved for evaluation only.
                # Never passed into model fitting.
                result["e4_PRIVATE_EVAL_ONLY"] = {
                    name: probes[name]
                    for name in E4_NAMES
                }

            rows.append(result)

        if len(scene_a) > 1:
            a_gap = float(
                np.max(
                    np.ptp(
                        np.asarray(scene_a),
                        axis=0,
                    )
                )
            )

            physics["max_A_gap_m"] = max(
                physics["max_A_gap_m"],
                a_gap,
            )

    assert physics["max_passive_gap_m"] < 1e-8

    return rows, physics


def bootstrap(rows, value_fn, seed):
    """
    Scene-cluster bootstrap.
    Multiple mechanisms in one scene are not independent.
    """
    by_scene = defaultdict(list)

    for row in rows:
        by_scene[row["scene_id"]].append(
            float(value_fn(row))
        )

    if not by_scene:
        return None

    scene_values = np.array([
        np.mean(by_scene[scene_id])
        for scene_id in sorted(by_scene)
    ])

    rng = np.random.default_rng(seed)

    draws = scene_values[
        rng.integers(
            len(scene_values),
            size=(2000, len(scene_values)),
        )
    ].mean(axis=1)

    return {
        "mean": float(scene_values.mean()),
        "scene_bootstrap_ci95": list(
            map(
                float,
                np.quantile(
                    draws,
                    [0.025, 0.975],
                ),
            )
        ),
        "independent_scenes": len(scene_values),
        "episodes": len(rows),
    }


def evaluate_split(rows, cutoff, api, seed):
    private = []

    for row in rows:

        truth = row["truth_PRIVATE"]

        e4 = row["e4_PRIVATE_EVAL_ONLY"]

        unrestricted = min(
            (row["known"], row["edit"]),
            key=lambda m: m["bic"],
        )

        shared_scores = {
            "known_12": {},
            "unrestricted_12": {},
        }

        for name, probe in e4.items():
            shared_scores["known_12"][name] = (
                score_rms(
                    probe,
                    row["known"],
                    api,
                )
            )

            shared_scores[
                "unrestricted_12"
            ][name] = score_rms(
                probe,
                unrestricted,
                api,
            )

        arms = {}

        for arm in E3_ARMS:

            early = row["arms"][arm]
            threshold = cutoff[arm]

            method_values = {}

            for method in METHODS:

                status, selected = select(
                    early,
                    threshold,
                    method,
                )

                chosen_fit = row[selected]

                method_values[method] = {
                    "decision": status,
                    "final_kind":
                        chosen_fit["kind"],
                    "e4_m": {
                        name: score_rms(
                            probe,
                            chosen_fit,
                            api,
                        )
                        for name, probe in e4.items()
                    },
                }

            known_three = early["known_three"]

            method_values["known_123"] = {
                "decision": "known_refit",
                "final_kind":
                    known_three["kind"],
                "e4_m": {
                    name: score_rms(
                        probe,
                        known_three,
                        api,
                    )
                    for name, probe in e4.items()
                },
            }

            arms[arm] = {
                "known_e3_m":
                    early["known_e3"],
                "edit_e3_m":
                    early["edit_e3"],
                "gain_e3_m":
                    early["gain_e3"],
                "bic_advantage":
                    early["bic_advantage"],
                "e3_cost":
                    row["costs"][arm],
                "methods":
                    method_values,
            }

        private.append({
            "scene_id": row["scene_id"],
            "truth_PRIVATE_EVAL_ONLY": truth,
            "known_kind":
                row["known"]["kind"],
            "proposal_kind":
                row["edit"]["kind"],
            "shared_E4_baselines_m":
                shared_scores,
            "arms": arms,
        })

    report = {}

    for arm in E3_ARMS:

        arm_report = {}

        for method in METHODS:

            def entry(r):
                return (
                    r["arms"][arm]
                    ["methods"][method]
                )

            known = [
                r for r in private
                if r["truth_PRIVATE_EVAL_ONLY"]
                in KNOWN
            ]

            unknown = [
                r for r in private
                if r["truth_PRIVATE_EVAL_ONLY"]
                in TRUTH_TO_EDIT
            ]

            arm_report[method] = {
                "decision_counts": dict(
                    Counter(
                        entry(r)["decision"]
                        for r in private
                    )
                ),

                "known_false_edit": bootstrap(
                    known,
                    lambda r:
                        entry(r)["decision"] == "edit",
                    seed + 10,
                ),

                "known_inconclusive": bootstrap(
                    known,
                    lambda r:
                        entry(r)["decision"]
                        == "inconclusive",
                    seed + 11,
                ),

                "unknown_correct_edit": bootstrap(
                    unknown,
                    lambda r: (
                        entry(r)["decision"] == "edit"
                        and entry(r)["final_kind"]
                        == TRUTH_TO_EDIT[
                            r["truth_PRIVATE_EVAL_ONLY"]
                        ]
                    ),
                    seed + 12,
                ),

                "unknown_wrong_edit": bootstrap(
                    unknown,
                    lambda r: (
                        entry(r)["decision"] == "edit"
                        and entry(r)["final_kind"]
                        != TRUTH_TO_EDIT[
                            r["truth_PRIVATE_EVAL_ONLY"]
                        ]
                    ),
                    seed + 13,
                ),

                "correct_edit_by_unknown_kind": {
                    truth: bootstrap(
                        [
                            r for r in unknown
                            if r[
                                "truth_PRIVATE_EVAL_ONLY"
                            ] == truth
                        ],
                        lambda r: (
                            entry(r)["decision"] == "edit"
                            and entry(r)["final_kind"]
                            == TRUTH_TO_EDIT[truth]
                        ),
                        seed + 20 + i,
                    )
                    for i, truth in enumerate(
                        TRUTH_TO_EDIT
                    )
                },

                "e4_mean_rms_m_by_probe": {
                    name: float(
                        np.mean([
                            entry(r)["e4_m"][name]
                            for r in private
                        ])
                    )
                    for name in E4_NAMES
                },

                "e4_mean_rms_m_by_truth": {
                    truth: float(
                        np.mean([
                            np.mean(
                                list(
                                    entry(r)["e4_m"].values()
                                )
                            )
                            for r in private
                            if r[
                                "truth_PRIVATE_EVAL_ONLY"
                            ] == truth
                        ])
                    )
                    for truth in sorted(
                        set(
                            r["truth_PRIVATE_EVAL_ONLY"]
                            for r in private
                        )
                    )
                },
            }

        arm_report[
            "paired_relative_minus_absolute_e4_m"
        ] = bootstrap(
            private,
            lambda r: (
                np.mean(
                    list(
                        r["arms"][arm]["methods"]
                        ["relative_first"]
                        ["e4_m"].values()
                    )
                )
                -
                np.mean(
                    list(
                        r["arms"][arm]["methods"]
                        ["absolute_first"]
                        ["e4_m"].values()
                    )
                )
            ),
            seed + 14,
        )

        arm_report["mean_E3_abs_work_J"] = float(
            np.mean([
                r["arms"][arm]["e3_cost"]
                ["work_proxy_J"]
                for r in private
            ])
        )

        arm_report["mean_E3_impulse_Ns"] = float(
            np.mean([
                r["arms"][arm]["e3_cost"]
                ["impulse_Ns"]
                for r in private
            ])
        )

        arm_report[
            "nominal_E3_budget_m_per_s"
        ] = private[0]["arms"][arm][
            "e3_cost"
        ]["command_budget_m_per_s"]

        report[arm] = arm_report

    report["shared_baselines_E4_mean_m"] = {
        name: float(
            np.mean([
                np.mean(
                    list(
                        r["shared_E4_baselines_m"]
                        [name].values()
                    )
                )
                for r in private
            ])
        )
        for name in (
            "known_12",
            "unrestricted_12",
        )
    }

    return report, private


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

    sim = run_path(
        str(
            HERE /
            "16_p2d1b_probe_diversity.py"
        )
    )

    api = run_path(
        str(
            HERE /
            "22_p3a1b_structural_revision.py"
        )
    )

    assert tuple(sim["MODES"]) == MODES

    profiles = sim["make_profiles"]()

    if args.smoke:
        ncal, ndev, nshift = 2, 2, 2
        nfev = 20
    else:
        ncal, ndev, nshift = 12, 18, 18
        nfev = 50

    cal, pc = make_split(
        sim,
        api,
        profiles,
        ncal,
        KNOWN,
        SEED + 11,
        evaluation=False,
        nfev=nfev,
    )

    dev, pd = make_split(
        sim,
        api,
        profiles,
        ndev,
        MODES,
        SEED + 22,
        evaluation=True,
        nfev=nfev,
    )

    shift, ps = make_split(
        sim,
        api,
        profiles,
        nshift,
        KNOWN,
        SEED + 33,
        mass_shift=True,
        evaluation=True,
        nfev=nfev,
    )

    cutoff = {
        arm: calibrate(cal, arm)
        for arm in E3_ARMS
    }

    dev_report, dev_rows = evaluate_split(
        dev, cutoff, api, SEED + 100
    )

    shift_report, shift_rows = evaluate_split(
        shift, cutoff, api, SEED + 200
    )

    report = {
        "protocol": (
            "P3-A2A new-scene gate-priority "
            "x fixed-E3-strength ablation"
        ),
        "development_only": True,
        "seed": SEED,
        "counts": {
            "cal_known": ncal,
            "dev_all": ndev,
            "shift_known": nshift,
        },
        "E1_E2": "pulse_1p0,pulse_2p0",
        "E3_arms": list(E3_ARMS),
        "E4_evaluation_only": [
            "pulse_2p75",
            "sweep_2p25",
        ],
        "thresholds_known_only": cutoff,
        "development": dev_report,
        "mass_shift": shift_report,
        "physics_audit": {
            "max_passive_gap_m": max(
                x["max_passive_gap_m"]
                for x in (pc, pd, ps)
            ),
            "max_A_mode_gap_m": max(
                x["max_A_gap_m"]
                for x in (pc, pd, ps)
            ),
        },
        "source_sha256": {
            name: hashlib.sha256(
                (HERE / name).read_bytes()
            ).hexdigest()
            for name in (
                "16_p2d1b_probe_diversity.py",
                "20_p3a0_parameter_refit.py",
                "22_p3a1b_structural_revision.py",
            )
        },
        "notes": [
            "All E3 actions are fixed; not an adaptive policy.",
            "Calibration contains known mechanisms only.",
            "E4 is evaluation-only.",
            "E3 strengths have different real costs.",
            "Paired counterfactual E3 observations are not combined by an agent.",
            "B prediction conditions on observed E4 A.",
            "BIC is a heuristic with correlated trajectory residuals.",
            "Development only, not final confirmatory evidence.",
        ],
    }

    assert (
        report["physics_audit"]
        ["max_passive_gap_m"] < 1e-8
    )

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
        OUT / f"p3a2a_{suffix}_report.json"
    )

    audit_path = (
        OUT / f"p3a2a_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps({
            "development": dev_rows,
            "mass_shift": shift_rows,
        }, indent=2),
        encoding="utf-8",
    )

    print(
        "P3-A2A", suffix,
        "cal/dev/shift scenes",
        ncal, ndev, nshift,
    )

    for group in (
        "development",
        "mass_shift",
    ):
        print("\n", group)

        for arm in E3_ARMS:
            result = report[group][arm]

            print(
                " ", arm,
                "cost J",
                round(
                    result["mean_E3_abs_work_J"],
                    4,
                ),
            )

            for method in METHODS:
                stats = result[method]

                correct_unknown = stats[
                    "unknown_correct_edit"
                ]

                print(
                    "   ",
                    method,
                    "false_edit",
                    stats["known_false_edit"]["mean"],
                    "correct_unknown",
                    (
                        None
                        if correct_unknown is None
                        else correct_unknown["mean"]
                    ),
                    "E4mm",
                    {
                        name: round(value * 1000, 3)
                        for name, value in stats[
                            "e4_mean_rms_m_by_probe"
                        ].items()
                    },
                )

    print(
        "P3-A2A DEVELOPMENT AUDIT COMPLETE "
        "(not active policy or final test)"
    )

    print("Saved:", report_path)


if __name__ == "__main__":
    main()
