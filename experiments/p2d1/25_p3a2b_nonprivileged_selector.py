
"""
P3-A2B: Nonprivileged E3 action selection.

E1/E2: Pulse 1.0x / 2.0x
E3: Pulse 1.5x OR 2.25x
E4: Pulse 2.75x AND Sweep 2.25x

The action chooser reads ONLY E1/E2 BIC scores
and a threshold calibrated on known-only scenes.

All potential E3 results are generated offline
for paired evaluation, but unchosen observations
are not inputs to the policy.

DEVELOPMENT ONLY. Not unrestricted concept discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a2b"

SEED = 20261121

KNOWN = (
    "independent",
    "rigid",
    "elastic",
)

UNKNOWN = {
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

WEAK = "weak_pulse_1p5"
STRONG = "strong_pulse_2p25"

E4 = (
    "e4_pulse_2p75",
    "e4_sweep_2p25",
)

POLICIES = (
    "early_only",
    "fixed_weak",
    "fixed_strong",
    "random50",
    "bic_adaptive",
)


def early_score(known_bic, edit_bic):
    return float(known_bic - edit_bic)


def choose_action(
    known_bic,
    edit_bic,
    known_only_cut
):
    """
    Nonprivileged action selection.

    Inputs:
      - BIC values estimated using E1/E2 only.
      - Threshold from known-only calibration.

    No true mechanism or E3/E4 observation.
    """
    score = early_score(
        known_bic,
        edit_bic
    )

    if score > known_only_cut:
        return STRONG

    return WEAK


def check_unit():
    assert choose_action(
        100.0, 0.0, 20.0
    ) == STRONG

    assert choose_action(
        0.0, 100.0, 20.0
    ) == WEAK

    assert choose_action(
        25.0, 5.0, 20.0
    ) == WEAK

    print(
        "P3-A2B PUBLIC-EVIDENCE ACTION "
        "UNIT TEST PASS "
        "(not MuJoCo validation)"
    )


def auc_pairwise(pos, neg):
    """
    AUROC from pairwise comparisons.
    No classifier training.
    """
    if not pos or not neg:
        return None

    a = np.asarray(pos, float)[:, None]
    b = np.asarray(neg, float)[None, :]

    return float(np.mean(
        (a > b) + 0.5 * (a == b)
    ))


def cluster_bootstrap(
    rows,
    fn,
    seed,
    boot=2000
):
    """
    Scene-cluster bootstrap.
    Multiple mechanisms share one scene.
    """
    by_scene = {}

    for row in rows:
        by_scene.setdefault(
            row["scene_id"], []
        ).append(float(fn(row)))

    if not by_scene:
        return None

    values = np.asarray([
        np.mean(by_scene[k])
        for k in sorted(by_scene)
    ])

    rng = np.random.default_rng(seed)

    samples = values[
        rng.integers(
            len(values),
            size=(boot, len(values))
        )
    ].mean(axis=1)

    return {
        "mean": float(values.mean()),
        "scene_bootstrap_95_ci": (
            np.quantile(
                samples,
                [0.025, 0.975]
            ).tolist()
        ),
        "scenes": len(values),
        "episodes": len(rows),
    }


def evaluate(
    rows,
    bic_cut,
    e3_cut,
    api,
    stage,
    seed
):
    """
    All decisions are frozen before E4 scoring.

    Each policy uses only its selected E3 arm.
    """
    rng = np.random.default_rng(seed)
    data = []

    for episode in rows:

        known_bic = float(
            episode["known"]["bic"]
        )

        edit_bic = float(
            episode["edit"]["bic"]
        )

        gap = early_score(
            known_bic,
            edit_bic
        )

        adaptive_arm = choose_action(
            known_bic,
            edit_bic,
            bic_cut
        )

        random_arm = (
            STRONG if rng.random() < 0.5
            else WEAK
        )

        decisions = {}

        for policy in POLICIES:

            if policy == "early_only":

                if gap > bic_cut:
                    fit = episode["edit"]
                    status = "edit"
                else:
                    fit = episode["known"]
                    status = "keep"

                arm = None

            else:
                arm = {
                    "fixed_weak": WEAK,
                    "fixed_strong": STRONG,
                    "random50": random_arm,
                    "bic_adaptive": adaptive_arm,
                }[policy]

                # Read ONLY the chosen E3 observation.
                public_e3 = episode["arms"][arm]

                status, which = stage["select"](
                    public_e3,
                    e3_cut[arm],
                    "relative_first"
                )

                fit = episode[which]

            decisions[policy] = (
                status,
                fit,
                arm
            )

        # E4 is first accessed AFTER policy decisions.
        e4_probes = episode[
            "e4_PRIVATE_EVAL_ONLY"
        ]

        result = {
            "scene_id": episode["scene_id"],
            "truth_PRIVATE_EVAL_ONLY":
                episode["truth_PRIVATE"],
            "early_bic_advantage": gap,
            "policies": {},
        }

        for policy, (
            status, fit, arm
        ) in decisions.items():

            cost = (
                episode["costs"][arm]
                if arm is not None
                else {}
            )

            e4_errors = {
                name: float(
                    stage["score_rms"](
                        probe,
                        fit,
                        api
                    )
                )
                for name, probe
                in e4_probes.items()
            }

            result["policies"][policy] = {
                "E3_action": (
                    arm if arm is not None
                    else "none"
                ),
                "decision": status,
                "final_kind": fit["kind"],
                "E3_work_proxy_J": float(
                    cost.get(
                        "work_proxy_J", 0.0
                    )
                ),
                "E3_impulse_Ns": float(
                    cost.get(
                        "impulse_Ns", 0.0
                    )
                ),
                "E4_rms_m": e4_errors,
            }

        data.append(result)

    summaries = {}

    for policy_index, policy in enumerate(POLICIES):

        def values(row):
            return row["policies"][policy]

        known_rows = [
            row for row in data
            if row["truth_PRIVATE_EVAL_ONLY"]
            in KNOWN
        ]

        unknown_rows = [
            row for row in data
            if row["truth_PRIVATE_EVAL_ONLY"]
            in UNKNOWN
        ]

        summaries[policy] = {
            "actions": dict(Counter(
                values(row)["E3_action"]
                for row in data
            )),

            "decisions": dict(Counter(
                values(row)["decision"]
                for row in data
            )),

            "mean_E3_work_proxy_J": float(
                np.mean([
                    values(row)["E3_work_proxy_J"]
                    for row in data
                ])
            ),

            "mean_E3_impulse_Ns": float(
                np.mean([
                    values(row)["E3_impulse_Ns"]
                    for row in data
                ])
            ),

            "E4_mean_rms_mm_by_probe": {
                name: 1000 * float(np.mean([
                    values(row)["E4_rms_m"][name]
                    for row in data
                ]))
                for name in E4
            },

            "known_false_edit": cluster_bootstrap(
                known_rows,
                lambda row: (
                    values(row)["decision"] == "edit"
                ),
                seed + 10 + policy_index
            ),

            "unknown_correct_edit": cluster_bootstrap(
                unknown_rows,
                lambda row: (
                    values(row)["decision"] == "edit"
                    and values(row)["final_kind"]
                    == UNKNOWN[
                        row["truth_PRIVATE_EVAL_ONLY"]
                    ]
                ),
                seed + 20 + policy_index
            ),

            "unknown_wrong_edit": cluster_bootstrap(
                unknown_rows,
                lambda row: (
                    values(row)["decision"] == "edit"
                    and values(row)["final_kind"]
                    != UNKNOWN[
                        row["truth_PRIVATE_EVAL_ONLY"]
                    ]
                ),
                seed + 30 + policy_index
            ),
        }

    # Paired evaluation:
    # Same scenes, same true mechanisms,
    # same held-out E4 observations.
    for other in (
        "early_only",
        "fixed_weak",
        "fixed_strong",
        "random50"
    ):

        comparison = cluster_bootstrap(
            data,

            lambda row: 1000 * np.mean([
                (
                    row["policies"]["bic_adaptive"]
                    ["E4_rms_m"][name]
                )
                -
                (
                    row["policies"][other]
                    ["E4_rms_m"][name]
                )
                for name in E4
            ]),

            seed + 110 + POLICIES.index(other)
        )

        summaries["bic_adaptive"][
            "paired_E4_mean_delta_mm_vs_" + other
        ] = comparison

    return summaries, data


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

    check_unit()

    if args.self_test_only:
        return

    # Reuse earlier validated implementations.
    # These files are not modified.
    sim = run_path(str(
        HERE /
        "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE /
        "22_p3a1b_structural_revision.py"
    ))

    stage = run_path(str(
        HERE /
        "24_p3a2a_gate_and_excitation.py"
    ))

    assert tuple(sim["MODES"]) == (
        KNOWN + tuple(UNKNOWN)
    )

    profiles = sim["make_profiles"]()

    if args.smoke:
        ncal, ndev, nshift = 2, 2, 2
        nfev = 20
    else:
        ncal, ndev, nshift = 12, 18, 18
        nfev = 50

    generate = stage["make_split"]

    cal, physics_cal = generate(
        sim,
        api,
        profiles,
        ncal,
        KNOWN,
        SEED + 11,
        evaluation=False,
        nfev=nfev
    )

    dev, physics_dev = generate(
        sim,
        api,
        profiles,
        ndev,
        sim["MODES"],
        SEED + 22,
        evaluation=True,
        nfev=nfev
    )

    shift, physics_shift = generate(
        sim,
        api,
        profiles,
        nshift,
        KNOWN,
        SEED + 33,
        mass_shift=True,
        evaluation=True,
        nfev=nfev
    )

    # Calibration uses known mechanisms only.
    cal_bic = [
        early_score(
            row["known"]["bic"],
            row["edit"]["bic"]
        )
        for row in cal
    ]

    bic_cut = max(
        2.0,
        float(np.quantile(cal_bic, 0.95))
    )

    e3_cut = {
        arm: stage["calibrate"](cal, arm)
        for arm in (WEAK, STRONG)
    }

    dev_report, dev_rows = evaluate(
        dev,
        bic_cut,
        e3_cut,
        api,
        stage,
        SEED + 100
    )

    shift_report, shift_rows = evaluate(
        shift,
        bic_cut,
        e3_cut,
        api,
        stage,
        SEED + 200
    )

    # Private evaluation labels are used ONLY here.
    unknown_scores = [
        row["early_bic_advantage"]
        for row in dev_rows
        if row["truth_PRIVATE_EVAL_ONLY"]
        in UNKNOWN
    ]

    known_scores = [
        row["early_bic_advantage"]
        for row in dev_rows
        if row["truth_PRIVATE_EVAL_ONLY"]
        in KNOWN
    ]

    shift_scores = [
        row["early_bic_advantage"]
        for row in shift_rows
    ]

    auc_id = auc_pairwise(
        unknown_scores,
        known_scores
    )

    auc_shift = auc_pairwise(
        unknown_scores,
        shift_scores
    )

    physics = {
        "max_passive_gap_m": max(
            q["max_passive_gap_m"]
            for q in (
                physics_cal,
                physics_dev,
                physics_shift
            )
        ),

        "max_A_mode_gap_m": max(
            q["max_A_gap_m"]
            for q in (
                physics_cal,
                physics_dev,
                physics_shift
            )
        ),
    }

    report = {
        "protocol": (
            "P3-A2B E1/E2-only evidence "
            "vs nonprivileged E3 action selection"
        ),

        "development_only": True,

        "seed": SEED,

        "scene_counts": {
            "calibration": ncal,
            "development": ndev,
            "mass_shift": nshift,
        },

        "policy": (
            "Strong E3 iff known_BIC - edit_BIC "
            "exceeds known-only calibration q95; "
            "otherwise Weak E3."
        ),

        "calibration_known_only": {
            "bic_margin_cutoff": bic_cut,
            "E3_thresholds": e3_cut,
        },

        "early_BIC_unknown_AUROC_vs_ID_known":
            auc_id,

        "early_BIC_unknown_AUROC_vs_mass_shift_known":
            auc_shift,

        "development": dev_report,

        "mass_shift": shift_report,

        "physics": physics,

        "source_sha256": {
            name: hashlib.sha256(
                (HERE / name).read_bytes()
            ).hexdigest()
            for name in (
                "16_p2d1b_probe_diversity.py",
                "22_p3a1b_structural_revision.py",
                "24_p3a2a_gate_and_excitation.py"
            )
        },

        "notes": [
            (
                "The action chooser does not access "
                "true mechanisms or future observations."
            ),
            (
                "Alternative E3 outcomes are generated "
                "offline only for paired evaluation."
            ),
            (
                "Early-only is a no-E3 baseline."
            ),
            (
                "Random50 is not necessarily "
                "work-budget matched."
            ),
            (
                "BIC is a heuristic with "
                "temporally correlated residuals."
            ),
            (
                "E4 B predictions are conditioned "
                "on observed E4 A trajectories."
            ),
            (
                "Candidate structural edits are "
                "manually specified and include "
                "the tested unknown mechanisms."
            ),
            (
                "This is a development experiment, "
                "not a final confirmatory test."
            )
        ]
    }

    assert (
        physics["max_passive_gap_m"] < 1e-8
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "development"
    )

    report_path = (
        OUT /
        f"p3a2b_{suffix}_report.json"
    )

    audit_path = (
        OUT /
        f"p3a2b_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    audit_path.write_text(
        json.dumps(
            {
                "development": dev_rows,
                "mass_shift": shift_rows
            },
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        f"P3-A2B {suffix}: "
        f"cal/dev/shift = "
        f"{ncal}/{ndev}/{nshift}"
    )

    print(
        "Known-only BIC cutoff:",
        round(bic_cut, 3)
    )

    print(
        "Early BIC unknown AUROC "
        "(ID / mass shift):",
        auc_id,
        auc_shift
    )

    for group in (
        "development",
        "mass_shift"
    ):

        print("\n", group)

        for policy in POLICIES:

            stats = report[group][policy]

            correct = stats[
                "unknown_correct_edit"
            ]

            print(
                f"{policy:>13s}",
                f"actions={stats['actions']}",
                f"work={stats['mean_E3_work_proxy_J']:.3f}J",
                f"false_edit="
                f"{stats['known_false_edit']['mean']:.3f}",
                f"correct_unknown="
                f"{None if correct is None else round(correct['mean'], 3)}",
                "E4mm=",
                {
                    k: round(v, 3)
                    for k, v in stats[
                        "E4_mean_rms_mm_by_probe"
                    ].items()
                }
            )

    print(
        "\nP3-A2B DEVELOPMENT-ONLY AUDIT FINISHED"
    )

    print(
        "Not learned active concept discovery."
    )

    print("Saved:", report_path)


if __name__ == "__main__":
    main()
