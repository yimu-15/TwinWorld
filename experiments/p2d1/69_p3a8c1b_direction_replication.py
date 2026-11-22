
"""
P3-A8C1B — Independent Direction Replication

Primary comparison (predeclared):
    Reverse-High vs High

Secondary cost reference:
    Biphasic-High

Reuses exact action definitions and fitting
interface from P3-A8C1, but generates fully
new calibration and test base scenes.

No E4. No test threshold tuning.
No adaptive action policy yet.
Truth is used only for private evaluation.

Paired bootstrap resamples base scene IDs,
not individual mode observations.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a8c1b"

HIST = (
    HERE / "results" / "p3a8c1"
    / "p3a8c1_fresh_report.json"
)

CAL_SEED = 20280321
TEST_SEED = 20280322
CAL_NOISE = 2028032101
TEST_NOISE = 2028032201

N_CAL = 79
N_TEST = 72

BANDS = (
    "original",
    "above_original_range",
)

KNOWN = (
    "independent", "rigid",
    "elastic", "deadzone", "nonlinear",
)

MODES = KNOWN + ("saturating",)

TRUTH = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

ARMS = (
    "high",
    "reverse_high",
    "biphasic_high",
)

ALPHA_MID = 0.05
ALPHA_ACTION = 0.025

# These indices preserve P3-A8C1's
# fixed challenge-noise seed mapping.
SEED_INDEX = {
    "high": 5,
    "reverse_high": 6,
    "biphasic_high": 7,
}


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def read(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def initialize():
    historic = read(HIST)

    if historic["smoke"]:
        raise RuntimeError(
            "Formal P3-A8C1 report required"
        )

    for name, expected in (
        historic["source_sha256"].items()
    ):
        if digest(HERE / name) != expected:
            raise RuntimeError(
                "Historical source changed: " + name
            )

    p68 = run_path(str(
        HERE /
        "68_p3a8c1_temporal_action_feasibility.py"
    ))

    (
        core, src, sim,
        api, commands, budgets
    ) = p68["initialize"]()

    for arm in ARMS:
        expected = (
            historic["command_budgets"][arm]
            ["command_sha256"]
        )

        actual = budgets[arm]["command_sha256"]

        if actual != expected:
            raise RuntimeError(
                "Action definition changed: " + arm
            )

    return core, src, sim, api, commands


def collect(
    core, src, sim, api,
    commands, scene, sid,
    mode, band, noise_seed
):
    state = core["collect_first"](
        src, sim, api,
        commands, scene, sid,
        mode, band, noise_seed,
    )

    scores = {"mid": state["mid_scores"]}
    work = {}

    for arm in ARMS:
        observation, cost = core["probe"](
            src, sim, api,
            state["model"],
            scene, mode,
            commands[arm],
            state["seedbase"] + SEED_INDEX[arm],
        )

        scores[arm] = {
            law: float(
                api["prediction_rms"](
                    observation, state["fits"][law]
                )
            )
            for law in src["LAWS"]
        }

        work[arm] = float(cost)

    return {
        "scores": scores,
        "work": work,
        "base_work": float(state["base_work"]),
    }


def calibrate(
    core, src, sim, api,
    commands, scenes, noise_seed, smoke
):
    actions = ("mid",) + ARMS

    raw = {
        band: {
            law: {
                action: []
                for action in actions
            }
            for law in src["LAWS"]
        }
        for band in BANDS
    }

    for sid, scene_set in enumerate(scenes):
        for band in BANDS:
            for mode in KNOWN:
                result = collect(
                    core, src, sim, api,
                    commands, scene_set[band],
                    sid, mode, band, noise_seed
                )

                law = TRUTH[mode]

                for action in actions:
                    raw[band][law][action].append(
                        result["scores"][action][law]
                    )

    thresholds = {}

    for band in BANDS:
        thresholds[band] = {}

        for action in actions:
            alpha = (
                ALPHA_MID
                if action == "mid"
                else ALPHA_ACTION
            )

            thresholds[band][action] = {}

            for law in src["LAWS"]:
                thresholds[band][action][law] = (
                    core["quantile"](
                        raw[band][law][action],
                        alpha, smoke,
                    )
                )

    return thresholds


def evaluate(
    core, src, sim, api,
    commands, scene, sid,
    mode, band, noise_seed, thresholds
):
    result = collect(
        core, src, sim, api,
        commands, scene, sid,
        mode, band, noise_seed
    )

    first = core["allowed"](
        result["scores"]["mid"],
        thresholds["mid"],
    )

    final_sets = {}
    outcomes = {}
    costs = {}

    for arm in ARMS:
        if not first:
            final_set = []
            cost = result["base_work"]
        else:
            compatible = core["allowed"](
                result["scores"][arm],
                thresholds[arm],
            )

            final_set = sorted(
                set(first).intersection(
                    compatible
                )
            )

            cost = (
                result["base_work"]
                + result["work"][arm]
            )

        final_sets[arm] = final_set
        outcomes[arm] = core["classify"](
            final_set, mode
        )
        costs[arm] = float(cost)

    return {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "first_set": first,
        "final_sets": final_sets,
        "outcomes_PRIVATE": outcomes,
        "costs_J": costs,
        "challenge_work_J_PRIVATE": result["work"],
    }


def summarize(rows):
    known = [
        r for r in rows
        if r["truth_PRIVATE"] in KNOWN
    ]

    unknown = [
        r for r in rows
        if r["truth_PRIVATE"] == "saturating"
    ]

    summary = {}

    for arm in ARMS:
        summary[arm] = {
            "known_correct": sum(
                r["outcomes_PRIVATE"][arm]
                == "correct_singleton"
                for r in known
            ),
            "known_wrong": sum(
                r["outcomes_PRIVATE"][arm]
                == "wrong_singleton"
                for r in known
            ),
            "known_contains": sum(
                TRUTH[r["truth_PRIVATE"]]
                in r["final_sets"][arm]
                for r in known
            ),
            "unknown_empty": sum(
                r["outcomes_PRIVATE"][arm]
                == "unknown_empty"
                for r in unknown
            ),
            "unknown_false_singleton": sum(
                r["outcomes_PRIVATE"][arm]
                == "unknown_false_singleton"
                for r in unknown
            ),
            "total_work_J": float(
                sum(
                    r["costs_J"][arm]
                    for r in rows
                )
            ),
            "per_mode_correct_PRIVATE": {
                mode: sum(
                    r["truth_PRIVATE"] == mode
                    and r["outcomes_PRIVATE"][arm]
                    == "correct_singleton"
                    for r in known
                )
                for mode in KNOWN
            },
        }

    paired = {
        "reverse_only_correct": sum(
            r["outcomes_PRIVATE"]["reverse_high"]
            == "correct_singleton"
            and r["outcomes_PRIVATE"]["high"]
            != "correct_singleton"
            for r in known
        ),
        "high_only_correct": sum(
            r["outcomes_PRIVATE"]["high"]
            == "correct_singleton"
            and r["outcomes_PRIVATE"]["reverse_high"]
            != "correct_singleton"
            for r in known
        ),
        "reverse_only_unknown_empty": sum(
            r["outcomes_PRIVATE"]["reverse_high"]
            == "unknown_empty"
            and r["outcomes_PRIVATE"]["high"]
            != "unknown_empty"
            for r in unknown
        ),
        "high_only_unknown_empty": sum(
            r["outcomes_PRIVATE"]["high"]
            == "unknown_empty"
            and r["outcomes_PRIVATE"]["reverse_high"]
            != "unknown_empty"
            for r in unknown
        ),
    }

    return summary, paired


def bootstrap_by_scene(rows, n_scene, seed):
    """
    Paired, scene-cluster bootstrap.

    Every resampled base scene carries all
    six mechanisms together.
    """
    by_scene = defaultdict(list)

    for row in rows:
        by_scene[row["scene_id"]].append(row)

    if len(by_scene) != n_scene:
        raise RuntimeError(
            "Unexpected scene count"
        )

    vectors = []

    for sid in range(n_scene):
        group = by_scene[sid]

        if len(group) != len(MODES):
            raise RuntimeError(
                "Incomplete mechanism group"
            )

        known = [
            r for r in group
            if r["truth_PRIVATE"] in KNOWN
        ]

        unknown = [
            r for r in group
            if r["truth_PRIVATE"] == "saturating"
        ]

        correct_delta = sum(
            int(
                r["outcomes_PRIVATE"]["reverse_high"]
                == "correct_singleton"
            ) - int(
                r["outcomes_PRIVATE"]["high"]
                == "correct_singleton"
            )
            for r in known
        )

        contains_delta = sum(
            int(
                TRUTH[r["truth_PRIVATE"]]
                in r["final_sets"]["reverse_high"]
            ) - int(
                TRUTH[r["truth_PRIVATE"]]
                in r["final_sets"]["high"]
            )
            for r in known
        )

        unknown_delta = sum(
            int(
                r["outcomes_PRIVATE"]["reverse_high"]
                == "unknown_empty"
            ) - int(
                r["outcomes_PRIVATE"]["high"]
                == "unknown_empty"
            )
            for r in unknown
        )

        high_work = sum(
            r["costs_J"]["high"]
            for r in group
        )

        reverse_work = sum(
            r["costs_J"]["reverse_high"]
            for r in group
        )

        vectors.append([
            correct_delta,
            contains_delta,
            unknown_delta,
            high_work,
            reverse_work,
        ])

    values = np.asarray(vectors, dtype=float)
    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0, n_scene,
        size=(10000, n_scene)
    )

    sample = values[indices].sum(axis=1)

    savings = (
        1.0 - sample[:, 4]
        / np.maximum(sample[:, 3], 1e-12)
    )

    def interval(arr):
        return [
            float(x)
            for x in np.quantile(
                arr, [0.025, 0.975]
            )
        ]

    return {
        "correct_delta_95pct": interval(
            sample[:, 0]
        ),
        "contains_delta_95pct": interval(
            sample[:, 1]
        ),
        "unknown_empty_delta_95pct": interval(
            sample[:, 2]
        ),
        "work_saving_fraction_95pct": interval(
            savings
        ),
        "bootstrap_unit": "base_scene_id",
        "replications": 10000,
        "interpretation": (
            "Exploratory clustered uncertainty; "
            "no real-system generalization guarantee"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true"
    )
    args = parser.parse_args()

    (
        core, src, sim, api, commands
    ) = initialize()

    offset = 100000 if args.smoke else 0
    n_cal = 4 if args.smoke else N_CAL
    n_test = 2 if args.smoke else N_TEST

    print("\nP3-A8C1B DIRECTION REPLICATION")
    print(
        "Calibration:", n_cal,
        "| Test:", n_test,
    )
    print(
        "Primary comparison: Reverse-High vs High"
    )

    cal_scenes = src["make_scenes"](
        sim, CAL_SEED + offset, n_cal
    )
    test_scenes = src["make_scenes"](
        sim, TEST_SEED + offset, n_test
    )

    thresholds = calibrate(
        core, src, sim, api, commands,
        cal_scenes, CAL_NOISE + offset,
        args.smoke,
    )

    reports = {}
    audits = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(
            test_scenes
        ):
            for mode in MODES:
                rows.append(
                    evaluate(
                        core, src, sim, api,
                        commands,
                        scene_set[band],
                        sid, mode, band,
                        TEST_NOISE + offset,
                        thresholds[band],
                    )
                )

        if len(rows) != n_test * len(MODES):
            raise RuntimeError(
                "Incomplete test matrix"
            )

        summary, paired = summarize(rows)

        bootstrap = bootstrap_by_scene(
            rows, n_test,
            TEST_SEED + offset
        )

        reports[band] = {
            "summary": summary,
            "paired": paired,
            "cluster_bootstrap": bootstrap,
        }

        audits[band] = rows

        print("\nBAND:", band)

        for arm in ARMS:
            s = summary[arm]
            print(
                f"  {arm:15s}",
                "correct=", s["known_correct"],
                "wrong=", s["known_wrong"],
                "contains=", s["known_contains"],
                "unknown_empty=", s["unknown_empty"],
                "unknown_false=",
                s["unknown_false_singleton"],
                "work_J=",
                round(s["total_work_J"], 3),
            )

        print("  Paired:", paired)
        print(
            "  Bootstrap correct delta 95%:",
            bootstrap["correct_delta_95pct"],
        )
        print(
            "  Bootstrap work saving 95%:",
            bootstrap["work_saving_fraction_95pct"],
        )

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "fresh"

    report_path = (
        OUT / f"p3a8c1b_{suffix}_report.json"
    )
    audit_path = (
        OUT / f"p3a8c1b_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P3-A8C1B independent direction replication",
        "smoke": args.smoke,
        "n_cal": n_cal,
        "n_test": n_test,
        "cal_seed": CAL_SEED + offset,
        "test_seed": TEST_SEED + offset,
        "alpha_mid": ALPHA_MID,
        "alpha_action": ALPHA_ACTION,
        "primary_comparison":
            "Reverse-High vs High",
        "secondary_action":
            "Biphasic-High",
        "thresholds_m": thresholds,
        "results": reports,
        "E4_accessed": False,
        "test_threshold_tuned": False,
        "online_uses_truth": False,
        "adaptive_action_policy": False,
        "limitations": [
            "Synthetic MuJoCo-only confirmation",
            "Same base scenes across gap bands",
            "Only one held-out unknown family",
            "Counterfactual actions simulated offline",
            "One selected action charged per baseline",
            "Noise streams differ between action arms",
            "Equal command L1 does not imply equal work",
            "Actuator clipping requires separate audit",
            "Calibration validity needs exchangeability",
            "No joint adaptive-policy coverage claim",
        ],
        "source_sha256": {
            name: digest(HERE / name)
            for name in (
                "60_p3a7e1_action_family_pilot.py",
                "68_p3a8c1_temporal_action_feasibility.py",
            )
        },
    }

    report_path.write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            audits, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\nP3-A8C1B REPLICATION COMPLETE")
    print("No E4. No test threshold tuning.")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
