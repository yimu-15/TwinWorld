
"""
P3-A8B0 — Pre-Action Disagreement Pilot

Goal:
Test whether candidate-model disagreement can
select Reverse vs High WITHOUT accessing the
future observed A trajectory.

All strategies use the same E1/E2/Weak/Strong/Mid
observations and frozen calibrated cutoffs.

Policies:
  mid_only
  always_reverse
  always_high
  predicted_disagreement
  cost_normalized_disagreement

Planner inputs ONLY:
  previous Weak observation,
  fitted candidate models,
  current Mid-compatible hypothesis set,
  known command profiles.

The future A and B observations are generated
ONLY AFTER the action choice.

New calibration/test scenes. No E4.
No test threshold tuning.
Not yet parameter-uncertainty-aware.
"""

import argparse
import copy
import hashlib
import json

from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.interpolate import CubicSpline


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a8b0"

HISTORY = (
    HERE / "results" / "p3a7e1"
    / "p3a7e1_fresh_report.json"
)

CAL_SEED = 20271001
TEST_SEED = 20271002

CAL_NOISE = 2027100101
TEST_NOISE = 2027100201

N_CAL = 79
N_TEST = 36

SIGMA = 0.005
REFERENCE_COMMAND_BUDGET = 0.4

BANDS = (
    "original",
    "above_original_range",
)

KNOWN = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear",
)

MODES = KNOWN + ("saturating",)

TRUTH = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

POLICIES = (
    "mid_only",
    "always_reverse",
    "always_high",
    "predicted_disagreement",
    "cost_normalized_disagreement",
)


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def initialize():
    core = run_path(str(
        HERE / "60_p3a7e1_action_family_pilot.py"
    ))

    src, sim, api, commands = (
        core["initialize"]()
    )

    return core, src, sim, api, commands


def construct_future_a_proxy(
    weak_probe, command, sim
):
    """
    Uses previous observed initial A position and
    velocity plus a known acceleration command.

    Assumes ideal target tracking by controlled A.
    Does NOT use actual future A or true mass.

    Its physical approximation error is evaluated
    separately after the action is executed.
    """
    dt = float(sim["DT"])
    steps = int(sim["STEPS"])
    every = int(sim["SAMPLE_EVERY"])

    cmd = np.asarray(command, dtype=float)

    if cmd.shape != (steps,):
        raise RuntimeError(
            "Unexpected command shape"
        )

    position = float(weak_probe.a0)
    velocity = float(weak_probe.va_half[0])

    samples = [position]

    for step in range(steps):
        velocity += float(cmd[step]) * dt
        position += velocity * dt

        if (step + 1) % every == 0:
            samples.append(position)

    if len(samples) != 51:
        raise RuntimeError(
            "Expected 51 trajectory samples"
        )

    observed_dt = dt * every

    t = np.arange(51) * observed_dt
    half_t = np.arange(101) * (
        observed_dt / 2.0
    )

    interpolator = CubicSpline(
        t, samples,
        bc_type="natural"
    )

    # Copy a previously observed probe,
    # but replace its future A trajectory
    # with a command-derived projection.
    projected = copy.copy(weak_probe)

    projected.a_half = np.asarray(
        interpolator(half_t), dtype=float
    )

    projected.va_half = np.asarray(
        interpolator.derivative()(half_t),
        dtype=float
    )

    projected.a0 = float(
        projected.a_half[0]
    )

    return projected


def structural_disagreement(
    projected_probe, fits, laws
):
    """
    Pre-action separation of fitted predictions.

    Returns 25th percentile of pairwise,
    sigma-normalized trajectory RMS distances.

    This deliberately does not claim to account
    for uncertainty over fitted parameters.
    """
    predictions = {}

    for law in laws:
        fit = fits[law]

        y = np.asarray(
            projected_probe.predict(
                fit["kind"],
                fit["params"]
            ),
            dtype=float,
        )

        if not np.isfinite(y).all():
            raise RuntimeError(
                "Nonfinite projected trajectory"
            )

        predictions[law] = y

    distances = []

    for i, first in enumerate(laws):
        for second in laws[i + 1:]:
            a = predictions[first]
            b = predictions[second]

            if a.shape != b.shape:
                raise RuntimeError(
                    "Model prediction shape mismatch"
                )

            distances.append(
                float(np.sqrt(np.mean(
                    ((a - b) / SIGMA) ** 2
                )))
            )

    if not distances:
        return 0.0

    return float(
        np.quantile(distances, 0.25)
    )


def select_actions(
    first_set, fits, weak_probe,
    commands, sim
):
    """
    The only action-planning function.

    IMPORTANT:
    It takes no simulator scene, truth label,
    future A, future B, or future residual.
    """
    if len(first_set) == 0:
        return {
            "predicted_disagreement": None,
            "cost_normalized_disagreement": None,
        }, {}, {}

    # For a singleton, test whether rejected
    # competitors could still offer an
    # alternative explanation.
    comparison_laws = (
        list(first_set)
        if len(first_set) >= 2
        else sorted(fits)
    )

    scores = {}
    projected = {}

    for action in ("reverse", "high"):
        projected[action] = (
            construct_future_a_proxy(
                weak_probe,
                commands[action],
                sim,
            )
        )

        scores[action] = (
            structural_disagreement(
                projected[action],
                fits,
                comparison_laws,
            )
        )

    def nominal_budget(action):
        return float(
            np.sum(
                np.abs(commands[action])
            )
            * float(sim["DT"])
        )

    raw_choice = max(
        ("reverse", "high"),
        key=lambda action: (
            scores[action],
            -nominal_budget(action)
        )
    )

    cost_choice = max(
        ("reverse", "high"),
        key=lambda action: (
            scores[action] /
            (
                nominal_budget(action)
                / REFERENCE_COMMAND_BUDGET
            ),
            -nominal_budget(action)
        )
    )

    return {
        "predicted_disagreement": raw_choice,
        "cost_normalized_disagreement":
            cost_choice,
    }, scores, projected


def evaluate_episode(
    core, src, sim, api, commands,
    scene, sid, mode, band,
    noise_seed, cutoffs,
):
    # Observations through Mid are completed.
    state = core["collect_first"](
        src, sim, api, commands,
        scene, sid, mode, band,
        noise_seed,
    )

    first = core["allowed"](
        state["mid_scores"],
        cutoffs["mid"],
    )

    # Reproduce the already-observed Weak probe
    # using its original seed.
    # No extra experimental action is charged.
    weak_probe, _ = core["probe"](
        src, sim, api,
        state["model"],
        scene,
        mode,
        commands["weak"],
        state["seedbase"] + 2,
    )

    selected, planned_scores, projected = (
        select_actions(
            first,
            state["fits"],
            weak_probe,
            commands,
            sim,
        )
    )

    # Decisions made and frozen at this point.
    planned = {
        "mid_only": None,
        "always_reverse": "reverse",
        "always_high": "high",
        **selected,
    }

    # Generate future observations ONLY AFTER
    # planning. These are counterfactual
    # executions for paired policy evaluation.
    challenge_scores = {}
    challenge_costs = {}
    proxy_errors_mm = {}

    for index, action in (
        (5, "reverse"),
        (6, "high"),
    ):
        actual, work = core["probe"](
            src, sim, api,
            state["model"],
            scene,
            mode,
            commands[action],
            state["seedbase"] + index,
        )

        challenge_costs[action] = float(work)

        challenge_scores[action] = {}

        for law in src["LAWS"]:
            fit = state["fits"][law]

            challenge_scores[action][law] = (
                float(api["prediction_rms"](
                    actual, fit
                ))
            )

        if action in projected:
            # PRIVATE diagnostic only:
            # compare planned A with actual A.
            predicted_a = (
                projected[action]
                .a_half[::2]
            )

            observed_a = (
                actual.a_half[::2]
            )

            proxy_errors_mm[action] = float(
                1000 * np.sqrt(np.mean(
                    (predicted_a - observed_a) ** 2
                ))
            )

    compatible = {
        action: core["allowed"](
            challenge_scores[action],
            cutoffs[action],
        )
        for action in ("reverse", "high")
    }

    final_sets = {}
    outcomes = {}
    costs = {}

    for policy, action in planned.items():
        if action is None:
            final_set = list(first)
            cost = state["base_work"]
        else:
            final_set = sorted(
                set(first).intersection(
                    compatible[action]
                )
            )

            cost = (
                state["base_work"]
                + challenge_costs[action]
            )

        final_sets[policy] = final_set

        outcomes[policy] = (
            core["classify"](
                final_set, mode
            )
        )

        costs[policy] = float(cost)

    return {
        "scene_id": sid,
        "truth_PRIVATE": mode,
        "band": band,
        "initial_set": first,
        "selected_actions": planned,
        "planned_scores": planned_scores,
        "proxy_A_errors_mm_PRIVATE":
            proxy_errors_mm,
        "final_sets": final_sets,
        "outcomes_PRIVATE": outcomes,
        "costs_J": costs,
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

    result = {}

    reference_cost = sum(
        r["costs_J"]["always_high"]
        for r in rows
    )

    for policy in POLICIES:
        count = Counter(
            r["outcomes_PRIVATE"][policy]
            for r in rows
        )

        actions = Counter(
            str(r["selected_actions"][policy])
            for r in rows
        )

        cost = sum(
            r["costs_J"][policy]
            for r in rows
        )

        result[policy] = {
            "known_correct_unique": sum(
                r["outcomes_PRIVATE"][policy]
                == "correct_singleton"
                for r in known
            ),

            "known_wrong_unique": sum(
                r["outcomes_PRIVATE"][policy]
                == "wrong_singleton"
                for r in known
            ),

            "known_contains_true": sum(
                TRUTH[r["truth_PRIVATE"]]
                in r["final_sets"][policy]
                for r in known
            ),

            "unknown_false_singleton": sum(
                r["outcomes_PRIVATE"][policy]
                == "unknown_false_singleton"
                for r in unknown
            ),

            "unknown_empty": sum(
                r["outcomes_PRIVATE"][policy]
                == "unknown_empty"
                for r in unknown
            ),

            "actions": dict(actions),
            "outcomes": dict(count),

            "cost_J": float(cost),

            "saving_vs_always_high": float(
                (reference_cost - cost)
                / max(reference_cost, 1e-12)
            ),
        }

        item = result[policy]

        print(
            f"  {policy:33s}",
            "correct=", item["known_correct_unique"],
            "wrong=", item["known_wrong_unique"],
            "contains=", item["known_contains_true"],
            "unknown_empty=", item["unknown_empty"],
            "unknown_false=", item[
                "unknown_false_singleton"
            ],
        )

        print(
            "    actions=", item["actions"],
            "cost_J=", round(item["cost_J"], 3),
            "saving_vs_high=",
            round(
                100 * item["saving_vs_always_high"],
                3
            ),
            "%",
        )

    # Surrogate validation is an OFFLINE audit;
    # it cannot affect the online action choice.
    proxy_summary = {}

    for action in ("reverse", "high"):
        values = [
            r["proxy_A_errors_mm_PRIVATE"][action]
            for r in rows
            if action in r["proxy_A_errors_mm_PRIVATE"]
        ]

        proxy_summary[action] = {
            "n": len(values),
            "median_mm": (
                float(np.median(values))
                if values else None
            ),
            "p90_mm": (
                float(np.quantile(values, 0.9))
                if values else None
            ),
        }

    print(
        "\n  Pre-action A proxy accuracy:",
        proxy_summary,
    )

    return {
        "policies": result,
        "A_proxy_diagnostic_PRIVATE":
            proxy_summary,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    historic = load_json(HISTORY)

    if historic["smoke"]:
        raise RuntimeError(
            "Expected formal source report"
        )

    for filename, expected in (
        historic["source_sha256"].items()
    ):
        if digest(HERE / filename) != expected:
            raise RuntimeError(
                "Historical source changed: "
                + filename
            )

    core, src, sim, api, commands = (
        initialize()
    )

    offset = 100000 if args.smoke else 0

    n_cal = (
        4 if args.smoke else N_CAL
    )

    n_test = (
        2 if args.smoke else N_TEST
    )

    print(
        "\nP3-A8B0 PRE-ACTION "
        "DISAGREEMENT PILOT"
    )

    print(
        "Calibration:", n_cal,
        "| Test:", n_test,
    )

    cal_scenes = src["make_scenes"](
        sim, CAL_SEED + offset, n_cal
    )

    test_scenes = src["make_scenes"](
        sim, TEST_SEED + offset, n_test
    )

    # Same predefined 0.05 / 0.025 / 0.025
    # per-law calibration protocol as A7E1.
    cutoffs = core["calibrate"](
        src, sim, api, commands,
        cal_scenes,
        CAL_NOISE + offset,
        args.smoke,
    )

    summaries = {}
    audits = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(
            test_scenes
        ):
            for mode in MODES:
                rows.append(
                    evaluate_episode(
                        core, src, sim, api,
                        commands,
                        scene_set[band],
                        sid, mode, band,
                        TEST_NOISE + offset,
                        cutoffs[band],
                    )
                )

        if len(rows) != (
            n_test * len(MODES)
        ):
            raise RuntimeError(
                "Incomplete experimental matrix"
            )

        print("\nBAND:", band)

        summaries[band] = summarize(rows)
        audits[band] = rows

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report = {
        "protocol":
            "P3-A8B0 non-oracle pre-action "
            "model disagreement pilot",

        "smoke": args.smoke,
        "n_cal_scenes": n_cal,
        "n_test_scenes": n_test,
        "calibration_seed": CAL_SEED + offset,
        "test_seed": TEST_SEED + offset,

        "planning_actions": [
            "reverse",
            "high",
        ],

        "planning_uses_future_observed_A": False,
        "planning_uses_future_observed_B": False,
        "planning_uses_true_physics": False,

        "planner_A_model":
            "Ideal command integration initialized "
            "from previous Weak A observation",

        "planner_score":
            "25th percentile pairwise "
            "sigma-normalized prediction RMS",

        "cost_normalization":
            "Absolute commanded acceleration budget",

        "calibration_alpha": {
            "mid": 0.05,
            "reverse": 0.025,
            "high": 0.025,
        },

        "cutoffs_m": cutoffs,
        "summary": summaries,

        "E4_accessed": False,
        "test_tuning": False,

        "limitations": [
            "Planning uses approximate future A",
            "Only point estimates for model parameters",
            "No calibrated confidence for planning score",
            "No true-actuator dynamic identification",
            "Candidate pool can exclude true law",
            "Planned costs use nominal command budget",
            "Reported actual costs are offline proxies",
            "Both counterfactual actions simulated",
            "One unknown Saturating mechanism",
            "Gap bands share base scene physics",
            "Synthetic MuJoCo and numeric positions only",
            "No RGB or real robot validation",
            "Smoke thresholds are not inferential",
        ],
    }

    report_path = OUT / (
        f"p3a8b0_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a8b0_{suffix}_audit_PRIVATE.json"
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
            audits,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A8B0 PRE-ACTION "
        "PILOT COMPLETE"
    )

    print(
        "No E4. No oracle future trajectory "
        "for planning."
    )

    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
