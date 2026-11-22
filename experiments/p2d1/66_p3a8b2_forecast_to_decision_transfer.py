
"""
P3-A8B2: Forecast-to-Decision Transfer

Compare old A forecast vs historical affine A
forecast under identical frozen action scoring.

Calibration thresholds:
  Reuse P3-A8B0 formal calibration.

Evaluation:
  New scene and noise seeds only.

No E4.
No future A/B used before action selection.
No truth-dependent online action choice.
No test-set threshold tuning.

Simulates BOTH challenge actions for paired
offline evaluation, charging only the
action selected by each policy.

This tests the planning component, not a
certified open-world discovery algorithm.
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
OUT = HERE / "results" / "p3a8b2"

OLD_REPORT = (
    HERE / "results" / "p3a8b0"
    / "p3a8b0_fresh_report.json"
)

B1_REPORT = (
    HERE / "results" / "p3a8b1"
    / "p3a8b1_fresh_report.json"
)

TEST_SEED = 20271222
NOISE_SEED = 2027122201

N_SCENES = 36
NFEV = 60

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

TRAIN = ("e1", "e2", "weak", "strong")

HISTORY = (
    "e1", "e2", "weak",
    "strong", "mid",
)

CHALLENGES = ("reverse", "high")

METHODS = (
    "mid_only",
    "always_reverse",
    "always_high",
    "random_action",
    "old_planner",
    "affine_planner",
    "affine_cost_planner",
)


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def build_projected_probe(
    reference_probe, positions
):
    """
    Positions are predicted from historical
    observations BEFORE the future action.

    The B initial state remains estimated
    from a previously observed Weak probe.
    """
    y = np.asarray(
        positions, dtype=float
    ).ravel()

    if y.shape != (51,):
        raise RuntimeError(
            "Expected 51 predicted A positions"
        )

    if not np.isfinite(y).all():
        raise RuntimeError(
            "Nonfinite A forecast"
        )

    t = np.arange(51) * 0.02
    tm = np.arange(101) * 0.01

    spline = CubicSpline(
        t, y, bc_type="natural"
    )

    projected = copy.copy(reference_probe)

    projected.a_half = np.asarray(
        spline(tm), dtype=float
    )

    projected.va_half = np.asarray(
        spline.derivative()(tm),
        dtype=float,
    )

    projected.a0 = float(
        projected.a_half[0]
    )

    return projected


def replace_with_observed_A(
    reference_probe, future_probe
):
    """
    PRIVATE retrospective upper-reference.

    Never used to make an online action
    decision. It provides the true observed
    A trajectory after that action.
    """
    projected = copy.copy(reference_probe)

    projected.a_half = (
        future_probe.a_half.copy()
    )

    projected.va_half = (
        future_probe.va_half.copy()
    )

    projected.a0 = float(
        future_probe.a0
    )

    return projected


def choose_from_scores(
    scores, commands, sim,
    normalize_cost=False,
):
    if not scores:
        return None

    if normalize_cost:
        budgets = {
            name: float(
                np.sum(
                    np.abs(commands[name])
                ) * sim["DT"]
            )
            for name in CHALLENGES
        }

        return max(
            CHALLENGES,
            key=lambda name: (
                scores[name]
                / max(budgets[name], 1e-12),
                -budgets[name],
            )
        )

    return max(
        CHALLENGES,
        key=lambda name: scores[name]
    )


def planning_scores(
    planner, fits, comparison,
    probes,
):
    return {
        action: planner[
            "structural_disagreement"
        ](
            probes[action],
            fits,
            comparison,
        )
        for action in CHALLENGES
    }


def prepare_episode(
    core, src, sim, api,
    commands, scene, sid,
    mode, band, noise_seed,
):
    model = sim["make_model"](scene)

    seedbase = (
        noise_seed
        + sid * 1000
        + src["MODES"].index(mode) * 20
        + BANDS.index(band) * 100000
    )

    prior = {}
    works = {}

    for index, action in enumerate(
        HISTORY
    ):
        prior[action], works[action] = (
            core["probe"](
                src, sim, api,
                model, scene, mode,
                commands[action],
                seedbase + index,
            )
        )

    # Mid does NOT participate in fitting.
    _, fits = api["fit_library"](
        [prior[action] for action in TRAIN],
        src["LAWS"],
        NFEV,
    )

    mid_scores = {
        law: float(
            api["prediction_rms"](
                prior["mid"], fits[law]
            )
        )
        for law in src["LAWS"]
    }

    return {
        "model": model,
        "scene": scene,
        "mode_PRIVATE": mode,
        "seedbase": seedbase,
        "prior": prior,
        "fits": fits,
        "mid_scores": mid_scores,
        "base_work_J": float(
            sum(works.values())
        ),
    }


def evaluate_case(
    planner, forecaster,
    core, src, sim, api,
    commands, scene, sid,
    mode, band, noise_seed,
    thresholds,
):
    state = prepare_episode(
        core, src, sim, api,
        commands, scene, sid,
        mode, band, noise_seed,
    )

    first = core["allowed"](
        state["mid_scores"],
        thresholds["mid"],
    )

    planned_scores = {}
    proxy_forecasts = {}
    plans = {
        "mid_only": None,
        "always_reverse": "reverse",
        "always_high": "high",
    }

    # This random policy depends only on the
    # scene identifier and band, NOT truth.
    rng = np.random.default_rng(
        TEST_SEED + sid * 100
        + BANDS.index(band)
    )

    plans["random_action"] = (
        CHALLENGES[int(rng.integers(2))]
        if first else None
    )

    if not first:
        plans["old_planner"] = None
        plans["affine_planner"] = None
        plans["affine_cost_planner"] = None

    else:
        comparison = (
            list(first)
            if len(first) >= 2
            else sorted(state["fits"])
        )

        # Old planner: original P3-A8B0 method.
        old_projected = {
            action: planner[
                "construct_future_a_proxy"
            ](
                state["prior"]["weak"],
                commands[action],
                sim,
            )
            for action in CHALLENGES
        }

        # Affine forecast from the five
        # previously executed actions.
        # NO future challenge information.
        forecasts, diagnostics = (
            forecaster["prepare_forecasters"](
                planner,
                sim,
                commands,
                state["prior"],
            )
        )

        affine_projected = {
            action: build_projected_probe(
                state["prior"]["weak"],
                forecasts[action]["affine"],
            )
            for action in CHALLENGES
        }

        planned_scores["old"] = (
            planning_scores(
                planner,
                state["fits"],
                comparison,
                old_projected,
            )
        )

        planned_scores["affine"] = (
            planning_scores(
                planner,
                state["fits"],
                comparison,
                affine_projected,
            )
        )

        plans["old_planner"] = (
            choose_from_scores(
                planned_scores["old"],
                commands, sim,
            )
        )

        plans["affine_planner"] = (
            choose_from_scores(
                planned_scores["affine"],
                commands, sim,
            )
        )

        plans["affine_cost_planner"] = (
            choose_from_scores(
                planned_scores["affine"],
                commands, sim,
                normalize_cost=True,
            )
        )

        proxy_forecasts = {
            action: forecasts[action]["affine"]
            for action in CHALLENGES
        }

    # ------------------------------
    # ONLINE PLANNING ENDS HERE.
    # ------------------------------

    future = {}
    challenge_work = {}

    for index, action in enumerate(
        CHALLENGES, start=5
    ):
        future[action], challenge_work[action] = (
            core["probe"](
                src, sim, api,
                state["model"],
                scene, mode,
                commands[action],
                state["seedbase"] + index,
            )
        )

    # Observations are available only now.
    compatible = {}

    for action in CHALLENGES:
        scores = {
            law: float(
                api["prediction_rms"](
                    future[action],
                    state["fits"][law],
                )
            )
            for law in src["LAWS"]
        }

        compatible[action] = (
            core["allowed"](
                scores,
                thresholds[action],
            )
        )

    final_sets = {}
    outcomes = {}
    costs = {}

    for method in METHODS:
        action = plans[method]

        if action is None:
            final_set = list(first)
            cost = state["base_work_J"]

        else:
            final_set = sorted(
                set(first).intersection(
                    compatible[action]
                )
            )

            cost = (
                state["base_work_J"]
                + challenge_work[action]
            )

        final_sets[method] = final_set

        # PRIVATE evaluation only.
        outcomes[method] = (
            core["classify"](
                final_set, mode
            )
        )

        costs[method] = float(cost)

    # Post-action diagnostic:
    # Compare planned score rankings against
    # rankings using actually observed A.
    actual_A_scores = {}
    proxy_A_errors = {}

    if first:
        comparison = (
            list(first)
            if len(first) >= 2
            else sorted(state["fits"])
        )

        conditioned = {
            action: replace_with_observed_A(
                state["prior"]["weak"],
                future[action],
            )
            for action in CHALLENGES
        }

        actual_A_scores = planning_scores(
            planner,
            state["fits"],
            comparison,
            conditioned,
        )

        for action in CHALLENGES:
            observed_positions = np.asarray(
                future[action].a_half[::2],
                dtype=float,
            )

            proxy_A_errors[action] = float(
                1000 * np.sqrt(np.mean(
                    (
                        proxy_forecasts[action]
                        - observed_positions
                    ) ** 2
                ))
            )

    return {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "initial_set": first,
        "selected_actions": plans,
        "planned_scores": planned_scores,
        "actual_A_scores_PRIVATE":
            actual_A_scores,
        "proxy_A_errors_mm_PRIVATE":
            proxy_A_errors,
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

    reference = sum(
        r["costs_J"]["always_high"]
        for r in rows
    )

    summary = {}

    for method in METHODS:
        cost = float(sum(
            r["costs_J"][method]
            for r in rows
        ))

        item = {
            "known_correct_singleton": sum(
                r["outcomes_PRIVATE"][method]
                == "correct_singleton"
                for r in known
            ),

            "known_wrong_singleton": sum(
                r["outcomes_PRIVATE"][method]
                == "wrong_singleton"
                for r in known
            ),

            "known_contains_true": sum(
                TRUTH[r["truth_PRIVATE"]]
                in r["final_sets"][method]
                for r in known
            ),

            "unknown_empty": sum(
                r["outcomes_PRIVATE"][method]
                == "unknown_empty"
                for r in unknown
            ),

            "unknown_false_singleton": sum(
                r["outcomes_PRIVATE"][method]
                == "unknown_false_singleton"
                for r in unknown
            ),

            "actions": dict(Counter(
                str(r["selected_actions"][method])
                for r in rows
            )),

            "cost_J": cost,

            "saving_vs_always_high": float(
                (reference - cost)
                / max(reference, 1e-12)
            ),
        }

        summary[method] = item

        print(
            f"  {method:24s}",
            "correct=", item[
                "known_correct_singleton"
            ],
            "wrong=", item[
                "known_wrong_singleton"
            ],
            "contains=", item[
                "known_contains_true"
            ],
            "unknown empty=", item[
                "unknown_empty"
            ],
            "unknown false=", item[
                "unknown_false_singleton"
            ],
        )

        print(
            "    actions=", item["actions"],
            "cost=", round(cost, 3),
            "saving=", round(
                100 * item[
                    "saving_vs_always_high"
                ], 3
            ),
            "%",
        )

    eligible = [
        r for r in rows
        if len(r["initial_set"]) > 0
    ]

    changed = sum(
        r["selected_actions"]["old_planner"]
        != r["selected_actions"]["affine_planner"]
        for r in eligible
    )

    # Retrospective actual-A score ranking:
    # NOT an online action policy.
    oracle_agreements = {}

    for method, score_key in (
        ("old", "old"),
        ("affine", "affine"),
    ):
        match = sum(
            r["selected_actions"][
                "old_planner"
                if method == "old"
                else "affine_planner"
            ] == max(
                r["actual_A_scores_PRIVATE"],
                key=r[
                    "actual_A_scores_PRIVATE"
                ].get,
            )
            for r in eligible
        )

        oracle_agreements[method] = {
            "match_count": match,
            "n": len(eligible),
        }

    proxy_errors = {}

    for action in CHALLENGES:
        values = [
            r["proxy_A_errors_mm_PRIVATE"][action]
            for r in eligible
        ]

        proxy_errors[action] = {
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
        "  Old -> Affine action changes:",
        changed, "/", len(eligible)
    )

    print(
        "  Actual-A ranking agreement:",
        oracle_agreements
    )

    print(
        "  Affine A forecast accuracy:",
        proxy_errors
    )

    return {
        "policies": summary,
        "eligible_action_cases": len(eligible),
        "old_affine_action_changes": changed,
        "actual_A_ranking_agreement_PRIVATE":
            oracle_agreements,
        "A_forecast_diagnostic_PRIVATE":
            proxy_errors,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    old = read_json(OLD_REPORT)
    b1 = read_json(B1_REPORT)

    if old["smoke"] or b1["smoke"]:
        raise RuntimeError(
            "Formal historical reports required"
        )

    # Freeze all previous physics/fitting
    # implementations using B1 provenance.
    for filename, expected in (
        b1["source_sha256"].items()
    ):
        if digest(HERE / filename) != expected:
            raise RuntimeError(
                "Historical source changed: "
                + filename
            )

    planner = run_path(str(
        HERE /
        "64_p3a8b0_preaction_disagreement_pilot.py"
    ))

    forecaster = run_path(str(
        HERE /
        "65_p3a8b1_observable_A_response_audit.py"
    ))

    core, src, sim, api, commands = (
        planner["initialize"]()
    )

    cutoffs = old["cutoffs_m"]

    offset = 100000 if args.smoke else 0
    n = 2 if args.smoke else N_SCENES

    scenes = src["make_scenes"](
        sim, TEST_SEED + offset, n
    )

    print(
        "\nP3-A8B2 FORECAST-TO-DECISION "
        "TRANSFER"
    )

    print("New scenes:", n)
    print("Calibration: frozen P3-A8B0")
    print("No future A before planning.")
    print("No E4 or test threshold tuning.")

    summaries = {}
    audits = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(
            scenes
        ):
            for mode in MODES:
                rows.append(
                    evaluate_case(
                        planner,
                        forecaster,
                        core, src, sim, api,
                        commands,
                        scene_set[band],
                        sid, mode, band,
                        NOISE_SEED + offset,
                        cutoffs[band],
                    )
                )

        if len(rows) != n * len(MODES):
            raise RuntimeError(
                "Incomplete test matrix"
            )

        print("\n" + "=" * 70)
        print("BAND:", band)
        print("=" * 70)

        summaries[band] = summarize(rows)
        audits[band] = rows

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report_path = OUT / (
        f"p3a8b2_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a8b2_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P3-A8B2 forecast-to-decision "
            "transfer",

        "smoke": args.smoke,
        "n_base_scenes": n,
        "test_seed": TEST_SEED + offset,
        "noise_seed": NOISE_SEED + offset,

        "calibration_reused_from":
            "P3-A8B0 formal report",

        "same_model_fits": True,
        "same_candidate_thresholds": True,
        "same_disagreement_score": True,

        "online_A_predictor":
            "Five historical actions; "
            "affine gain-response",

        "summary": summaries,

        "online_uses_future_A": False,
        "online_uses_future_B": False,
        "online_uses_truth": False,

        "actual_A_ranking_is_private": True,

        "E4_accessed": False,
        "test_tuning": False,

        "limitations": [
            "One fixed pulse action family",
            "Only Reverse and High choices",
            "Five historical probes already paid",
            "Future A prediction is approximate",
            "B initial state uses Weak probe",
            "Point estimates for structural laws",
            "No parametric uncertainty propagation",
            "Actual-A score ranking is not "
            "a structural identification oracle",
            "Counterfactual actions simulated",
            "Only selected actions charged",
            "Mechanical work is a proxy",
            "One held-out Saturating family",
            "Gap bands share base physics",
            "MuJoCo positions only; no RGB",
            "No real-robot validation",
        ],

        "source_sha256": {
            name: digest(HERE / name)
            for name in (
                "60_p3a7e1_action_family_pilot.py",
                "64_p3a8b0_preaction_disagreement_pilot.py",
                "65_p3a8b1_observable_A_response_audit.py",
            )
        },
    }

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
        "\nP3-A8B2 TRANSFER AUDIT COMPLETE"
    )

    print(
        "No E4. No future oracle in planning."
    )

    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
