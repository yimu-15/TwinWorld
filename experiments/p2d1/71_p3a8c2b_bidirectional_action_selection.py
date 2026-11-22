
"""
P3-A8C2B — Bidirectional Pre-Action Selection

New test scenes; frozen P3-A8C1B thresholds.

Policies:
  mid_only
  fixed_high
  fixed_reverse
  fixed_biphasic
  old_disagreement
  affine_disagreement

Both adaptive methods use the SAME
25th-percentile pairwise model disagreement
score, differing only in pre-action A forecast.

Future A/B and truth are unavailable to planning.
Counterfactual actions are simulated offline
for paired evaluation; only selected action
work is charged to each policy.

No E4; no test threshold tuning.
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
OUT = HERE / "results" / "p3a8c2b"

REPLICATION = (
    HERE / "results" / "p3a8c1b"
    / "p3a8c1b_fresh_report.json"
)
FORECAST = (
    HERE / "results" / "p3a8c2a"
    / "p3a8c2a_fresh_report.json"
)

TEST_SEED = 20280521
NOISE_SEED = 2028052101
N_TEST = 72

BANDS = ("original", "above_original_range")
KNOWN = (
    "independent", "rigid", "elastic",
    "deadzone", "nonlinear",
)
MODES = KNOWN + ("saturating",)

TRUTH = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

HISTORY = (
    ("e1", 0.25),
    ("e2", 0.50),
    ("weak", 0.75),
    ("strong", 2.25),
    ("mid", 1.75),
)

CHALLENGES = {
    "high": (2.75, 5),
    "reverse_high": (-2.75, 6),
    "biphasic_high": (None, 7),
}

POLICIES = (
    "mid_only",
    "fixed_high",
    "fixed_reverse",
    "fixed_biphasic",
    "old_disagreement",
    "affine_disagreement",
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_provenance():
    for report_path in (REPLICATION, FORECAST):
        report = load(report_path)
        if report["smoke"]:
            raise RuntimeError("Formal report required")
        for name, expected in report["source_sha256"].items():
            if sha(HERE / name) != expected:
                raise RuntimeError(
                    "Historical source mismatch: " + name
                )
    return load(REPLICATION)


def initialize():
    p68 = run_path(str(
        HERE / "68_p3a8c1_temporal_action_feasibility.py"
    ))
    p64 = run_path(str(
        HERE / "64_p3a8b0_preaction_disagreement_pilot.py"
    ))
    p65 = run_path(str(
        HERE / "65_p3a8b1_observable_A_response_audit.py"
    ))
    core, src, sim, api, commands, budgets = (
        p68["initialize"]()
    )
    return core, src, sim, api, commands, p64, p65, budgets


def observed_a(probe):
    y = np.asarray(probe.a_half[::2], dtype=float).ravel()
    if y.shape != (51,) or not np.isfinite(y).all():
        raise RuntimeError("Invalid A trajectory")
    return y


def project_probe(reference, positions, sim):
    values = np.asarray(positions, dtype=float).ravel()
    if values.shape != (51,) or not np.isfinite(values).all():
        raise RuntimeError("Invalid forecast shape")

    interval = float(sim["DT"]) * int(sim["SAMPLE_EVERY"])
    times = np.arange(51) * interval
    half_times = np.arange(101) * interval / 2.0

    spline = CubicSpline(times, values, bc_type="natural")

    projected = copy.copy(reference)
    projected.a_half = np.asarray(spline(half_times))
    projected.va_half = np.asarray(
        spline.derivative()(half_times)
    )
    projected.a0 = float(projected.a_half[0])
    return projected


def collect_history(core, src, sim, api, commands, state,
                    scene, mode):
    prior = {}
    for index, (name, _) in enumerate(HISTORY):
        probe, _ = core["probe"](
            src, sim, api, state["model"],
            scene, mode, commands[name],
            state["seedbase"] + index,
        )
        prior[name] = probe

    # Ensure replay reproduces the Mid observation
    # used by collect_first for candidate filtering.
    for law in src["LAWS"]:
        replay = float(api["prediction_rms"](
            prior["mid"], state["fits"][law]
        ))
        recorded = float(state["mid_scores"][law])
        if abs(replay - recorded) > 1e-9:
            raise RuntimeError("Mid replay mismatch")

    return prior


def prepare_forecasts(prior, commands, sim, p64, p65):
    gains = np.asarray([g for _, g in HISTORY], dtype=float)
    positions = np.asarray(
        [float(prior[name].a0) for name, _ in HISTORY]
    )
    anchor = float(np.median(positions))

    displacements = np.stack([
        observed_a(prior[name]) - float(prior[name].a0)
        for name, _ in HISTORY
    ])

    coeff = p65["fit_response"](
        gains, displacements, degree=1
    )

    projected = {"old": {}, "affine": {}}
    predicted_A = {}

    for action in ("high", "reverse_high"):
        gain, _ = CHALLENGES[action]

        affine_A = anchor + p65["forecast_gain"](
            coeff, gain
        )

        projected["affine"][action] = project_probe(
            prior["weak"], affine_A, sim
        )
        projected["old"][action] = p64[
            "construct_future_a_proxy"
        ](
            prior["weak"],
            commands[action],
            sim,
        )
        predicted_A[action] = affine_A

    return projected, predicted_A


def choose_actions(first_set, fits, projected, p64):
    comparison = (
        list(first_set)
        if len(first_set) >= 2
        else sorted(fits)
    )
    scores = {}
    selections = {}

    for method in ("old", "affine"):
        scores[method] = {
            action: float(p64["structural_disagreement"](
                projected[method][action],
                fits,
                comparison,
            ))
            for action in ("high", "reverse_high")
        }

        # Deterministic tie-break: High.
        selections[method] = max(
            ("high", "reverse_high"),
            key=lambda action: (
                scores[method][action],
                action == "high",
            ),
        )

    return selections, scores


def run_case(core, src, sim, api, commands,
             p64, p65, scene, sid, mode, band,
             thresholds, noise_seed):
    state = core["collect_first"](
        src, sim, api, commands,
        scene, sid, mode, band, noise_seed,
    )

    first = list(core["allowed"](
        state["mid_scores"], thresholds["mid"]
    ))

    plans = {policy: None for policy in POLICIES}
    plans["fixed_high"] = "high"
    plans["fixed_reverse"] = "reverse_high"
    plans["fixed_biphasic"] = "biphasic_high"

    score_audit = {}
    forecast_error = {}

    if first:
        prior = collect_history(
            core, src, sim, api, commands,
            state, scene, mode,
        )

        # Planning ends before challenge simulation.
        projected, predicted_A = prepare_forecasts(
            prior, commands, sim, p64, p65,
        )

        selections, score_audit = choose_actions(
            first, state["fits"], projected, p64
        )
        plans["old_disagreement"] = selections["old"]
        plans["affine_disagreement"] = selections["affine"]
    else:
        for policy in POLICIES:
            plans[policy] = None

    # All action choices are now frozen.
    # Future observations are generated below.
    compatible = {}
    action_cost = {}

    if first:
        for action, (_, index) in CHALLENGES.items():
            future, work = core["probe"](
                src, sim, api,
                state["model"], scene, mode,
                commands[action],
                state["seedbase"] + index,
            )

            scores = {
                law: float(api["prediction_rms"](
                    future, state["fits"][law]
                ))
                for law in src["LAWS"]
            }

            compatible[action] = set(core["allowed"](
                scores, thresholds[action]
            ))
            action_cost[action] = float(work)

            if action in predicted_A:
                forecast_error[action] = float(
                    1000 * np.sqrt(np.mean(
                        (predicted_A[action]
                         - observed_a(future)) ** 2
                    ))
                )

    final_sets = {}
    outcomes = {}
    costs = {}

    for policy in POLICIES:
        action = plans[policy]

        if action is None:
            final_set = sorted(first)
            cost = float(state["base_work"])
        else:
            final_set = sorted(
                set(first).intersection(compatible[action])
            )
            cost = float(
                state["base_work"] + action_cost[action]
            )

        final_sets[policy] = final_set
        outcomes[policy] = core["classify"](
            final_set, mode
        )
        costs[policy] = cost

    return {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "first_set": first,
        "plans": plans,
        "scores_PRIVATE": score_audit,
        "A_forecast_errors_mm_PRIVATE": forecast_error,
        "final_sets": final_sets,
        "outcomes_PRIVATE": outcomes,
        "costs_J": costs,
    }


def summarize(rows):
    known = [
        r for r in rows if r["truth_PRIVATE"] in KNOWN
    ]
    unknown = [
        r for r in rows
        if r["truth_PRIVATE"] == "saturating"
    ]

    summary = {}

    for policy in POLICIES:
        item = {
            "correct": sum(
                r["outcomes_PRIVATE"][policy]
                == "correct_singleton"
                for r in known
            ),
            "wrong": sum(
                r["outcomes_PRIVATE"][policy]
                == "wrong_singleton"
                for r in known
            ),
            "contains_true": sum(
                TRUTH[r["truth_PRIVATE"]]
                in r["final_sets"][policy]
                for r in known
            ),
            "unknown_empty": sum(
                r["outcomes_PRIVATE"][policy]
                == "unknown_empty"
                for r in unknown
            ),
            "unknown_false_singleton": sum(
                r["outcomes_PRIVATE"][policy]
                == "unknown_false_singleton"
                for r in unknown
            ),
            "work_J": float(sum(
                r["costs_J"][policy] for r in rows
            )),
            "actions": dict(Counter(
                str(r["plans"][policy]) for r in rows
            )),
        }

        summary[policy] = item

        print(
            f" {policy:20s}"
            f" correct={item['correct']}"
            f" wrong={item['wrong']}"
            f" contains={item['contains_true']}"
            f" unknown_empty={item['unknown_empty']}"
            f" unknown_false={item['unknown_false_singleton']}"
            f" work={item['work_J']:.3f}"
        )
        print("   actions:", item["actions"])

    eligible = [r for r in rows if r["first_set"]]

    changes = sum(
        r["plans"]["old_disagreement"]
        != r["plans"]["affine_disagreement"]
        for r in eligible
    )

    forecast_summary = {}
    for action in ("high", "reverse_high"):
        values = [
            r["A_forecast_errors_mm_PRIVATE"][action]
            for r in eligible
        ]
        forecast_summary[action] = {
            "n": len(values),
            "median_mm": float(np.median(values))
            if values else None,
            "p90_mm": float(np.quantile(values, 0.9))
            if values else None,
        }

    print(
        " Old/Affine action changes:",
        changes, "/", len(eligible)
    )
    print(" Affine forecast:", forecast_summary)

    return {
        "policies": summary,
        "eligible": len(eligible),
        "old_affine_action_changes": changes,
        "A_forecast_PRIVATE": forecast_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    historical = check_provenance()

    (
        core, src, sim, api, commands,
        p64, p65, budgets
    ) = initialize()

    offset = 100000 if args.smoke else 0
    n = 2 if args.smoke else N_TEST

    scenes = src["make_scenes"](
        sim, TEST_SEED + offset, n
    )

    print("\nP3-A8C2B BIDIRECTIONAL ACTION SELECTION")
    print("New base scenes:", n)
    print("Thresholds: frozen P3-A8C1B")
    print("No future A/B before planning.")
    print("No E4. No test tuning.")

    summaries, audits = {}, {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(scenes):
            for mode in MODES:
                rows.append(run_case(
                    core, src, sim, api, commands,
                    p64, p65,
                    scene_set[band], sid, mode, band,
                    historical["thresholds_m"][band],
                    NOISE_SEED + offset,
                ))

        if len(rows) != n * len(MODES):
            raise RuntimeError("Incomplete test matrix")

        print("\nBAND:", band)
        summaries[band] = summarize(rows)
        audits[band] = rows

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "fresh"

    report_path = OUT / f"p3a8c2b_{suffix}_report.json"
    audit_path = OUT / f"p3a8c2b_{suffix}_audit_PRIVATE.json"

    report = {
        "protocol": "P3-A8C2B bidirectional selection",
        "smoke": args.smoke,
        "n_test_scenes": n,
        "test_seed": TEST_SEED + offset,
        "noise_seed": NOISE_SEED + offset,
        "threshold_source": "P3-A8C1B formal",
        "alpha_mid": 0.05,
        "alpha_high": 0.025,
        "alpha_reverse_high": 0.025,
        "selection_score": (
            "25th-percentile pairwise normalized "
            "prediction RMS, same as P3-A8B0"
        ),
        "summary": summaries,
        "online_uses_future_A": False,
        "online_uses_future_B": False,
        "online_uses_truth": False,
        "E4_accessed": False,
        "test_tuning": False,
        "limitations": [
            "Only two selectable actions",
            "Biphasic included as fixed reference only",
            "Affine negative extrapolation remains noisy",
            "Point-estimate candidate models",
            "No calibrated planner uncertainty",
            "No wrong-singleton probability guarantee",
            "One held-out unknown mechanism",
            "Offline counterfactual execution",
            "Selected actions alone charged",
            "Mechanical work is a proxy",
            "Gap bands share base scenes",
            "Synthetic MuJoCo positions only",
            "No real hardware or RGB",
        ],
        "source_sha256": {
            name: sha(HERE / name)
            for name in (
                "64_p3a8b0_preaction_disagreement_pilot.py",
                "65_p3a8b1_observable_A_response_audit.py",
                "68_p3a8c1_temporal_action_feasibility.py",
                "71_p3a8c2b_bidirectional_action_selection.py",
            )
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(audits, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nP3-A8C2B COMPLETE")
    print("No E4. No future trajectory in planning.")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
