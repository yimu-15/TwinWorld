
"""
P3-A7E1: Action-Family Feasibility Pilot

New independent calibration/test scenes.
Compare reverse vs stronger positive intervention.

Truth used only for simulator and PRIVATE audit.
Action choice uses only the Vmid hypothesis set.

No E4. No test tuning.
No unknown-law calibration.
No claim of optimal experiment design.
"""

import argparse
import json
import hashlib
from math import ceil
from pathlib import Path
from runpy import run_path
from collections import Counter

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a7e1"

HIST = (
    HERE / "results" / "p3a7d"
    / "p3a7d_fresh_report.json"
)

CAL_SEED = 20270701
TEST_SEED = 20270702
CAL_NOISE = 2027070101
TEST_NOISE = 2027070201

N_CAL = 79
N_TEST = 36
NFEV = 60

BANDS = ("original", "above_original_range")
KNOWN = (
    "independent", "rigid",
    "elastic", "deadzone", "nonlinear"
)
MODES = KNOWN + ("saturating",)

TRUTH = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

ALPHA = {
    "mid": 0.05,
    "reverse": 0.025,
    "high": 0.025,
}

TRAIN = ("e1", "e2", "weak", "strong")
METHODS = (
    "mid_only",
    "always_reverse",
    "reverse_if_ambiguous",
    "high_if_ambiguous",
    "candidate_aware",
)


def load(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def quantile(values, alpha, smoke):
    s = sorted(float(x) for x in values)
    n = len(s)
    k = ceil((n + 1) * (1 - alpha))

    if k > n:
        if not smoke:
            raise RuntimeError(
                "Insufficient formal calibration size"
            )
        # Smoke is functionality-only;
        # it has no coverage interpretation.
        k = n

    return s[k - 1]


def allowed(scores, thresholds):
    return sorted(
        law for law, value in scores.items()
        if value <= thresholds[law]
    )


def choose_action(first):
    # Predeclared physical heuristic.
    # No truth and no challenge observation.
    if len(first) == 0:
        return None
    if len(first) == 1:
        return "reverse"
    if "rigid" in first:
        return "reverse"
    return "high"


def initialize():
    core = run_path(str(
        HERE / "58_p3a7d_hypothesis_set_validation.py"
    ))

    src, sim, api, _ = core["init"]()

    pulse = sim["make_profiles"]()["pulse"]

    commands = {
        "e1": pulse * 0.25,
        "e2": pulse * 0.50,
        "weak": pulse * 0.75,
        "strong": pulse * 2.25,
        "mid": pulse * 1.75,
        "reverse": pulse * -1.75,
        "high": pulse * 2.75,
    }

    return src, sim, api, commands


def probe(src, sim, api, model, scene,
          mode, command, seed):
    p, work = src["get_probe"](
        sim, api, model, scene,
        mode, command, seed
    )
    return p, float(work)


def collect_first(
    src, sim, api, commands,
    scene, sid, mode, band, noise_seed,
):
    model = sim["make_model"](scene)

    seedbase = (
        noise_seed
        + sid * 1000
        + src["MODES"].index(mode) * 20
        + BANDS.index(band) * 100000
    )

    observed = {}
    work = {}

    for index, name in enumerate(TRAIN):
        observed[name], work[name] = probe(
            src, sim, api, model, scene,
            mode, commands[name],
            seedbase + index,
        )

    _, fits = api["fit_library"](
        [observed[name] for name in TRAIN],
        src["LAWS"], NFEV,
    )

    mid, work["mid"] = probe(
        src, sim, api, model, scene,
        mode, commands["mid"],
        seedbase + 4,
    )

    scores = {
        law: float(
            api["prediction_rms"](mid, fits[law])
        )
        for law in src["LAWS"]
    }

    if not np.isfinite(list(scores.values())).all():
        raise RuntimeError("Invalid Vmid scores")

    return {
        "model": model,
        "fits": fits,
        "scene": scene,
        "mode": mode,
        "seedbase": seedbase,
        "mid_scores": scores,
        "base_work": sum(work.values()),
    }


def collect_challenges(
    src, sim, api, commands, state
):
    # Called after online action selection
    # on the test path. Both potential actions
    # are simulated for paired offline audit.
    scores = {}
    works = {}

    for idx, name in enumerate(
        ("reverse", "high"), start=5
    ):
        p, works[name] = probe(
            src, sim, api,
            state["model"],
            state["scene"],
            state["mode"],
            commands[name],
            state["seedbase"] + idx,
        )

        scores[name] = {
            law: float(
                api["prediction_rms"](
                    p, state["fits"][law]
                )
            )
            for law in src["LAWS"]
        }

        if not np.isfinite(
            list(scores[name].values())
        ).all():
            raise RuntimeError(
                f"Invalid {name} scores"
            )

    return scores, works


def calibrate(
    src, sim, api, commands,
    scenes, noise_seed, smoke,
):
    raw = {
        band: {
            law: {
                "mid": [],
                "reverse": [],
                "high": [],
            }
            for law in src["LAWS"]
        }
        for band in BANDS
    }

    for sid, scene_set in enumerate(scenes):
        for band in BANDS:
            for mode in KNOWN:
                state = collect_first(
                    src, sim, api, commands,
                    scene_set[band], sid, mode,
                    band, noise_seed,
                )

                extra, _ = collect_challenges(
                    src, sim, api, commands, state
                )

                law = TRUTH[mode]

                raw[band][law]["mid"].append(
                    state["mid_scores"][law]
                )

                for action in ("reverse", "high"):
                    raw[band][law][action].append(
                        extra[action][law]
                    )

    cutoffs = {}

    for band in BANDS:
        cutoffs[band] = {
            action: {}
            for action in ALPHA
        }

        print("\nCALIBRATION:", band)

        for action, alpha in ALPHA.items():
            for law in src["LAWS"]:
                q = quantile(
                    raw[band][law][action],
                    alpha, smoke
                )
                cutoffs[band][action][law] = q

                print(
                    f"  {action:8s} {law:12s}"
                    f" q={1000*q:.4f} mm"
                )

    return cutoffs


def classify(group, truth):
    if truth == "saturating":
        if len(group) == 0:
            return "unknown_empty"
        if len(group) == 1:
            return "unknown_false_singleton"
        return "unknown_ambiguous"

    true_law = TRUTH[truth]

    if len(group) == 0:
        return "known_empty"

    if len(group) == 1:
        return (
            "correct_singleton"
            if group[0] == true_law
            else "wrong_singleton"
        )

    return (
        "ambiguous_contains_true"
        if true_law in group
        else "ambiguous_excludes_true"
    )


def evaluate_episode(
    src, sim, api, commands,
    scene, sid, mode, band,
    noise_seed, cutoffs,
):
    state = collect_first(
        src, sim, api, commands,
        scene, sid, mode, band, noise_seed
    )

    first = allowed(
        state["mid_scores"],
        cutoffs["mid"]
    )

    # All online action decisions are fixed HERE,
    # before either new challenge is generated.
    planned = {
        "mid_only": None,
        "always_reverse": "reverse",
        "reverse_if_ambiguous": (
            "reverse" if len(first) > 1 else None
        ),
        "high_if_ambiguous": (
            "high" if len(first) > 1 else None
        ),
        "candidate_aware": choose_action(first),
    }

    extra, extra_work = collect_challenges(
        src, sim, api, commands, state
    )

    compatible = {
        action: allowed(
            extra[action],
            cutoffs[action]
        )
        for action in ("reverse", "high")
    }

    final_sets = {}
    outcomes = {}
    costs = {}

    for method, action in planned.items():
        if action is None:
            group = list(first)
            cost = state["base_work"]
        else:
            group = sorted(
                set(first).intersection(
                    compatible[action]
                )
            )
            cost = (
                state["base_work"]
                + extra_work[action]
            )

        final_sets[method] = group
        outcomes[method] = classify(
            group, mode
        )
        costs[method] = float(cost)

    return {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "first_set": first,
        "planned_actions": planned,
        "challenge_sets": compatible,
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

    for method in METHODS:
        outcomes = [
            r["outcomes_PRIVATE"][method]
            for r in rows
        ]

        costs = [
            r["costs_J"][method]
            for r in rows
        ]

        actions = Counter(
            r["planned_actions"][method]
            for r in rows
        )

        s = {
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
            "unknown_false_singleton": sum(
                r["outcomes_PRIVATE"][method]
                == "unknown_false_singleton"
                for r in unknown
            ),
            "unknown_empty": sum(
                r["outcomes_PRIVATE"][method]
                == "unknown_empty"
                for r in unknown
            ),
            "ambiguous_total": sum(
                len(r["final_sets"][method]) > 1
                for r in rows
            ),
            "actions": {
                str(k): v
                for k, v in actions.items()
            },
            "cost_J": float(sum(costs)),
            "outcomes": dict(Counter(outcomes)),
        }

        result[method] = s

    reference_cost = (
        result["always_reverse"]["cost_J"]
    )

    for method in METHODS:
        s = result[method]

        s["work_reduction_vs_reverse"] = (
            (reference_cost - s["cost_J"])
            / max(reference_cost, 1e-12)
        )

        print(
            f"  {method:23s}",
            "known unique",
            s["known_correct_singleton"],
            "| wrong unique",
            s["known_wrong_singleton"],
            "| contains",
            s["known_contains_true"],
            "| unknown false unique",
            s["unknown_false_singleton"],
            "| unknown empty",
            s["unknown_empty"],
            "| actions",
            s["actions"],
            "| cost",
            round(s["cost_J"], 3),
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true"
    )
    args = parser.parse_args()

    historic = load(HIST)

    if historic["smoke"]:
        raise RuntimeError(
            "Formal historical provenance required"
        )

    for name, expected in (
        historic["source_sha256"].items()
    ):
        if sha(HERE / name) != expected:
            raise RuntimeError(
                "Historical source hash mismatch: "
                + name
            )

    src, sim, api, commands = initialize()

    offset = 100000 if args.smoke else 0
    n_cal = 4 if args.smoke else N_CAL
    n_test = 2 if args.smoke else N_TEST

    print("\nP3-A7E1 ACTION FAMILY PILOT")
    print(
        "Calibration:", n_cal,
        "Test:", n_test
    )
    print(
        "Smoke thresholds have NO coverage "
        "interpretation." if args.smoke
        else "Formal alpha allocation: 0.05/0.025/0.025"
    )

    cal_scenes = src["make_scenes"](
        sim, CAL_SEED + offset, n_cal
    )

    test_scenes = src["make_scenes"](
        sim, TEST_SEED + offset, n_test
    )

    cutoffs = calibrate(
        src, sim, api, commands,
        cal_scenes,
        CAL_NOISE + offset,
        args.smoke,
    )

    summaries = {}
    audits = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(test_scenes):
            for mode in MODES:
                rows.append(
                    evaluate_episode(
                        src, sim, api, commands,
                        scene_set[band],
                        sid, mode, band,
                        TEST_NOISE + offset,
                        cutoffs[band],
                    )
                )

        if len(rows) != n_test * len(MODES):
            raise RuntimeError(
                "Incomplete test matrix"
            )

        print("\nBAND:", band)

        summaries[band] = summarize(rows)
        audits[band] = rows

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "fresh"

    report = {
        "protocol":
            "P3-A7E1 preregistered action-family pilot",
        "smoke": args.smoke,
        "n_cal": n_cal,
        "n_test": n_test,
        "calibration_seed": CAL_SEED + offset,
        "test_seed": TEST_SEED + offset,
        "commands": {
            "mid_gain": 1.75,
            "reverse_gain": -1.75,
            "high_gain": 2.75,
        },
        "alpha_allocation": ALPHA,
        "cutoffs_m": cutoffs,
        "summary": summaries,
        "E4_accessed": False,
        "test_threshold_tuned": False,
        "truth_used_for_online_action": False,
        "action_selected_before_challenge": True,
        "smoke_has_valid_coverage": False,
        "limitations": [
            "Physics-informed fixed action heuristic",
            "Not optimized model-disagreement planning",
            "MuJoCo only, no RGB vision",
            "Unknown law excluded from calibration",
            "One specific unknown Saturating law",
            "Possible nested/equivalent candidate laws",
            "Counterfactual actions simulated offline",
            "Only selected action charged per policy",
            "No safety guarantee for unknown laws",
            "Conditional coverage needs exchangeability",
            "Two bands share base scenes",
        ],
        "source_sha256": {
            name: sha(HERE / name)
            for name in (
                "16_p2d1b_probe_diversity.py",
                "22_p3a1b_structural_revision.py",
                "27_p3a3b_evidence_vs_structure.py",
                "47_p3a6h_six_mechanism_openworld_audit.py",
                "58_p3a7d_hypothesis_set_validation.py",
            )
        },
    }

    report_path = OUT / (
        f"p3a7e1_{suffix}_report.json"
    )
    audit_path = OUT / (
        f"p3a7e1_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )
    audit_path.write_text(
        json.dumps(
            audits, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nP3-A7E1 PILOT COMPLETE")
    print("No E4. No test threshold tuning.")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
