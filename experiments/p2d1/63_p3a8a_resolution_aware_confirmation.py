
"""
P3-A8A: Resolution-Aware Concept Confirmation

First proposed decision-rule pilot.

Same observations for both baselines:
    E1 E2 Weak Strong -> fit five laws
    Mid + High -> independent validation

Baseline:
    One calibrated compatible law => unique

Resolution-aware:
    One compatible law AND sufficient
    predictive separation => confirmed
    Otherwise => abstain / ambiguous

New calibration and test worlds.
No E4, no oracle in online decision.
No evaluation-set threshold tuning.

IMPORTANT:
Gaussian separation is a diagnostic proxy,
NOT a certified probability bound for
composite fitted models.
"""

import argparse
import hashlib
import json
from collections import Counter
from math import ceil
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a8a"

HISTORY = (
    HERE / "results" / "p3a7e1"
    / "p3a7e1_fresh_report.json"
)

CAL_SEED = 20270901
TEST_SEED = 20270902
CAL_NOISE = 2027090101
TEST_NOISE = 2027090201

N_CAL = 79
N_TEST = 36

ALPHA_MID = 0.05
ALPHA_HIGH = 0.05

# For simple, fixed Gaussian hypotheses only.
SIMPLE_HYPOTHESIS_ERROR = 0.05

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


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def qscore(values, alpha, smoke):
    arr = sorted(float(x) for x in values)
    n = len(arr)
    k = ceil((n + 1) * (1 - alpha))

    if k > n:
        if not smoke:
            raise RuntimeError(
                "Insufficient calibration"
            )
        k = n

    return arr[k - 1]


def initialize():
    previous = run_path(str(
        HERE /
        "60_p3a7e1_action_family_pilot.py"
    ))

    src, sim, api, commands = (
        previous["initialize"]()
    )

    fit_one = (
        api["fit_library"]
        .__globals__["fit_one"]
    )

    bounds = fit_one.__globals__["BOUNDS"]
    sigma = float(
        fit_one.__globals__["SIGMA"]
    )

    if abs(sigma - 0.005) > 1e-12:
        raise RuntimeError(
            "Unexpected measurement sigma"
        )

    return (
        previous, src, sim,
        api, commands, bounds, sigma
    )


def run_case(
    previous, src, sim, api, commands,
    scene, sid, mode, band, noise_seed,
    sigma,
):
    # Fits only use already executed
    # E1/E2/Weak/Strong observations.
    state = previous["collect_first"](
        src, sim, api, commands,
        scene, sid, mode, band, noise_seed,
    )

    # Reconstruct Mid from its original
    # deterministic noise seed, solely
    # to obtain the observed A trajectory.
    mid, mid_work = previous["probe"](
        src, sim, api,
        state["model"],
        scene, mode,
        commands["mid"],
        state["seedbase"] + 4,
    )

    high, high_work = previous["probe"](
        src, sim, api,
        state["model"],
        scene, mode,
        commands["high"],
        state["seedbase"] + 6,
    )

    probes = {
        "mid": mid,
        "high": high,
    }

    scores = {}
    predictions = {}

    for action, p in probes.items():
        scores[action] = {}
        predictions[action] = {}

        for law in src["LAWS"]:
            fit = state["fits"][law]

            pred = np.asarray(
                p.predict(
                    fit["kind"],
                    fit["params"],
                ),
                dtype=float,
            ).ravel()

            if (
                pred.size == 0
                or not np.isfinite(pred).all()
            ):
                raise RuntimeError(
                    "Invalid prediction"
                )

            predictions[action][law] = pred

            scores[action][law] = float(
                api["prediction_rms"](p, fit)
            )

    # Replay integrity: reconstructed Mid
    # must match the scores already used
    # by the earlier experimental pipeline.
    for law in src["LAWS"]:
        if abs(
            scores["mid"][law]
            - state["mid_scores"][law]
        ) > 1e-8:
            raise RuntimeError(
                "Mid deterministic replay failed"
            )

    separation = {}

    for i, first in enumerate(src["LAWS"]):
        separation[first] = {}

        for second in src["LAWS"]:
            if first == second:
                separation[first][second] = 0.0
                continue

            terms = []

            for action in ("mid", "high"):
                a = predictions[action][first]
                b = predictions[action][second]

                if a.shape != b.shape:
                    raise RuntimeError(
                        "Prediction shape mismatch"
                    )

                terms.append(
                    float(np.sum(
                        ((a - b) / sigma) ** 2
                    ))
                )

            separation[first][second] = float(
                np.sqrt(sum(terms))
            )

    return {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "scores_m": scores,
        "separation": separation,
        "work_J": float(
            state["base_work"] + high_work
        ),
        "mid_replay_work_J_PRIVATE":
            float(mid_work),
    }


def calibrate(
    previous, src, sim, api, commands,
    scenes, seed, sigma, smoke,
):
    values = {
        band: {
            law: {"mid": [], "high": []}
            for law in src["LAWS"]
        }
        for band in BANDS
    }

    for sid, scene_set in enumerate(scenes):
        for band in BANDS:
            for mode in KNOWN:
                r = run_case(
                    previous, src, sim,
                    api, commands,
                    scene_set[band],
                    sid, mode, band,
                    seed, sigma,
                )

                law = TRUTH[mode]

                for action in ("mid", "high"):
                    values[band][law][action].append(
                        r["scores_m"][action][law]
                    )

    cutoffs = {
        band: {
            "mid": {},
            "high": {},
        }
        for band in BANDS
    }

    for band in BANDS:
        print("\nCALIBRATION:", band)

        for law in src["LAWS"]:
            for action, alpha in (
                ("mid", ALPHA_MID),
                ("high", ALPHA_HIGH),
            ):
                q = qscore(
                    values[band][law][action],
                    alpha, smoke,
                )

                cutoffs[band][action][law] = q

                print(
                    f" {action:5s} {law:12s}"
                    f" q={q * 1000:.4f} mm"
                )

    return cutoffs


def evaluate_case(
    r, cutoffs, separation_cutoff
):
    laws = list(r["scores_m"]["mid"])

    accepted = sorted([
        h for h in laws
        if (
            r["scores_m"]["mid"][h]
            <= cutoffs["mid"][h]
            and r["scores_m"]["high"][h]
            <= cutoffs["high"][h]
        )
    ])

    baseline = (
        accepted[0]
        if len(accepted) == 1
        else (
            "unresolved_none"
            if len(accepted) == 0
            else "ambiguous"
        )
    )

    min_separation = None
    nearest = None

    if len(accepted) == 1:
        selected = accepted[0]

        distances = {
            h: r["separation"][selected][h]
            for h in laws if h != selected
        }

        nearest = min(
            distances, key=distances.get
        )

        min_separation = distances[nearest]

        if min_separation >= separation_cutoff:
            new_decision = selected
        else:
            new_decision = "resolution_ambiguous"

    elif len(accepted) == 0:
        new_decision = "unresolved_none"

    else:
        new_decision = "ambiguous"

    return {
        "scene_id": r["scene_id"],
        "truth_PRIVATE": r["truth_PRIVATE"],
        "accepted_set": accepted,
        "baseline_decision": baseline,
        "resolution_decision": new_decision,
        "nearest_competitor": nearest,
        "minimum_separation": min_separation,
        "cost_J": r["work_J"],
    }


def summarize(rows):
    result = {}

    for method in (
        "baseline",
        "resolution",
    ):
        decision_key = (
            method + "_decision"
        )

        known_rows = [
            r for r in rows
            if r["truth_PRIVATE"] in KNOWN
        ]

        unknown_rows = [
            r for r in rows
            if r["truth_PRIVATE"] == "saturating"
        ]

        def is_unique(d):
            return d in TRUTH.values()

        correct = sum(
            r[decision_key]
            == TRUTH[r["truth_PRIVATE"]]
            for r in known_rows
        )

        wrong = sum(
            is_unique(r[decision_key])
            and r[decision_key]
            != TRUTH[r["truth_PRIVATE"]]
            for r in known_rows
        )

        unknown_false = sum(
            is_unique(r[decision_key])
            for r in unknown_rows
        )

        counts = dict(Counter(
            r[decision_key] for r in rows
        ))

        result[method] = {
            "known_correct_unique": correct,
            "known_wrong_unique": wrong,
            "unknown_false_unique": unknown_false,
            "known_contains_true": sum(
                TRUTH[r["truth_PRIVATE"]]
                in r["accepted_set"]
                for r in known_rows
            ),
            "unknown_empty": sum(
                len(r["accepted_set"]) == 0
                for r in unknown_rows
            ),
            "decisions": counts,
        }

        print(
            f" {method:10s}",
            "correct unique:", correct,
            "| wrong unique:", wrong,
            "| unknown false unique:", unknown_false,
        )

    result["total_work_J"] = float(
        sum(r["cost_J"] for r in rows)
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true"
    )
    args = parser.parse_args()

    history = json.loads(
        HISTORY.read_text(encoding="utf-8")
    )

    if history["smoke"]:
        raise RuntimeError(
            "Formal provenance required"
        )

    for name, expected in (
        history["source_sha256"].items()
    ):
        if sha(HERE / name) != expected:
            raise RuntimeError(
                "Source provenance mismatch: "
                + name
            )

    (
        previous, src, sim, api,
        commands, bounds, sigma
    ) = initialize()

    separation_cutoff = float(
        2.0 * norm.ppf(
            1.0 - SIMPLE_HYPOTHESIS_ERROR
        )
    )

    # Report actual optimizer bounds.
    bound_report = {}

    for law in ("elastic", "deadzone", "cubic"):
        lower, upper = bounds[law]

        bound_report[law] = {
            "lower": np.asarray(
                lower, dtype=float
            ).tolist(),
            "upper": np.asarray(
                upper, dtype=float
            ).tolist(),
        }

    offset = 100000 if args.smoke else 0
    n_cal = 4 if args.smoke else N_CAL
    n_test = 2 if args.smoke else N_TEST

    print(
        "\nP3-A8A RESOLUTION-AWARE CONFIRMATION"
    )

    print("Calibration scenes:", n_cal)
    print("Test scenes:", n_test)
    print(
        "Simple-hypothesis separation cutoff:",
        round(separation_cutoff, 6)
    )

    print(
        "\nOPTIMIZER BOUNDS:\n",
        json.dumps(bound_report, indent=2)
    )

    cal_scenes = src["make_scenes"](
        sim, CAL_SEED + offset, n_cal
    )

    test_scenes = src["make_scenes"](
        sim, TEST_SEED + offset, n_test
    )

    cutoffs = calibrate(
        previous, src, sim, api,
        commands, cal_scenes,
        CAL_NOISE + offset,
        sigma, args.smoke,
    )

    summaries = {}
    audits = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(
            test_scenes
        ):
            for mode in MODES:
                r = run_case(
                    previous, src, sim, api,
                    commands,
                    scene_set[band],
                    sid, mode, band,
                    TEST_NOISE + offset,
                    sigma,
                )

                rows.append(
                    evaluate_case(
                        r,
                        cutoffs[band],
                        separation_cutoff,
                    )
                )

        if len(rows) != (
            n_test * len(MODES)
        ):
            raise RuntimeError(
                "Incomplete test matrix"
            )

        print("\nBAND:", band)
        summaries[band] = summarize(rows)
        audits[band] = rows

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report = {
        "protocol":
            "P3-A8A resolution-aware confirmation",
        "smoke": args.smoke,
        "n_cal_scenes": n_cal,
        "n_test_scenes": n_test,
        "calibration_seed": CAL_SEED + offset,
        "test_seed": TEST_SEED + offset,
        "optimizer_bounds_PRIVATE": bound_report,
        "measurement_sigma_m": sigma,
        "alpha_mid": ALPHA_MID,
        "alpha_high": ALPHA_HIGH,
        "separation_cutoff": separation_cutoff,
        "cutoffs_m": cutoffs,
        "summary": summaries,
        "E4_accessed": False,
        "test_threshold_tuned": False,
        "online_uses_truth": False,
        "claim_of_certified_error_control": False,
        "limitations": [
            "Fixed Mid and High interventions",
            "No action planning yet",
            "Separation assumes simple Gaussian models",
            "Fitted model uncertainty is not integrated",
            "Observed A used after action completion",
            "No frequentist false-confirmation guarantee",
            "MuJoCo synthetic data only",
            "No RGB perception",
            "Only one unknown mechanism",
            "Gap bands share physical scenes",
            "Test data cannot retune this rule",
        ],
    }

    report_path = OUT / (
        f"p3a8a_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a8a_{suffix}_audit_PRIVATE.json"
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
            audits,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "\nP3-A8A RESOLUTION-AWARE PILOT COMPLETE"
    )
    print(
        "No E4. No threshold tuning."
    )
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
