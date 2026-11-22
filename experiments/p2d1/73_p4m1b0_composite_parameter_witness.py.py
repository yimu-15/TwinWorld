
"""
P4-M1B0 — Legal Composite-Parameter Witness Search

One-sided profile:
  True Elastic    -> Alternative Deadzone
  True Deadzone   -> Alternative Elastic

Alternative parameters remain inside
physical generation bounds.

Two distinct audits:
  1. Fit the alternative on historical
     clean trajectory means, then hold
     parameters FIXED for all challenges.
  2. Profile the alternative separately
     for each completed experiment design.

The second audit does NOT establish
one common pair hard under all actions.

No E4.
No learned online policy.
Oracle trajectories are PRIVATE.
Finite optimization is not global proof.
"""

import argparse
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p4m1b0"

HISTORY_REPORT = (
    HERE / "results" / "p3a8c1b"
    / "p3a8c1b_fresh_report.json"
)

SIGMA = 0.005
SEED = 20290201

N_SCENES = 8
MAX_NFEV = 25

BANDS = (
    "original",
    "above_original_range",
)

HISTORY = (
    "e1", "e2", "weak", "strong", "mid"
)

CHALLENGES = (
    "mid", "high", "reverse_high"
)

DIRECTIONS = (
    ("elastic", "deadzone"),
    ("deadzone", "elastic"),
)

HARD_ERROR = 0.10


def read(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def initialize():
    old = read(HISTORY_REPORT)

    if old.get("smoke") is not False:
        raise RuntimeError(
            "Formal source report required"
        )

    for filename, expected in (
        old["source_sha256"].items()
    ):
        if sha(HERE / filename) != expected:
            raise RuntimeError(
                "Historical source changed: "
                + filename
            )

    p68 = run_path(str(
        HERE /
        "68_p3a8c1_temporal_action_feasibility.py"
    ))

    (
        core, src, sim,
        api, commands, budgets
    ) = p68["initialize"]()

    return src, sim, commands


def parameter_spec(law, band):
    if law == "elastic":
        return (
            ("k", "c"),
            np.array([5.0, 0.5]),
            np.array([16.0, 2.2]),
        )

    if law == "deadzone":
        gap_limits = (
            (0.015, 0.12)
            if band == "original"
            else (0.12, 0.18)
        )

        return (
            ("k", "c", "gap"),
            np.array([
                5.0, 0.5, gap_limits[0]
            ]),
            np.array([
                16.0, 2.2, gap_limits[1]
            ]),
        )

    raise ValueError(law)


def make_alternative(
    scene, law, band, theta
):
    names, lower, upper = (
        parameter_spec(law, band)
    )

    theta = np.asarray(
        theta, dtype=float
    )

    if theta.shape != lower.shape:
        raise RuntimeError(
            "Parameter dimension mismatch"
        )

    if np.any(theta < lower - 1e-9) or \
       np.any(theta > upper + 1e-9):
        raise RuntimeError(
            "Illegal alternative parameter"
        )

    alt = dict(scene)

    for name, value in zip(names, theta):
        alt[name] = float(value)

    return alt


def observe_clean(
    sim, model, scene,
    law, command
):
    passive, active, _, _ = (
        sim["simulate"](
            model, scene, law, command
        )
    )

    result = np.stack(
        (passive, active), axis=0
    )

    if result.shape != (2, 51, 2):
        raise RuntimeError(
            "Unexpected trajectory shape"
        )

    if not np.isfinite(result).all():
        raise RuntimeError(
            "Nonfinite simulation"
        )

    return result.reshape(-1)


def get_target(
    sim, model, scene,
    law, commands
):
    names = set(
        HISTORY + CHALLENGES
    )

    return {
        name: observe_clean(
            sim, model, scene,
            law, commands[name]
        )
        for name in names
    }


def residual_vector(
    theta, sim, model,
    scene, alt_law, band,
    commands, targets, action_names
):
    alt_scene = make_alternative(
        scene, alt_law, band, theta
    )

    residuals = []

    for name in action_names:
        prediction = observe_clean(
            sim, model, alt_scene,
            alt_law, commands[name]
        )

        residuals.append(
            (prediction - targets[name])
            / SIGMA
        )

    return np.concatenate(residuals)


def optimize_alternative(
    sim, model, scene,
    alt_law, band,
    commands, targets,
    action_names
):
    names, lo, hi = (
        parameter_spec(alt_law, band)
    )

    original = np.array([
        scene[name] for name in names
    ], dtype=float)

    starts = (
        np.clip(original, lo, hi),
        (lo + hi) / 2,
    )

    solutions = []

    for start in starts:
        fit = least_squares(
            residual_vector,
            x0=start,
            bounds=(lo, hi),
            args=(
                sim, model, scene,
                alt_law, band,
                commands, targets,
                action_names,
            ),
            max_nfev=MAX_NFEV,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )

        solutions.append(fit)

    best = min(
        solutions,
        key=lambda x: np.dot(
            x.fun, x.fun
        ),
    )

    return {
        "theta": best.x.tolist(),
        "distance": float(
            np.linalg.norm(best.fun)
        ),
        "success": bool(best.success),
        "nfev": int(best.nfev),
    }


def distance_for(
    theta, sim, model,
    scene, alt_law, band,
    commands, targets,
    actions
):
    residual = residual_vector(
        theta, sim, model,
        scene, alt_law, band,
        commands, targets, actions
    )

    return float(
        np.linalg.norm(residual)
    )


def error_metrics(distance):
    d = float(distance)

    return {
        "d": d,
        "log10_optimal_error": float(
            norm.logsf(d / 2.0)
            / np.log(10.0)
        ),
        "hard_at_10pct": bool(
            d <= 2.563103
        ),
    }


def run_one(
    sim, commands,
    scene, band, sid,
    true_law, alt_law
):
    model = sim["make_model"](scene)

    targets = get_target(
        sim, model, scene,
        true_law, commands
    )

    historical = optimize_alternative(
        sim, model, scene,
        alt_law, band,
        commands, targets,
        HISTORY,
    )

    fixed_theta = historical["theta"]

    challenges = {}

    for name in CHALLENGES:
        # One fixed alternative fitted only
        # against the historical trajectories.
        fixed_d = distance_for(
            fixed_theta,
            sim, model, scene,
            alt_law, band,
            commands, targets,
            HISTORY + (name,),
        )

        # PRIVATE oracle profile including
        # the future challenge's true mean.
        # This is NOT a deployable predictor.
        profiled = optimize_alternative(
            sim, model, scene,
            alt_law, band,
            commands, targets,
            HISTORY + (name,),
        )

        challenges[name] = {
            "fixed_historical_parameters":
                error_metrics(fixed_d),

            "separately_profiled":
                error_metrics(
                    profiled["distance"]
                ),

            "profiled_theta_PRIVATE":
                profiled["theta"],

            "profiled_optimizer_success":
                profiled["success"],

            "profiled_nfev":
                profiled["nfev"],
        }

    return {
        "scene_id": sid,
        "band": band,
        "true_law_PRIVATE": true_law,
        "alternative_law": alt_law,
        "true_scene_PRIVATE": scene,
        "historical_fit_PRIVATE":
            historical,
        "challenges": challenges,
        "same_parameters_hard_all": all(
            challenges[name][
                "fixed_historical_parameters"
            ]["hard_at_10pct"]
            for name in CHALLENGES
        ),
    }


def summarize(rows):
    summary = {}

    for direction in DIRECTIONS:
        name = "_to_".join(direction)

        subset = [
            row for row in rows
            if (
                row["true_law_PRIVATE"],
                row["alternative_law"],
            ) == direction
        ]

        arms = {}

        for action in CHALLENGES:
            fixed = [
                row["challenges"][action][
                    "fixed_historical_parameters"
                ]["d"]
                for row in subset
            ]

            profiled = [
                row["challenges"][action][
                    "separately_profiled"
                ]["d"]
                for row in subset
            ]

            arms[action] = {
                "fixed_min_d": float(
                    min(fixed)
                ),
                "profiled_min_d": float(
                    min(profiled)
                ),
                "fixed_median_d": float(
                    np.median(fixed)
                ),
                "profiled_median_d": float(
                    np.median(profiled)
                ),
                "fixed_hard_count": sum(
                    d <= 2.563103
                    for d in fixed
                ),
                "profiled_hard_count": sum(
                    d <= 2.563103
                    for d in profiled
                ),
            }

        summary[name] = {
            "n_scenes": len(subset),
            "same_pair_hard_all_count": sum(
                row["same_parameters_hard_all"]
                for row in subset
            ),
            "actions": arms,
        }

        print("\n", name)
        print(
            " Same-pair hard for every action:",
            summary[name][
                "same_pair_hard_all_count"
            ],
            "/", len(subset)
        )

        for action, values in arms.items():
            print(
                f"  {action:13s}",
                "fixed min d=",
                round(values["fixed_min_d"], 4),
                "profiled min d=",
                round(values["profiled_min_d"], 4),
                "profiled hard=",
                values["profiled_hard_count"],
            )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true"
    )
    args = parser.parse_args()

    src, sim, commands = initialize()

    n = 1 if args.smoke else N_SCENES
    offset = 100000 if args.smoke else 0

    scenes = src["make_scenes"](
        sim, SEED + offset, n
    )

    print(
        "\nP4-M1B0 COMPOSITE PARAMETER SEARCH"
    )
    print("New scenes:", n)
    print("Legal physical parameter bounds only")
    print("Fixed parameters vs separately profiled")
    print("No E4. Private oracle audit only.")

    audits = {}
    summaries = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(
            scenes
        ):
            scene = scene_set[band]

            for true_law, alt_law in DIRECTIONS:
                rows.append(
                    run_one(
                        sim, commands,
                        scene, band, sid,
                        true_law, alt_law,
                    )
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

    report_path = OUT / (
        f"p4m1b0_{suffix}_report.json"
    )

    private_path = OUT / (
        f"p4m1b0_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P4-M1B0 legal composite parameter witness",
        "smoke": args.smoke,
        "n_base_scenes": n,
        "scene_seed": SEED + offset,
        "noise_sigma_m": SIGMA,
        "comparison_directions": DIRECTIONS,
        "history": HISTORY,
        "challenges": CHALLENGES,
        "summary": summaries,
        "E4_accessed": False,
        "online_uses_truth": False,
        "limitations": [
            "One-sided parameter optimization only",
            "Masses and initial velocity held equal",
            "Physical scene truth used offline",
            "Optimization may find local minima",
            "No global minimum certified",
            "Separate profiles do not define one"
            " world pair hard under all actions",
            "No learned action choice",
            "No statistical abstention policy",
            "No test-set threshold fitting",
            "Synthetic original Gaussian noise only",
        ],
        "source_sha256": {
            "68_p3a8c1_temporal_action_feasibility.py":
                sha(
                    HERE /
                    "68_p3a8c1_temporal_action_feasibility.py"
                ),
            "73_p4m1b0_composite_parameter_witness.py":
                sha(
                    HERE /
                    "73_p4m1b0_composite_parameter_witness.py"
                ),
        },
    }

    report_path.write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    private_path.write_text(
        json.dumps(
            audits, indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nP4-M1B0 COMPLETE")
    print("Saved:", report_path)
    print("Saved:", private_path)


if __name__ == "__main__":
    main()
