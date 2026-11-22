
"""
P4-M1A — Statistical Identifiability Limits

Scientific targets:
1. Find difficult lawful physical-world pairs.
2. Compute exact equal-covariance Gaussian
   simple-hypothesis Bayes error.
3. Compare Mid-repeat / High / Reverse-High.
4. Separate raw independent sensor noise
   from an AR(1) correlation stress test.

This is a PRIVATE offline oracle audit.
It is NOT a deployable action-selection policy.
It does NOT solve composite-parameter minimax
optimization or selective abstention.

No E4. No thresholds tuned on evaluation data.
"""

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p4m1a"

HISTORIC = (
    HERE / "results" / "p3a8c1b"
    / "p3a8c1b_fresh_report.json"
)

DEV_SEED = 20290111
EVAL_SEED = 20290112

N_DEV = 24
N_EVAL = 72

SIGMA = 0.005
RHO_VALUES = (0.0, 0.6)

# Precommitted simple-hypothesis error threshold.
HARD_ERROR = 0.10

LAWS = ("independent", "elastic", "deadzone")
PAIRS = tuple(combinations(LAWS, 2))

BANDS = ("original", "above_original_range")

HISTORY = ("e1", "e2", "weak", "strong", "mid")
CHALLENGES = ("mid", "high", "reverse_high")

NEEDED = tuple(dict.fromkeys(HISTORY + CHALLENGES))


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def initialize():
    hist = read_json(HISTORIC)

    if hist.get("smoke") is not False:
        raise RuntimeError(
            "Formal historical report required"
        )

    # Check the physics/action source files
    # previously frozen by P3-A8C1B.
    for name, expected in (
        hist["source_sha256"].items()
    ):
        path = HERE / name

        if digest(path) != expected:
            raise RuntimeError(
                "Source provenance mismatch: " + name
            )

    p68 = run_path(str(
        HERE /
        "68_p3a8c1_temporal_action_feasibility.py"
    ))

    core, src, sim, api, commands, budgets = (
        p68["initialize"]()
    )

    for name in NEEDED:
        if name not in commands:
            raise RuntimeError(
                "Missing command: " + name
            )

    for key in ("simulate", "make_model"):
        if key not in sim:
            raise RuntimeError(
                "Missing simulator interface: " + key
            )

    if (
        "SIGMA" in api
        and abs(float(api["SIGMA"]) - SIGMA) > 1e-12
    ):
        raise RuntimeError(
            "Noise sigma mismatch"
        )

    return src, sim, commands


def assert_original_physics(scene, band):
    ranges = {
        "mass_a": (0.8, 1.6),
        "mass_b": (0.8, 1.6),
        "velocity": (0.05, 0.25),
        "k": (5.0, 16.0),
        "c": (0.5, 2.2),
        "k3": (100.0, 850.0),
    }

    if band == "original":
        ranges["gap"] = (0.015, 0.12)

    for name, (lo, hi) in ranges.items():
        value = float(scene[name])

        if not lo - 1e-9 <= value <= hi + 1e-9:
            raise RuntimeError(
                f"Invalid physical parameter {name}: "
                f"{value} in band {band}"
            )


def collect_clean_worlds(
    sim, commands, scene
):
    """
    Same physical scene/nuisance parameters
    across different law families.

    Full raw observations:
      (passive/active, 51 times, A/B positions)

    All future law labels/physical parameters
    are PRIVATE for oracle auditing.
    """
    model = sim["make_model"](scene)

    trajectories = {}
    works = {}

    for law in LAWS:
        trajectories[law] = {}
        works[law] = {}

        for action in NEEDED:
            p, a, work, impulse = sim["simulate"](
                model,
                scene,
                law,
                commands[action],
            )

            p = np.asarray(p, dtype=float)
            a = np.asarray(a, dtype=float)

            if p.shape != (51, 2) or a.shape != (51, 2):
                raise RuntimeError(
                    "Expected 51x2 raw positions"
                )

            obs = np.stack((p, a), axis=0)

            if not np.isfinite(obs).all():
                raise RuntimeError(
                    "Nonfinite simulated trajectory"
                )

            trajectories[law][action] = obs
            works[law][action] = float(work)

    return trajectories, works


def mahalanobis_squared(diff, rho):
    """
    Gaussian covariance assumptions:

    Each of the four channels:
      passive-A, passive-B,
      active-A, active-B

    has AR(1) time covariance:
       sigma^2 * rho^|i-j|

    Channels and separate action trials
    are independent.

    rho=0 exactly matches the raw sensor
    noise model in the existing code.

    rho=0.6 is a hypothetical stress test,
    NOT an assertion about existing noise.
    """

    delta = np.asarray(diff, dtype=float)

    if delta.shape != (2, 51, 2):
        raise RuntimeError(
            "Invalid raw trajectory difference"
        )

    x = delta.transpose(0, 2, 1).reshape(4, 51)

    if not 0 <= rho < 1:
        raise RuntimeError("Invalid AR1 rho")

    # Closed-form AR(1) inverse precision.
    numerator = (
        np.sum(x[:, 0] ** 2)
        + np.sum(x[:, -1] ** 2)
        + (1.0 + rho ** 2)
        * np.sum(x[:, 1:-1] ** 2)
        - 2.0 * rho
        * np.sum(x[:, :-1] * x[:, 1:])
    )

    d2 = float(
        numerator
        / (SIGMA ** 2 * (1.0 - rho ** 2))
    )

    if d2 < -1e-7:
        raise RuntimeError(
            "Negative squared Mahalanobis distance"
        )

    return max(0.0, d2)


def error_from_d2(d2):
    # Exact equal-prior optimal error for
    # two fixed Gaussian means and equal covariance.
    return float(
        norm.sf(0.5 * np.sqrt(max(0.0, d2)))
    )


def record_pair(
    trajectories, works,
    law_a, law_b
):
    records = {}

    for rho in RHO_VALUES:
        key = f"rho_{rho:g}"

        d2_by_action = {}

        for action in NEEDED:
            delta = (
                trajectories[law_a][action]
                - trajectories[law_b][action]
            )

            d2_by_action[action] = (
                mahalanobis_squared(delta, rho)
            )

        # Five independent historical probes.
        # The Mid challenge below represents
        # another independent Mid measurement.
        d2_prior = sum(
            d2_by_action[action]
            for action in HISTORY
        )

        by_challenge = {}

        for action in CHALLENGES:
            only = d2_by_action[action]
            total = d2_prior + only

            by_challenge[action] = {
                "single_d": float(np.sqrt(only)),
                "single_error": error_from_d2(only),
                "prior_plus_action_d": float(
                    np.sqrt(total)
                ),
                "prior_plus_action_error":
                    error_from_d2(total),
                "additional_kl": 0.5 * only,
                "mean_challenge_work_J": float(
                    0.5 * (
                        works[law_a][action]
                        + works[law_b][action]
                    )
                ),
            }

        # Oracle action: offline diagnostic ONLY.
        best_action = min(
            CHALLENGES,
            key=lambda a: (
                by_challenge[a][
                    "prior_plus_action_error"
                ],
                a,
            )
        )

        best_error = by_challenge[
            best_action
        ]["prior_plus_action_error"]

        records[key] = {
            "prior_d": float(np.sqrt(d2_prior)),
            "prior_error": error_from_d2(d2_prior),
            "by_action": by_challenge,
            "oracle_best_action_PRIVATE": best_action,
            "oracle_best_error_PRIVATE": best_error,
            "all_three_hard": (
                best_error >= HARD_ERROR
            ),
        }

    return records


def run_group(
    src, sim, commands, scenes,
    phase
):
    output = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(scenes):
            scene = scene_set[band]

            assert_original_physics(scene, band)

            trajectories, works = (
                collect_clean_worlds(
                    sim, commands, scene
                )
            )

            for law_a, law_b in PAIRS:
                rows.append({
                    "scene_id": sid,
                    "pair": f"{law_a}_vs_{law_b}",
                    "parameter_scene_PRIVATE": scene,
                    "metrics": record_pair(
                        trajectories,
                        works,
                        law_a,
                        law_b,
                    ),
                })

        if len(rows) != len(scenes) * len(PAIRS):
            raise RuntimeError(
                "Incomplete pairwise matrix"
            )

        output[band] = rows

    return output


def summary_group(group):
    output = {}

    for band, rows in group.items():
        output[band] = {}

        for pair in (
            f"{a}_vs_{b}" for a, b in PAIRS
        ):
            selected = [
                r for r in rows
                if r["pair"] == pair
            ]

            pair_summary = {}

            for rho in RHO_VALUES:
                key = f"rho_{rho:g}"

                hardest = max(
                    selected,
                    key=lambda r: r["metrics"][key][
                        "oracle_best_error_PRIVATE"
                    ]
                )

                hard_count = sum(
                    r["metrics"][key]["all_three_hard"]
                    for r in selected
                )

                action_summary = {}

                for action in CHALLENGES:
                    vals = [
                        r["metrics"][key][
                            "by_action"
                        ][action][
                            "prior_plus_action_error"
                        ]
                        for r in selected
                    ]

                    single = [
                        r["metrics"][key][
                            "by_action"
                        ][action]["single_error"]
                        for r in selected
                    ]

                    action_summary[action] = {
                        "single_hard_count": sum(
                            x >= HARD_ERROR
                            for x in single
                        ),
                        "after_prior_hard_count": sum(
                            x >= HARD_ERROR
                            for x in vals
                        ),
                        "median_oracle_error":
                            float(np.median(vals)),
                        "max_oracle_error":
                            float(np.max(vals)),
                    }

                pair_summary[key] = {
                    "n_scenes": len(selected),
                    "all_actions_hard_count": hard_count,
                    "max_best_oracle_error":
                        hardest["metrics"][key][
                            "oracle_best_error_PRIVATE"
                        ],
                    "hardest_scene_id_PRIVATE":
                        hardest["scene_id"],
                    "best_action_counts_PRIVATE":
                        dict(Counter(
                            r["metrics"][key][
                                "oracle_best_action_PRIVATE"
                            ]
                            for r in selected
                        )),
                    "actions": action_summary,
                }

            output[band][pair] = pair_summary

    return output


def validate_gaussian_oracle():
    """
    Test the 1-D sufficient statistic induced
    by the exact two-Gaussian likelihood ratio.
    No trajectory or test labels are used here.
    """
    rng = np.random.default_rng(20290113)
    n = 10000

    for d in (0.0, 1.0, 2.0, 3.29):
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)

        empirical = 0.5 * (
            np.mean(z1 > d / 2.0)
            + np.mean(z2 < -d / 2.0)
        )

        theoretical = float(norm.sf(d / 2.0))

        if abs(empirical - theoretical) > 0.025:
            raise RuntimeError(
                "Gaussian oracle MC mismatch"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    src, sim, commands = initialize()
    validate_gaussian_oracle()

    if args.smoke:
        n_dev, n_eval = 2, 2
        offset = 100000
    else:
        n_dev, n_eval = N_DEV, N_EVAL
        offset = 0

    print("\nP4-M1A IDENTIFIABILITY LIMITS")
    print(
        "Development:", n_dev,
        "| Independent evaluation:", n_eval
    )
    print(
        "Noise: raw iid 5mm; AR1 rho=0.6 stress"
    )
    print(
        "All oracle comparisons: PRIVATE only"
    )
    print(
        "No E4. No threshold tuning."
    )

    dev = src["make_scenes"](
        sim, DEV_SEED + offset, n_dev
    )

    evaluation = src["make_scenes"](
        sim, EVAL_SEED + offset, n_eval
    )

    raw_dev = run_group(
        src, sim, commands, dev, "development"
    )

    raw_eval = run_group(
        src, sim, commands, evaluation, "evaluation"
    )

    report = {
        "development": summary_group(raw_dev),
        "evaluation": summary_group(raw_eval),
    }

    for band in BANDS:
        print("\nBAND:", band)

        for pair, item in (
            report["evaluation"][band].items()
        ):
            print("\n", pair)

            for rho in RHO_VALUES:
                key = f"rho_{rho:g}"
                data = item[key]

                print(
                    f"  rho={rho:g}",
                    "hard under every action=",
                    data["all_actions_hard_count"],
                    "/",
                    data["n_scenes"],
                    "max best-error=",
                    round(
                        data["max_best_oracle_error"],
                        5
                    ),
                )

                for action in CHALLENGES:
                    a = data["actions"][action]

                    print(
                        f"    {action:12s}",
                        "single hard=",
                        a["single_hard_count"],
                        "after-history hard=",
                        a["after_prior_hard_count"],
                    )

    OUT.mkdir(parents=True, exist_ok=True)

    suffix = (
        "smoke" if args.smoke else "fresh"
    )

    report_path = OUT / (
        f"p4m1a_{suffix}_report.json"
    )

    private_path = OUT / (
        f"p4m1a_{suffix}_audit_PRIVATE.json"
    )

    metadata = {
        "protocol":
            "P4-M1A statistical identifiability",
        "smoke": args.smoke,
        "n_dev": n_dev,
        "n_eval": n_eval,
        "development_seed": DEV_SEED + offset,
        "evaluation_seed": EVAL_SEED + offset,
        "noise_sigma_m": SIGMA,
        "rho_values": RHO_VALUES,
        "hard_error_threshold": HARD_ERROR,
        "laws": LAWS,
        "historical_actions": HISTORY,
        "candidate_actions": CHALLENGES,
        "results": report,
        "E4_accessed": False,
        "test_tuning": False,
        "online_action_policy": False,
        "limitations": [
            "Simple fixed-parameter Gaussian pairs",
            "Matched nuisance parameters only",
            "Finite random scene search",
            "Not a global composite minimax result",
            "AR1 correlation is hypothetical stress",
            "Prior probes assumed independent",
            "Mid challenge is an independent repeat",
            "Oracle parameters used offline only",
            "Only the declared action menu tested",
            "No structure-fitting pipeline evaluated",
            "No abstention policy evaluated",
            "Original and extended gaps kept separate",
        ],
        "source_sha256": {
            name: digest(HERE / name)
            for name in (
                "60_p3a7e1_action_family_pilot.py",
                "68_p3a8c1_temporal_action_feasibility.py",
                "72_p4m1a_identifiability_limits.py",
            )
        },
    }

    report_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8"
    )

    private_path.write_text(
        json.dumps(
            {
                "development": raw_dev,
                "evaluation": raw_eval,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8"
    )

    print("\nP4-M1A COMPLETE")
    print("Saved:", report_path)
    print("Saved:", private_path)


if __name__ == "__main__":
    main()
