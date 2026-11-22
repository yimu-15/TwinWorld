
"""
P3-A6G: Fresh-seed Cross-Intervention Structural Validation

New scenes, same simulation generator.

Training:
  E1/E2 + Weak -> evaluate Strong
  E1/E2 + Strong -> evaluate Weak

Compare:
  Independent vs Deadzone on heldout action.

No E4 access.
No threshold fitting.
No candidate-set refitting from P3-A6B.
No action selection policy.

Smoke scenes 0-1 are EXCLUDED from full results.
"""

from pathlib import Path
from runpy import run_path
import argparse
import hashlib
import json

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a6g"

SCENE_SEED = 20261213
NOISE_SEED = 2026121301

N_SMOKE = 2
N_FULL = 36

NFEV = 80

MODES = ("independent", "deadzone")

BANDS = (
    "original",
    "high_in_range",
    "above_original_range",
)

ACTIONS = (
    "e1",
    "e2",
    "weak",
    "strong",
)


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def median_mm(rows, key):
    return round(
        1000.0 * float(np.median([
            r[key] for r in rows
        ])),
        4,
    )


def describe_margin(values):
    values = np.asarray(values, dtype=float)

    return {
        "min_mm": float(np.min(values)),
        "median_mm": float(np.median(values)),
        "max_mm": float(np.max(values)),
        "positive_count": int(np.sum(values > 0)),
        "over_1mm_count": int(np.sum(values > 1.0)),
    }


def make_scenes(sim):
    rng = np.random.default_rng(SCENE_SEED)

    scenes = []

    for _ in range(N_SMOKE + N_FULL):
        base = sim["sample_scene"](rng)

        original = dict(base)

        high = dict(base)
        high["gap"] = float(
            rng.uniform(0.09, 0.12)
        )

        wide = dict(base)
        wide["gap"] = float(
            rng.uniform(0.12, 0.18)
        )

        scenes.append({
            "original": original,
            "high_in_range": high,
            "above_original_range": wide,
        })

    return scenes


def get_commands(sim):
    profiles = sim["make_profiles"]()

    return {
        "e1": profiles["pulse"] * 0.25,
        "e2": profiles["pulse"] * 0.50,
        "weak": profiles["pulse"] * 1.50,
        "strong": profiles["pulse"] * 2.25,
    }


def observation(sim, api, model, scene,
                mode, command, sid,
                mode_index, action_index):
    """
    Independent noise stream per scene/mode/action.

    Same observation noise is reused across
    gap bands for paired comparisons.
    """
    passive, active, work, _ = sim[
        "simulate"
    ](
        model,
        scene,
        mode,
        command,
    )

    rng = np.random.default_rng(
        NOISE_SEED
        + sid * 1000
        + mode_index * 100
        + action_index
    )

    noisy_p, noisy_a = api[
        "noisy_observation"
    ](
        passive, active, rng
    )

    probe = api["RevisionProbe"](
        noisy_p, noisy_a
    )

    return probe, float(work)


def fit_law(api, probes, law, nfev):
    fit, _ = api["fit_library"](
        probes, (law,), nfev
    )

    return fit


def rms(api, probe, fit):
    return float(
        api["prediction_rms"](probe, fit)
    )


def run_episode(sim, api, scene, sid,
                mode, mode_index, commands):
    """
    Only observed probes enter physical inference.

    Truth scene parameters are used solely for
    creating the simulated world.
    """
    model = sim["make_model"](scene)

    observations = {}
    works = {}

    # Generate early and Weak observations.
    for index, name in enumerate(
        ("e1", "e2", "weak")
    ):
        probe, work = observation(
            sim, api, model, scene,
            mode, commands[name],
            sid, mode_index, index,
        )
        observations[name] = probe
        works[name] = work

    early = [
        observations["e1"],
        observations["e2"],
    ]

    null = fit_law(
        api, early, "independent", 40
    )

    early_deadzone = fit_law(
        api, early, "deadzone", 40
    )

    # Weak-augmented model is frozen BEFORE
    # Strong observations are used.
    weak_updated = fit_law(
        api,
        early + [observations["weak"]],
        "deadzone",
        NFEV,
    )

    strong_probe, strong_work = observation(
        sim, api, model, scene,
        mode, commands["strong"],
        sid, mode_index, 3,
    )

    observations["strong"] = strong_probe
    works["strong"] = strong_work

    # First independent heldout evaluation.
    null_strong = rms(
        api, strong_probe, null
    )

    deadzone_strong = rms(
        api, strong_probe, weak_updated
    )

    early_strong = rms(
        api, strong_probe, early_deadzone
    )

    # Reverse-direction fitting.
    # The Weak observation is NOT part of this fit.
    strong_updated = fit_law(
        api,
        early + [observations["strong"]],
        "deadzone",
        NFEV,
    )

    weak_probe = observations["weak"]

    null_weak = rms(
        api, weak_probe, null
    )

    deadzone_weak = rms(
        api, weak_probe, strong_updated
    )

    early_weak = rms(
        api, weak_probe, early_deadzone
    )

    weak_params = np.asarray(
        weak_updated["params"], dtype=float
    )

    strong_params = np.asarray(
        strong_updated["params"], dtype=float
    )

    if not (
        np.isfinite(weak_params).all()
        and np.isfinite(strong_params).all()
    ):
        raise RuntimeError(
            "Nonfinite model parameters"
        )

    return {
        "scene_id": sid,

        "truth_PRIVATE": mode,

        "early_to_strong_m":
            early_strong,

        "weak_to_strong_m":
            deadzone_strong,

        "null_to_strong_m":
            null_strong,

        "strong_margin_m":
            null_strong - deadzone_strong,

        "early_to_weak_m":
            early_weak,

        "strong_to_weak_m":
            deadzone_weak,

        "null_to_weak_m":
            null_weak,

        "weak_margin_m":
            null_weak - deadzone_weak,

        "weak_update_params":
            weak_params.tolist(),

        "strong_update_params":
            strong_params.tolist(),

        "weak_optimizer_success":
            bool(
                weak_updated["optimizer_success"]
            ),

        "strong_optimizer_success":
            bool(
                strong_updated["optimizer_success"]
            ),

        "weak_work_proxy_J":
            works["weak"],

        "strong_work_proxy_J":
            works["strong"],
    }


def summary(rows):
    result = {}

    for mode in MODES:
        sub = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        strong_margin = [
            1000 * r["strong_margin_m"]
            for r in sub
        ]

        weak_margin = [
            1000 * r["weak_margin_m"]
            for r in sub
        ]

        result[mode] = {
            "n": len(sub),

            "early_to_strong_mm":
                median_mm(
                    sub, "early_to_strong_m"
                ),

            "weak_to_strong_mm":
                median_mm(
                    sub, "weak_to_strong_m"
                ),

            "null_to_strong_mm":
                median_mm(
                    sub, "null_to_strong_m"
                ),

            "early_to_weak_mm":
                median_mm(
                    sub, "early_to_weak_m"
                ),

            "strong_to_weak_mm":
                median_mm(
                    sub, "strong_to_weak_m"
                ),

            "strong_test_margin":
                describe_margin(strong_margin),

            "weak_test_margin":
                describe_margin(weak_margin),

            "weak_improves_strong_count":
                sum(
                    r["weak_to_strong_m"]
                    < r["early_to_strong_m"]
                    for r in sub
                ),

            "strong_improves_weak_count":
                sum(
                    r["strong_to_weak_m"]
                    < r["early_to_weak_m"]
                    for r in sub
                ),

            "converged_both":
                sum(
                    r["weak_optimizer_success"]
                    and r["strong_optimizer_success"]
                    for r in sub
                ),
        }

        # Model-degeneracy diagnostic.
        # These are descriptive boundaries,
        # not learned structural thresholds.
        if mode == "independent":
            result[mode][
                "weak_near_zero_coupling_count"
            ] = sum(
                (
                    r["weak_update_params"][0]
                    <= 0.011
                    and
                    r["weak_update_params"][1]
                    <= 0.011
                )
                or
                (
                    r["weak_update_params"][2]
                    >= 0.18
                )
                for r in sub
            )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke", action="store_true"
    )

    args = parser.parse_args()

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    scenes = make_scenes(sim)
    commands = get_commands(sim)

    # Smoke and formal evaluation DO NOT overlap.
    ids = (
        range(0, N_SMOKE)
        if args.smoke
        else range(
            N_SMOKE,
            N_SMOKE + N_FULL
        )
    )

    all_rows = {}
    all_summary = {}

    print(
        "P3-A6G FRESH STRUCTURAL VALIDATION"
    )

    print(
        "Scene seed:", SCENE_SEED,
        "Noise seed:", NOISE_SEED,
        "Smoke:", args.smoke,
    )

    for band in BANDS:
        rows = []

        for sid in ids:
            scene = scenes[sid][band]

            for mi, mode in enumerate(MODES):
                rows.append(
                    run_episode(
                        sim, api, scene, sid,
                        mode, mi, commands
                    )
                )

        stats = summary(rows)

        all_rows[band] = rows
        all_summary[band] = stats

        print("\n" + "=" * 70)
        print("BAND:", band)
        print("BASE SCENES:", len(ids))
        print("=" * 70)

        for mode in MODES:
            s = stats[mode]

            print("\nMODE:", mode)
            print(
                "E1/E2 -> Strong:",
                s["early_to_strong_mm"],
                "mm",
            )
            print(
                "E1/E2+Weak -> Strong:",
                s["weak_to_strong_mm"],
                "mm",
            )
            print(
                "Independent -> Strong:",
                s["null_to_strong_mm"],
                "mm",
            )

            print(
                "Strong heldout margin:",
                s["strong_test_margin"],
            )

            print(
                "Weak heldout margin:",
                s["weak_test_margin"],
            )

            print(
                "Weak improves Strong:",
                f"{s['weak_improves_strong_count']}"
                f"/{s['n']}",
            )

            print(
                "Strong improves Weak:",
                f"{s['strong_improves_weak_count']}"
                f"/{s['n']}",
            )

            if mode == "independent":
                print(
                    "Near-null Deadzone fits:",
                    s[
                        "weak_near_zero_coupling_count"
                    ],
                    "/", s["n"],
                )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke
        else "prospective"
    )

    report_path = (
        OUT / f"p3a6g_{suffix}_report.json"
    )

    audit_path = (
        OUT
        / f"p3a6g_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P3-A6G fresh structural validation",
        "scene_seed": SCENE_SEED,
        "noise_seed": NOISE_SEED,
        "smoke_excluded_from_full": True,
        "base_scene_ids": list(ids),
        "prediction":
            "B conditional on observed A",
        "E4_accessed": False,
        "thresholds_tuned": False,
        "new_action_policy": False,
        "source_sha256": {
            name: sha256(HERE / name)
            for name in (
                "16_p2d1b_probe_diversity.py",
                "22_p3a1b_structural_revision.py",
            )
        },
        "summary": all_summary,
        "limitations": [
            "New seed but same simulation generator",
            "Only Independent and Deadzone evaluated",
            "Strong is heldout from Weak-based fitting",
            "Reverse Weak test is heldout from Strong fitting",
            "Both A trajectories are observed at test time",
            "No online action-selection policy",
            "No E4 results are used",
            "Prediction margins are descriptive, not calibrated",
        ],
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
            all_rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A6G FRESH STRUCTURAL "
        "VALIDATION PASS"
    )
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
