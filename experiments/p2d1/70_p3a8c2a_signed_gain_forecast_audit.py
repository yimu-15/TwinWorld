
"""
P3-A8C2A — Signed-Gain A Forecast Audit

Pre-action forecast for:
    High         (+2.75)
    Reverse-High (-2.75)

Methods:
    old_ideal_integrator
    historical_affine

History:
    E1, E2, Weak, Strong, Mid
All are observed before future action prediction.

The fitted affine predictor sees NO future
A trajectory, B trajectory, truth label or
true physics parameters.

No E4.
No new structural thresholds.
No adaptive policy claimed.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a8c2a"

HISTORIC = (
    HERE / "results" / "p3a8c1b"
    / "p3a8c1b_fresh_report.json"
)

SEED = 20280411
NOISE_SEED = 2028041101

N_SCENES = 36

BANDS = (
    "original",
    "above_original_range",
)

MODES = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear",
    "saturating",
)

HISTORY = (
    ("e1", 0.25),
    ("e2", 0.50),
    ("weak", 0.75),
    ("strong", 2.25),
    ("mid", 1.75),
)

ACTIONS = (
    ("high", 2.75, 5),
    ("reverse_high", -2.75, 6),
)


def read(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def get_A(probe):
    y = np.asarray(
        probe.a_half[::2],
        dtype=float
    ).ravel()

    if y.shape != (51,):
        raise RuntimeError(
            "Expected 51 A positions"
        )

    if not np.isfinite(y).all():
        raise RuntimeError(
            "Nonfinite A observation"
        )

    return y


def rms_mm(a, b):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)

    if x.shape != y.shape:
        raise RuntimeError(
            "Forecast shape mismatch"
        )

    return float(
        1000 * np.sqrt(
            np.mean((x - y) ** 2)
        )
    )


def summarize_values(values):
    arr = np.asarray(values, dtype=float)

    return {
        "n": len(arr),
        "median_mm": float(np.median(arr)),
        "mean_mm": float(np.mean(arr)),
        "p90_mm": float(
            np.quantile(arr, 0.9)
        ),
        "max_mm": float(np.max(arr)),
    }


def initialize():
    history = read(HISTORIC)

    if history["smoke"]:
        raise RuntimeError(
            "Formal historical source required"
        )

    for filename, expected in (
        history["source_sha256"].items()
    ):
        if sha(HERE / filename) != expected:
            raise RuntimeError(
                "Historical physics source changed: "
                + filename
            )

    p68 = run_path(str(
        HERE /
        "68_p3a8c1_temporal_action_feasibility.py"
    ))

    p64 = run_path(str(
        HERE /
        "64_p3a8b0_preaction_disagreement_pilot.py"
    ))

    p65 = run_path(str(
        HERE /
        "65_p3a8b1_observable_A_response_audit.py"
    ))

    (
        core, src, sim,
        api, commands, budgets
    ) = p68["initialize"]()

    for name in ("high", "reverse_high"):
        actual = budgets[name]["command_sha256"]

        expected = history.get(
            "source_action_hashes", {}
        ).get(name)

        # The P3-A8C1B report does not store
        # command hashes separately; p68 source
        # hash is verified above.
        if expected is not None and actual != expected:
            raise RuntimeError(
                "Action hash mismatch"
            )

    return (
        core, src, sim, api,
        commands, p64, p65, budgets
    )


def forecast_before_action(
    prior, commands, sim, p64, p65
):
    gains = np.asarray([
        gain for _, gain in HISTORY
    ], dtype=float)

    initial_A = np.asarray([
        float(prior[name].a0)
        for name, _ in HISTORY
    ])

    anchor = float(
        np.median(initial_A)
    )

    displacements = np.stack([
        get_A(prior[name])
        - float(prior[name].a0)
        for name, _ in HISTORY
    ])

    # Exactly the affine response fitting
    # design used in P3-A8B1.
    coeff = p65["fit_response"](
        gains,
        displacements,
        degree=1,
    )

    forecasts = {}

    for action, gain, _ in ACTIONS:
        affine = (
            anchor
            + p65["forecast_gain"](
                coeff, gain
            )
        )

        old_probe = p64[
            "construct_future_a_proxy"
        ](
            prior["weak"],
            commands[action],
            sim,
        )

        old = get_A(old_probe)

        if not np.isfinite(affine).all():
            raise RuntimeError(
                "Nonfinite affine A forecast"
            )

        forecasts[action] = {
            "old": old,
            "affine": affine,
        }

    return forecasts


def run_case(
    core, src, sim, api,
    commands, p64, p65,
    scene, sid, mode, band, noise_seed
):
    model = sim["make_model"](scene)

    seedbase = (
        noise_seed
        + sid * 1000
        + src["MODES"].index(mode) * 20
        + BANDS.index(band) * 100000
    )

    prior = {}

    for index, (name, _) in enumerate(
        HISTORY
    ):
        prior[name], _ = core["probe"](
            src, sim, api,
            model, scene, mode,
            commands[name],
            seedbase + index,
        )

    # All forecasts are complete BEFORE
    # either challenge is simulated.
    predicted = forecast_before_action(
        prior, commands, sim, p64, p65
    )

    errors = {}

    for action, gain, index in ACTIONS:
        future, _ = core["probe"](
            src, sim, api,
            model, scene, mode,
            commands[action],
            seedbase + index,
        )

        observed = get_A(future)

        errors[action] = {
            method: rms_mm(
                values, observed
            )
            for method, values
            in predicted[action].items()
        }

    return {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "errors_mm": errors,
    }


def summarize(rows):
    summary = {}

    for action, _, _ in ACTIONS:
        old = [
            r["errors_mm"][action]["old"]
            for r in rows
        ]

        affine = [
            r["errors_mm"][action]["affine"]
            for r in rows
        ]

        old_result = summarize_values(old)
        affine_result = summarize_values(affine)

        improved = sum(
            a < b for a, b
            in zip(affine, old)
        )

        reduction = (
            1.0
            - affine_result["median_mm"]
            / max(
                old_result["median_mm"],
                1e-12
            )
        )

        summary[action] = {
            "old": old_result,
            "affine": affine_result,
            "paired_affine_better": improved,
            "median_reduction": reduction,
        }

        print("\nACTION:", action)

        print(
            " Old:",
            round(
                old_result["median_mm"], 4
            ),
            "mm median;",
            round(
                old_result["p90_mm"], 4
            ),
            "mm p90"
        )

        print(
            " Affine:",
            round(
                affine_result["median_mm"], 4
            ),
            "mm median;",
            round(
                affine_result["p90_mm"], 4
            ),
            "mm p90"
        )

        print(
            " Paired improvement:",
            improved, "/", len(rows)
        )

        print(
            " Median reduction:",
            round(100 * reduction, 3),
            "%"
        )

    per_mode = {}

    for mode in MODES:
        subset = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        per_mode[mode] = {
            action: summarize_values([
                r["errors_mm"][action]["affine"]
                for r in subset
            ])
            for action, _, _ in ACTIONS
        }

    summary["per_mode_PRIVATE"] = per_mode

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args = parser.parse_args()

    (
        core, src, sim, api,
        commands, p64, p65, budgets
    ) = initialize()

    offset = (
        100000 if args.smoke else 0
    )

    n = 2 if args.smoke else N_SCENES

    scenes = src["make_scenes"](
        sim,
        SEED + offset,
        n,
    )

    print(
        "\nP3-A8C2A SIGNED-GAIN "
        "FORECAST AUDIT"
    )

    print("Independent base scenes:", n)

    print(
        "Future A/B unavailable to predictor."
    )

    summaries = {}
    audits = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(
            scenes
        ):
            for mode in MODES:
                rows.append(
                    run_case(
                        core, src, sim, api,
                        commands, p64, p65,
                        scene_set[band],
                        sid, mode, band,
                        NOISE_SEED + offset,
                    )
                )

        if len(rows) != n * len(MODES):
            raise RuntimeError(
                "Incomplete experiment"
            )

        print("\nBAND:", band)

        summaries[band] = summarize(rows)
        audits[band] = rows

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report_path = OUT / (
        f"p3a8c2a_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a8c2a_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P3-A8C2A signed-gain forecast audit",
        "smoke": args.smoke,
        "n_base_scenes": n,
        "scene_seed": SEED + offset,
        "noise_seed": NOISE_SEED + offset,
        "actions": {
            name: gain
            for name, gain, _ in ACTIONS
        },
        "summary": summaries,
        "no_future_A_used": True,
        "no_future_B_used": True,
        "no_true_parameters_used_for_forecast":
            True,
        "no_structural_threshold_tuning": True,
        "E4_accessed": False,
        "limitations": [
            "Only one fixed pulse family",
            "Reverse-High is negative extrapolation",
            "A observations contain measurement noise",
            "No B forecast validation",
            "No active action choice",
            "Only synthetic MuJoCo scenes",
            "Gap bands share physical base scenes",
            "Actuator clipping not audited",
        ],
        "source_sha256": {
            name: sha(HERE / name)
            for name in (
                "64_p3a8b0_preaction_disagreement_pilot.py",
                "65_p3a8b1_observable_A_response_audit.py",
                "68_p3a8c1_temporal_action_feasibility.py",
                "70_p3a8c2a_signed_gain_forecast_audit.py",
            )
        },
    }

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

    print("\nP3-A8C2A AUDIT COMPLETE")
    print("No E4. No future observation leakage.")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
