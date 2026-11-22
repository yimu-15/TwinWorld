
"""
P3-A8B1: Observable A-Response Prediction Audit.

Compare:
  original  : old ideal acceleration integration
  affine    : linear response-vs-command gain
  quadratic : quadratic response-vs-command gain
  loo       : historical leave-one-out model choice

History: E1/E2/Weak/Strong/Mid only.
Future: Reverse/High (not seen by planner).

All A-response predictions are frozen BEFORE
generating future challenge observations.

No E4.
No truth input to the predictor.
No future A/B used to choose the predictor.
No test-set threshold tuning.

Only tests A forecasting, NOT policy accuracy.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a8b1"

SEED = 20271116
NOISE_SEED = 2027111601
N_TEST = 36

BANDS = ("original", "above_original_range")

MODES = (
    "independent", "rigid", "elastic",
    "deadzone", "nonlinear", "saturating",
)

HISTORY = (
    ("e1", 0.25),
    ("e2", 0.50),
    ("weak", 0.75),
    ("strong", 2.25),
    ("mid", 1.75),
)

CHALLENGES = (
    ("reverse", -1.75),
    ("high", 2.75),
)

METHODS = (
    "original", "affine", "quadratic", "loo",
)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def observed_a(probe):
    result = np.asarray(
        probe.a_half[::2], dtype=float
    ).ravel()

    if result.shape != (51,):
        raise RuntimeError(
            "Unexpected A observation shape"
        )

    if not np.isfinite(result).all():
        raise RuntimeError(
            "Nonfinite A observations"
        )

    return result


def rms_mm(pred, actual):
    a = np.asarray(pred, dtype=float)
    b = np.asarray(actual, dtype=float)

    if a.shape != b.shape:
        raise RuntimeError("A shape mismatch")

    return float(
        np.sqrt(np.mean((a - b) ** 2))
        * 1000.0
    )


def fit_response(
    gains, displacement, degree
):
    x = np.vander(
        np.asarray(gains, dtype=float),
        N=degree + 1,
        increasing=True,
    )

    y = np.asarray(
        displacement, dtype=float
    )

    if x.shape[0] < x.shape[1]:
        raise RuntimeError(
            "Insufficient historical actions"
        )

    coeff, _, _, _ = np.linalg.lstsq(
        x, y, rcond=None
    )

    return coeff


def forecast_gain(
    coeff, gain
):
    features = np.array([
        gain ** degree
        for degree in range(coeff.shape[0])
    ])

    return features @ coeff


def choose_degree_from_history(
    gains, displacement
):
    """
    Leave-one-history-out prediction error.

    This function sees only previously
    executed and observed actions.
    """
    n = len(gains)
    errors = {}

    for degree in (1, 2):
        squared_errors = []

        for holdout in range(n):
            keep = [
                i for i in range(n)
                if i != holdout
            ]

            coeff = fit_response(
                gains[keep],
                displacement[keep],
                degree,
            )

            predicted = forecast_gain(
                coeff,
                gains[holdout],
            )

            observed = displacement[holdout]

            squared_errors.append(
                float(np.mean(
                    (predicted - observed) ** 2
                ))
            )

        errors[degree] = float(
            np.mean(squared_errors)
        )

    # Tie conservatively favors affine.
    selected = min(
        (1, 2),
        key=lambda d: (errors[d], d)
    )

    return selected, errors


def prepare_forecasters(
    base, sim, commands,
    history_probes
):
    """
    No future observations or world truth
    are passed into this function.
    """
    weak = history_probes["weak"]

    gains = np.array([
        gain for _, gain in HISTORY
    ], dtype=float)

    # All episodes begin from the same
    # passive-state condition.
    initial_positions = np.array([
        float(history_probes[name].a0)
        for name, _ in HISTORY
    ])

    anchor = float(
        np.median(initial_positions)
    )

    # Fit displacements, not absolute
    # positions, to reduce anchoring noise.
    displacement = np.stack([
        observed_a(history_probes[name])
        - float(history_probes[name].a0)
        for name, _ in HISTORY
    ])

    affine = fit_response(
        gains, displacement, degree=1
    )

    quadratic = fit_response(
        gains, displacement, degree=2
    )

    degree, loo_scores = (
        choose_degree_from_history(
            gains, displacement
        )
    )

    selected_coeff = (
        affine if degree == 1
        else quadratic
    )

    forecasts = {}

    for action, gain in CHALLENGES:
        old_proxy = base[
            "construct_future_a_proxy"
        ](
            weak,
            commands[action],
            sim,
        )

        forecasts[action] = {
            "original": observed_a(old_proxy),

            "affine": (
                anchor + forecast_gain(
                    affine, gain
                )
            ),

            "quadratic": (
                anchor + forecast_gain(
                    quadratic, gain
                )
            ),

            "loo": (
                anchor + forecast_gain(
                    selected_coeff, gain
                )
            ),
        }

        for method, values in (
            forecasts[action].items()
        ):
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"Nonfinite forecast: {method}"
                )

    diagnostics = {
        "selected_degree": degree,
        "historical_loo_mse": {
            str(key): value
            for key, value
            in loo_scores.items()
        },
        "weak_initial_spline_velocity":
            float(weak.va_half[0]),
        "historical_a0_spread_mm":
            float(
                1000 * np.ptp(
                    initial_positions
                )
            ),
    }

    return forecasts, diagnostics


def run_case(
    base, core, src, sim, api,
    commands, scene, sid, mode, band,
    noise_seed
):
    model = sim["make_model"](scene)

    seedbase = (
        noise_seed
        + sid * 1000
        + src["MODES"].index(mode) * 20
        + BANDS.index(band) * 100000
    )

    prior = {}

    # 1. Only historical observations.
    for index, (name, _) in enumerate(
        HISTORY
    ):
        prior[name], _ = core["probe"](
            src, sim, api,
            model, scene, mode,
            commands[name],
            seedbase + index,
        )

    # 2. Freeze all future forecasts.
    predictions, diagnostics = (
        prepare_forecasters(
            base, sim, commands, prior
        )
    )

    # 3. Generate future actions only
    # AFTER all predictions are frozen.
    results = {}

    for index, (action, _) in enumerate(
        CHALLENGES, start=5
    ):
        future, _ = core["probe"](
            src, sim, api,
            model, scene, mode,
            commands[action],
            seedbase + index,
        )

        # This is the processed OBSERVED A.
        # It is not an oracle noiseless state.
        actual = observed_a(future)

        results[action] = {
            method: rms_mm(
                predictions[action][method],
                actual,
            )
            for method in METHODS
        }

    return {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "A_forecast_RMS_mm": results,
        "history_diagnostic": diagnostics,
    }


def metrics(values):
    arr = np.asarray(
        values, dtype=float
    )

    return {
        "n": len(arr),
        "median_mm": float(
            np.median(arr)
        ),
        "mean_mm": float(
            np.mean(arr)
        ),
        "p90_mm": float(
            np.quantile(arr, 0.9)
        ),
        "max_mm": float(
            np.max(arr)
        ),
    }


def summarize(rows):
    output = {}

    for action, _ in CHALLENGES:
        output[action] = {}

        print("\nACTION:", action)

        for method in METHODS:
            values = [
                row["A_forecast_RMS_mm"][
                    action
                ][method]
                for row in rows
            ]

            result = metrics(values)
            output[action][method] = result

            print(
                f"  {method:12s}",
                f"median={result['median_mm']:.4f} mm",
                f"p90={result['p90_mm']:.4f} mm",
                f"max={result['max_mm']:.4f} mm",
            )

        baseline = output[
            action
        ]["original"]["median_mm"]

        affine = output[
            action
        ]["affine"]["median_mm"]

        print(
            "  Median affine reduction:",
            round(
                100 * (
                    1 - affine / max(
                        baseline, 1e-12
                    )
                ),
                3,
            ),
            "%",
        )

    output["per_mode_PRIVATE"] = {}

    for mode in MODES:
        subset = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        output[
            "per_mode_PRIVATE"
        ][mode] = {
            action: {
                method: metrics([
                    r["A_forecast_RMS_mm"][
                        action
                    ][method]
                    for r in subset
                ])
                for method in METHODS
            }
            for action, _ in CHALLENGES
        }

    selected = Counter(
        r["history_diagnostic"][
            "selected_degree"
        ]
        for r in rows
    )

    output["loo_degree_counts"] = {
        str(k): v
        for k, v in selected.items()
    }

    print(
        "Historical LOO selections:",
        output["loo_degree_counts"]
    )

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke", action="store_true"
    )

    args = parser.parse_args()

    base = run_path(str(
        HERE /
        "64_p3a8b0_preaction_disagreement_pilot.py"
    ))

    core, src, sim, api, commands = (
        base["initialize"]()
    )

    offset = (
        100000 if args.smoke else 0
    )

    n = 2 if args.smoke else N_TEST

    scenes = src["make_scenes"](
        sim, SEED + offset, n
    )

    print(
        "\nP3-A8B1 OBSERVABLE "
        "A-RESPONSE AUDIT"
    )
    print(
        "Scenes:", n,
        "per band and mechanism"
    )

    print(
        "No E4. Forecasts frozen "
        "before challenge observations."
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
                        base, core,
                        src, sim, api,
                        commands,
                        scene_set[band],
                        sid, mode, band,
                        NOISE_SEED + offset,
                    )
                )

        expected = n * len(MODES)

        if len(rows) != expected:
            raise RuntimeError(
                "Incomplete scene matrix"
            )

        print("\n" + "=" * 66)
        print("BAND:", band)
        print("=" * 66)

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
        f"p3a8b1_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a8b1_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P3-A8B1 observable A-response audit",
        "smoke": args.smoke,
        "scene_seed": SEED + offset,
        "noise_seed": NOISE_SEED + offset,
        "n_base_scenes": n,
        "history_gain_values": {
            name: gain
            for name, gain in HISTORY
        },
        "challenge_gain_values": {
            name: gain
            for name, gain in CHALLENGES
        },
        "summary": summaries,
        "planning_uses_future_A": False,
        "planning_uses_future_B": False,
        "forecast_uses_true_parameters": False,
        "forecast_uses_true_mechanism": False,
        "challenge_used_in_model_selection":
            False,
        "E4_accessed": False,
        "thresholds_tuned": False,
        "limitations": [
            "All commands share one pulse shape",
            "Reverse is negative-gain extrapolation",
            "High is extrapolation beyond 2.25 gain",
            "Actual A is processed noisy observation",
            "No active model-selection policy tested",
            "No guarantee for different pulse shapes",
            "No independent real-robot verification",
            "Both gap bands share base scenes",
            "MuJoCo only; no RGB or E4",
        ],
        "source_sha256": {
            name: sha(HERE / name)
            for name in (
                "16_p2d1b_probe_diversity.py",
                "20_p3a0_parameter_refit.py",
                "22_p3a1b_structural_revision.py",
                "60_p3a7e1_action_family_pilot.py",
                "64_p3a8b0_preaction_disagreement_pilot.py",
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

    print(
        "\nP3-A8B1 FORECAST AUDIT COMPLETE"
    )
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
