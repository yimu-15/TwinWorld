
"""
P3-A5B: Nonprivileged model-disagreement feasibility audit.

Uses:
  - E1/E2 noisy observations
  - fitted candidate physical models
  - planned action commands

Does NOT use future observed A, E3 outcomes, truth
parameters or E4 outcomes to predict disagreement.

Oracle comparisons are retrospective only.

No learned action policy.
No threshold tuning.
No historical files modified.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from runpy import run_path
from collections import Counter

import numpy as np

from scipy.signal import savgol_filter


HERE = Path(__file__).resolve().parent

AUDIT_PATH = (
    HERE / "results" / "p3a4d"
    / "p3a4d_stress_audit_PRIVATE.json"
)

REPORT_PATH = (
    HERE / "results" / "p3a4d"
    / "p3a4d_stress_report.json"
)

STRESS_SEED = 20261128

SIGMA = 0.005
NFEV = 40

BANDS = (
    "original",
    "high_in_range",
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

FOCUS = (
    "independent",
    "deadzone",
)

ACTIONS = (
    "pulse_1p5",
    "pulse_2p25",
    "sweep_2p25",
)

ALL_COMMANDS = (
    "e1",
    "e2",
    *ACTIONS,
    "pulse_2p75",
    "oscillation_2p25",
)


def command_bank(profiles):
    return {
        "e1": profiles["pulse"] * 0.25,
        "e2": profiles["pulse"] * 0.50,

        "pulse_1p5":
            profiles["pulse"] * 1.50,

        "pulse_2p25":
            profiles["pulse"] * 2.25,

        "sweep_2p25":
            profiles["sweep"] * 2.25,

        "pulse_2p75":
            profiles["pulse"] * 2.75,

        "oscillation_2p25":
            profiles["oscillation"] * 2.25,
    }


def forced_displacement(command, sim):
    """
    Double integration of public acceleration command.

    This is a planned kinematic displacement,
    not an observed future A trajectory.
    """
    dx = 0.0
    dv = 0.0

    sampled = [0.0]

    for step, acc in enumerate(command):

        dv += float(acc) * sim["DT"]
        dx += dv * sim["DT"]

        if (
            (step + 1) % sim["SAMPLE_EVERY"]
            == 0
        ):
            sampled.append(dx)

    assert len(sampled) == 51

    return np.asarray(sampled)


def forecast_A(early_pairs, commands, sim):
    """
    Estimate:
        A(t) = A0 + v0*t + gain*command_displacement

    Fit gain ONLY from E1/E2 noisy observations.

    Limitations:
        The scalar gain cannot fully describe
        nonlinear, coupled A dynamics.
    """
    t = np.arange(51) * 0.02

    intercepts = []
    velocities = []

    for passive, active in early_pairs:

        velocity, intercept = np.polyfit(
            t - 1.0,
            passive[:, 0],
            deg=1,
        )

        velocities.append(float(velocity))
        intercepts.append(float(intercept))

    a0 = float(np.mean(intercepts))
    v0 = float(np.mean(velocities))

    numerator = 0.0
    denominator = 0.0

    for (
        passive, active
    ), name in zip(
        early_pairs,
        ("e1", "e2"),
    ):

        smooth = savgol_filter(
            active[:, 0],
            window_length=11,
            polyorder=3,
        )

        smooth = smooth - smooth[0] + a0

        forced = forced_displacement(
            commands[name],
            sim,
        )

        observed_displacement = (
            smooth - (a0 + v0 * t)
        )

        numerator += float(
            np.dot(
                forced,
                observed_displacement,
            )
        )

        denominator += float(
            np.dot(forced, forced)
        )

    gain = float(np.clip(
        numerator / max(denominator, 1e-12),
        0.0,
        2.0,
    ))

    forecasts = {
        name: (
            a0
            + v0 * t
            + gain * forced_displacement(
                commands[name],
                sim,
            )
        )
        for name in ACTIONS
    }

    return forecasts, gain


def model_predictions(
    early_pairs,
    fitted,
    future_A,
    api,
):
    """
    Compare the early best known model against
    the early fitted Deadzone candidate.

    Only forecasted A is supplied to the probe.
    """
    passive = np.mean(
        [
            pair[0]
            for pair in early_pairs
        ],
        axis=0,
    )

    placeholder_B = np.full(
        51,
        passive[-1, 1],
    )

    output = {}

    for action, A in future_A.items():

        synthetic_probe = api[
            "RevisionProbe"
        ](
            passive,
            np.column_stack([
                A,
                placeholder_B,
            ]),
        )

        known_prediction = (
            synthetic_probe.predict(
                fitted["known"]["kind"],
                fitted["known"]["params"],
            )
        )

        deadzone_prediction = (
            synthetic_probe.predict(
                "deadzone",
                fitted["deadzone"]["params"],
            )
        )

        output[action] = float(np.sqrt(
            np.mean(
                (
                    deadzone_prediction
                    - known_prediction
                ) ** 2
            )
        ))

    return output


def recover_early(
    scene,
    model,
    sim,
    api,
    commands,
    noise_rng,
    expected,
    scene_id,
):
    """
    Reconstruct the P3-A4D early observation stream.

    Skip future observations by advancing the noise
    generator without simulating or reading them.

    Verify E1/E2 BIC and RMS match old audit data.
    """
    results = []

    for mode in MODES:

        early_pairs = []

        for name in ALL_COMMANDS:

            if (
                mode in FOCUS
                and name in ("e1", "e2")
            ):

                passive, active, _, _ = (
                    sim["simulate"](
                        model,
                        scene,
                        mode,
                        commands[name],
                    )
                )

                noisy_p, noisy_a = api[
                    "noisy_observation"
                ](
                    passive,
                    active,
                    noise_rng,
                )

                early_pairs.append(
                    (noisy_p, noisy_a)
                )

            else:

                # Match two Gaussian arrays consumed
                # by each unused original observation.
                noise_rng.normal(
                    0,
                    SIGMA,
                    (51, 2),
                )

                noise_rng.normal(
                    0,
                    SIGMA,
                    (51, 2),
                )

        if mode not in FOCUS:
            continue

        probes = [
            api["RevisionProbe"](p, a)
            for p, a in early_pairs
        ]

        known, _ = api["fit_library"](
            probes,
            (
                "independent",
                "rigid",
                "elastic",
            ),
            NFEV,
        )

        edit, all_edits = api[
            "fit_library"
        ](
            probes,
            ("deadzone", "cubic"),
            NFEV,
        )

        margin = float(
            known["bic"] - edit["bic"]
        )

        old_evidence = expected[
            (scene_id, mode)
        ]["early_evidence"]

        assert (
            known["kind"]
            == old_evidence["known_kind"]
        ), (
            scene_id,
            mode,
            "kind mismatch",
        )

        assert abs(
            margin
            - old_evidence["bic_advantage"]
        ) < 1e-3, (
            scene_id,
            mode,
            "BIC replay mismatch",
        )

        assert abs(
            known["fit_rms_m"]
            - old_evidence["known_fit_rms_m"]
        ) < 1e-6, (
            scene_id,
            mode,
            "RMS replay mismatch",
        )

        fitted = {
            "known": known,
            "deadzone": all_edits["deadzone"],
        }

        future_A, gain = forecast_A(
            early_pairs,
            commands,
            sim,
        )

        disagreements = model_predictions(
            early_pairs,
            fitted,
            future_A,
            api,
        )

        results.append({
            "scene_id": scene_id,
            "mode": mode,
            "predicted": disagreements,
            "A_forecast": future_A,
            "A_gain": gain,
            "fitted_deadzone_gap":
                fitted["deadzone"]["params"][2],
            "old_stopped": (
                expected[(scene_id, mode)]
                ["policies"]
                ["old_stop_or_strong"]
                ["action"] == "none"
            ),
        })

    return results


def retrospective_oracle(
    scene,
    model,
    sim,
    commands,
    records,
):
    """
    PRIVILEGED EVALUATION ONLY.

    All early-model forecasts must already
    exist before this function is called.
    """
    oracle = {}
    true_A = {}

    for action in ACTIONS:

        _, independent, _, _ = (
            sim["simulate"](
                model,
                scene,
                "independent",
                commands[action],
            )
        )

        _, deadzone, _, _ = (
            sim["simulate"](
                model,
                scene,
                "deadzone",
                commands[action],
            )
        )

        oracle[action] = float(np.sqrt(
            np.mean(
                (
                    deadzone[:, 1]
                    - independent[:, 1]
                ) ** 2
            )
        ))

        true_A[action] = {
            "independent":
                independent[:, 0],
            "deadzone":
                deadzone[:, 0],
        }

    for record in records:

        record["oracle_PRIVATE"] = dict(
            oracle
        )

        record["A_forecast_RMS_PRIVATE"] = {
            action: float(np.sqrt(
                np.mean(
                    (
                        record["A_forecast"][action]
                        - true_A[action][
                            record["mode"]
                        ]
                    ) ** 2
                )
            ))
            for action in ACTIONS
        }


def med(values):
    if not values:
        return None

    return round(
        float(np.median(values)),
        4,
    )


def summarize(band, rows):
    print("\n" + "=" * 70)
    print("BAND:", band)
    print("EPISODES:", len(rows))
    print("=" * 70)

    subsets = (
        (
            "Independent",
            [
                r for r in rows
                if r["mode"] == "independent"
            ],
        ),
        (
            "Deadzone",
            [
                r for r in rows
                if r["mode"] == "deadzone"
            ],
        ),
        (
            "Deadzone old-stop",
            [
                r for r in rows
                if (
                    r["mode"] == "deadzone"
                    and r["old_stopped"]
                )
            ],
        ),
    )

    for label, subset in subsets:

        if not subset:
            continue

        print(
            "\nGROUP:", label,
            "N =", len(subset),
        )

        print(
            "Early fitted gap median (m):",
            med([
                r["fitted_deadzone_gap"]
                for r in subset
            ]),
        )

        for action in ACTIONS:

            predicted_mm = [
                1000 * r["predicted"][action]
                for r in subset
            ]

            oracle_mm = [
                1000 * r[
                    "oracle_PRIVATE"
                ][action]
                for r in subset
            ]

            a_error_mm = [
                1000 * r[
                    "A_forecast_RMS_PRIVATE"
                ][action]
                for r in subset
            ]

            print(
                f"  {action:14s}",
                "predicted_B_mm=",
                med(predicted_mm),
                "oracle_B_mm=",
                med(oracle_mm),
                "A_forecast_RMS_mm=",
                med(a_error_mm),
            )

        predicted_winners = Counter(
            max(
                ACTIONS,
                key=lambda a:
                    r["predicted"][a],
            )
            for r in subset
        )

        oracle_winners = Counter(
            max(
                ACTIONS,
                key=lambda a:
                    r["oracle_PRIVATE"][a],
            )
            for r in subset
        )

        agreements = sum(
            (
                max(
                    ACTIONS,
                    key=lambda a:
                        r["predicted"][a],
                )
                ==
                max(
                    ACTIONS,
                    key=lambda a:
                        r["oracle_PRIVATE"][a],
                )
            )
            for r in subset
        )

        print(
            "Predicted best actions:",
            dict(predicted_winners),
        )

        print(
            "Oracle best actions:",
            dict(oracle_winners),
        )

        print(
            "Top-1 agreement:",
            f"{agreements}/{len(subset)}",
        )

        print(
            "A forecast gain median:",
            med([
                r["A_gain"]
                for r in subset
            ]),
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    audit = json.loads(
        AUDIT_PATH.read_text(
            encoding="utf-8"
        )
    )

    report = json.loads(
        REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    sim = run_path(str(
        HERE /
        "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE /
        "22_p3a1b_structural_revision.py"
    ))

    commands = command_bank(
        sim["make_profiles"]()
    )

    n = (
        2 if args.smoke
        else int(report["scenes_per_band"])
    )

    for band in BANDS:

        scenes = audit[band][
            "scene_parameters_PRIVATE"
        ][:n]

        expected = {
            (
                row["scene_id"],
                row["truth_PRIVATE_EVAL_ONLY"],
            ): row
            for row in audit[band]["trajectories"]
        }

        # Match the original P3-A4D noise seed.
        noise_rng = np.random.default_rng(
            STRESS_SEED + 222 + 100000
        )

        records = []

        for sid, scene in enumerate(scenes):

            model = sim["make_model"](
                scene
            )

            # Predict E3 disagreement from
            # E1/E2 observations alone.
            scene_records = recover_early(
                scene,
                model,
                sim,
                api,
                commands,
                noise_rng,
                expected,
                sid,
            )

            # Oracle accessed only AFTER
            # pre-action predictions are fixed.
            retrospective_oracle(
                scene,
                model,
                sim,
                commands,
                scene_records,
            )

            records.extend(
                scene_records
            )

        summarize(
            band,
            records,
        )

    print(
        "\nP3-A5B NONPRIVILEGED "
        "EARLY-MODEL AUDIT PASS"
    )

    print(
        "Model scores use E1/E2 observations "
        "and planned commands only."
    )

    print(
        "Oracle and actual future A are "
        "used only for retrospective evaluation."
    )

    print(
        "No new policy training, "
        "no threshold tuning, "
        "no historical result modification."
    )


if __name__ == "__main__":
    main()
