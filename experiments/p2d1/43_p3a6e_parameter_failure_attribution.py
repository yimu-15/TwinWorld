
"""
P3-A6E: Parameter / Forward-Model Failure Attribution.

Diagnose P3-A6D candidate collapse using:
1. E1/E2 frozen fitted candidate parameters.
2. Oracle physical parameters on noisy E3.
3. Oracle physical parameters on clean E3.
4. Fixed-gap E3-only parameter refit.

No thresholds are changed.
No historical experiment files are modified.
No E4 information is accessed.
True parameters are used for private audit ONLY.
E3-only refit is IN-SAMPLE, not policy evidence.
"""

from pathlib import Path
from runpy import run_path
import argparse
import json

import numpy as np
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"

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

ORDER = (
    "e1",
    "e2",
    "pulse_1p5",
    "pulse_2p25",
    "sweep_2p25",
    "pulse_2p75",
    "oscillation_2p25",
)

ACTIONS = (
    "pulse_1p5",
    "pulse_2p25",
)

SIGMA = 0.005
SEED = 20261128 + 222 + 100000


def read(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def skip_noise(rng):
    rng.normal(0, SIGMA, (51, 2))
    rng.normal(0, SIGMA, (51, 2))


def deadzone_rms(probe, params):
    predicted = np.asarray(
        probe.predict("deadzone", params),
        dtype=float,
    )

    observed = probe.y_b

    return float(np.sqrt(
        np.mean(
            (predicted - observed) ** 2
        )
    ))


def refit_E3_only(
    probe,
    gap,
    initial,
    max_nfev,
):
    """
    Upper-bound diagnostic for model flexibility.

    Uses observed E3 B for parameter optimization.

    This is NOT an independent E3 predictive test.
    """
    lower = np.array([2.0, 0.0])
    upper = np.array([100.0, 10.0])

    def residual(theta):
        params = [
            float(theta[0]),
            float(theta[1]),
            float(gap),
        ]

        return (
            probe.predict(
                "deadzone", params
            ) - probe.y_b
        )

    starts = (
        np.asarray(initial[:2], dtype=float),
        np.array([8.0, 1.0]),
    )

    best = float("inf")

    for start in starts:
        x0 = np.clip(
            start,
            lower + 1e-7,
            upper - 1e-7,
        )

        fit = least_squares(
            residual,
            x0=x0,
            bounds=(lower, upper),
            max_nfev=max_nfev,
        )

        error = residual(fit.x)

        fit_rms = float(np.sqrt(
            np.mean(error ** 2)
        ))

        best = min(best, fit_rms)

    return best


def replay_band(
    band,
    scenes,
    saved,
    previous,
    sim,
    api,
    commands,
    nfev,
):
    frozen_index = {
        (
            int(r["scene_id"]),
            r["truth_PRIVATE"],
        ): r
        for r in saved[band]
    }

    previous_index = {
        (
            int(r["scene_id"]),
            r["mode_PRIVATE"],
        ): r
        for r in previous[band]
    }

    rng = np.random.default_rng(SEED)

    records = []

    for sid, scene in enumerate(scenes):
        model = sim["make_model"](scene)

        for mode in MODES:
            measured = {}
            clean = {}

            for action in ORDER:
                if (
                    mode in FOCUS
                    and action in (
                        "e1",
                        "e2",
                        *ACTIONS,
                    )
                ):
                    passive, active, _, _ = (
                        sim["simulate"](
                            model,
                            scene,
                            mode,
                            commands[action],
                        )
                    )

                    p, a = api[
                        "noisy_observation"
                    ](
                        passive,
                        active,
                        rng,
                    )

                    measured[action] = api[
                        "RevisionProbe"
                    ](p, a)

                    if (
                        action in ACTIONS
                        and mode == "deadzone"
                    ):
                        clean[action] = api[
                            "RevisionProbe"
                        ](
                            passive,
                            active,
                        )
                else:
                    skip_noise(rng)

            if mode not in FOCUS:
                continue

            early_probes = [
                measured["e1"],
                measured["e2"],
            ]

            # E1/E2-only Independent reference.
            null, _ = api["fit_library"](
                early_probes,
                ("independent",),
                40,
            )

            frozen = frozen_index[
                (sid, mode)
            ]

            old = previous_index[
                (sid, mode)
            ]

            profiles = [
                p for p in frozen["profiles"]
                if (
                    p["rms_m"]
                    <= frozen["frozen_cutoff_m"]
                )
            ]

            assert len(profiles) == (
                frozen["retained_count"]
            )

            for action in ACTIONS:
                probe = measured[action]
                prior = old["actions"][action]

                cutoff = float(
                    prior["threshold_m"]
                )

                # Exact replay of P3-A6D.
                null_error = float(
                    api["prediction_rms"](
                        probe, null
                    )
                )

                if not np.isclose(
                    null_error,
                    prior["independent_RMS_m"],
                    atol=1e-8,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        "Noise replay mismatch: "
                        f"{band}, {sid}, "
                        f"{mode}, {action}"
                    )

                frozen_errors = np.asarray([
                    deadzone_rms(
                        probe,
                        [
                            p["k"],
                            p["c"],
                            p["gap_m"],
                        ],
                    )
                    for p in profiles
                ])

                retained = int(np.sum(
                    frozen_errors <= cutoff
                ))

                if retained != prior["post_n"]:
                    raise RuntimeError(
                        "Candidate replay mismatch: "
                        f"{band}, {sid}, "
                        f"{mode}, {action}"
                    )

                row = {
                    "scene_id": sid,
                    "mode": mode,
                    "action": action,
                    "cutoff_m": cutoff,
                    "null_RMS_m": null_error,
                    "frozen_best_RMS_m":
                        float(np.min(
                            frozen_errors
                        )),
                    "frozen_post_n": retained,
                }

                if mode == "deadzone":
                    true_gap = float(
                        scene["gap"]
                    )

                    mass_b = float(
                        scene["mass_b"]
                    )

                    # RevisionProbe expects
                    # effective acceleration
                    # parameters alpha=k/mB,
                    # beta=c/mB.
                    true_parameters = [
                        float(
                            scene["k"] / mass_b
                        ),
                        float(
                            scene["c"] / mass_b
                        ),
                        true_gap,
                    ]

                    nearest_gap = float(
                        frozen[
                            "true_gap_nearest_grid_m_PRIVATE"
                        ]
                    )

                    nearest = min(
                        profiles,
                        key=lambda p:
                            abs(
                                p["gap_m"]
                                - nearest_gap
                            ),
                    )

                    fitted_params = [
                        nearest["k"],
                        nearest["c"],
                        nearest["gap_m"],
                    ]

                    row.update({
                        "true_gap_PRIVATE":
                            true_gap,

                        "oracle_true_noisy_RMS_m":
                            deadzone_rms(
                                probe,
                                true_parameters,
                            ),

                        "oracle_true_clean_RMS_m":
                            deadzone_rms(
                                clean[action],
                                true_parameters,
                            ),

                        "frozen_nearest_RMS_m":
                            deadzone_rms(
                                probe,
                                fitted_params,
                            ),

                        "E3_only_refit_nearest_RMS_m":
                            refit_E3_only(
                                probe,
                                nearest_gap,
                                fitted_params,
                                nfev,
                            ),

                        "nearest_alpha":
                            float(
                                nearest["k"]
                            ),

                        "nearest_beta":
                            float(
                                nearest["c"]
                            ),

                        "true_alpha_PRIVATE":
                            true_parameters[0],

                        "true_beta_PRIVATE":
                            true_parameters[1],
                    })

                records.append(row)

    return records


def print_summary(band, rows):
    print("\n" + "=" * 72)
    print("BAND:", band)
    print("=" * 72)

    for mode in FOCUS:
        for action in ACTIONS:
            selected = [
                r for r in rows
                if (
                    r["mode"] == mode
                    and r["action"] == action
                )
            ]

            def median_mm(key):
                return round(
                    1000 * float(np.median([
                        r[key] for r in selected
                    ])),
                    3,
                )

            print(
                "\nMODE:", mode,
                "ACTION:", action,
                "N:", len(selected),
            )

            print(
                "  Null RMS median:",
                median_mm("null_RMS_m"),
                "mm",
            )

            print(
                "  Frozen best RMS median:",
                median_mm(
                    "frozen_best_RMS_m"
                ),
                "mm",
            )

            if mode != "deadzone":
                continue

            diagnostics = (
                "oracle_true_clean_RMS_m",
                "oracle_true_noisy_RMS_m",
                "frozen_nearest_RMS_m",
                "E3_only_refit_nearest_RMS_m",
            )

            for key in diagnostics:
                within = sum(
                    r[key] <= r["cutoff_m"]
                    for r in selected
                )

                print(
                    " ", key,
                    "median_mm=",
                    median_mm(key),
                    "within_null_Q95=",
                    f"{within}/{len(selected)}",
                )

            print(
                "  Frozen all-candidate empty:",
                sum(
                    r["frozen_post_n"] == 0
                    for r in selected
                ),
                "/",
                len(selected),
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    saved = read(
        ROOT / "p3a6b"
        / "p3a6b_development_audit_PRIVATE.json"
    )

    old = read(
        ROOT / "p3a4d"
        / "p3a4d_stress_audit_PRIVATE.json"
    )

    previous = read(
        ROOT / "p3a6d"
        / "p3a6d_development_audit_PRIVATE.json"
    )

    sim = run_path(str(
        HERE /
        "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE /
        "22_p3a1b_structural_revision.py"
    ))

    helper35 = run_path(str(
        HERE /
        "35_p3a5b_preaction_model_disagreement.py"
    ))

    commands = helper35[
        "command_bank"
    ](
        sim["make_profiles"]()
    )

    n = 2 if args.smoke else 24

    for band in BANDS:
        scenes = old[band][
            "scene_parameters_PRIVATE"
        ][:n]

        rows = replay_band(
            band,
            scenes,
            saved,
            previous,
            sim,
            api,
            commands,
            25 if args.smoke else 50,
        )

        print_summary(band, rows)

    print(
        "\nP3-A6E PARAMETER/FORWARD-MODEL "
        "ATTRIBUTION PASS"
    )

    print(
        "No files written. "
        "Oracle parameters are evaluation-only."
    )

    print(
        "E3-only refitting is in-sample "
        "and must not be reported as "
        "heldout validation."
    )


if __name__ == "__main__":
    main()
