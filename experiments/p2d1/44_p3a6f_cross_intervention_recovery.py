
"""
P3-A6F: Cross-Intervention Parameter Recovery

Compare:
  A. Frozen best E1/E2 Deadzone profile
  B. E1/E2 + Weak fit -> test on Strong
  C. E1/E2 + Strong fit -> test on Weak
  D. Oracle true parameters (evaluation only)

Also test Independent as a negative control.

No E4 access.
No online action policy.
No threshold tuning.
No reuse of E3 training residuals as test scores.
"""

from pathlib import Path
from runpy import run_path
import argparse
import json

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"

A4D = ROOT / "p3a4d" / "p3a4d_stress_audit_PRIVATE.json"
A6B = ROOT / "p3a6b" / "p3a6b_development_audit_PRIVATE.json"
A6D = ROOT / "p3a6d" / "p3a6d_development_audit_PRIVATE.json"

OUT = ROOT / "p3a6f"

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

FOCUS = ("independent", "deadzone")
WEAK = "pulse_1p5"
STRONG = "pulse_2p25"

ORDER = (
    "e1",
    "e2",
    WEAK,
    STRONG,
    "sweep_2p25",
    "pulse_2p75",
    "oscillation_2p25",
)

SEED = 20261128 + 222 + 100000
SIGMA = 0.005
NFEV = 80


def load(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def skip_noise(rng):
    for _ in range(2):
        rng.normal(0.0, SIGMA, (51, 2))


def predict_rms(probe, params, kind="deadzone"):
    predicted = np.asarray(
        probe.predict(kind, params),
        dtype=float,
    ).reshape(-1)

    target = np.asarray(
        probe.y_b, dtype=float
    ).reshape(-1)

    if (
        predicted.shape != (51,)
        or target.shape != (51,)
        or not np.all(np.isfinite(predicted))
    ):
        raise RuntimeError("Invalid predicted trajectory")

    return float(np.sqrt(
        np.mean((predicted - target) ** 2)
    ))


def median_mm(rows, key):
    return round(
        1000 * float(np.median([
            r[key] for r in rows
        ])),
        3,
    )


def paired_improvement(rows, baseline, updated):
    return sum(
        r[updated] < r[baseline]
        for r in rows
    )


def replay_band(band, scenes, archive, b6, d6,
                sim, api, commands):

    rng = np.random.default_rng(SEED)

    old_idx = {
        (int(r["scene_id"]),
         r["truth_PRIVATE_EVAL_ONLY"]): r
        for r in archive[band]["trajectories"]
    }

    b6_idx = {
        (int(r["scene_id"]), r["truth_PRIVATE"]): r
        for r in b6[band]
    }

    d6_idx = {
        (int(r["scene_id"]), r["mode_PRIVATE"]): r
        for r in d6[band]
    }

    rows = []

    for sid, scene in enumerate(scenes):
        model = sim["make_model"](scene)

        for mode in MODES:
            observed = {}

            for action in ORDER:
                if (
                    mode in FOCUS
                    and action in ("e1", "e2", WEAK, STRONG)
                ):
                    passive, active, _, _ = sim["simulate"](
                        model, scene, mode, commands[action]
                    )

                    p, a = api["noisy_observation"](
                        passive, active, rng
                    )

                    observed[action] = api[
                        "RevisionProbe"
                    ](p, a)
                else:
                    skip_noise(rng)

            if mode not in FOCUS:
                continue

            early = [
                observed["e1"],
                observed["e2"],
            ]

            # Verify replay of the original early data.
            known, _ = api["fit_library"](
                early,
                ("independent", "rigid", "elastic"),
                40,
            )

            edited, _ = api["fit_library"](
                early,
                ("deadzone", "cubic"),
                40,
            )

            expected = old_idx[(sid, mode)][
                "early_evidence"
            ]

            margin = known["bic"] - edited["bic"]

            if (
                known["kind"] != expected["known_kind"]
                or not np.isclose(
                    margin,
                    expected["bic_advantage"],
                    atol=1e-3,
                    rtol=0,
                )
                or not np.isclose(
                    known["fit_rms_m"],
                    expected["known_fit_rms_m"],
                    atol=1e-6,
                    rtol=0,
                )
            ):
                raise RuntimeError(
                    f"E1/E2 replay mismatch: {band}, "
                    f"scene={sid}, mode={mode}"
                )

            old = b6_idx[(sid, mode)]

            candidates = [
                c for c in old["profiles"]
                if c["rms_m"] <= old["frozen_cutoff_m"]
            ]

            if len(candidates) != old["retained_count"]:
                raise RuntimeError("Frozen set mismatch")

            if not candidates:
                raise RuntimeError("Empty frozen set")

            # Select baseline using E1/E2 ONLY.
            # No true gap is used for this choice.
            best = min(
                candidates,
                key=lambda c: c["rms_m"],
            )

            frozen_params = [
                best["k"], best["c"], best["gap_m"]
            ]

            # Fit with Weak; Strong is excluded.
            fit_weak, _ = api["fit_library"](
                early + [observed[WEAK]],
                ("deadzone",),
                NFEV,
            )

            # Fit with Strong; Weak is excluded.
            fit_strong, _ = api["fit_library"](
                early + [observed[STRONG]],
                ("deadzone",),
                NFEV,
            )

            # Negative-control reference.
            independent_fit, _ = api["fit_library"](
                early,
                ("independent",),
                40,
            )

            d6_row = d6_idx[(sid, mode)]

            for action in (WEAK, STRONG):
                check = float(
                    api["prediction_rms"](
                        observed[action],
                        independent_fit,
                    )
                )

                original = d6_row["actions"][action][
                    "independent_RMS_m"
                ]

                if not np.isclose(
                    check, original,
                    atol=1e-8,
                    rtol=0,
                ):
                    raise RuntimeError(
                        f"A6D observation replay mismatch: "
                        f"{band}, scene={sid}, mode={mode}, "
                        f"action={action}"
                    )

            strong_probe = observed[STRONG]
            weak_probe = observed[WEAK]

            record = {
                "scene_id": sid,
                "truth_PRIVATE": mode,

                "frozen_best_gap_m":
                    float(best["gap_m"]),

                "updated_weak_params":
                    list(map(float, fit_weak["params"])),

                "updated_strong_params":
                    list(map(float, fit_strong["params"])),

                "weak_fit_success":
                    bool(fit_weak["optimizer_success"]),

                "strong_fit_success":
                    bool(fit_strong["optimizer_success"]),

                # Both baseline scores are held out
                # from the E1/E2-only frozen fit.
                "frozen_to_strong_m":
                    predict_rms(
                        strong_probe, frozen_params
                    ),

                "frozen_to_weak_m":
                    predict_rms(
                        weak_probe, frozen_params
                    ),

                # Critical cross-intervention tests.
                "weak_updated_to_strong_m":
                    predict_rms(
                        strong_probe, fit_weak["params"]
                    ),

                "strong_updated_to_weak_m":
                    predict_rms(
                        weak_probe, fit_strong["params"]
                    ),

                "independent_to_strong_m":
                    float(api["prediction_rms"](
                        strong_probe, independent_fit
                    )),

                "independent_to_weak_m":
                    float(api["prediction_rms"](
                        weak_probe, independent_fit
                    )),
            }

            # Oracle is attached AFTER fitting and
            # heldout predictions are complete.
            if mode == "deadzone":
                true_params = [
                    scene["k"] / scene["mass_b"],
                    scene["c"] / scene["mass_b"],
                    scene["gap"],
                ]

                record.update({
                    "oracle_to_strong_PRIVATE_m":
                        predict_rms(
                            strong_probe, true_params
                        ),

                    "oracle_to_weak_PRIVATE_m":
                        predict_rms(
                            weak_probe, true_params
                        ),

                    "true_params_PRIVATE":
                        list(map(float, true_params)),
                })

            rows.append(record)

    if len(rows) != 2 * len(scenes):
        raise RuntimeError("Unexpected episode count")

    return rows


def print_summary(band, rows):
    print("\n" + "=" * 72)
    print("P3-A6F BAND:", band)
    print("=" * 72)

    for mode in FOCUS:
        sub = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        print("\nMODE:", mode, "N:", len(sub))

        print(
            "  Frozen E1/E2 -> Strong:",
            median_mm(sub, "frozen_to_strong_m"),
            "mm"
        )

        print(
            "  E1/E2+Weak -> Strong:",
            median_mm(sub, "weak_updated_to_strong_m"),
            "mm"
        )

        print(
            "  Improved Strong predictions:",
            paired_improvement(
                sub,
                "frozen_to_strong_m",
                "weak_updated_to_strong_m",
            ),
            "/", len(sub),
        )

        print(
            "  Frozen E1/E2 -> Weak:",
            median_mm(sub, "frozen_to_weak_m"),
            "mm"
        )

        print(
            "  E1/E2+Strong -> Weak:",
            median_mm(sub, "strong_updated_to_weak_m"),
            "mm"
        )

        print(
            "  Improved Weak predictions:",
            paired_improvement(
                sub,
                "frozen_to_weak_m",
                "strong_updated_to_weak_m",
            ),
            "/", len(sub),
        )

        print(
            "  Converged update fits:",
            sum(
                r["weak_fit_success"]
                and r["strong_fit_success"]
                for r in sub
            ),
            "/", len(sub),
        )

        if mode == "independent":
            false_preference = sum(
                r["weak_updated_to_strong_m"]
                < r["independent_to_strong_m"]
                for r in sub
            )
            print(
                "  Deadzone beats Independent on "
                "heldout Strong:",
                false_preference, "/", len(sub),
            )
        else:
            print(
                "  Oracle -> Strong:",
                median_mm(
                    sub, "oracle_to_strong_PRIVATE_m"
                ),
                "mm",
            )

            print(
                "  Oracle -> Weak:",
                median_mm(
                    sub, "oracle_to_weak_PRIVATE_m"
                ),
                "mm",
            )

            print(
                "  Updated Weak->Strong within 10 mm:",
                sum(
                    r["weak_updated_to_strong_m"]
                    <= 0.010
                    for r in sub
                ),
                "/", len(sub),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    archive = load(A4D)
    b6 = load(A6B)
    d6 = load(A6D)

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    helper = run_path(str(
        HERE /
        "35_p3a5b_preaction_model_disagreement.py"
    ))

    commands = helper["command_bank"](
        sim["make_profiles"]()
    )

    n = 2 if args.smoke else 24

    all_rows = {}

    for band in BANDS:
        scenes = archive[band][
            "scene_parameters_PRIVATE"
        ][:n]

        rows = replay_band(
            band,
            scenes,
            archive,
            b6,
            d6,
            sim,
            api,
            commands,
        )

        print_summary(band, rows)
        all_rows[band] = rows

    OUT.mkdir(parents=True, exist_ok=True)

    name = (
        "p3a6f_smoke_audit_PRIVATE.json"
        if args.smoke
        else "p3a6f_development_audit_PRIVATE.json"
    )

    output = OUT / name

    output.write_text(
        json.dumps(
            {
                "protocol":
                    "P3-A6F cross-intervention recovery",
                "training":
                    "E1/E2 plus one E3 action",
                "testing":
                    "the alternative E3 action",
                "conditional_on_observed_A": True,
                "E4_accessed": False,
                "oracle_evaluation_only": True,
                "limitations": [
                    "Previously explored development scenes",
                    "Only Independent and Deadzone",
                    "True parameters are evaluation-only",
                    "E3 actions are independently reset",
                    "This is not autonomous A-B forecasting",
                    "The 10mm diagnostic is not a decision threshold",
                    "No online adaptive policy is validated",
                ],
                "bands": all_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A6F CROSS-INTERVENTION "
        "RECOVERY AUDIT PASS"
    )
    print("No E4 access, no policy tuning.")
    print("Saved:", output)


if __name__ == "__main__":
    main()
