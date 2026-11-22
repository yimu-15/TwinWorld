
"""
P4-M2A — Generative / Inference Fidelity Audit

Separate:
  1. clean physical-model discrepancy
  2. noisy-observation discrepancy
  3. clean parameter-fitting discrepancy
  4. noisy parameter-fitting discrepancy

Also inspect whether alternative candidate
fits to Independent require physically
inadmissible effective parameters.

Development only. Private oracle audit.
No E4. No future action planning.
No test threshold fitting.
"""

import argparse
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p4m2a"

PRIOR_REPORT = (
    HERE / "results" / "p4m1b0"
    / "p4m1b0_fresh_report.json"
)

SEED = 20290317
NOISE_SEED = 2029031701
N_DEV = 12

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
)

TRAIN = ("e1", "e2", "weak", "strong")
VALID = "mid"
SIGMA = 0.005


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def rms_mm(a, b):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)

    return float(
        1000 * np.sqrt(
            np.mean((x - y) ** 2)
        )
    )


def quant(values):
    x = np.asarray(values, dtype=float)

    return {
        "n": len(x),
        "median_mm": float(np.median(x)),
        "p90_mm": float(np.quantile(x, 0.9)),
        "max_mm": float(np.max(x)),
    }


def truth_fit(mode, scene):
    """
    Private oracle parameters.
    These must never enter online fitting
    or experiment selection.
    """
    mb = float(scene["mass_b"])

    if mode in ("independent", "rigid"):
        return {
            "kind": mode,
            "params": None,
        }

    alpha = float(scene["k"]) / mb
    beta = float(scene["c"]) / mb

    if mode == "elastic":
        return {
            "kind": "elastic",
            "params": [alpha, beta],
        }

    if mode == "deadzone":
        return {
            "kind": "deadzone",
            "params": [
                alpha, beta,
                float(scene["gap"]),
            ],
        }

    return {
        "kind": "cubic",
        "params": [
            alpha, beta,
            float(scene["k3"]) / mb,
        ],
    }


def physical_box_check(kind, pars, band):
    """
    Necessary physical outer-box constraints.
    Not a sufficient condition for physical
    feasibility: alpha/beta/gamma share mass_B.
    """
    p = np.asarray(pars, dtype=float)

    alpha_lo = 5.0 / 1.6
    alpha_hi = 16.0 / 0.8
    beta_lo = 0.5 / 1.6
    beta_hi = 2.2 / 0.8

    reasons = []

    if not alpha_lo <= p[0] <= alpha_hi:
        reasons.append("alpha")

    if not beta_lo <= p[1] <= beta_hi:
        reasons.append("beta")

    if kind == "deadzone":
        lo, hi = (
            (0.015, 0.12)
            if band == "original"
            else (0.12, 0.18)
        )

        if not lo <= p[2] <= hi:
            reasons.append("gap")

    if kind == "cubic":
        if not 100 / 1.6 <= p[2] <= 850 / 0.8:
            reasons.append("gamma")

    return reasons


def initialize():
    hist = json.loads(
        PRIOR_REPORT.read_text(
            encoding="utf-8"
        )
    )

    if hist["smoke"]:
        raise RuntimeError(
            "Expected formal P4-M1B0 report"
        )

    name = (
        "68_p3a8c1_temporal_action_feasibility.py"
    )

    expected = hist["source_sha256"][name]

    if sha(HERE / name) != expected:
        raise RuntimeError(
            "Frozen physics/action source changed"
        )

    p68 = run_path(str(HERE / name))

    core, src, sim, api, commands, _ = (
        p68["initialize"]()
    )

    for name in TRAIN + (VALID,):
        if name not in commands:
            raise RuntimeError(
                "Missing command: " + name
            )

    for name in ("RevisionProbe", "fit_one"):
        if name not in api:
            raise RuntimeError(
                "Missing inference API: " + name
            )

    return src, sim, api, commands


def run_case(
    sim, api, commands,
    scene, sid, band, mode, noise_seed
):
    model = sim["make_model"](scene)

    rng = np.random.default_rng(
        noise_seed
        + sid * 101
        + MODES.index(mode) * 7907
        + BANDS.index(band) * 1000003
    )

    clean = {}
    noisy = {}

    for action in TRAIN + (VALID,):
        passive, active, _, _ = (
            sim["simulate"](
                model,
                scene,
                mode,
                commands[action],
            )
        )

        passive = np.asarray(
            passive, dtype=float
        )

        active = np.asarray(
            active, dtype=float
        )

        if passive.shape != (51, 2):
            raise RuntimeError(
                "Invalid passive shape"
            )

        if active.shape != (51, 2):
            raise RuntimeError(
                "Invalid active shape"
            )

        clean[action] = api["RevisionProbe"](
            passive, active
        )

        noisy[action] = api["RevisionProbe"](
            passive + rng.normal(
                0, SIGMA, passive.shape
            ),
            active + rng.normal(
                0, SIGMA, active.shape
            ),
        )

    truth = truth_fit(mode, scene)
    law = truth["kind"]

    # 1. Oracle dynamics with clean A/B.
    predicted_clean = clean[VALID].predict(
        law, truth["params"]
    )

    # 2. Oracle dynamics with noisy A.
    predicted_noisy = noisy[VALID].predict(
        law, truth["params"]
    )

    # 3. Fit only historical clean observations.
    fitted_clean = api["fit_one"](
        [clean[x] for x in TRAIN],
        law,
        40,
    )

    # 4. Fit only historical noisy observations.
    fitted_noisy = api["fit_one"](
        [noisy[x] for x in TRAIN],
        law,
        40,
    )

    predicted_clean_fit = (
        clean[VALID].predict(
            law, fitted_clean["params"]
        )
    )

    predicted_noisy_fit = (
        noisy[VALID].predict(
            law, fitted_noisy["params"]
        )
    )

    result = {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,

        "true_effective_params_PRIVATE":
            truth["params"],

        "oracle_clean_mm": rms_mm(
            predicted_clean,
            clean[VALID].y_b,
        ),

        "oracle_noisy_A_to_clean_B_mm": rms_mm(
            predicted_noisy,
            clean[VALID].y_b,
        ),

        "oracle_noisy_to_noisy_mm": rms_mm(
            predicted_noisy,
            noisy[VALID].y_b,
        ),

        "clean_fit_to_clean_mm": rms_mm(
            predicted_clean_fit,
            clean[VALID].y_b,
        ),

        "noisy_fit_to_noisy_mm": rms_mm(
            predicted_noisy_fit,
            noisy[VALID].y_b,
        ),

        "truth_fit_noisy_optimizer_success":
            bool(
                fitted_noisy["optimizer_success"]
            ),

        "truth_fit_noisy_params_PRIVATE":
            fitted_noisy["params"],
    }

    # Diagnose the wide candidate library.
    # This does not modify its boundaries.
    if mode == "independent":
        aliases = {}

        for alt in (
            "elastic", "deadzone", "cubic"
        ):
            fit = api["fit_one"](
                [noisy[x] for x in TRAIN],
                alt,
                40,
            )

            aliases[alt] = {
                "theta_PRIVATE": fit["params"],

                "heldout_rms_mm": rms_mm(
                    noisy[VALID].predict(
                        alt, fit["params"]
                    ),
                    noisy[VALID].y_b,
                ),

                "outside_physical_outer_box_PRIVATE":
                    physical_box_check(
                        alt, fit["params"], band
                    ),

                "optimizer_success": bool(
                    fit["optimizer_success"]
                ),
            }

        result[
            "independent_aliases_PRIVATE"
        ] = aliases

    return result


def summarize(rows):
    output = {}

    fields = (
        "oracle_clean_mm",
        "oracle_noisy_A_to_clean_B_mm",
        "oracle_noisy_to_noisy_mm",
        "clean_fit_to_clean_mm",
        "noisy_fit_to_noisy_mm",
    )

    for mode in MODES:
        subset = [
            row for row in rows
            if row["truth_PRIVATE"] == mode
        ]

        item = {
            key: quant([
                r[key] for r in subset
            ])
            for key in fields
        }

        item["true_fit_optimizer_failures"] = sum(
            not r[
                "truth_fit_noisy_optimizer_success"
            ]
            for r in subset
        )

        if mode == "independent":
            item["aliases"] = {}

            for alt in (
                "elastic", "deadzone", "cubic"
            ):
                item["aliases"][alt] = {
                    "outside_outer_box_n": sum(
                        bool(
                            r[
                                "independent_aliases_PRIVATE"
                            ][alt][
                                "outside_physical_outer_box_PRIVATE"
                            ]
                        )
                        for r in subset
                    ),

                    "heldout_rms": quant([
                        r[
                            "independent_aliases_PRIVATE"
                        ][alt]["heldout_rms_mm"]
                        for r in subset
                    ]),
                }

        output[mode] = item

    return output


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--smoke",
        action="store_true",
    )

    args = ap.parse_args()

    src, sim, api, commands = initialize()

    n = 1 if args.smoke else N_DEV
    offset = 100000 if args.smoke else 0

    scenes = src["make_scenes"](
        sim, SEED + offset, n
    )

    print(
        "\nP4-M2A GENERATOR / INFERENCE "
        "FIDELITY AUDIT"
    )

    print(
        "New development scenes:", n
    )

    print(
        "No E4. No test threshold tuning."
    )

    reports = {}
    audits = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(
            scenes
        ):
            for mode in MODES:
                rows.append(
                    run_case(
                        sim, api, commands,
                        scene_set[band],
                        sid, band, mode,
                        NOISE_SEED + offset,
                    )
                )

        audits[band] = rows
        reports[band] = summarize(rows)

        print("\nBAND:", band)

        for mode, metrics in (
            reports[band].items()
        ):
            clean = metrics[
                "oracle_clean_mm"
            ]["median_mm"]

            noisy = metrics[
                "oracle_noisy_to_noisy_mm"
            ]["median_mm"]

            fitted = metrics[
                "noisy_fit_to_noisy_mm"
            ]["median_mm"]

            print(
                f"  {mode:11s}"
                f" clean truth={clean:.3f}mm"
                f" noisy truth={noisy:.3f}mm"
                f" noisy fitted={fitted:.3f}mm"
            )

            if mode == "independent":
                print(
                    "  Aliases outside physical box:",
                    {
                        key: val["outside_outer_box_n"]
                        for key, val in
                        metrics["aliases"].items()
                    }
                )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke
        else "dev"
    )

    report = {
        "protocol":
            "P4-M2A generator/inference fidelity",

        "smoke": args.smoke,
        "n_dev_scenes": n,
        "scene_seed": SEED + offset,
        "noise_seed": NOISE_SEED + offset,

        "fit_history": TRAIN,
        "validation_action": VALID,
        "noise_sigma_m": SIGMA,

        "summary": reports,

        "test_threshold_tuning": False,
        "E4_accessed": False,

        "source_sha256": {
            filename: sha(HERE / filename)
            for filename in (
                "16_p2d1b_probe_diversity.py",
                "20_p3a0_parameter_refit.py",
                "22_p3a1b_structural_revision.py",
                "68_p3a8c1_temporal_action_feasibility.py",
                "74_p4m2a_generator_inference_fidelity.py",
            )
        },

        "limitations": [
            "Development-only diagnostics",
            "No final test claims",
            "Oracle physical parameters private",
            "Physical outer box is not sufficient",
            "One noisy realization per case",
            "No calibrated coverage claim",
            "A-conditioned approximate model",
            "No new action selection policy",
            "Gap bands share base scenes",
        ],
    }

    report_path = OUT / (
        f"p4m2a_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2a_{suffix}_audit_PRIVATE.json"
    )

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
            audits,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nP4-M2A COMPLETE")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
