
"""
P4-M2B0 — Physical-Domain Fit Pilot

Compare legacy wide parameter fitting against
an explicitly physically admissible fit.

Uses SAME development scenes and noise seeds
as P4-M2A. No independent test claim.

True law is used to generate scenes and
for PRIVATE auditing, never by the fitter.

No E4. No test-set threshold selection.
No structural confirmation claims.
"""

import argparse
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p4m2b0"

P74 = HERE / (
    "74_p4m2a_generator_inference_fidelity.py"
)

REFERENCE = (
    HERE / "results" / "p4m2a"
    / "p4m2a_dev_audit_PRIVATE.json"
)

SCENE_SEED = 20290317
NOISE_SEED = 2029031701
SIGMA = 0.005

N_DEV = 12
MAX_NFEV = 60

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

LAWS = (
    "elastic",
    "deadzone",
    "cubic",
)

TRAIN = ("e1", "e2", "weak", "strong")
VALID = "mid"

TRUE_LAW = {
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def rms_mm(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    return float(
        1000 * np.sqrt(np.mean((x-y)**2))
    )


def median(values):
    return float(np.median(
        np.asarray(values, dtype=float)
    ))


def gap_bounds(band):
    if band == "original":
        return 0.015, 0.12

    return 0.12, 0.18


def parameterization(law, band):
    # Physical parameters, not independently
    # bounded effective coefficients.
    names = ["mass_b", "k", "c"]
    lo = [0.8, 5.0, 0.5]
    hi = [1.6, 16.0, 2.2]

    if law == "deadzone":
        gap_lo, gap_hi = gap_bounds(band)

        names.append("gap")
        lo.append(gap_lo)
        hi.append(gap_hi)

    elif law == "cubic":
        names.append("k3")
        lo.append(100.0)
        hi.append(850.0)

    elif law != "elastic":
        raise ValueError(law)

    return (
        names,
        np.asarray(lo, dtype=float),
        np.asarray(hi, dtype=float)
    )


def effective_params(law, z):
    z = np.asarray(z, dtype=float)
    mass, k, c = z[:3]

    if not np.isfinite(z).all() or mass <= 0:
        raise ValueError("Invalid physical vector")

    result = [k/mass, c/mass]

    if law == "deadzone":
        result.append(z[3])
    elif law == "cubic":
        result.append(z[3]/mass)

    return result


def exactly_feasible(law, params, band):
    p = np.asarray(params, dtype=float)

    if not np.isfinite(p).all():
        return False

    alpha, beta = p[:2]

    if alpha <= 0 or beta <= 0:
        return False

    # A common mass must explain k and c.
    lower = max(
        0.8,
        5.0/alpha,
        0.5/beta,
    )

    upper = min(
        1.6,
        16.0/alpha,
        2.2/beta,
    )

    if law == "deadzone":
        g0, g1 = gap_bounds(band)

        if not g0-1e-9 <= p[2] <= g1+1e-9:
            return False

    if law == "cubic":
        gamma = p[2]

        if gamma <= 0:
            return False

        lower = max(lower, 100.0/gamma)
        upper = min(upper, 850.0/gamma)

    return bool(lower <= upper+1e-9)


def fit_physical(probes, law, band):
    names, lo, hi = parameterization(
        law, band
    )

    # Fixed, truth-blind starting points.
    starts = (
        lo + 0.30*(hi-lo),
        lo + 0.70*(hi-lo),
    )

    def residual(z):
        params = effective_params(law, z)

        return np.concatenate([
            (
                np.asarray(
                    probe.predict(law, params),
                    dtype=float
                ).ravel()
                -
                np.asarray(
                    probe.y_b,
                    dtype=float
                ).ravel()
            ) / SIGMA
            for probe in probes
        ])

    results = []

    for x0 in starts:
        result = least_squares(
            residual,
            x0=x0,
            bounds=(lo, hi),
            max_nfev=MAX_NFEV,
            x_scale="jac",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )

        results.append(result)

    best = min(
        results,
        key=lambda fit: float(
            np.dot(fit.fun, fit.fun)
        )
    )

    params = effective_params(law, best.x)

    if not exactly_feasible(
        law, params, band
    ):
        raise RuntimeError(
            "Physical fit violated its domain"
        )

    return {
        "params": params,
        "physical_PRIVATE": {
            name: float(value)
            for name, value
            in zip(names, best.x)
        },
        "train_sse": float(
            np.dot(best.fun, best.fun)
        ),
        "optimizer_success": bool(
            best.success
        ),
        "nfev": int(best.nfev),
    }


def create_noisy_probes(
    sim, api, commands,
    scene, sid, band, mode, noise_seed
):
    model = sim["make_model"](scene)

    rng = np.random.default_rng(
        noise_seed
        + sid*101
        + MODES.index(mode)*7907
        + BANDS.index(band)*1000003
    )

    probes = {}

    for name in TRAIN + (VALID,):
        passive, active, _, _ = sim["simulate"](
            model, scene, mode,
            commands[name]
        )

        passive = np.asarray(
            passive, dtype=float
        )
        active = np.asarray(
            active, dtype=float
        )

        probes[name] = api["RevisionProbe"](
            passive + rng.normal(
                0, SIGMA, passive.shape
            ),
            active + rng.normal(
                0, SIGMA, active.shape
            ),
        )

    return probes


def run_case(
    sim, api, commands,
    scene, sid, band, mode, noise_seed
):
    probes = create_noisy_probes(
        sim, api, commands,
        scene, sid, band,
        mode, noise_seed
    )

    training = [probes[x] for x in TRAIN]
    validation = probes[VALID]

    fits = {}

    for law in LAWS:
        legacy = api["fit_one"](
            training, law, 40
        )

        physical = fit_physical(
            training, law, band
        )

        legacy_params = legacy["params"]
        physical_params = physical["params"]

        legacy_rms = rms_mm(
            validation.predict(
                law, legacy_params
            ),
            validation.y_b,
        )

        physical_rms = rms_mm(
            validation.predict(
                law, physical_params
            ),
            validation.y_b,
        )

        fits[law] = {
            "legacy_rms_mm": legacy_rms,
            "physical_rms_mm": physical_rms,
            "delta_physical_minus_legacy_mm":
                physical_rms-legacy_rms,

            "legacy_exact_feasible": (
                exactly_feasible(
                    law, legacy_params, band
                )
            ),

            "physical_exact_feasible": (
                exactly_feasible(
                    law, physical_params, band
                )
            ),

            "legacy_params_PRIVATE":
                legacy_params,

            "physical_params_PRIVATE":
                physical_params,

            "physical_base_params_PRIVATE":
                physical["physical_PRIVATE"],

            "physical_optimizer_success":
                physical["optimizer_success"],

            "physical_nfev":
                physical["nfev"],
        }

    return {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "fits": fits,
    }


def summarize(rows):
    result = {}

    for mode in MODES:
        selected = [
            row for row in rows
            if row["truth_PRIVATE"] == mode
        ]

        result[mode] = {}

        for law in LAWS:
            fits = [
                row["fits"][law]
                for row in selected
            ]

            item = {
                "n": len(fits),

                "legacy_median_mm": median([
                    x["legacy_rms_mm"]
                    for x in fits
                ]),

                "physical_median_mm": median([
                    x["physical_rms_mm"]
                    for x in fits
                ]),

                "median_paired_delta_mm": median([
                    x[
                        "delta_physical_minus_legacy_mm"
                    ]
                    for x in fits
                ]),

                "legacy_physically_feasible_n": sum(
                    x["legacy_exact_feasible"]
                    for x in fits
                ),

                "physical_fit_failures": sum(
                    not x["physical_optimizer_success"]
                    for x in fits
                ),
            }

            result[mode][law] = item

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke", action="store_true"
    )

    args = parser.parse_args()

    p74 = run_path(str(P74))

    src, sim, api, commands = (
        p74["initialize"]()
    )

    n = 1 if args.smoke else N_DEV

    offset = 100000 if args.smoke else 0

    scenes = src["make_scenes"](
        sim, SCENE_SEED + offset, n
    )

    print(
        "\nP4-M2B0 PHYSICAL DOMAIN FIT PILOT"
    )
    print(
        "Scenes:", n,
        "| Paired development comparison"
    )
    print(
        "No E4; no structural threshold tuning"
    )

    audits = {}
    summary = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(scenes):
            scene = scene_set[band]

            for mode in MODES:
                rows.append(
                    run_case(
                        sim, api, commands,
                        scene, sid, band, mode,
                        NOISE_SEED + offset,
                    )
                )

        audits[band] = rows
        summary[band] = summarize(rows)

        print("\nBAND:", band)

        for mode in MODES:
            print(" ", mode)

            display_laws = (
                LAWS if mode == "independent"
                else (
                    (TRUE_LAW[mode],)
                    if mode in TRUE_LAW
                    else LAWS
                )
            )

            for law in display_laws:
                s = summary[band][mode][law]

                print(
                    f"   {law:9s}",
                    "wide=",
                    round(s["legacy_median_mm"], 3),
                    "physical=",
                    round(s["physical_median_mm"], 3),
                    "wide feasible=",
                    s["legacy_physically_feasible_n"],
                    "/",
                    s["n"],
                    "physical fit failures=",
                    s["physical_fit_failures"],
                )

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "dev"
    )

    report_path = OUT / (
        f"p4m2b0_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2b0_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P4-M2B0 physical domain fit pilot",
        "smoke": args.smoke,
        "n_scenes": n,
        "scene_seed": SCENE_SEED + offset,
        "noise_seed": NOISE_SEED + offset,
        "summary": summary,

        "test_tuning": False,
        "E4_accessed": False,
        "uses_test_truth_online": False,

        "source_sha256": {
            "74_p4m2a_generator_inference_fidelity.py":
                sha(P74),
            "75_p4m2b0_physical_domain_fit_pilot.py":
                sha(
                    HERE /
                    "75_p4m2b0_physical_domain_fit_pilot.py"
                ),
        },

        "limitations": [
            "Reuses P4-M2A development scenes",
            "Not an independent test",
            "Legacy and physical solvers differ",
            "Physical fit has redundant mass dimension",
            "Finite local optimization, no global proof",
            "Only held-out RMS compared",
            "No structural acceptance threshold",
            "No calibrated coverage claim",
            "No new active experiment selection",
            "Physical domain depends on generator prior",
            "No robustness to incorrect domain prior tested",
        ],
    }

    report_path.write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            audits, indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nP4-M2B0 COMPLETE")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
