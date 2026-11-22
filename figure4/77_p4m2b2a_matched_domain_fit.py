
"""
P4-M2B2A — Matched-Solver Domain Ablation

Compare three nested effective-parameter domains:
    WIDE
    OUTER_BOX
    SHARED_MASS

Same data, objective, starts, solver and budget.
The exact shared-mass domain is represented
by linear inequalities in effective coefficients.

Development-only experiment, same scenes as P4-M2B1.
No E4, no future action, no confirmation thresholds.
"""

import argparse
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.optimize import (
    Bounds, LinearConstraint, minimize
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p4m2b2a"

P74 = HERE / "74_p4m2a_generator_inference_fidelity.py"
P75 = HERE / "75_p4m2b0_physical_domain_fit_pilot.py"
REF_DIR = HERE / "results" / "p4m2b1"

SEED = 20290317
NOISE_SEED = 2029031701
N_DEV = 12

SIGMA = 0.005
MAXITER = 150

BANDS = ("original", "above_original_range")

MODES = (
    "independent", "rigid", "elastic",
    "deadzone", "nonlinear",
)

LAWS = ("elastic", "deadzone", "cubic")
DOMAINS = ("wide", "outer_box", "shared_mass")

TRUTH_TO_LAW = {
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

TRAIN = ("e1", "e2", "weak", "strong")
VALID = "mid"

MASS_LO, MASS_HI = 0.8, 1.6

PHYSICAL_RANGES = {
    "alpha": (5.0, 16.0),
    "beta": (0.5, 2.2),
    "gamma": (100.0, 850.0),
}

WIDE_RANGES = {
    "alpha": (0.01, 70.0),
    "beta": (0.001, 15.0),
    "gamma": (0.01, 2500.0),
    "gap": (0.001, 0.2),
}

GAP_RANGE = (0.015, 0.18)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def structure(law):
    if law == "elastic":
        return ("alpha", "beta")

    if law == "deadzone":
        return ("alpha", "beta", "gap")

    if law == "cubic":
        return ("alpha", "beta", "gamma")

    raise ValueError(law)


def specifications(law, domain):
    names = structure(law)

    lows, highs = [], []

    for name in names:
        if domain == "wide":
            lo, hi = WIDE_RANGES[name]

        elif name == "gap":
            lo, hi = GAP_RANGE

        else:
            p_lo, p_hi = PHYSICAL_RANGES[name]
            lo = p_lo / MASS_HI
            hi = p_hi / MASS_LO

        lows.append(lo)
        highs.append(hi)

    return (
        np.array(lows, dtype=float),
        np.array(highs, dtype=float),
    )


def shared_mass_constraint(law):
    """
    For q_i = p_i / mass, existence of one
    common mass is equivalent to:

      U_j*q_i - L_i*q_j >= 0

    for all ordered pairs i != j, together
    with the physical outer-box bounds.

    Gap is independent of the shared mass.
    """
    names = structure(law)

    coupled = (
        ("alpha", "beta", "gamma")
        if law == "cubic"
        else ("alpha", "beta")
    )

    rows = []

    for i, ni in enumerate(coupled):
        for j, nj in enumerate(coupled):
            if i == j:
                continue

            Li = PHYSICAL_RANGES[ni][0]
            Uj = PHYSICAL_RANGES[nj][1]

            row = np.zeros(len(names))
            row[i] = Uj
            row[j] = -Li
            rows.append(row)

    A = np.stack(rows)

    return LinearConstraint(
        A,
        np.zeros(len(rows)),
        np.full(len(rows), np.inf),
    )


def is_feasible(law, params, domain, tol=1e-5):
    q = np.asarray(params, dtype=float)
    lo, hi = specifications(law, domain)

    if not np.isfinite(q).all():
        return False

    if np.any(q < lo-tol) or np.any(q > hi+tol):
        return False

    if domain == "shared_mass":
        names = structure(law)

        lower = MASS_LO
        upper = MASS_HI

        for name, value in zip(names, q):
            if name == "gap":
                continue

            p_lo, p_hi = PHYSICAL_RANGES[name]

            if value <= 0:
                return False

            lower = max(lower, p_lo/value)
            upper = min(upper, p_hi/value)

        return lower <= upper+tol

    return True


def initial_points(law):
    """
    Identical truth-blind starts for all domains.
    Each initial point is physically feasible.
    """
    starts = []

    for frac in (0.30, 0.70):
        mass = 1.2

        k = 5.0 + frac*(16.0-5.0)
        c = 0.5 + frac*(2.2-0.5)

        values = [k/mass, c/mass]

        if law == "deadzone":
            values.append(
                GAP_RANGE[0]
                + frac*(GAP_RANGE[1]-GAP_RANGE[0])
            )

        elif law == "cubic":
            k3 = 100.0 + frac*(850.0-100.0)
            values.append(k3/mass)

        starts.append(np.array(values, dtype=float))

    return starts


def fit_domain(probes, law, domain):
    lo, hi = specifications(law, domain)

    observations = [
        np.asarray(p.y_b, dtype=float).ravel()
        for p in probes
    ]

    def objective(q):
        residuals = []

        for probe, observed in zip(
            probes, observations
        ):
            pred = np.asarray(
                probe.predict(law, q.tolist()),
                dtype=float,
            ).ravel()

            residuals.append((pred-observed)/SIGMA)

        r = np.concatenate(residuals)
        return float(np.mean(r*r))

    constraints = (
        [shared_mass_constraint(law)]
        if domain == "shared_mass"
        else []
    )

    attempts = []

    for x0 in initial_points(law):
        if not is_feasible(law, x0, domain):
            raise RuntimeError(
                "Predefined start is infeasible"
            )

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=constraints,
            options={
                "maxiter": MAXITER,
                "ftol": 1e-9,
                "disp": False,
            },
        )

        valid = (
            is_feasible(law, result.x, domain)
            and np.isfinite(result.fun)
        )

        attempts.append((result, valid))

    valid_results = [
        r for r, valid in attempts if valid
    ]

    if not valid_results:
        return {
            "valid": False,
            "converged": False,
            "params_PRIVATE": None,
            "train_mse": None,
            "nfev": sum(
                int(r.nfev) for r, _ in attempts
            ),
        }

    best = min(valid_results, key=lambda r: r.fun)

    return {
        "valid": True,
        "converged": bool(best.success),
        "params_PRIVATE": best.x.tolist(),
        "train_mse": float(best.fun),
        "nfev": sum(
            int(r.nfev) for r, _ in attempts
        ),
    }


def run_case(
    sim, api, commands,
    p75, scene, sid, band, mode,
    noise_seed, reference
):
    probes = p75["create_noisy_probes"](
        sim, api, commands,
        scene, sid, band, mode,
        noise_seed,
    )

    training = [probes[name] for name in TRAIN]
    validation = probes[VALID]

    # Confirm exactly the same observed data
    # as P4-M2B1, using the legacy fit.
    legacy = api["fit_one"](
        training, "deadzone", 40
    )

    legacy_rms = p75["rms_mm"](
        validation.predict(
            "deadzone", legacy["params"]
        ),
        validation.y_b,
    )

    expected = reference["legacy_deadzone_mm"]

    if not np.isclose(
        legacy_rms, expected,
        atol=1e-5, rtol=1e-9
    ):
        raise RuntimeError(
            "Observation parity failed: "
            f"{band}/{sid}/{mode}"
        )

    results = {}

    for law in LAWS:
        results[law] = {}

        for domain in DOMAINS:
            fitted = fit_domain(
                training, law, domain
            )

            if fitted["valid"]:
                rms = p75["rms_mm"](
                    validation.predict(
                        law,
                        fitted["params_PRIVATE"],
                    ),
                    validation.y_b,
                )
            else:
                rms = None

            fitted["validation_rms_mm"] = rms

            results[law][domain] = fitted

    return {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "fits": results,
    }


def summarise(rows):
    result = {}

    for mode in MODES:
        subset = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        result[mode] = {}

        for domain in DOMAINS:
            scores = {}

            for law in LAWS:
                values = [
                    r["fits"][law][domain][
                        "validation_rms_mm"
                    ]
                    for r in subset
                ]

                complete = [
                    float(x) for x in values
                    if x is not None
                ]

                scores[law] = {
                    "n_valid": len(complete),
                    "median_mm": (
                        float(np.median(complete))
                        if complete else None
                    ),
                    "failures": sum(
                        not r["fits"][law][domain]["valid"]
                        for r in subset
                    ),
                    "nonconverged": sum(
                        r["fits"][law][domain]["valid"]
                        and not r["fits"][law][domain]["converged"]
                        for r in subset
                    ),
                }

            winner_count = None

            if mode in TRUTH_TO_LAW:
                true_law = TRUTH_TO_LAW[mode]
                winner_count = 0

                for r in subset:
                    local = {
                        law: r["fits"][law][domain][
                            "validation_rms_mm"
                        ]
                        for law in LAWS
                    }

                    if any(v is None for v in local.values()):
                        continue

                    winner_count += (
                        min(local, key=local.get)
                        == true_law
                    )

            result[mode][domain] = {
                "candidates": scores,
                "correct_winner_count": winner_count,
            }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true"
    )
    args = parser.parse_args()

    suffix = "smoke" if args.smoke else "dev"
    n = 1 if args.smoke else N_DEV
    offset = 100000 if args.smoke else 0

    p74 = run_path(str(P74))
    p75 = run_path(str(P75))

    src, sim, api, commands = p74["initialize"]()

    previous = read(
        REF_DIR /
        f"p4m2b1_{suffix}_audit_PRIVATE.json"
    )

    reference = {
        (band, r["scene_id"], r["truth_PRIVATE"]): r
        for band, rows in previous.items()
        for r in rows
    }

    scenes = src["make_scenes"](
        sim, SEED+offset, n
    )

    print("\nP4-M2B2A MATCHED DOMAIN FIT")
    print("Scenes:", n)
    print("Domains:", DOMAINS)
    print("Solver: SLSQP | budget:", MAXITER)
    print("Band-blind physical prior")
    print("No E4. No threshold tuning.")

    audits = {}
    summaries = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(scenes):
            for mode in MODES:
                rows.append(run_case(
                    sim, api, commands,
                    p75,
                    scene_set[band],
                    sid, band, mode,
                    NOISE_SEED+offset,
                    reference[(band, sid, mode)],
                ))

        audits[band] = rows
        summaries[band] = summarise(rows)

        print("\nBAND:", band)

        for mode in MODES:
            print(" ", mode)

            for domain in DOMAINS:
                item = summaries[band][mode][domain]

                parts = []

                for law in LAWS:
                    stat = item["candidates"][law]

                    val = stat["median_mm"]
                    shown = (
                        "NA" if val is None
                        else f"{val:.3f}"
                    )

                    parts.append(f"{law}={shown}")

                print(
                    f"   {domain:12s}",
                    " ".join(parts),
                    "winner=",
                    item["correct_winner_count"],
                )

    OUT.mkdir(parents=True, exist_ok=True)

    report_path = OUT / (
        f"p4m2b2a_{suffix}_report.json"
    )
    audit_path = OUT / (
        f"p4m2b2a_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol": "P4-M2B2A matched-domain ablation",
        "smoke": args.smoke,
        "n_scenes": n,
        "scene_seed": SEED+offset,
        "noise_seed": NOISE_SEED+offset,
        "domains": DOMAINS,
        "solver": "SLSQP",
        "maxiter_per_start": MAXITER,
        "starts_per_fit": 2,
        "band_blind_gap_range": GAP_RANGE,
        "summary": summaries,
        "E4_accessed": False,
        "test_threshold_tuning": False,
        "independent_evaluation": False,
        "source_sha256": {
            P74.name: digest(P74),
            P75.name: digest(P75),
            Path(__file__).name:
                digest(Path(__file__)),
        },
        "limitations": [
            "Same previously examined development scenes",
            "Finite local optimization",
            "Success status is not a global certificate",
            "Only Mid validation RMS",
            "No calibrated candidate sets",
            "No unknown mechanism evaluation",
            "No mistaken-prior stress test",
            "No independent generalization claim",
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

    print("\nP4-M2B2A COMPLETE")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
