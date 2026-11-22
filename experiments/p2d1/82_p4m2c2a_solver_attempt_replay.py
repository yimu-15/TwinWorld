
"""
P4-M2C2A — Targeted SLSQP Attempt-Level Replay

Investigate the 21 nested-domain violations
reported by P4-M2C1.

Scope:
    Deadzone candidate law only
    Original 21 violating cases

Audit:
    Per-start SLSQP status
    Per-start objective values
    Feasibility
    Iteration counts
    Outer-to-Wide feasible incumbent replay

No E4.
No Mid validation.
No threshold tuning.
No modification of historical files.
"""

import argparse
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.optimize import Bounds, minimize


HERE = Path(__file__).resolve().parent

P80 = HERE / "80_p4m2c0_prior_mismatch_stress.py"
P81 = HERE / "81_p4m2c1_prior_conflict_audit.py"

SOURCE_DIR = HERE / "results" / "p4m2c1"
P80_DIR = HERE / "results" / "p4m2c0"

OUT = HERE / "results" / "p4m2c2a"

P80_REPORT = P80_DIR / "p4m2c0_dev_report.json"
P80_AUDIT = P80_DIR / "p4m2c0_dev_audit_PRIVATE.json"

P81_REPORT = SOURCE_DIR / "p4m2c1_dev_report.json"
P81_AUDIT = SOURCE_DIR / "p4m2c1_dev_audit_PRIVATE.json"

DOMAINS = ("wide", "outer_box")

LAW = "deadzone"

NESTED_TOL = 1e-6

MAXITER = 150
FTOL = 1e-9


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def verify_provenance():

    report = load_json(P81_REPORT)

    if report["smoke"] is not False:
        raise RuntimeError("Formal P4-M2C1 required")

    if report["nested_violation_count"] != 21:
        raise RuntimeError("Expected 21 violations")

    if report["nested_optimization_violations"].get(
        "deadzone", 0
    ) != 21:
        raise RuntimeError("Unexpected violation distribution")

    if report["E4_accessed"] is not False:
        raise RuntimeError("Unexpected E4 access")

    for path, key in (
        (P80_REPORT, "report_sha256"),
        (P80_AUDIT, "audit_sha256"),
    ):

        expected = report["source_inputs"][key]

        if sha(path) != expected:
            raise RuntimeError(
                "P4-M2C0 provenance mismatch: "
                + path.name
            )

    expected_81 = report["source_sha256"].get(
        P81.name
    )

    if expected_81 and sha(P81) != expected_81:
        raise RuntimeError("P4-M2C1 source mismatch")

    rows = load_json(P81_AUDIT)

    cases = [
        r for r in rows
        if LAW in r["nested_violations"]
    ]

    if len(cases) != 21:
        raise RuntimeError(
            f"Expected 21 cases, found {len(cases)}"
        )

    return report, cases


def make_objective(probes, p77):

    sigma = float(p77["SIGMA"])

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
                probe.predict(
                    LAW,
                    np.asarray(q).tolist()
                ),
                dtype=float,
            ).ravel()

            if pred.shape != observed.shape:
                raise RuntimeError(
                    "Prediction shape mismatch"
                )

            residuals.append(
                (pred - observed) / sigma
            )

        residual = np.concatenate(residuals)

        if not np.isfinite(residual).all():
            return float("inf")

        return float(
            np.mean(residual * residual)
        )

    return objective


def detailed_fit(probes, domain, p77):

    lo, hi = p77["specifications"](
        LAW, domain
    )

    objective = make_objective(probes, p77)

    starts = p77["initial_points"](LAW)

    attempts = []

    for index, x0 in enumerate(starts):

        x0 = np.asarray(x0, dtype=float)

        if not p77["is_feasible"](
            LAW, x0, domain
        ):
            raise RuntimeError(
                f"Infeasible initial point: {domain}/{index}"
            )

        initial_fun = objective(x0)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=[],
            options={
                "maxiter": MAXITER,
                "ftol": FTOL,
                "disp": False,
            },
        )

        params = np.asarray(
            result.x, dtype=float
        )

        feasible = bool(
            p77["is_feasible"](
                LAW, params, domain
            )
        )

        finite = bool(
            np.isfinite(params).all()
            and np.isfinite(result.fun)
        )

        recomputed = (
            objective(params)
            if finite else None
        )

        valid = feasible and finite

        attempts.append({
            "start_index": index,
            "initial_params": x0.tolist(),
            "initial_fun": initial_fun,
            "final_params": params.tolist(),
            "result_fun": (
                float(result.fun)
                if np.isfinite(result.fun)
                else None
            ),
            "recomputed_fun": recomputed,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "njev": (
                int(result.njev)
                if hasattr(result, "njev")
                else None
            ),
            "feasible": feasible,
            "finite": finite,
            "valid": valid,
        })

    valid_attempts = [
        x for x in attempts
        if x["valid"]
    ]

    successful_attempts = [
        x for x in valid_attempts
        if x["success"]
    ]

    best = (
        min(
            valid_attempts,
            key=lambda x: x["result_fun"]
        )
        if valid_attempts else None
    )

    best_success = (
        min(
            successful_attempts,
            key=lambda x: x["result_fun"]
        )
        if successful_attempts else None
    )

    return {
        "attempts": attempts,
        "n_attempts": len(attempts),
        "n_success": len(successful_attempts),
        "n_valid": len(valid_attempts),
        "best_valid": best,
        "best_success": best_success,
    }


def replay_case(case, scenes, p74, p75, p77,
                sim, api, commands, noise_seed):

    sid = int(case["scene_id"])
    band = case["band"]
    mode = case["mode_PRIVATE"]
    axis = case["axis_PRIVATE"]
    level = case["level_PRIVATE"]

    base_scene = scenes[sid][band]

    if axis == "matched":
        scene = base_scene
    else:
        scene, _ = p80_stress_scene(
            p74, p77,
            base_scene,
            mode,
            axis,
            level,
        )

    probes = p75["create_noisy_probes"](
        sim,
        api,
        commands,
        scene,
        sid,
        band,
        mode,
        noise_seed,
    )

    training = [
        probes[name]
        for name in p77["TRAIN"]
    ]

    fits = {
        domain: detailed_fit(
            training, domain, p77
        )
        for domain in DOMAINS
    }

    # Verify reproduction of historical losses.

    for domain in DOMAINS:

        expected = float(
            case["per_law_losses"][LAW][
                domain
            ]
        )

        selected = fits[domain]["best_valid"]

        if selected is None:
            raise RuntimeError(
                "No valid replay fit: " + domain
            )

        if not np.isclose(
            selected["result_fun"],
            expected,
            rtol=1e-8,
            atol=1e-8,
        ):
            raise RuntimeError(
                "Historical objective parity failed: "
                f"{sid}/{band}/{mode}/{axis}/{level}/{domain}"
            )

    wide_best = fits["wide"]["best_valid"]
    outer_best = fits["outer_box"]["best_valid"]

    if not wide_best["success"]:
        raise RuntimeError(
            "Historical Wide selected fit did not converge"
        )

    if not outer_best["success"]:
        raise RuntimeError(
            "Historical Outer selected fit did not converge"
        )

    # The Outer solution must also be feasible in Wide.

    outer_params = np.asarray(
        outer_best["final_params"],
        dtype=float,
    )

    outer_feasible_in_wide = bool(
        p77["is_feasible"](
            LAW,
            outer_params,
            "wide",
        )
    )

    if not outer_feasible_in_wide:
        raise RuntimeError(
            "Outer solution is not Wide-feasible"
        )

    objective = make_objective(
        training, p77
    )

    outer_replay_fun = objective(
        outer_params
    )

    if not np.isclose(
        outer_replay_fun,
        outer_best["result_fun"],
        rtol=1e-9,
        atol=1e-9,
    ):
        raise RuntimeError(
            "Cross-domain objective parity failed"
        )

    wide_original_fun = wide_best["result_fun"]

    replay_gap = (
        wide_original_fun - outer_replay_fun
    )

    legacy_violation = (
        outer_best["result_fun"]
        < wide_best["result_fun"] - NESTED_TOL
    )

    replay_closes_violation = (
        outer_replay_fun
        <= wide_original_fun + NESTED_TOL
    )

    return {
        "scene_id": sid,
        "band": band,
        "mode": mode,
        "axis": axis,
        "level": level,

        "historical_violation": True,

        "fits": fits,

        "cross_feasible_replay": {
            "outer_feasible_in_wide":
                outer_feasible_in_wide,

            "wide_original_best_fun":
                wide_original_fun,

            "outer_best_fun":
                outer_best["result_fun"],

            "outer_solution_replayed_in_wide_fun":
                outer_replay_fun,

            "wide_minus_outer_replay":
                replay_gap,

            "replay_closes_violation":
                replay_closes_violation,

            "optimizer_search_gap":
                bool(
                    legacy_violation
                    and replay_closes_violation
                ),
        },
    }


def p80_stress_scene(
    p74, p77, base_scene, mode, axis, level
):
    return P80_MODULE["make_stress_scene"](
        p74,
        p77,
        base_scene,
        mode,
        axis,
        level,
    )


def main():

    global P80_MODULE

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("P4-M2C2A — SOLVER ATTEMPT REPLAY")
    print("=" * 72)

    print("Smoke:", args.smoke)
    print("Target: P4-M2C1 nested violations")
    print("Law:", LAW)
    print("No E4. No Mid validation.")

    print("\n[1] Provenance verification")

    report, cases = verify_provenance()

    print("[PASS] Historical report")
    print("[PASS] 21 violation records")
    print("[PASS] P4-M2C0 source hashes")

    print("\n[2] Loading frozen implementations")

    P80_MODULE = run_path(str(P80))

    P80_MODULE["check_provenance"]()

    (
        p74, p75, p77, p79,
        src, sim, api, commands
    ) = P80_MODULE["initialize"]()

    print("[PASS] P74/P75/P77/P79")

    print("\n[3] Regenerating original development scenes")

    scenes = src["make_scenes"](
        sim,
        P80_MODULE["SCENE_SEED"],
        P80_MODULE["N_DEV"],
    )

    if len(scenes) != 12:
        raise RuntimeError("Scene count mismatch")

    print("Scenes:", len(scenes))

    selected_cases = (
        cases[:1] if args.smoke else cases
    )

    print("\n[4] Per-start optimization replay")

    results = []

    for i, case in enumerate(selected_cases):

        result = replay_case(
            case,
            scenes,
            p74,
            p75,
            p77,
            sim,
            api,
            commands,
            P80_MODULE["NOISE_SEED"],
        )

        results.append(result)

        replay = result["cross_feasible_replay"]

        print(
            f"Case {i+1}/{len(selected_cases)} | "
            f"scene={result['scene_id']} | "
            f"{result['mode']} | "
            f"{result['axis']} | "
            f"{result['level']}"
        )

        print(
            "  Wide starts:",
            result["fits"]["wide"]["n_success"],
            "/",
            result["fits"]["wide"]["n_attempts"],
        )

        print(
            "  Outer starts:",
            result["fits"]["outer_box"]["n_success"],
            "/",
            result["fits"]["outer_box"]["n_attempts"],
        )

        print(
            "  Replay gap:",
            replay["wide_minus_outer_replay"],
        )

        print(
            "  Violation closed:",
            replay["replay_closes_violation"],
        )

    print("\n[5] Summary")

    total = len(results)

    closed = sum(
        r["cross_feasible_replay"][
            "replay_closes_violation"
        ]
        for r in results
    )

    search_gap = sum(
        r["cross_feasible_replay"][
            "optimizer_search_gap"
        ]
        for r in results
    )

    print("Cases:", total)
    print("Replay closes violations:", closed)
    print("Optimizer search-gap cases:", search_gap)

    print("\n[6] Saving reports")

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke else "dev"
    )

    report_path = OUT / (
        f"p4m2c2a_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2c2a_{suffix}_audit_PRIVATE.json"
    )

    output = {
        "protocol": (
            "P4-M2C2A targeted SLSQP attempt replay"
        ),

        "smoke": args.smoke,
        "development_only": True,

        "n_cases": total,
        "historical_violation_count": 21,

        "law": LAW,

        "solver": "SLSQP",
        "maxiter": MAXITER,
        "ftol": FTOL,
        "starts_per_fit": 2,

        "replay_closes_violations": closed,
        "optimizer_search_gap_cases": search_gap,

        "uses_truth_in_fitting": False,
        "uses_mid_validation": False,
        "E4_accessed": False,

        "source_sha256": {
            P80.name: sha(P80),
            P81.name: sha(P81),
            "77_p4m2b2a_matched_domain_fit.py":
                sha(HERE / "77_p4m2b2a_matched_domain_fit.py"),
            P80_REPORT.name: sha(P80_REPORT),
            P80_AUDIT.name: sha(P80_AUDIT),
            P81_REPORT.name: sha(P81_REPORT),
            P81_AUDIT.name: sha(P81_AUDIT),
        },

        "limitations": [
            "Only 21 historical violation cases",
            "Development data only",
            "Two original SLSQP starts",
            "No global optimality certificate",
            "No independent evaluation",
            "No adaptive policy",
        ],
    }

    report_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nP4-M2C2A COMPLETE")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
