
"""
P4-M2C2B — Warm-start Transfer Audit

Question:
Can the Outer Box solution be transferred into
the Wide domain and further optimized?

Target:
All 21 P4-M2C2A historical cases.

Fixed:
    Candidate law: deadzone
    Optimizer: SLSQP
    maxiter: 150
    ftol: 1e-9
    Same development observations
    Same Wide bounds
    Same training objective

New diagnostic:
    Outer Box best parameters -> Wide initial point

No:
    E4
    Mid validation
    Truth-assisted fitting
    Random starts
    Historical result modification
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
P77 = HERE / "77_p4m2b2a_matched_domain_fit.py"

A_DIR = HERE / "results" / "p4m2c2a"
A_REPORT = A_DIR / "p4m2c2a_dev_report.json"
A_AUDIT = A_DIR / "p4m2c2a_dev_audit_PRIVATE.json"

OUT = HERE / "results" / "p4m2c2b"

LAW = "deadzone"
TOL = 1e-6
FTOL = 1e-9
MAXITER = 150


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )



def verify_inputs():

    report = read_json(A_REPORT)
    audit = read_json(A_AUDIT)

    if report["smoke"] is not False:
        raise RuntimeError("Formal 82A report required")

    if report["n_cases"] != 21:
        raise RuntimeError("Expected 21 cases")

    if report["replay_closes_violations"] != 21:
        raise RuntimeError("82A replay verification failed")

    if report["E4_accessed"] is not False:
        raise RuntimeError("Unexpected E4 access")

    def resolve_path(name):

        if name.endswith(".py"):
            return HERE / name

        if name.startswith("p4m2c0_"):
            return HERE / "results" / "p4m2c0" / name

        if name.startswith("p4m2c1_"):
            return HERE / "results" / "p4m2c1" / name

        raise RuntimeError(
            f"Unknown provenance file: {name}"
        )

    for name, expected in report["source_sha256"].items():

        path = resolve_path(name)

        if not path.is_file():
            raise FileNotFoundError(path)

        actual = sha(path)

        if actual != expected:
            raise RuntimeError(
                f"Source hash mismatch: {name}"
            )

    if len(audit) != 21:
        raise RuntimeError("82A audit size mismatch")

    return report, audit



def make_objective(training, p77):

    sigma = float(p77["SIGMA"])

    observations = [
        np.asarray(p.y_b, dtype=float).ravel()
        for p in training
    ]

    def objective(q):

        residuals = []

        for probe, observed in zip(
            training, observations
        ):

            pred = np.asarray(
                probe.predict(
                    LAW,
                    np.asarray(q, dtype=float).tolist()
                ),
                dtype=float
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
            np.mean(residual ** 2)
        )

    return objective


def prepare_probes(row, scenes, p74, p75,
                   sim, api, commands, noise_seed):

    sid = int(row["scene_id"])
    band = row["band"]
    mode = row["mode"]
    axis = row["axis"]
    level = row["level"]

    scene = scenes[sid][band]

    if axis != "matched":

        scene, _ = p80["make_stress_scene"](
            p74,
            p77,
            scene,
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

    return training


def run_case(index, row, scenes, p74, p75,
             p77, sim, api, commands, noise_seed):

    training = prepare_probes(
        row,
        scenes,
        p74,
        p75,
        sim,
        api,
        commands,
        noise_seed,
    )

    objective = make_objective(
        training, p77
    )

    # Historical solutions from 82A.

    outer_params = np.asarray(
        row["fits"]["outer_box"]
        ["best_valid"]["final_params"],
        dtype=float,
    )

    historical_wide = float(
        row["fits"]["wide"]
        ["best_valid"]["result_fun"]
    )

    historical_outer = float(
        row["fits"]["outer_box"]
        ["best_valid"]["result_fun"]
    )

    # Verify the transferred solution.

    feasible = bool(
        p77["is_feasible"](
            LAW,
            outer_params,
            "wide",
        )
    )

    if not feasible:
        raise RuntimeError(
            f"Case {index}: Outer solution not Wide-feasible"
        )

    seed_fun = objective(outer_params)

    if not np.isclose(
        seed_fun,
        historical_outer,
        rtol=1e-8,
        atol=1e-8,
    ):
        raise RuntimeError(
            f"Case {index}: 82A objective mismatch"
        )

    # Optimize using Wide bounds.

    lo, hi = p77["specifications"](
        LAW, "wide"
    )

    result = minimize(
        objective,
        outer_params.copy(),
        method="SLSQP",
        bounds=Bounds(lo, hi),
        constraints=[],
        options={
            "maxiter": MAXITER,
            "ftol": FTOL,
            "disp": False,
        },
    )

    final_params = np.asarray(
        result.x,
        dtype=float,
    )

    final_fun = objective(final_params)

    final_feasible = bool(
        p77["is_feasible"](
            LAW,
            final_params,
            "wide",
        )
    )

    if not np.isfinite(final_fun):
        raise RuntimeError(
            f"Case {index}: Nonfinite final objective"
        )

    if not final_feasible:
        raise RuntimeError(
            f"Case {index}: Infeasible final solution"
        )

    # Best-so-far incumbent:
    # never discard a better feasible initial point.

    if final_fun <= seed_fun:
        incumbent_fun = final_fun
        incumbent_params = final_params
        incumbent_source = "warm_start_optimization"
    else:
        incumbent_fun = seed_fun
        incumbent_params = outer_params
        incumbent_source = "outer_initial_incumbent"

    return {
        "case": index,
        "scene_id": int(row["scene_id"]),
        "band": row["band"],
        "mode": row["mode"],
        "axis": row["axis"],
        "level": row["level"],

        "historical_wide_fun": historical_wide,
        "historical_outer_fun": historical_outer,

        "warm_start": {
            "initial_params": outer_params.tolist(),
            "initial_fun": float(seed_fun),

            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),

            "nit": int(result.nit),
            "nfev": int(result.nfev),

            "final_params": final_params.tolist(),
            "final_fun": float(final_fun),
            "final_feasible": final_feasible,

            "improvement_from_seed": float(
                seed_fun - final_fun
            ),

            "not_worse_than_seed": bool(
                final_fun <= seed_fun + TOL
            ),

            "beats_historical_wide": bool(
                final_fun < historical_wide - TOL
            ),
        },

        "best_so_far": {
            "fun": float(incumbent_fun),
            "params": incumbent_params.tolist(),
            "source": incumbent_source,

            "closes_historical_gap": bool(
                incumbent_fun <= historical_wide + TOL
            ),
        },
    }


def main():

    global p80, p77

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("P4-M2C2B — WARM-START TRANSFER AUDIT")
    print("=" * 72)

    print("Smoke:", args.smoke)
    print("Candidate law:", LAW)
    print("Target: 21 P4-M2C2A cases")
    print("E4: NOT ACCESSED")

    print("\n[1] Verify 82A")

    report, cases = verify_inputs()

    print("[PASS] 21 historical cases")
    print("[PASS] Source hashes")
    print("[PASS] 82A nested replay")

    print("\n[2] Load frozen implementations")

    p80 = run_path(str(P80))

    p80["check_provenance"]()

    (
        p74, p75, p77, p79,
        src, sim, api, commands
    ) = p80["initialize"]()

    if p77["MAXITER"] != MAXITER:
        raise RuntimeError("MAXITER mismatch")

    if not np.isclose(
        p77["SIGMA"], 0.005
    ):
        raise RuntimeError("SIGMA mismatch")

    print("[PASS] P74/P75/P77/P79")

    print("\n[3] Reconstruct original development data")

    scenes = src["make_scenes"](
        sim,
        p80["SCENE_SEED"],
        p80["N_DEV"],
    )

    if len(scenes) != 12:
        raise RuntimeError("Scene count mismatch")

    noise_seed = p80["NOISE_SEED"]

    print("[PASS] 12 original development scenes")

    print("\n[4] Warm-start transfer")

    selected = cases[:1] if args.smoke else cases

    results = []

    for i, row in enumerate(selected, 1):

        result = run_case(
            i,
            row,
            scenes,
            p74,
            p75,
            p77,
            sim,
            api,
            commands,
            noise_seed,
        )

        results.append(result)

        w = result["warm_start"]
        b = result["best_so_far"]

        print(
            f"Case {i}/{len(selected)} | "
            f"scene={result['scene_id']} | "
            f"{result['mode']} | "
            f"{result['axis']} | "
            f"{result['level']}"
        )

        print(
            f"  Seed objective: {w['initial_fun']:.10f}"
        )

        print(
            f"  Warm objective: {w['final_fun']:.10f}"
        )

        print(
            f"  SLSQP success: {w['success']}"
        )

        print(
            f"  Not worse than seed: "
            f"{w['not_worse_than_seed']}"
        )

        print(
            f"  Beats historical Wide: "
            f"{w['beats_historical_wide']}"
        )

        print(
            f"  Best incumbent: {b['fun']:.10f}"
        )

    print("\n[5] Summary")

    n = len(results)

    success = sum(
        r["warm_start"]["success"]
        for r in results
    )

    not_worse = sum(
        r["warm_start"]["not_worse_than_seed"]
        for r in results
    )

    beats_wide = sum(
        r["warm_start"]["beats_historical_wide"]
        for r in results
    )

    closes = sum(
        r["best_so_far"]["closes_historical_gap"]
        for r in results
    )

    print("Cases:", n)
    print("SLSQP success:", success, "/", n)
    print("Not worse than seed:", not_worse, "/", n)
    print("Warm final beats historical Wide:", beats_wide, "/", n)
    print("Best-so-far closes gap:", closes, "/", n)

    print("\n[6] Save")

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = "smoke" if args.smoke else "dev"

    summary = {
        "protocol": "P4-M2C2B warm-start transfer audit",
        "smoke": args.smoke,
        "development_only": True,

        "law": LAW,
        "optimizer": "SLSQP",
        "maxiter": MAXITER,
        "ftol": FTOL,

        "n_cases": n,

        "warm_success": success,
        "warm_not_worse_than_seed": not_worse,
        "warm_beats_historical_wide": beats_wide,
        "best_so_far_closes_gap": closes,

        "uses_truth_in_fitting": False,
        "uses_mid_validation": False,
        "E4_accessed": False,

        "source_sha256": {
            P80.name: sha(P80),
            P81.name: sha(P81),
            P77.name: sha(P77),
            A_REPORT.name: sha(A_REPORT),
            A_AUDIT.name: sha(A_AUDIT),
        },

        "limitations": [
            "Development data only",
            "Only 21 historical cases",
            "Local SLSQP optimization",
            "No global optimality certificate",
            "No independent evaluation",
            "No adaptive policy",
        ],
    }

    report_path = OUT / (
        f"p4m2c2b_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2c2b_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(
            summary,
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

    print("Saved:", report_path)
    print("Saved:", audit_path)

    print("\nP4-M2C2B COMPLETE")


if __name__ == "__main__":
    main()
