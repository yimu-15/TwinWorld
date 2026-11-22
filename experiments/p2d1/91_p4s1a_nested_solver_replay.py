"""P4-S1A — Frozen solver replay and nested-domain diagnosis.

Truth-blind *solver* selection: target cases are chosen only from the
frozen P90 v2 solver inconsistency / nonconvergence flags.

For each target and for the DEADZONE candidate alone:
  1) reconstruct identical P90 training probes and replay the original
     two-start, maxiter=150 fit in Wide and Outer Box;
  2) verify the replayed training MSE / validation RMS against P90 v2;
  3) add exactly three shared, deterministic, feasible starts to BOTH
     domains with the SAME per-start options;
  4) audit a feasible Outer-to-Wide objective witness, and record the
     remaining nested-domain optimizer gap and actual function calls.

This is NOT a revised model-selection benchmark. No classifier scores
are changed in P90. Full 384-case, all-candidate solver-budget parity
must be checked separately in P4-S1B before SCI claims.

No E4, no threshold selection, no refitting on MID/HALF/FULL.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.optimize import Bounds, minimize

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
INPUT_REPORT = RESULTS / "p4m2d2b" / "p4m2d2b_v2_fresh_report.json"
INPUT_AUDIT = RESULTS / "p4m2d2b" / "p4m2d2b_v2_fresh_audit_PRIVATE.json"
P80 = HERE / "80_p4m2c0_prior_mismatch_stress.py"
P90 = HERE / "90_p4m2d2b_amplitude_dose_response_v2.py"
OUT = RESULTS / "p4s1a"

LAW = "deadzone"
DOMAINS = ("wide", "outer_box")
EXTRA_FRACTIONS = (0.10, 0.50, 0.90)
PER_START_MAXITER = 150
FTOL = 1e-9
N_BASE = 24
EXPECTED_CASES = 384
GAP_TOL = 1e-6
REPLAY_ATOL = 1e-4
REPLAY_RTOL = 1e-5


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    require(path.is_file(), f"Missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_key(row):
    return (
        int(row["scene_id"]),
        row["band"],
        row["mode_PRIVATE"],
        row["axis_PRIVATE"],
        row["level_PRIVATE"],
    )


def check_frozen_inputs():
    report, rows = read(INPUT_REPORT), read(INPUT_AUDIT)
    require(report["smoke"] is False, "Formal P90 v2 required")
    require(report["n_cases"] == EXPECTED_CASES, "Case count changed")
    require(report["n_base_scenes"] == N_BASE, "Scene count changed")
    require(report["scene_seed"] == 20291117, "Scene seed changed")
    require(report["fit_noise_seed"] == 2029111701, "Training seed changed")
    require(report["E4_accessed"] is False, "E4 was used")
    require(report["action_used_for_fit"] is False, "Heldout action was fitted")
    require(report["threshold_tuning"] is False, "Threshold was tuned")
    require(sha(P80) == report["source_provenance"][P80.name],
            "Frozen P80 source mismatch")
    require(sha(P90) == report["source_provenance"][P90.name],
            "Frozen P90 v2 source mismatch")
    require(len(rows) == EXPECTED_CASES, "Missing records")
    require(len(set(map(case_key, rows))) == EXPECTED_CASES,
            "Duplicate case keys")
    counts = Counter(row["scene_id"] for row in rows)
    require(set(counts) == set(range(N_BASE)) and
            all(counts[x] == 16 for x in counts), "Incorrect condition grid")

    violated = []
    failures = []
    for row in rows:
        flags = row["nested_violations_PRIVATE"]
        for law in ("elastic", "deadzone", "cubic"):
            w = row["fit_status_PRIVATE"]["wide"][law]
            o = row["fit_status_PRIVATE"]["outer_box"][law]
            numeric_gap = (
                w["train_mse"] is not None and o["train_mse"] is not None
                and w["train_mse"] - o["train_mse"] > GAP_TOL
                and w["valid"] and o["valid"]
            )
            require(numeric_gap == (law in flags),
                    f"Nested flag mismatch at {case_key(row)} / {law}")
            if numeric_gap:
                violated.append((case_key(row), law))
            for domain, fit in (("wide", w), ("outer_box", o)):
                if not fit["available"]:
                    failures.append((case_key(row), domain, law))
    require(len(violated) == 36 and all(law == LAW for _, law in violated),
            "Frozen P90 v2 nested-violation manifest changed")
    require(len(failures) == 1 and failures[0][2] == LAW,
            "Frozen P90 v2 failure manifest changed")
    require(report["nested_solver_violations"] == {LAW: 36},
            "Source nested counts mismatch")
    require(report["rows_with_failed_candidates"] == 1,
            "Source failure counts mismatch")

    targets = [row for row in rows if LAW in row["nested_violations_PRIVATE"]
               or any(not row["fit_status_PRIVATE"][d][LAW]["available"]
                      for d in DOMAINS)]
    require(len(targets) == 37, "Expected 37 unique solver-target rows")
    return report, rows, sorted(targets, key=case_key)


def verify_nested_bounds(p77):
    lows = {}
    highs = {}
    for domain in DOMAINS:
        lo, hi = p77["specifications"](LAW, domain)
        lows[domain] = np.asarray(lo, dtype=float)
        highs[domain] = np.asarray(hi, dtype=float)
    require(np.all(lows["wide"] <= lows["outer_box"]) and
            np.all(highs["wide"] >= highs["outer_box"]),
            "Outer Box must be contained in Wide")
    require(tuple(p77["structure"](LAW)) == ("alpha", "beta", "gap"),
            "Unexpected deadzone parameterization")
    require(p77["MAXITER"] == PER_START_MAXITER and
            np.isclose(p77["SIGMA"], .005), "Solver protocol mismatch")
    require(tuple(p77["TRAIN"]) == ("e1", "e2", "weak", "strong")
            and p77["VALID"] == "mid", "Action protocol mismatch")
    return {d: {"lower": lows[d].tolist(), "upper": highs[d].tolist()}
            for d in DOMAINS}


def objective_factory(training, sigma):
    seen = [np.asarray(p.y_b, dtype=float).ravel() for p in training]

    def objective(q):
        q = np.asarray(q, dtype=float)
        residuals = []
        for probe, actual in zip(training, seen):
            predicted = np.asarray(probe.predict(LAW, q.tolist()),
                                   dtype=float).ravel()
            if predicted.shape != actual.shape or not np.all(np.isfinite(predicted)):
                return 1e30
            residuals.append((predicted - actual) / sigma)
        r = np.concatenate(residuals)
        return float(np.mean(r * r))

    return objective


def extra_start_points(p77):
    pr = p77["PHYSICAL_RANGES"]
    gl, gh = p77["GAP_RANGE"]
    mass = 1.2
    result = []
    for fraction in EXTRA_FRACTIONS:
        alpha = (pr["alpha"][0] + fraction *
                 (pr["alpha"][1] - pr["alpha"][0])) / mass
        beta = (pr["beta"][0] + fraction *
                (pr["beta"][1] - pr["beta"][0])) / mass
        gap = gl + fraction * (gh - gl)
        x0 = np.array([alpha, beta, gap], dtype=float)
        require(all(p77["is_feasible"](LAW, x0, domain)
                    for domain in DOMAINS), "Common extra start infeasible")
        result.append(x0)
    originals = p77["initial_points"](LAW)
    require(len(originals) == 2, "Original initial-point budget changed")
    for x0 in result:
        require(not any(np.allclose(x0, old) for old in originals),
                "Extra start duplicates original")
    return result


def recreate_case_probes(row, scenes, p80, p74, p75, p77,
                         sim, api, commands, noise_seed):
    sid, band, mode, axis, level = case_key(row)
    base = scenes[sid][band]
    if level == "M0":
        require(axis == "matched", "Unexpected M0 axis")
        scene = base
    else:
        scene, _ = p80["make_stress_scene"](
            p74, p77, base, mode, axis, level
        )
    probes = p75["create_noisy_probes"](
        sim, api, commands, scene, sid, band, mode, noise_seed
    )
    return [probes[name] for name in ("e1", "e2", "weak", "strong")], probes["mid"]


def numeric_replay(p77, p75, training, validation, row):
    originals = {}
    for domain in DOMAINS:
        new = p77["fit_domain"](training, LAW, domain)
        old = row["fit_status_PRIVATE"][domain][LAW]
        require(bool(new["valid"]) == old["valid"],
                f"Replay valid mismatch: {case_key(row)} {domain}")
        require(bool(new["converged"]) == old["converged"],
                f"Replay convergence mismatch: {case_key(row)} {domain}")
        old_mse = old["train_mse"]
        if old_mse is None:
            require(new["train_mse"] is None, "Replay expected no MSE")
        else:
            require(new["train_mse"] is not None and np.isclose(
                new["train_mse"], old_mse,
                atol=REPLAY_ATOL, rtol=REPLAY_RTOL),
                f"Replay MSE mismatch: {case_key(row)} {domain}: "
                f"{new['train_mse']} vs {old_mse}")
        if new["valid"]:
            mid_rms = float(p75["rms_mm"](
                validation.predict(LAW, new["params_PRIVATE"]),
                validation.y_b,
            ))
            require(np.isclose(mid_rms, old["validation_rms_mm"],
                               atol=REPLAY_ATOL, rtol=REPLAY_RTOL),
                    f"Replay validation mismatch: {case_key(row)} {domain}")
        originals[domain] = new
    return originals


def extra_optimizations(p77, objective, extra_starts):
    output = {}
    for domain in DOMAINS:
        lo, hi = p77["specifications"](LAW, domain)
        attempts = []
        for i, x0 in enumerate(extra_starts):
            res = minimize(
                objective, x0, method="SLSQP",
                bounds=Bounds(lo, hi),
                options={"maxiter": PER_START_MAXITER,
                         "ftol": FTOL, "disp": False},
            )
            feasible = bool(
                p77["is_feasible"](LAW, res.x, domain)
                and np.isfinite(res.fun)
                and np.all(np.isfinite(res.x))
            )
            attempts.append({
                "start_index": i,
                "start_params_PRIVATE": x0.tolist(),
                "valid": feasible,
                "converged": bool(res.success),
                "mse": float(res.fun) if np.isfinite(res.fun) else None,
                "nfev": int(res.nfev),
                "nit": int(res.nit),
                "status": int(res.status),
                "message": str(res.message),
                "params_PRIVATE": res.x.tolist() if np.all(np.isfinite(res.x)) else None,
            })
        output[domain] = attempts
    return output


def combine_best(originals, extra, objective, p77):
    combined = {}
    for domain in DOMAINS:
        a = originals[domain]
        candidates = []
        if a["valid"] and a["train_mse"] is not None:
            candidates.append({"origin": "original_two_starts",
                               "mse": float(a["train_mse"]),
                               "converged": bool(a["converged"]),
                               "params_PRIVATE": a["params_PRIVATE"]})
        for record in extra[domain]:
            if record["valid"] and record["mse"] is not None:
                candidates.append({"origin": f"extra_{record['start_index']}",
                                   "mse": record["mse"],
                                   "converged": record["converged"],
                                   "params_PRIVATE": record["params_PRIVATE"]})
        best_any = min(candidates, key=lambda x: x["mse"]) if candidates else None
        successes = [x for x in candidates if x["converged"]]
        best_success = min(successes, key=lambda x: x["mse"]) if successes else None
        for item in (best_any, best_success):
            if item is not None:
                require(p77["is_feasible"](LAW, item["params_PRIVATE"], domain),
                        "Infeasible best point")
                require(np.isclose(objective(item["params_PRIVATE"]), item["mse"],
                                   rtol=1e-6, atol=1e-5),
                        "Objective replay mismatch")
        combined[domain] = {"best_valid": best_any,
                            "best_converged": best_success,
                            "nfev_extra": sum(r["nfev"] for r in extra[domain]),
                            "nfev_original": int(a["nfev"]),
                            "nfev_combined": int(a["nfev"]) +
                            sum(r["nfev"] for r in extra[domain])}
    return combined


def evaluate_target(row, scenes, p80, p74, p75, p77,
                    sim, api, commands, fit_seed, extra_starts):
    training, validation = recreate_case_probes(
        row, scenes, p80, p74, p75, p77, sim, api, commands, fit_seed
    )
    objective = objective_factory(training, float(p77["SIGMA"]))
    originals = numeric_replay(p77, p75, training, validation, row)
    extras = extra_optimizations(p77, objective, extra_starts)
    combined = combine_best(originals, extras, objective, p77)

    outer_best = combined["outer_box"]["best_valid"]
    require(outer_best is not None, "Cannot form feasible outer witness")
    require(p77["is_feasible"](LAW, outer_best["params_PRIVATE"], "wide"),
            "Outer point not Wide-feasible")
    wide_at_outer = float(objective(outer_best["params_PRIVATE"]))
    require(np.isclose(wide_at_outer, outer_best["mse"],
                       atol=1e-5, rtol=1e-6), "Objective differs by domain")

    wide_best = combined["wide"]["best_valid"]
    new_gap = None if wide_best is None else float(
        wide_best["mse"] - outer_best["mse"]
    )
    old_w = row["fit_status_PRIVATE"]["wide"][LAW]["train_mse"]
    old_o = row["fit_status_PRIVATE"]["outer_box"][LAW]["train_mse"]
    old_gap = None if old_w is None or old_o is None else float(old_w-old_o)

    return {
        "key_PRIVATE": list(case_key(row)),
        "old_violation": LAW in row["nested_violations_PRIVATE"],
        "old_nonconverged": [d for d in DOMAINS if not
                             row["fit_status_PRIVATE"][d][LAW]["available"]],
        "original": {d: {
            "valid": bool(originals[d]["valid"]),
            "converged": bool(originals[d]["converged"]),
            "train_mse": originals[d]["train_mse"],
            "nfev": int(originals[d]["nfev"]),
            "params_PRIVATE": originals[d]["params_PRIVATE"],
        } for d in DOMAINS},
        "extra_attempts_PRIVATE": extras,
        "combined_PRIVATE": combined,
        "old_wide_minus_outer_mse": old_gap,
        "augmented_wide_minus_outer_mse": new_gap,
        "remaining_nested_violation": new_gap is not None and new_gap > GAP_TOL,
        "outer_point_is_feasible_in_wide": True,
        "wide_objective_at_outer_witness": wide_at_outer,
        "wide_feasible_upper_bound_mse": min(
            wide_at_outer, wide_best["mse"] if wide_best else float("inf")
        ),
    }


def summarise(records):
    old_violations = [x for x in records if x["old_violation"]]
    before = [x["old_wide_minus_outer_mse"] for x in records
              if x["old_wide_minus_outer_mse"] is not None]
    after = [x["augmented_wide_minus_outer_mse"] for x in records
             if x["augmented_wide_minus_outer_mse"] is not None]
    return {
        "n_target_cases": len(records),
        "original_violations_in_subset": len(old_violations),
        "original_unavailable_in_subset": sum(bool(x["old_nonconverged"])
                                              for x in records),
        "remaining_augmented_nested_violations": sum(
            x["remaining_nested_violation"] for x in records),
        "original_gap_positive_median": (
            float(np.median([x for x in before if x > GAP_TOL]))
            if any(x > GAP_TOL for x in before) else None),
        "augmented_gap_positive_median": (
            float(np.median([x for x in after if x > GAP_TOL]))
            if any(x > GAP_TOL for x in after) else None),
        "total_function_evaluations": {
            domain: sum(x["combined_PRIVATE"][domain]["nfev_combined"]
                        for x in records)
            for domain in DOMAINS
        },
        "cases_with_converged_augmented_wide": sum(
            x["combined_PRIVATE"]["wide"]["best_converged"] is not None
            for x in records),
        "cases_with_converged_augmented_outer": sum(
            x["combined_PRIVATE"]["outer_box"]["best_converged"] is not None
            for x in records),
        "feasible_outer_to_wide_witnesses": sum(
            bool(x["outer_point_is_feasible_in_wide"]) for x in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("P4-S1A — NESTED-DOMAIN SOLVER REPLAY / TARGETED DIAGNOSIS")
    print("=" * 72)
    print("Smoke:", args.smoke, "| No E4 | No classifier revision")
    report, rows, targets = check_frozen_inputs()
    print("[PASS] Frozen P90 v2: 384 cases, 36 gaps, 1 failed candidate")

    p80 = run_path(str(P80))
    p80["check_provenance"]()
    p74, p75, p77, p79, src, sim, api, commands = p80["initialize"]()
    bounds = verify_nested_bounds(p77)
    common_starts = extra_start_points(p77)
    print("[PASS] Outer bounds lie within Wide bounds")
    print("[SOLVER] 2 frozen + 3 extra common starts; 150 iter/start")

    scenes = src["make_scenes"](sim, report["scene_seed"], N_BASE)
    fps = p79["fingerprint_scenes"](scenes)
    require(fps == report["scene_independence"]["fingerprints"],
            "Scene reconstruction fingerprint mismatch")
    print("[PASS] Exact P90 scene fingerprint replay:", len(fps))

    if args.smoke:
        first_gap = next(x for x in targets if LAW in
                         x["nested_violations_PRIVATE"])
        first_failure = next(x for x in targets if any(
            not x["fit_status_PRIVATE"][d][LAW]["available"]
            for d in DOMAINS))
        selected = [first_gap] + ([first_failure]
                                  if case_key(first_failure) != case_key(first_gap)
                                  else [])
    else:
        selected = targets
    print("[TARGETS]", len(selected), "of 37 frozen manifest cases")

    records = []
    for index, row in enumerate(selected, 1):
        result = evaluate_target(
            row, scenes, p80, p74, p75, p77, sim, api, commands,
            report["fit_noise_seed"], common_starts
        )
        records.append(result)
        print(
            f"[{index:02d}/{len(selected)}] {tuple(result['key_PRIVATE'])} "
            f"gap {result['old_wide_minus_outer_mse']:+.6g} -> "
            f"{result['augmented_wide_minus_outer_mse']:+.6g} "
            f"remaining={result['remaining_nested_violation']}",
            flush=True,
        )

    summary = summarise(records)
    print("\n[SUMMARY]")
    for key, val in summary.items():
        print(key, val)

    OUT.mkdir(parents=True, exist_ok=True)
    name = "smoke" if args.smoke else "diagnostic"
    report_path = OUT / f"p4s1a_{name}_report.json"
    audit_path = OUT / f"p4s1a_{name}_audit_PRIVATE.json"

    out = {
        "protocol": "P4-S1A frozen target solver replay, NOT classification benchmark",
        "smoke": args.smoke,
        "target_manifest_size": len(targets),
        "processed_targets": len(records),
        "selection": "All 36 P90 nested violations plus the sole failed candidate",
        "selection_uses_truth_labels": False,
        "law": LAW,
        "domains": DOMAINS,
        "domain_bounds": bounds,
        "frozen_starts_per_domain": 2,
        "added_starts_per_domain": len(common_starts),
        "added_start_fractions": EXTRA_FRACTIONS,
        "maxiter_per_start": PER_START_MAXITER,
        "ftol": FTOL,
        "optimizer": "SLSQP",
        "solver_objective": "mean(((train prediction - train observation)/0.005)^2)",
        "summary": summary,
        "processed_keys_PRIVATE": [list(case_key(x)) for x in selected],
        "source_sha256": {
            INPUT_REPORT.name: sha(INPUT_REPORT),
            INPUT_AUDIT.name: sha(INPUT_AUDIT),
            P80.name: sha(P80),
            P90.name: sha(P90),
            Path(__file__).name: sha(Path(__file__)),
        },
        "E4_accessed": False,
        "new_actions_used_for_fitting": False,
        "truth_used_for_solver_selection": False,
        "model_selection_changed": False,
        "limitations": [
            "Targeted post-hoc solver diagnosis only; not an unbiased performance test",
            "All three additional starts are common and feasible in both domains",
            "SLSQP local solver does not certify global optimality",
            "Observed function evaluations differ even with matched start/iteration budgets",
            "Only Deadzone candidate was reoptimized in this stage",
            "Must run full 384-case all-law matched-budget P4-S1B before performance claims",
            "Original P90 v2 accuracy remains frozen",
        ],
    }
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding="utf-8")
    audit_path.write_text(json.dumps(records, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")
    print("Saved:", report_path)
    print("Saved:", audit_path)
    print("P4-S1A COMPLETE")


if __name__ == "__main__":
    main()
