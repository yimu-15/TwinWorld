"""P4-S1B — complete-case, all-law paired solver-budget sensitivity.

Reads frozen P90 v2 (384 rows) and P91 diagnostic; neither is modified.
Every law/domain gets the original two SLSQP starts plus exactly three
pre-fixed common additional starts, maxiter=150/extra start, ftol=1e-9.

Three independent reporting arms:
 FROZEN: immutable original P90 decisions.
 MATCHED5: best converged ORIGINAL result or 3 common new starts, no witness.
 WITNESS: diagnostic only, allow Wide to reuse a converged Outer result
          without optimizing it again. NOT a fair independent-fit baseline.

All observation data and fitting methods remain frozen; no E4, no truth
in fitting/selection of solver attempts. The test labels are used only
in offline evaluation. Paired HALF/FULL observations are never fitted.

The full run can be long: per-scene SHA-guarded atomic checkpoints allow
safe resumption with unchanged source and frozen inputs.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import scipy
from collections import Counter, defaultdict
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.optimize import Bounds, minimize

HERE = Path(__file__).resolve().parent
BASE = HERE / "results"
OUT = BASE / "p4s1b"
P80 = HERE / "80_p4m2c0_prior_mismatch_stress.py"
P90 = HERE / "90_p4m2d2b_amplitude_dose_response_v2.py"
P91 = HERE / "91_p4s1a_nested_solver_replay.py"
R90 = BASE / "p4m2d2b" / "p4m2d2b_v2_fresh_report.json"
A90 = BASE / "p4m2d2b" / "p4m2d2b_v2_fresh_audit_PRIVATE.json"
R91 = BASE / "p4s1a" / "p4s1a_diagnostic_report.json"
A91 = BASE / "p4s1a" / "p4s1a_diagnostic_audit_PRIVATE.json"

LAWS = ("elastic", "deadzone", "cubic")
DOMAINS = ("wide", "outer_box")
CONFIGS = ("WWW", "OOO", "WOO")
METHODS = ("MID", "HALF_ONLY", "FULL_ONLY", "MID_PLUS_HALF", "MID_PLUS_FULL")
ARMS = ("FROZEN", "MATCHED5", "WITNESS")
TRUTH = {"elastic": "elastic", "deadzone": "deadzone", "nonlinear": "cubic"}
START_FRACTIONS = (0.10, 0.50, 0.90)
MAXITER = 150
FTOL = 1e-9
GAP_TOL = 1e-6
REPLAY_ATOL = 1e-4
REPLAY_RTOL = 1e-5
BOOT_N = 2000
BOOT_SEED = 2029111901


def check(x, message):
    if not x:
        raise RuntimeError(message)


def read(path):
    check(path.is_file(), f"Missing frozen file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                              allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def key(row):
    return (int(row["scene_id"]), row["band"], row["mode_PRIVATE"],
            row["axis_PRIVATE"], row["level_PRIVATE"])


def provenance():
    r90, a90, r91, a91 = read(R90), read(A90), read(R91), read(A91)
    check(not r90["smoke"] and not r91["smoke"], "Formal reports required")
    check(r90["n_cases"] == 384 and len(a90) == 384, "P90 matrix changed")
    check(r91["processed_targets"] == 37 and len(a91) == 37,
          "P91 diagnostic incomplete")
    check(not r90["E4_accessed"] and not r91["E4_accessed"], "E4 flag mismatch")
    check(not r90["action_used_for_fit"] and not r91["truth_used_for_solver_selection"],
          "Historical fitting protocol flag")
    check(digest(P80) == r90["source_provenance"][P80.name], "P80 SHA mismatch")
    check(digest(P90) == r90["source_provenance"][P90.name], "P90 SHA mismatch")
    for path in (R90, A90, P80, P90, P91):
        check(path.name in r91["source_sha256"],
              f"P91 provenance missing: {path.name}")
        check(digest(path) == r91["source_sha256"][path.name],
              f"P91 provenance mismatch: {path.name}")
    check(digest(P91) == r91["source_sha256"][P91.name], "P91 SHA mismatch")
    check(len(set(map(key, a90))) == 384, "Duplicate P90 case keys")
    counts = Counter(int(row["scene_id"]) for row in a90)
    check(set(counts) == set(range(24)) and all(v == 16 for v in counts.values()),
          "P90 scene matrix changed")
    check(r90["nested_solver_violations"] == {"deadzone": 36},
          "P90 nested manifest changed")
    check(r90["rows_with_failed_candidates"] == 1,
          "P90 failure manifest changed")
    frozen = {key(r): r for r in a90}
    for diag in a91:
        k = tuple(diag["key_PRIVATE"])
        check(k in frozen, f"P91 target not in P90: {k}")
        check(diag["old_violation"] ==
              ("deadzone" in frozen[k]["nested_violations_PRIVATE"]),
              f"P91 target mismatch: {k}")
    manifest = {name: digest(path) for name, path in
                (("P80", P80), ("P90", P90), ("P91", P91),
                 ("R90", R90), ("A90", A90), ("R91", R91),
                 ("A91", A91), ("SELF", Path(__file__)))}
    return r90, frozen, manifest


def common_starts(p77, law):
    names = tuple(p77["structure"](law))
    out = []
    for f in START_FRACTIONS:
        vals = []
        for name in names:
            if name == "gap":
                lo, hi = p77["GAP_RANGE"]
                vals.append(float(lo + f * (hi - lo)))
            else:
                lo, hi = p77["PHYSICAL_RANGES"][name]
                vals.append(float((lo + f * (hi - lo)) / 1.2))
        x = np.array(vals, dtype=float)
        check(all(p77["is_feasible"](law, x, d) for d in DOMAINS),
              f"Invalid common start for {law}: {x}")
        check(not any(np.allclose(x, old) for old in p77["initial_points"](law)),
              f"Duplicated original start for {law}")
        out.append(x)
    return out


def objective(training, law, sigma):
    observations = [np.asarray(p.y_b, dtype=float).ravel() for p in training]
    def loss(q):
        q = np.asarray(q, dtype=float)
        residuals = []
        for p, actual in zip(training, observations):
            pred = np.asarray(p.predict(law, q.tolist()), dtype=float).ravel()
            if pred.shape != actual.shape or not np.isfinite(pred).all():
                return 1e30
            residuals.append((pred - actual) / sigma)
        r = np.concatenate(residuals)
        return float(np.mean(r * r))
    return loss


def extra_fit(loss, p77, law, domain, starts):
    lo, hi = p77["specifications"](law, domain)
    attempts = []
    for i, x0 in enumerate(starts):
        res = minimize(loss, x0, method="SLSQP", bounds=Bounds(lo, hi),
                       options={"maxiter": MAXITER, "ftol": FTOL, "disp": False})
        ok = bool(np.isfinite(res.fun) and np.all(np.isfinite(res.x))
                  and p77["is_feasible"](law, res.x, domain))
        attempts.append({
            "start": i, "valid": ok, "converged": bool(res.success),
            "mse": float(res.fun) if np.isfinite(res.fun) else None,
            "params_PRIVATE": res.x.tolist() if np.isfinite(res.x).all() else None,
            "nfev": int(res.nfev), "nit": int(res.nit),
            "status": int(res.status), "message": str(res.message),
        })
    return attempts


def selected_fit(original, attempts, loss, p77, law, domain):
    candidates = []
    if original["valid"] and original["train_mse"] is not None:
        candidates.append({"origin": "frozen_two_starts",
                           "converged": bool(original["converged"]),
                           "mse": float(original["train_mse"]),
                           "params_PRIVATE": original["params_PRIVATE"]})
    for item in attempts:
        if item["valid"] and item["mse"] is not None:
            candidates.append({"origin": f"extra_{item['start']}",
                               "converged": item["converged"],
                               "mse": item["mse"],
                               "params_PRIVATE": item["params_PRIVATE"]})
    feasible = [x for x in candidates if x["converged"]]
    chosen = min(feasible, key=lambda x: x["mse"]) if feasible else None
    if chosen is not None:
        check(p77["is_feasible"](law, chosen["params_PRIVATE"], domain),
              "Selected fit infeasible")
        check(np.isclose(loss(chosen["params_PRIVATE"]), chosen["mse"],
                         rtol=1e-6, atol=1e-5), "Selected objective mismatch")
    return chosen


def rms(p75, probe, law, params):
    pred = np.asarray(probe.predict(law, params), dtype=float)
    check(pred.shape == probe.y_b.shape and np.isfinite(pred).all(),
          "Prediction shape/finite check failed")
    val = float(p75["rms_mm"](pred, probe.y_b))
    check(np.isfinite(val) and val >= 0, "Nonfinite RMS")
    return val


def score(mid, half, full, method):
    if method == "MID": return mid
    if method == "HALF_ONLY": return half
    if method == "FULL_ONLY": return full
    if method == "MID_PLUS_HALF": return float(np.sqrt((mid**2 + half**2)/2.0))
    if method == "MID_PLUS_FULL": return float(np.sqrt((mid**2 + full**2)/2.0))
    raise ValueError(method)


def decisions_from(fitted, cfg):
    decisions = {}
    for method in METHODS:
        scores = {}
        for i, law in enumerate(LAWS):
            domain = "outer_box" if cfg[i] == "O" else "wide"
            item = fitted[domain][law]
            if item is None:
                scores = None
                break
            scores[law] = score(item["mid"], item["half"], item["full"], method)
        decisions[method] = {"available": scores is not None,
                             "winner": min(LAWS, key=lambda x: scores[x])
                             if scores is not None else None}
    return decisions


def eval_case(row, scenes, p80, p90, p91, p74, p75, p77, sim, api,
              commands, fit_seed, high_command, starts):
    sid, band, mode, axis, level = key(row)
    training, validation = p91["recreate_case_probes"](
        row, scenes, p80, p74, p75, p77, sim, api, commands, fit_seed)
    base = scenes[sid][band]
    if level == "M0":
        check(axis == "matched", "M0 axis invalid")
        scene = base
    else:
        scene, _ = p80["make_stress_scene"](p74, p77, base, mode, axis, level)
    paired = p90["pair_probes"](
        scene, sid, band, mode, sim, api, high_command, float(p77["SIGMA"]))
    fits = {d: {} for d in DOMAINS}
    frozen_comp = {d: {} for d in DOMAINS}
    matched_comp = {d: {} for d in DOMAINS}
    fit_audit = {d: {} for d in DOMAINS}
    for law in LAWS:
        fun = objective(training, law, float(p77["SIGMA"]))
        for d in DOMAINS:
            old = p77["fit_domain"](training, law, d)
            hist = row["fit_status_PRIVATE"][d][law]
            check(bool(old["valid"]) == hist["valid"] and
                  bool(old["converged"]) == hist["converged"],
                  f"Frozen status replay mismatch {key(row)}/{d}/{law}")
            check(old["train_mse"] is not None and np.isclose(
                  old["train_mse"], hist["train_mse"],
                  atol=REPLAY_ATOL, rtol=REPLAY_RTOL),
                  f"Frozen train MSE replay mismatch {key(row)}/{d}/{law}")
            if old["valid"]:
                old_mid = rms(p75, validation, law, old["params_PRIVATE"])
                check(np.isclose(old_mid, hist["validation_rms_mm"],
                                 atol=REPLAY_ATOL, rtol=REPLAY_RTOL),
                      f"Frozen MID replay mismatch {key(row)}/{d}/{law}")
            attempts = extra_fit(fun, p77, law, d, starts[law])
            chosen = selected_fit(old, attempts, fun, p77, law, d)
            if chosen is not None:
                vals = {"mid": rms(p75, validation, law, chosen["params_PRIVATE"]),
                        "half": rms(p75, paired["HALF"], law, chosen["params_PRIVATE"]),
                        "full": rms(p75, paired["FULL"], law, chosen["params_PRIVATE"])}
                matched_comp[d][law] = {**chosen, **vals}
            else:
                matched_comp[d][law] = None
            if hist["available"]:
                old_params = old["params_PRIVATE"]
                half = rms(p75, paired["HALF"], law, old_params)
                full = rms(p75, paired["FULL"], law, old_params)
                check(np.isclose(half, row["heldout_rms_PRIVATE"]["HALF"][d][law],
                                 atol=REPLAY_ATOL, rtol=REPLAY_RTOL)
                      and np.isclose(full, row["heldout_rms_PRIVATE"]["FULL"][d][law],
                                     atol=REPLAY_ATOL, rtol=REPLAY_RTOL),
                      f"Frozen heldout parity mismatch {key(row)}/{d}/{law}")
                frozen_comp[d][law] = {"mid": old_mid, "half": half, "full": full}
            else:
                frozen_comp[d][law] = None
            fit_audit[d][law] = {
                "original": {"valid": old["valid"], "converged": old["converged"],
                             "train_mse": old["train_mse"], "nfev": old["nfev"]},
                "extra": attempts,
                "selected": chosen,
                "extra_nfev": sum(x["nfev"] for x in attempts),
                "total_nfev": int(old["nfev"])+sum(x["nfev"] for x in attempts),
            }
    witness_comp = {d: dict(items) for d, items in matched_comp.items()}
    nested = {}
    for law in LAWS:
        wide = matched_comp["wide"][law]
        outer = matched_comp["outer_box"][law]
        gap = None if wide is None or outer is None else float(wide["mse"] - outer["mse"])
        witness = outer is not None and p77["is_feasible"](
            law, outer["params_PRIVATE"], "wide")
        check(outer is None or witness, "Outer fit not feasible in Wide")
        if outer is not None:
            # Never call an optimizer here. This is a diagnostic transfer of
            # a known feasible parameter point; not an equal-cost WIDE fit.
            if wide is None or outer["mse"] < wide["mse"]:
                witness_comp["wide"][law] = {**outer, "origin": "outer_witness"}
        nested[law] = {"matched5_gap": gap,
                       "matched5_violation": gap is not None and gap > GAP_TOL,
                       "outer_feasible_in_wide": bool(witness),
                       "witness_gap": (None if witness_comp["wide"][law] is None
                                       or outer is None else
                                       witness_comp["wide"][law]["mse"] - outer["mse"]),
                       "witness_transferred": witness_comp["wide"][law] is not None
                       and witness_comp["wide"][law].get("origin") == "outer_witness"}
    decisions = {"FROZEN": {}, "MATCHED5": {}, "WITNESS": {}}
    for cfg in CONFIGS:
        for arm, data in (("FROZEN", frozen_comp), ("MATCHED5", matched_comp),
                          ("WITNESS", witness_comp)):
            decisions[arm][cfg] = decisions_from(data, cfg)
        for method in METHODS:
            old = row["decisions"][cfg][method]
            calc = decisions["FROZEN"][cfg][method]
            check(calc["available"] == old["available"] and
                  calc["winner"] == old["winner"],
                  f"Frozen selection parity mismatch: {key(row)} {cfg} {method}")
    return {"key_PRIVATE": list(key(row)), "decisions": decisions,
            "fit_audit_PRIVATE": fit_audit, "nested_audit_PRIVATE": nested,
            "legacy_nested_violations_PRIVATE": row["nested_violations_PRIVATE"]}


def summarize(rows):
    metrics = {}
    totals = len(rows)
    for arm in ARMS:
        metrics[arm] = {}
        for cfg in CONFIGS:
            metrics[arm][cfg] = {}
            for method in METHODS:
                available = [r for r in rows if r["decisions"][arm][cfg][method]["available"]]
                hits = sum(r["decisions"][arm][cfg][method]["winner"] ==
                           TRUTH[r["key_PRIVATE"][2]] for r in available)
                metrics[arm][cfg][method] = {"correct": hits, "n": totals,
                    "available": len(available), "accuracy_all": hits/totals,
                    "coverage": len(available)/totals,
                    "accuracy_available": hits/len(available) if available else None}
    nonlinear = [r for r in rows if r["key_PRIVATE"][2] == "nonlinear"
                 and r["key_PRIVATE"][4] == "M2"]
    check(len(nonlinear) in (2,96), "Nonlinear-M2 count incorrect")
    primary = {}
    for arm in ARMS:
        def acc(method):
            return sum(r["decisions"][arm]["WOO"][method]["winner"] == "cubic"
                       for r in nonlinear) / len(nonlinear)
        cov = sum(r["decisions"][arm]["WOO"]["MID_PLUS_FULL"]["available"]
                  for r in nonlinear)
        check(cov == sum(r["decisions"][arm]["WOO"]["MID_PLUS_HALF"]["available"]
                         for r in nonlinear), "Paired availability mismatch")
        primary[arm] = {"full_minus_half": acc("MID_PLUS_FULL")-acc("MID_PLUS_HALF"),
                        "n": len(nonlinear), "available": cov, "coverage": cov/len(nonlinear)}
    nested_counts = {law: sum(r["nested_audit_PRIVATE"][law]["matched5_violation"]
                              for r in rows) for law in LAWS}
    prior_violations = {law: sum(law in r["legacy_nested_violations_PRIVATE"]
                                 for r in rows) for law in LAWS}
    return metrics, primary, nested_counts, prior_violations


def bootstrap(rows):
    cluster = defaultdict(list)
    for r in rows:
        cluster[r["key_PRIVATE"][0]].append(r)
    check(set(cluster) == set(range(24)), "Bootstrap missing base scene")
    arrays = {arm: [] for arm in ARMS}
    rng = np.random.default_rng(BOOT_SEED)
    ids = np.array(sorted(cluster))
    for _ in range(BOOT_N):
        sample = [r for sid in rng.choice(ids, size=24, replace=True)
                  for r in cluster[int(sid)]
                  if r["key_PRIVATE"][2] == "nonlinear" and r["key_PRIVATE"][4] == "M2"]
        check(len(sample) == 96, "Incorrect bootstrap Nonlinear-M2 denominator")
        for arm in ARMS:
            hits_full = sum(r["decisions"][arm]["WOO"]["MID_PLUS_FULL"]["winner"] == "cubic"
                            for r in sample)
            hits_half = sum(r["decisions"][arm]["WOO"]["MID_PLUS_HALF"]["winner"] == "cubic"
                            for r in sample)
            arrays[arm].append((hits_full - hits_half)/96.0)
    return {"unit": "base_scene_id", "n_clusters": 24, "n_resamples": BOOT_N,
            "seed": BOOT_SEED,
            "primary_95CI": {arm: [float(np.quantile(v,.025)),
                                   float(np.quantile(v,.975))]
                             for arm,v in arrays.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    print("="*72)
    print("P4-S1B — FULL 384-CASE ALL-LAW MATCHED SOLVER BUDGET")
    print("="*72)
    print("Smoke:", args.smoke, "| E4 sealed | P90 remains frozen")
    r90, frozen, manifest = provenance()
    p80 = run_path(str(P80))
    p80["check_provenance"]()
    p90 = run_path(str(P90))
    p91 = run_path(str(P91))
    p74,p75,p77,p79,src,sim,api,commands = p80["initialize"]()
    check(int(p77["MAXITER"]) == MAXITER and np.isclose(p77["SIGMA"], .005),
          "Frozen optimizer/noise mismatch")
    check(tuple(p77["TRAIN"]) == ("e1", "e2", "weak", "strong")
          and p77["VALID"] == "mid", "Frozen action protocol changed")
    check(all(len(p77["initial_points"](law)) == 2 for law in LAWS),
          "Historical start budget changed")
    for law in LAWS:
        lo_w,hi_w = p77["specifications"](law,"wide")
        lo_o,hi_o = p77["specifications"](law,"outer_box")
        check(np.all(lo_w <= lo_o) and np.all(hi_w >= hi_o),
              f"Domain nesting changed: {law}")
    starts = {law: common_starts(p77, law) for law in LAWS}
    scenes = src["make_scenes"](sim,r90["scene_seed"],24)
    fps = p79["fingerprint_scenes"](scenes)
    check(fps == r90["scene_independence"]["fingerprints"],
          "P90 exact scene fingerprint replay mismatch")
    print("[PASS] Sources, action/noise, all three nested-domain bounds")
    print("[PASS] 48 exact scene fingerprints reconstructed")
    print("[SOLVER] 2 original + 3 common additional starts for 3 laws x 2 domains")
    high_command = np.asarray(commands["high"], dtype=float)
    check(high_command.ndim == 1, "HIGH command invalid")
    manifest_path = OUT / "p4s1b_source_manifest.json"
    if manifest_path.is_file():
        check(read(manifest_path)["source_sha256"] == manifest,
              "Checkpoint source manifest changed; do not mix runs")
    elif not args.smoke:
        atomic_json(manifest_path, {"source_sha256": manifest,
                  "protocol": "P4-S1B original2+extra3; all laws, all conditions"})
    rows = []
    if args.smoke:
        selection = [k for k in sorted(frozen) if k[0] == 0 and k[1] == "original"
                     and k[2] == "nonlinear" and k[4] == "M2"][:1]
        print("[SMOKE] Historical scene 0; one case, not a scientific subset")
        for k in selection:
            rows.append(eval_case(frozen[k],scenes,p80,p90,p91,p74,p75,p77,
                                  sim,api,commands,r90["fit_noise_seed"],
                                  high_command,starts))
        r = rows[0]
        check(len(r["decisions"]) == 3, "Smoke decisions incomplete")
        print("[PASS] Frozen parity / new prediction / three arms")
        print("Nested gaps:", {law: r["nested_audit_PRIVATE"][law]["matched5_gap"]
                                for law in LAWS})
        atomic_json(OUT / "p4s1b_smoke_report.json", {
            "protocol": "P4-S1B interface smoke: one previously observed P90 case",
            "smoke": True, "n_cases": len(rows), "full_inference": False,
            "source_sha256": manifest, "E4_accessed": False,
            "not_for_scientific_effect_estimation": True})
        atomic_json(OUT / "p4s1b_smoke_audit_PRIVATE.json", rows)
        print("Saved:", OUT / "p4s1b_smoke_report.json")
        print("Saved:", OUT / "p4s1b_smoke_audit_PRIVATE.json")
        print("[PASS] Smoke complete. Run without --smoke for 384 cases.")
        return
    for sid in range(24):
        path = OUT / "checkpoints" / f"scene_{sid:02d}.json"
        if path.is_file():
            saved = read(path)
            check(saved["source_sha256"] == manifest
                  and saved["scene_id"] == sid and len(saved["rows"]) == 16,
                  f"Checkpoint incompatible: {path}")
            chunk = saved["rows"]
            print(f"[RESUME] scene {sid+1}/24: 16 previously audited cases", flush=True)
        else:
            chunk = []
            for k in sorted(frozen):
                if k[0] == sid:
                    chunk.append(eval_case(frozen[k],scenes,p80,p90,p91,
                        p74,p75,p77,sim,api,commands,r90["fit_noise_seed"],
                        high_command,starts))
            check(len(chunk) == 16, f"Scene {sid} incomplete")
            atomic_json(path, {"scene_id":sid, "source_sha256":manifest,
                               "rows":chunk})
            print(f"[DONE] scene {sid+1}/24: 16 cases / 96 frozen+extra fits",
                  flush=True)
        rows.extend(chunk)
    check(len(rows) == 384 and len(set(tuple(r["key_PRIVATE"]) for r in rows)) == 384,
          "Incomplete full case grid")
    metrics, primary, gaps, prior_gaps = summarize(rows)
    check(metrics["FROZEN"]["WOO"]["MID_PLUS_FULL"]["correct"] == 351
          and metrics["FROZEN"]["OOO"]["MID_PLUS_FULL"]["correct"] == 334
          and metrics["FROZEN"]["WWW"]["MID_PLUS_FULL"]["correct"] == 308,
          "Frozen P90 total accuracy counts changed")
    check(np.isclose(primary["FROZEN"]["full_minus_half"], 19.0/96.0,
                     rtol=0, atol=1e-12),
          "Frozen P90 primary changed")
    check(prior_gaps == {"elastic":0,"deadzone":36,"cubic":0},
          "Historical nested gap count not reproduced")
    ci = bootstrap(rows)
    print("\nFULL 384 CASES / PRIMARY NONLINEAR-M2")
    for arm in ARMS:
        print(f"{arm:>9}: Full−Half={primary[arm]['full_minus_half']:+.6f}; "
              f"coverage={primary[arm]['available']}/{primary[arm]['n']}; "
              f"CI={ci['primary_95CI'][arm]}")
    print("Nested gaps: original",prior_gaps,"matched5",gaps)
    nfev = {d:sum(r["fit_audit_PRIVATE"][d][law]["total_nfev"]
                  for r in rows for law in LAWS) for d in DOMAINS}
    nfev_extra = {d:sum(r["fit_audit_PRIVATE"][d][law]["extra_nfev"]
                        for r in rows for law in LAWS) for d in DOMAINS}
    print("Function evaluations (both budgets):",nfev)
    print("Function evaluations (extra only):",nfev_extra)
    for cfg in CONFIGS:
        for arm in ARMS:
            res = metrics[arm][cfg]["MID_PLUS_FULL"]
            print(f"{arm}/{cfg} MID+FULL: {res['correct']}/384 "
                  f"acc={res['accuracy_all']:.4f} coverage={res['coverage']:.4f}")
    decision_changes = {
        arm: {cfg: {method: sum(
            r["decisions"][arm][cfg][method]["winner"] !=
            r["decisions"]["FROZEN"][cfg][method]["winner"] for r in rows)
            for method in METHODS} for cfg in CONFIGS}
        for arm in ("MATCHED5", "WITNESS")}
    witness_counts = {law: sum(r["nested_audit_PRIVATE"][law]
                                ["witness_transferred"] for r in rows)
                      for law in LAWS}
    report = {"protocol":"P4-S1B full all-law matched solver-budget sensitivity",
        "smoke":False, "n_base_scenes":24,"n_cases":len(rows),
        "source_sha256":manifest,"starts":{"historical":2,"common_extra":3,
        "fractions":START_FRACTIONS,"maxiter_per_start":MAXITER,"ftol":FTOL},
        "arms":{"FROZEN":"Unmodified P90 classification",
                "MATCHED5":"Best converged from original2+extra3 per law/domain",
                "WITNESS":"Diagnostic only: Wide may use Outer converged feasible point"},
        "metrics":metrics,"primary":primary,"bootstrap":ci,
        "nested_violations":{"original":prior_gaps,"matched5":gaps},
        "solver_nfev_total":nfev,"solver_nfev_extra":nfev_extra,
        "decision_changes_vs_frozen":decision_changes,
        "witness_transfers":witness_counts,
        "environment":{"python":sys.version,"platform":platform.platform(),
                       "numpy":np.__version__,"scipy":scipy.__version__},
        "E4_accessed":False,"truth_used_for_fitting":False,
        "heldout_used_for_fitting":False,"source_modified":False,
        "limitations":["All scenarios from same frozen simulator family",
          "Original P90 384-case results kept unchanged",
          "Extra starts selected after observing P91 solver failures; sensitivity analysis, not independent preregistered benchmark",
          "Same starts and maxiter, NOT identical actual nfev budgets",
          "Witness transfer violates independent-domain fit-budget parity and must not headline",
          "Finite-start SLSQP cannot establish global optimality",
          "Historical best unconverged result excluded in MATCHED5 unless a converged candidate exists",
          "Truth used only to compute offline classification metrics"]}
    atomic_json(OUT/"p4s1b_fresh_report.json",report)
    atomic_json(OUT/"p4s1b_fresh_audit_PRIVATE.json",rows)
    print("Saved:",OUT/"p4s1b_fresh_report.json")
    print("Saved:",OUT/"p4s1b_fresh_audit_PRIVATE.json")
    print("P4-S1B COMPLETE")


if __name__ == "__main__":
    main()
