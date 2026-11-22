"""P4-S1C (No-simulation) solver-sensitivity closure for the frozen P90/P92 data.

Reads P90/P92 reports and PRIVATE case audits, P92 checkpoint files and
frozen source files.  Does not import MuJoCo, fit parameters, simulate or
use labels in model choice. Labels are used only in offline scoring.

Outputs separate JSON files under results/p4s1c/. Does not rewrite history.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
R90 = RES / "p4m2d2b" / "p4m2d2b_v2_fresh_report.json"
A90 = RES / "p4m2d2b" / "p4m2d2b_v2_fresh_audit_PRIVATE.json"
R92 = RES / "p4s1b" / "p4s1b_fresh_report.json"
A92 = RES / "p4s1b" / "p4s1b_fresh_audit_PRIVATE.json"
M92 = RES / "p4s1b" / "p4s1b_source_manifest.json"
CHECKPOINTS = RES / "p4s1b" / "checkpoints"
OUT = RES / "p4s1c"
SOURCE = {
    "P80": HERE / "80_p4m2c0_prior_mismatch_stress.py",
    "P90": HERE / "90_p4m2d2b_amplitude_dose_response_v2.py",
    "P91": HERE / "91_p4s1a_nested_solver_replay.py",
    "P92": HERE / "92_p4s1b_full_solver_budget_audit.py",
    "R90": R90,
    "A90": A90,
    "R91": RES / "p4s1a" / "p4s1a_diagnostic_report.json",
    "A91": RES / "p4s1a" / "p4s1a_diagnostic_audit_PRIVATE.json",
}
LAWS = ("elastic", "deadzone", "cubic")
CONFIGS = ("WWW", "OOO", "WOO")
METHODS = ("MID", "HALF_ONLY", "FULL_ONLY", "MID_PLUS_HALF", "MID_PLUS_FULL")
ARMS = ("FROZEN", "MATCHED5", "WITNESS")
TRUTH = {"elastic": "elastic", "deadzone": "deadzone", "nonlinear": "cubic"}
TOL = 1e-6


def ensure(test, msg):
    if not test:
        raise RuntimeError(msg)


def read(path):
    ensure(path.is_file(), f"Missing required file: {path}")
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(value, fp, indent=2, ensure_ascii=False, allow_nan=False)


def idx(row):
    return tuple(row["key_PRIVATE"])


def verify_provenance(r90, r92):
    ensure(r90["smoke"] is False and r92["smoke"] is False, "Formal inputs required")
    ensure(r92["n_cases"] == 384 and r92["n_base_scenes"] == 24, "Wrong P92 case grid")
    ensure(r90["n_cases"] == 384, "Wrong P90 case grid")
    ensure(r90["E4_accessed"] is False and r92["E4_accessed"] is False, "E4 flag mismatch")
    ensure(r92["source_modified"] is False, "P92 source-modification flag mismatch")
    expected = r92["source_sha256"]
    for k, path in SOURCE.items():
        ensure(sha(path) == expected[k if k != "P92" else "SELF"], f"SHA mismatch: {k}")
    manifest = read(M92)
    ensure(manifest["source_sha256"] == expected, "Manifest mismatch")
    print("[PASS] Source hashes, immutable input reports, E4 sealed")
    return {k: sha(path) for k, path in SOURCE.items()}


def verify_rows(a90, a92, r92, check_checkpoints):
    ensure(len(a90) == len(a92) == 384, "Case count not 384")
    old = { (int(row["scene_id"]), row["band"], row["mode_PRIVATE"],
              row["axis_PRIVATE"], row["level_PRIVATE"]): row for row in a90 }
    fresh = {idx(row): row for row in a92}
    ensure(len(old) == len(fresh) == 384 and set(old) == set(fresh), "Missing/duplicate keys")
    ensure(set(k[0] for k in fresh) == set(range(24)), "Incomplete scenes")
    ensure(all(sum(k[0] == scene for k in fresh) == 16 for scene in range(24)),
           "A scene does not have 16 cases")
    for k, row in fresh.items():
        frozen = old[k]
        ensure(row["legacy_nested_violations_PRIVATE"] == frozen["nested_violations_PRIVATE"],
               f"Historical violation mismatch {k}")
        for c in CONFIGS:
            for m in METHODS:
                v90 = frozen["decisions"][c][m]
                v92 = row["decisions"]["FROZEN"][c][m]
                ensure(v90["winner"] == v92["winner"] and
                       bool(v90["available"]) == bool(v92["available"]),
                       f"Frozen decision mismatch {k} {c} {m}")
        for domain in ("wide", "outer_box"):
            for law in LAWS:
                hist = frozen["fit_status_PRIVATE"][domain][law]
                fit = row["fit_audit_PRIVATE"][domain][law]
                prev = fit["original"]
                ensure(bool(hist["valid"]) == bool(prev["valid"]) and
                       bool(hist["converged"]) == bool(prev["converged"]),
                       f"Original solver status mismatch {k} {domain} {law}")
                ensure(np.isclose(hist["train_mse"], prev["train_mse"],
                                  rtol=1e-8, atol=1e-9),
                       f"Original solver objective mismatch {k} {domain} {law}")
    if check_checkpoints:
        for scene in range(24):
            cp = read(CHECKPOINTS / f"scene_{scene:02d}.json")
            ensure(cp["source_sha256"] == r92["source_sha256"],
                   f"Checkpoint provenance mismatch scene={scene}")
            ensure(cp["scene_id"] == scene and len(cp["rows"]) == 16,
                   f"Checkpoint size mismatch scene={scene}")
            for row in cp["rows"]:
                ensure(row == fresh[idx(row)], f"Checkpoint/audit row mismatch: {idx(row)}")
    print("[PASS] 384 row identity, historical solver and decisions, 24 checkpoints")
    return fresh


def metrics(rows):
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for config in CONFIGS:
            out[arm][config] = {}
            for method in METHODS:
                avail = 0
                hits = 0
                for row in rows:
                    dec = row["decisions"][arm][config][method]
                    avail += int(dec["available"])
                    hits += int(dec["available"] and dec["winner"] == TRUTH[idx(row)[2]])
                out[arm][config][method] = {
                    "correct": hits, "n": len(rows), "available": avail,
                    "accuracy_all": hits / len(rows), "coverage": avail / len(rows),
                    "accuracy_available": hits / avail if avail else None,
                }
    return out


def primary(rows):
    sample = [row for row in rows if idx(row)[2] == "nonlinear" and idx(row)[4] == "M2"]
    ensure(len(sample) == 96, "Invalid primary sample size")
    results = {}
    for arm in ARMS:
        pairs = collections.Counter()
        for row in sample:
            dec = row["decisions"][arm]["WOO"]
            ensure(dec["MID_PLUS_HALF"]["available"] and
                   dec["MID_PLUS_FULL"]["available"], "Primary pair unavailable")
            half = dec["MID_PLUS_HALF"]["winner"] == "cubic"
            full = dec["MID_PLUS_FULL"]["winner"] == "cubic"
            pairs[(half, full)] += 1
        results[arm] = {
            "n": len(sample), "coverage": 1.0,
            "half_correct": pairs[(True, True)] + pairs[(True, False)],
            "full_correct": pairs[(True, True)] + pairs[(False, True)],
            "full_minus_half": (pairs[(False, True)] - pairs[(True, False)]) / 96,
            "fixed_correct": pairs[(True, True)],
            "corrected_by_full": pairs[(False, True)],
            "harmed_by_full": pairs[(True, False)],
            "both_wrong": pairs[(False, False)],
        }
    return results


def ci_replay(rows, r92):
    seed = r92["bootstrap"]["seed"]
    n_boot = r92["bootstrap"]["n_resamples"]
    ensure(seed == 2029111901 and n_boot == 2000, "Bootstrap protocol mismatch")
    per_scene = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for scene in range(24):
            grp = [row for row in rows if idx(row)[0] == scene and
                   idx(row)[2] == "nonlinear" and idx(row)[4] == "M2"]
            ensure(len(grp) == 4, f"Missing primary case scene {scene}")
            per_scene[arm][scene] = sum(
                int(row["decisions"][arm]["WOO"]["MID_PLUS_FULL"]["winner"] == "cubic") -
                int(row["decisions"][arm]["WOO"]["MID_PLUS_HALF"]["winner"] == "cubic")
                for row in grp
            )
    rng = np.random.default_rng(seed)
    sampled = {arm: [] for arm in ARMS}
    ids = np.arange(24)
    for _ in range(n_boot):
        draw = rng.choice(ids, size=24, replace=True)
        for arm in ARMS:
            sampled[arm].append(sum(per_scene[arm][int(scene)] for scene in draw) / 96)
    out = {arm: [float(np.quantile(sampled[arm], q)) for q in (0.025, 0.975)]
           for arm in ARMS}
    for arm in ARMS:
        ensure(np.allclose(out[arm], r92["bootstrap"]["primary_95CI"][arm],
                           rtol=0, atol=1e-12), f"Bootstrap CI mismatch: {arm}")
    return out


def nested_and_transitions(rows):
    old = set()
    new = set()
    by_law = collections.Counter()
    per_scene = collections.defaultdict(set)
    domain_feasible = 0
    witness_counts = collections.Counter()
    for row in rows:
        k = idx(row)
        for law in LAWS:
            v = row["nested_audit_PRIVATE"][law]
            f_w = row["fit_audit_PRIVATE"]["wide"][law]["selected"]
            f_o = row["fit_audit_PRIVATE"]["outer_box"][law]["selected"]
            if f_w is not None and f_o is not None:
                gap = float(f_w["mse"]) - float(f_o["mse"])
                ensure(np.isclose(gap, v["matched5_gap"], rtol=0, atol=1e-7),
                       f"Nested gap mismatch {k}/{law}")
                ensure(bool(gap > TOL) == bool(v["matched5_violation"]),
                       f"Nested sign mismatch {k}/{law}")
            ensure(v["outer_feasible_in_wide"] is True, f"Non-nested domains {k}/{law}")
            domain_feasible += 1
            ensure(v["witness_gap"] is None or v["witness_gap"] <= TOL,
                   f"Witness gap remains {k}/{law}")
            witness_counts[law] += int(v["witness_transferred"])
            if law in row["legacy_nested_violations_PRIVATE"]:
                old.add((k, law))
            if v["matched5_violation"]:
                new.add((k, law))
                by_law[law] += 1
                per_scene[k[0]].add(law)
    delta = {
        "original": len(old), "matched5": len(new),
        "resolved": len(old-new), "newly_appearing": len(new-old),
        "persistent": len(old & new),
        "matched5_by_law": dict(by_law), "affected_scenes": len(per_scene),
        "feasible_outer_witnesses": domain_feasible,
        "witness_transfers_diagnostic_only": dict(witness_counts),
    }
    ensure(len(old) == 36 and len(new) == 33 and len(old-new) == 11 and
           len(new-old) == 8, "Nested-gap reconciliation mismatch")
    return delta, [{"case_PRIVATE": list(k), "law":law,
                    "status": "resolved" if (k,law) in old-new else "new"
                   } for (k,law) in sorted((old-new)|(new-old))]


def transitions(rows):
    output = {}
    details = []
    for arm in ("MATCHED5", "WITNESS"):
        output[arm] = {}
        for cfg in CONFIGS:
            output[arm][cfg] = {}
            for meth in METHODS:
                changed = []
                for row in rows:
                    before = row["decisions"]["FROZEN"][cfg][meth]["winner"]
                    after = row["decisions"][arm][cfg][meth]["winner"]
                    if before != after:
                        changed.append((idx(row), before, after))
                        details.append({"case_PRIVATE": list(idx(row)), "arm":arm,
                                        "config":cfg, "method":meth,
                                        "old":before, "new":after})
                output[arm][cfg][meth] = len(changed)
    return output, details


def nfev(rows):
    result = {}
    for domain in ("wide", "outer_box"):
        result[domain] = {
            "total": sum(row["fit_audit_PRIVATE"][domain][law]["total_nfev"]
                         for row in rows for law in LAWS),
            "extra": sum(row["fit_audit_PRIVATE"][domain][law]["extra_nfev"]
                         for row in rows for law in LAWS),
        }
    return result


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--smoke", action="store_true")
    mode = args.parse_args().smoke
    print("="*70)
    print("P4-S1C — READ-ONLY SOLVER ROBUSTNESS CLOSURE")
    print("="*70)
    r90, r92 = read(R90), read(R92)
    provenance = verify_provenance(r90, r92)
    a90, a92 = read(A90), read(A92)
    mapped = verify_rows(a90, a92, r92, check_checkpoints=not mode)
    rows = [mapped[k] for k in sorted(mapped)]
    recalc = metrics(rows)
    for arm in ARMS:
        for cfg in CONFIGS:
            for meth in METHODS:
                x, y = recalc[arm][cfg][meth], r92["metrics"][arm][cfg][meth]
                ensure(x["correct"] == y["correct"] and x["available"] == y["available"],
                       f"Classification count mismatch: {arm}/{cfg}/{meth}")
    pp = primary(rows)
    for arm in ARMS:
        ensure(np.isclose(pp[arm]["full_minus_half"],
                          r92["primary"][arm]["full_minus_half"], atol=1e-12),
               f"Primary difference mismatch: {arm}")
    print("[PASS] 45 classification cells, all three 96-case primary contrasts")
    # Smoke tests source and parity, but not final interpretation artifacts.
    if mode:
        print("[PASS] Smoke read-only input audit, no new fits or simulations")
        return
    ci = ci_replay(rows, r92)
    nesting, gap_changes = nested_and_transitions(rows)
    counts, changed_rows = transitions(rows)
    ensure(counts == r92["decision_changes_vs_frozen"], "Decision change totals mismatch")
    total_nfev = nfev(rows)
    for d in ("wide", "outer_box"):
        ensure(total_nfev[d]["total"] == r92["solver_nfev_total"][d] and
               total_nfev[d]["extra"] == r92["solver_nfev_extra"][d],
               f"Function evaluation sum mismatch {d}")
    ratio = total_nfev["wide"]["total"]/total_nfev["outer_box"]["total"]
    print("[PASS] 2000 scene-cluster resamples and bootstrap CIs")
    print("Nested gaps original->MATCHED5:",nesting["original"],"->",nesting["matched5"])
    print("Persistent/resolved/new:",nesting["persistent"],nesting["resolved"],nesting["newly_appearing"])
    print("WOO Mid+Full Frozen/MATCHED5:",
          recalc["FROZEN"]["WOO"]["MID_PLUS_FULL"]["correct"],
          recalc["MATCHED5"]["WOO"]["MID_PLUS_FULL"]["correct"])
    print("Primary Frozen/MATCHED5:",pp["FROZEN"]["full_minus_half"],
          pp["MATCHED5"]["full_minus_half"])
    print("Function evaluation wide/outer ratio:",f"{ratio:.3f}")
    report = {
        "protocol":"P4-S1C post-hoc solver sensitivity closure, read-only",
        "n_scenes":24, "n_cases":384, "n_fits":2304,
        "source_sha256":provenance,
        "comparison_arms":r92["arms"], "matched_starts_not_equal_nfev":True,
        "metrics":recalc, "primary_paired":pp,
        "primary_scene_cluster_CI95":ci,
        "nested_gap_audit":nesting,
        "decision_change_counts":counts,
        "function_evaluations":total_nfev,
        "wide_to_outer_nfev_ratio":ratio,
        "numerical_caveat":"33 Deadzone nested violations survive matched five-start fitting; SLSQP cannot certify global minimum",
        "empirical_conclusion":"Primary Nonlinear-M2 Full-minus-Half effect unchanged at +19/96; WOO Mid+Full correct +1/384 under MATCHED5",
        "scope":"Sensitivity of P90 within the same MuJoCo simulator, not external validity or optimizer-independent proof",
        "E4_accessed":False, "new_fits":0,"new_simulations":0,
        "labels_used_for_fitting":False,
        "limitations":["Extra three starts were chosen after P91 results (post-hoc)",
                       "Equal start counts do not imply equal actual function-evaluation cost",
                       "Witness arm is diagnostic, not budget-matched",
                       "Remaining nested gaps are numerical, not a new physical-law property",
                       "No evidence of global optimizer convergence",
                       "Does not verify P88 interaction robustness to all alternative solvers"],
    }
    audit = {"nested_gap_status_changes_PRIVATE":gap_changes,
             "classification_switches_PRIVATE":changed_rows}
    report_path = OUT / "p4s1c_solver_closure_report.json"
    audit_path = OUT / "p4s1c_solver_closure_audit_PRIVATE.json"
    save(report_path,report)
    save(audit_path,audit)
    print("Saved:",report_path)
    print("Saved:",audit_path)
    print("P4-S1C COMPLETE")


if __name__ == "__main__":
    main()
