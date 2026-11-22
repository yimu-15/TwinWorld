"""P4-M2D1 — Fresh-scene action × prior interaction replication.

Prospectively confirm a pattern discovered in P4-M2D0, on new scenes.
Frozen 2×2: domain OOO/WOO × observation MID/MID_PLUS_HIGH.
HIGH_ONLY and WWW are retained as descriptive controls.

24 NEW base scenes; 2 bands; M0 and pre-specified M2 axes; 384 conditions.
Frozen TRAIN-fitting via P74/P75/P77/P79/P80; High is never used to fit.
No E4, no tuning, no action selection based on the new-scene results.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from runpy import run_path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "p4m2d1"
P80 = HERE / "80_p4m2c0_prior_mismatch_stress.py"
P86 = HERE / "86_p4m2c2e_fresh_scene_factorial.py"
P87 = HERE / "87_p4m2d0_action_discrimination.py"
R86 = RESULTS / "p4m2c2e" / "p4m2c2e_fresh_report.json"
R87 = RESULTS / "p4m2d0" / "p4m2d0_fresh_action_report.json"
R80 = RESULTS / "p4m2c0" / "p4m2c0_dev_report.json"

SCENE_SEED = 20291017
NOISE_SEED = 2029101701
HIGH_NOISE_SEED = 2029101702
BOOT_SEED = 2029101703
SMOKE_OFFSET = 100000
N_SCENES = 24
N_BOOT = 2000
LAWS = ("elastic", "deadzone", "cubic")
DOMAINS = ("WWW", "OOO", "WOO")
OBS = ("MID", "HIGH_ONLY", "MID_PLUS_HIGH")
MODES = {"elastic": "elastic", "deadzone": "deadzone", "nonlinear": "cubic"}
CONDITIONS = {
    "elastic": (("matched", "M0"), ("alpha_high", "M2")),
    "deadzone": (("matched", "M0"), ("alpha_high", "M2"), ("gap_high", "M2")),
    "nonlinear": (("matched", "M0"), ("alpha_high", "M2"), ("gamma_high", "M2")),
}


def require(flag, msg):
    if not flag:
        raise RuntimeError(msg)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    require(path.is_file(), f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def prior_provenance():
    r80, r86, r87 = load(R80), load(R86), load(R87)
    for r in (r80, r86, r87):
        require(r["smoke"] is False, "Formal history required")
        require(r["E4_accessed"] is False, "E4 boundary violation")
    require(r86["scene_seed"] == 20290617 and r86["n_cases"] == 384,
            "Unexpected P86 protocol")
    require(r87["n_cases"] == 384 and r87["new_fits"] == 0,
            "Unexpected P87 protocol")
    require(sha(P80) == r87["source_sha256"][P80.name], "P80 SHA mismatch")
    require(sha(P86) == r87["source_sha256"][P86.name], "P86 SHA mismatch")
    require(sha(P87) == r87["source_sha256"][P87.name], "P87 SHA mismatch")
    require(sha(R86) == r87["source_sha256"][R86.name], "R86 SHA mismatch")
    require(r86["source_sha256"][R80.name] == sha(R80),
            "P86/P80 provenance mismatch")
    require(r87["thresholds_tuned"] is False and
            r87["truth_used_for_selection"] is False,
            "Previous evaluation integrity mismatch")
    return r80, r86, r87


def verify_fingerprints(p79, src, sim, scenes, r80, r86, smoke):
    local = p79["fingerprint_scenes"](scenes)
    require(len(local) == len(set(local)) == 2 * len(scenes),
            "Duplicate or missing scene fingerprints")
    old = (set(r80["scene_audit"]["new_fingerprints"]) |
           set(r86["scene_independence"]["new_scene_fingerprints"]))
    require(not set(local).intersection(old),
            "Overlap with P80 or P86 physical scenes")
    prior = p79["verify_scene_independence"](src, sim, scenes)
    if not smoke:
        smoke_scenes = src["make_scenes"](sim, SCENE_SEED + SMOKE_OFFSET, 1)
        smoke_hashes = set(p79["fingerprint_scenes"](smoke_scenes))
        require(not smoke_hashes.intersection(local), "Smoke/formal scene overlap")
    return {"n_fingerprints": len(local), "exact_overlap": 0,
            "new_scene_fingerprints": local,
            "historical_audit": prior}


def stable_seed(scene_id, band, mode):
    payload = f"{HIGH_NOISE_SEED}|{scene_id}|{band}|{mode}|high"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def make_high_probe(p75, p77, sim, api, commands,
                    scene, sid, band, mode):
    model = sim["make_model"](scene)
    passive, active, _, _ = sim["simulate"](
        model, scene, mode, commands["high"])
    passive = np.asarray(passive, dtype=float)
    active = np.asarray(active, dtype=float)
    require(passive.shape == active.shape == (51, 2),
            "Unexpected probe shape")
    rng = np.random.default_rng(stable_seed(sid, band, mode))
    sigma = float(p77["SIGMA"])
    return api["RevisionProbe"](
        passive + rng.normal(0, sigma, passive.shape),
        active + rng.normal(0, sigma, active.shape),
    )


def domain_for(config, i):
    return "outer_box" if config[i] == "O" else "wide"


def choose(fits, high, config, obs):
    vals = {}
    for i, law in enumerate(LAWS):
        domain = domain_for(config, i)
        item = fits[domain][law]
        if not item["valid"] or not item["converged"]:
            return None
        m = item["validation_rms_mm"]
        h = high[domain][law]
        if m is None or h is None or not np.isfinite(m) or not np.isfinite(h):
            return None
        if obs == "MID":
            value = float(m)
        elif obs == "HIGH_ONLY":
            value = float(h)
        elif obs == "MID_PLUS_HIGH":
            value = float(np.sqrt((float(m)**2 + float(h)**2) / 2))
        else:
            raise RuntimeError(f"Unknown observation: {obs}")
        vals[law] = value
    return min(LAWS, key=lambda law: vals[law])


def process(p80, p75, p77, sim, api, commands,
            base, sid, band, mode, axis, level):
    if level == "M0":
        require(axis == "matched", "Unexpected M0 axis")
        scene = base
    else:
        scene, _ = p80["make_stress_scene"](
            p74, p77, base, mode, axis, level)
    fits = p80["evaluate_case"](
        p75, p77, sim, api, commands, scene,
        sid, band, mode, NOISE_SEED)
    probe = make_high_probe(p75, p77, sim, api, commands,
                            scene, sid, band, mode)
    high = {domain: {} for domain in ("wide", "outer_box")}
    for domain in high:
        for law in LAWS:
            item = fits[domain][law]
            if item["valid"] and item["converged"]:
                high[domain][law] = float(p75["rms_mm"](
                    probe.predict(law, item["params_PRIVATE"]), probe.y_b))
            else:
                high[domain][law] = None
    truth = MODES[mode]   # Private scoring only
    results = {}
    for obs in OBS:
        results[obs] = {}
        for config in DOMAINS:
            pred = choose(fits, high, config, obs)
            results[obs][config] = {
                "winner": pred,
                "available": pred is not None,
                "correct_PRIVATE": pred == truth,
            }
    require(results["MID"]["WWW"]["winner"] == p80["winner"](fits, "wide"),
            "Historical Wide-winner parity failed")
    require(results["MID"]["OOO"]["winner"] == p80["winner"](fits, "outer_box"),
            "Historical Outer-winner parity failed")
    return {
        "scene_id": sid, "band": band, "mode_PRIVATE": mode,
        "axis_PRIVATE": axis, "level_PRIVATE": level,
        "decisions": results,
        "nested_violations_PRIVATE": [law for law in LAWS
            if fits["wide"][law]["valid"] and fits["outer_box"][law]["valid"]
            and fits["outer_box"][law]["train_mse"] <
                fits["wide"][law]["train_mse"] - 1e-6],
        "train_provenance_PRIVATE": {
            domain: {law: fits[domain][law]["train_mse"] for law in LAWS}
            for domain in ("wide", "outer_box")},
        "mid_rms_PRIVATE": {
            domain: {law: fits[domain][law]["validation_rms_mm"] for law in LAWS}
            for domain in ("wide", "outer_box")},
        "high_rms_PRIVATE": high,
    }


def corr(row, obs, config):
    return int(row["decisions"][obs][config]["correct_PRIVATE"])


def acc(rows, obs, config, mode=None, level=None):
    selected = [r for r in rows if
                (mode is None or r["mode_PRIVATE"] == mode) and
                (level is None or r["level_PRIVATE"] == level)]
    require(bool(selected), "Empty subgroup")
    return float(np.mean([corr(r, obs, config) for r in selected]))


def macro(rows, obs, config):
    return float(np.mean([acc(rows, obs, config, mode=m) for m in MODES]))


def contrasts(rows):
    # PRIMARY: prospective replication of the interaction discovered in P87.
    mid_gap = acc(rows, "MID", "WOO") - acc(rows, "MID", "OOO")
    high_gap = acc(rows, "MID_PLUS_HIGH", "WOO") - acc(rows, "MID_PLUS_HIGH", "OOO")
    return {
        "PRIMARY_action_prior_interaction_overall": high_gap - mid_gap,
        "SECONDARY_WOO_minus_OOO_high_overall": high_gap,
        "SECONDARY_WOO_minus_OOO_high_macro":
            macro(rows, "MID_PLUS_HIGH", "WOO") - macro(rows, "MID_PLUS_HIGH", "OOO"),
        "SECONDARY_OOO_high_minus_mid_overall":
            acc(rows, "MID_PLUS_HIGH", "OOO") - acc(rows, "MID", "OOO"),
        "DIAGNOSTIC_WOO_high_minus_OOO_high_elastic_M2":
            acc(rows, "MID_PLUS_HIGH", "WOO", "elastic", "M2") -
            acc(rows, "MID_PLUS_HIGH", "OOO", "elastic", "M2"),
        "DIAGNOSTIC_OOO_high_minus_WOO_high_nonlinear_M2":
            acc(rows, "MID_PLUS_HIGH", "OOO", "nonlinear", "M2") -
            acc(rows, "MID_PLUS_HIGH", "WOO", "nonlinear", "M2"),
        "DIAGNOSTIC_OOO_high_only_minus_mid_plus_high":
            acc(rows, "HIGH_ONLY", "OOO") - acc(rows, "MID_PLUS_HIGH", "OOO"),
    }


def bootstrap(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["scene_id"])].append(row)
    ids = sorted(grouped)
    require(len(ids) == N_SCENES, "Cluster count mismatch")
    rng = np.random.default_rng(BOOT_SEED)
    dist = defaultdict(list)
    for _ in range(N_BOOT):
        picked = rng.choice(ids, size=len(ids), replace=True)
        sample = [row for sid in picked for row in grouped[int(sid)]]
        for name, value in contrasts(sample).items():
            dist[name].append(value)
    return {
        "unit": "base_scene_id", "n_clusters": len(ids),
        "replications": N_BOOT, "seed": BOOT_SEED,
        "paired_95CI": {name: [float(np.quantile(values, 0.025)),
                               float(np.quantile(values, 0.975))]
                        for name, values in dist.items()},
    }


def main():
    global p74
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    print("=" * 72)
    print("P4-M2D1 — FRESH-SCENE ACTION × PRIOR INTERACTION REPLICATION")
    print("=" * 72)
    print("Smoke:", args.smoke, "| E4 sealed | No tuning")

    print("[1] Prior evidence provenance")
    r80, r86, r87 = prior_provenance()
    p80 = run_path(str(P80))
    p80["check_provenance"]()
    p74, p75, p77, p79, src, sim, api, commands = p80["initialize"]()
    require(tuple(p77["TRAIN"]) == ("e1", "e2", "weak", "strong") and
            p77["VALID"] == "mid", "Frozen TRAIN/MID mismatch")
    require("high" in commands and "high" not in p77["TRAIN"],
            "HIGH must be a held-out action")
    high_command = np.asarray(commands["high"], dtype=float)
    require(np.isclose(np.sum(np.abs(high_command)),
                       r87["command_budgets"]["command_L1"]) and
            np.isclose(np.max(np.abs(high_command)),
                       r87["command_budgets"]["command_peak"]),
            "HIGH command changed")
    print("[PASS] Frozen sources and held-out HIGH action")

    print("[2] New scenes and exact-identity checks")
    n = 1 if args.smoke else N_SCENES
    seed = SCENE_SEED + (SMOKE_OFFSET if args.smoke else 0)
    scenes = src["make_scenes"](sim, seed, n)
    require(len(scenes) == n, "Scene count mismatch")
    fingerprints = verify_fingerprints(p79, src, sim, scenes, r80, r86, args.smoke)
    print("[PASS] New scene fingerprints:", fingerprints["n_fingerprints"])

    print("[3] Frozen fitting + held-out HIGH evaluation")
    cases = []
    for sid, scene_set in enumerate(scenes):
        for band in p80["BANDS"]:
            for mode, conditions in CONDITIONS.items():
                for axis, level in conditions:
                    cases.append(process(p80, p75, p77, sim, api, commands,
                                         scene_set[band], sid, band, mode, axis, level))
        print(f"Completed {sid + 1}/{n} base scenes")
    require(len(cases) == n * 16, "Case matrix incomplete")
    keys = [(row["scene_id"], row["band"], row["mode_PRIVATE"],
             row["axis_PRIVATE"], row["level_PRIVATE"]) for row in cases]
    require(len(keys) == len(set(keys)), "Duplicate case")

    print("[4] Fixed 2x2 and controls")
    metrics = {}
    for obs in OBS:
        metrics[obs] = {}
        for config in DOMAINS:
            coverage = sum(row["decisions"][obs][config]["available"] for row in cases)
            metrics[obs][config] = {
                "correct": int(sum(corr(row, obs, config) for row in cases)),
                "n": len(cases), "coverage": coverage / len(cases),
                "accuracy": acc(cases, obs, config),
                "macro": macro(cases, obs, config),
                "by_mode": {mode: acc(cases, obs, config, mode=mode)
                            for mode in MODES},
            }
            print(f"{obs:13s} {config}: {metrics[obs][config]['correct']:3d}/{len(cases)} "
                  f"Acc={metrics[obs][config]['accuracy']:.4f} "
                  f"Macro={metrics[obs][config]['macro']:.4f} "
                  f"Coverage={metrics[obs][config]['coverage']:.3f}")
    results = contrasts(cases)
    ci = None if args.smoke else bootstrap(cases)
    print("[5] Predefined comparisons")
    for k, val in results.items():
        print(f"{k}: {val:+.6f}",
              "95CI=" + str(ci["paired_95CI"][k]) if ci else "SMOKE_SKIP")
    nested = dict(Counter(law for row in cases
                          for law in row["nested_violations_PRIVATE"]))
    print("Nested solver gaps:", nested)
    min_coverage = min(v["coverage"] for sub in metrics.values() for v in sub.values())
    print("Minimum coverage:", min_coverage)

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "fresh"
    report_path = OUT / f"p4m2d1_{suffix}_report.json"
    audit_path = OUT / f"p4m2d1_{suffix}_audit_PRIVATE.json"
    report = {
        "protocol": "P4-M2D1 independent-seed action/prior interaction replication",
        "smoke": args.smoke, "same_simulator_family": True,
        "n_base_scenes": n, "n_cases": len(cases), "scene_seed": seed,
        "train_noise_seed": NOISE_SEED, "high_noise_seed": HIGH_NOISE_SEED,
        "scenario": "M0 + fixed M2 per applicable axis; 2 bands",
        "predeclared_primary": "PRIMARY_action_prior_interaction_overall",
        "methods": list(OBS), "configs": list(DOMAINS),
        "metrics": metrics, "comparisons": results, "bootstrap": ci,
        "nested_solver_violations": nested, "minimum_coverage": min_coverage,
        "scene_independence": fingerprints,
        "E4_accessed": False, "thresholds_tuned": False,
        "truth_used_for_fitting": False, "high_used_for_fitting": False,
        "source_sha256": {P80.name: sha(P80), P86.name: sha(P86),
                          P87.name: sha(P87), R80.name: sha(R80),
                          R86.name: sha(R86), R87.name: sha(R87),
                          Path(__file__).name: sha(Path(__file__))},
        "limitations": [
            "Same MuJoCo generator, new scene seed only",
            "P87 finding selected HIGH/WOO; this replication was motivated by it",
            "Only 24 independent base scene clusters",
            "More observation information for MID_PLUS_HIGH than MID",
            "Original finite-start SLSQP remains uncorrected",
            "No global optimality or real-hardware generalization certificate",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                      allow_nan=False), encoding="utf-8")
    audit_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding="utf-8")
    print("P4-M2D1 COMPLETE")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
