"""P4-M2D2B v2 — Frozen-model amplitude dose-response.

This is a PROTOCOL-PRESERVING failure-handling revision of experiment #90.
Scene seeds, paired HALF/FULL noise, action amplitudes, candidate fit procedure,
methods, contrasts, and 24-scene cluster bootstrap are unchanged.

An unavailable candidate is never silently dropped from a 3-way decision.
Its entire domain configuration abstains (counts as incorrect in all-case
accuracy), and conditional accuracy plus coverage are reported separately.

Nothing uses HIGH/HALF/FULL or ground truth for parameter optimization.
E4 is never accessed.
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
OUT = RESULTS / "p4m2d2b"
P80 = HERE / "80_p4m2c0_prior_mismatch_stress.py"
P88 = HERE / "88_p4m2d1_fresh_action_prior_interaction.py"
R88 = RESULTS / "p4m2d1" / "p4m2d1_fresh_report.json"
R89 = RESULTS / "p4m2d2a" / "p4m2d2a_fresh_report.json"

LAWS = ("elastic", "deadzone", "cubic")
CONFIGS = ("WWW", "OOO", "WOO")
TRUTH = {"elastic": "elastic", "deadzone": "deadzone", "nonlinear": "cubic"}
CONDITIONS = {
    "elastic": (("matched", "M0"), ("alpha_high", "M2")),
    "deadzone": (("matched", "M0"), ("alpha_high", "M2"), ("gap_high", "M2")),
    "nonlinear": (("matched", "M0"), ("alpha_high", "M2"), ("gamma_high", "M2")),
}
METHODS = ("MID", "HALF_ONLY", "FULL_ONLY", "MID_PLUS_HALF", "MID_PLUS_FULL")
SCENE_SEED = 20291117
FIT_NOISE_SEED = 2029111701
ACTION_NOISE_SEED = 2029111702
BOOT_SEED = 2029111703
N_SCENES = 24
SMOKE_OFFSET = 100000
BOOT_N = 2000
MIN_PRIMARY_COVERAGE = 0.90


def must(flag, message):
    if not flag:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    must(path.is_file(), f"Missing required input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def finite_params(value):
    if value is None:
        return False
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(a.ndim == 1 and a.size > 0 and np.all(np.isfinite(a)))


def stable_seed(sid, band, mode):
    message = f"{ACTION_NOISE_SEED}|{sid}|{band}|{mode}|amplitude_pair"
    raw = hashlib.sha256(message.encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "little")


def verify_sources():
    r88, r89 = read(R88), read(R89)
    must(r88["smoke"] is False and r89["smoke"] is False, "Formal reports required")
    must(r88["n_cases"] == 384 and r89["n_cases"] == 384, "History case-count mismatch")
    must(r88["E4_accessed"] is False and r89["E4_accessed"] is False,
         "Historical E4 flag")
    must(r89["new_fits"] == 0 and r89["new_simulations"] == 0,
         "P89 should be read-only")
    must(digest(R88) == r89["source_sha256"][R88.name], "R88 hash mismatch")
    must(digest(P88) == r89["source_sha256"][P88.name], "P88 hash mismatch")
    must(digest(P80) == r88["source_sha256"][P80.name], "P80 hash mismatch")
    return r88, r89


def verify_new_scenes(p79, src, sim, scenes, old, smoke):
    fingerprints = p79["fingerprint_scenes"](scenes)
    must(len(fingerprints) == 2 * len(scenes)
         and len(set(fingerprints)) == len(fingerprints),
         "Duplicate scene fingerprints")
    old_fps = set(old["scene_independence"]["new_scene_fingerprints"])
    must(not set(fingerprints) & old_fps, "Overlap with P88 scenes")
    p79["verify_scene_independence"](src, sim, scenes)
    if not smoke:
        smoke_scenes = src["make_scenes"](sim, SCENE_SEED + SMOKE_OFFSET, 1)
        smoke_fps = set(p79["fingerprint_scenes"](smoke_scenes))
        must(not set(fingerprints) & smoke_fps, "Smoke/formal overlap")
    return {"n_fingerprints": len(fingerprints),
            "vs_p88_exact_overlap": 0, "fingerprints": fingerprints}


def pair_probes(scene, sid, band, mode, sim, api, command, sigma):
    clean = {}
    for dose, multiplier in (("HALF", 0.5), ("FULL", 1.0)):
        model = sim["make_model"](scene)
        passive, active, _, _ = sim["simulate"](
            model, scene, mode, command * multiplier
        )
        pa, ac = np.asarray(passive, dtype=float), np.asarray(active, dtype=float)
        must(pa.shape == ac.shape == (51, 2),
             f"Unexpected observation shape: {dose}")
        clean[dose] = (pa, ac)
    rng = np.random.default_rng(stable_seed(sid, band, mode))
    pa_noise = rng.normal(0, sigma, (51, 2))
    ac_noise = rng.normal(0, sigma, (51, 2))
    return {
        dose: api["RevisionProbe"](pair[0] + pa_noise, pair[1] + ac_noise)
        for dose, pair in clean.items()
    }


def fit_availability(fits, sid, band, mode, axis, level):
    status = {"wide": {}, "outer_box": {}}
    for d in status:
        for law in LAWS:
            f = fits[d][law]
            reasons = []
            if f.get("valid") is not True:
                reasons.append("invalid")
            if f.get("converged") is not True:
                reasons.append("not_converged")
            if not finite_params(f.get("params_PRIVATE")):
                reasons.append("invalid_params")
            mid = finite_number(f.get("validation_rms_mm"))
            if mid is None or mid < 0:
                reasons.append("invalid_mid_rms")
            mse = finite_number(f.get("train_mse"))
            if mse is None or mse < 0:
                reasons.append("invalid_train_mse")
            status[d][law] = {
                "available": not reasons,
                "reasons": reasons,
                "valid": f.get("valid") is True,
                "converged": f.get("converged") is True,
                "train_mse": mse,
                "validation_rms_mm": mid,
                "nfev": f.get("nfev"),
            }
            if reasons:
                print("[FIT UNAVAILABLE]", f"scene={sid}", f"band={band}",
                      f"mode={mode}", f"axis={axis}", f"level={level}",
                      f"domain={d}", f"law={law}", f"reasons={reasons}",
                      flush=True)
    return status


def domain(config, i):
    return "outer_box" if config[i] == "O" else "wide"


def method_rms(fits, held, config, method, status):
    vals = {}
    for i, law in enumerate(LAWS):
        d = domain(config, i)
        if not status[d][law]["available"]:
            return None
        mid = float(fits[d][law]["validation_rms_mm"])
        half, full = held["HALF"][d][law], held["FULL"][d][law]
        if method == "MID":
            value = mid
        elif method == "HALF_ONLY":
            value = half
        elif method == "FULL_ONLY":
            value = full
        elif method == "MID_PLUS_HALF":
            value = float(np.sqrt((mid**2 + half**2) / 2.0))
        elif method == "MID_PLUS_FULL":
            value = float(np.sqrt((mid**2 + full**2) / 2.0))
        else:
            raise ValueError(method)
        must(value is not None and np.isfinite(value) and value >= 0,
             f"Invalid method score: {config}/{method}/{law}")
        vals[law] = float(value)
    return vals


def evaluate(p80, p74, p75, p77, sim, api, commands, base, sid, band,
             mode, axis, level, fit_seed, high_command):
    if level == "M0":
        must(axis == "matched", "M0 axis mismatch")
        scene = base
    else:
        scene, _ = p80["make_stress_scene"](p74, p77, base, mode, axis, level)

    fits = p80["evaluate_case"](
        p75, p77, sim, api, commands,
        scene, sid, band, mode, fit_seed
    )
    status = fit_availability(fits, sid, band, mode, axis, level)
    probes = pair_probes(scene, sid, band, mode, sim, api, high_command,
                         float(p77["SIGMA"]))

    held = {}
    predicted = {}
    for dose, probe in probes.items():
        held[dose] = {"wide": {}, "outer_box": {}}
        predicted[dose] = {"wide": {}, "outer_box": {}}
        for d in ("wide", "outer_box"):
            for law in LAWS:
                if not status[d][law]["available"]:
                    held[dose][d][law] = None
                    predicted[dose][d][law] = None
                    continue
                params = fits[d][law]["params_PRIVATE"]
                pred = np.asarray(probe.predict(law, params), dtype=float)
                must(pred.shape == probe.y_b.shape and np.all(np.isfinite(pred)),
                     f"Invalid prediction: {sid}/{band}/{mode}/{axis}/{d}/{law}")
                rms = finite_number(p75["rms_mm"](pred, probe.y_b))
                must(rms is not None and rms >= 0,
                     "Invalid held-out RMS")
                held[dose][d][law] = rms
                predicted[dose][d][law] = pred

    true_law = TRUTH[mode]
    separation = {}
    decisions = {}
    score_gap = {}
    for config in CONFIGS:
        ed, cd = domain(config, 0), domain(config, 2)
        separation[config] = {}
        for dose in ("HALF", "FULL"):
            pred_e, pred_c = predicted[dose][ed]["elastic"], predicted[dose][cd]["cubic"]
            if pred_e is None or pred_c is None:
                separation[config][dose] = None
            else:
                val = finite_number(p75["rms_mm"](pred_e, pred_c))
                must(val is not None and val >= 0, "Invalid pairwise separation")
                separation[config][dose] = val

        decisions[config] = {}
        score_gap[config] = {}
        for method in METHODS:
            vals = method_rms(fits, held, config, method, status)
            if vals is None:
                decisions[config][method] = {
                    "winner": None, "available": False, "correct_PRIVATE": False
                }
                score_gap[config][method] = None
            else:
                selection = min(LAWS, key=lambda law: vals[law])
                decisions[config][method] = {
                    "winner": selection,
                    "available": True,
                    "correct_PRIVATE": selection == true_law,
                }
                score_gap[config][method] = float(vals["elastic"] - vals["cubic"])

    for config, d in (("WWW", "wide"), ("OOO", "outer_box")):
        if decisions[config]["MID"]["available"]:
            must(decisions[config]["MID"]["winner"] == p80["winner"](fits, d),
                 f"Frozen Mid decision mismatch: {config}")

    nested = []
    both_converged_nested = []
    for law in LAWS:
        w = status["wide"][law]["train_mse"]
        o = status["outer_box"][law]["train_mse"]
        if (w is not None and o is not None
                and status["wide"][law]["valid"]
                and status["outer_box"][law]["valid"]
                and o < w - 1e-6):
            nested.append(law)
            if (status["wide"][law]["converged"]
                    and status["outer_box"][law]["converged"]):
                both_converged_nested.append(law)

    return {
        "scene_id": sid,
        "band": band,
        "mode_PRIVATE": mode,
        "axis_PRIVATE": axis,
        "level_PRIVATE": level,
        "fit_status_PRIVATE": status,
        "nested_violations_PRIVATE": nested,
        "both_converged_nested_violations_PRIVATE": both_converged_nested,
        "decisions": decisions,
        "separation_mm": separation,
        "score_gap_mm_PRIVATE": score_gap,
        "heldout_rms_PRIVATE": held,
    }


def correct(row, config, method):
    # Unavailable is a scored abstention (0), NOT a silently removed case.
    return int(row["decisions"][config][method]["correct_PRIVATE"])


def eligible_rows(rows, mode=None, level=None):
    return [r for r in rows
            if (mode is None or r["mode_PRIVATE"] == mode)
            and (level is None or r["level_PRIVATE"] == level)]


def accuracy(rows, config, method, mode=None, level=None):
    selected = eligible_rows(rows, mode, level)
    must(bool(selected), "Empty evaluation group")
    return float(np.mean([correct(row, config, method) for row in selected]))


def coverage(rows, config, method, mode=None, level=None):
    selected = eligible_rows(rows, mode, level)
    must(bool(selected), "Empty coverage group")
    available = sum(bool(r["decisions"][config][method]["available"])
                    for r in selected)
    return {"available": available, "total": len(selected),
            "coverage": available / len(selected)}


def median_or_none(values):
    return float(np.median(values)) if values else None


def effects(rows):
    nm2 = eligible_rows(rows, "nonlinear", "M2")
    em2 = eligible_rows(rows, "elastic", "M2")
    must(nm2 and em2, "Missing diagnostic strata")
    sep = []
    gap = []
    for row in nm2:
        s = row["separation_mm"]["WOO"]
        if s["FULL"] is not None and s["HALF"] is not None:
            sep.append(s["FULL"] - s["HALF"])
        g = row["score_gap_mm_PRIVATE"]["WOO"]
        if g["MID_PLUS_FULL"] is not None and g["MID_PLUS_HALF"] is not None:
            gap.append(g["MID_PLUS_FULL"] - g["MID_PLUS_HALF"])

    return {
        "PRIMARY_WOO_nonlinear_M2_recall_full_minus_half":
            accuracy(rows, "WOO", "MID_PLUS_FULL", "nonlinear", "M2")
            - accuracy(rows, "WOO", "MID_PLUS_HALF", "nonlinear", "M2"),
        "SECONDARY_WOO_nonlinear_M2_pairwise_sep_full_minus_half_mm":
            median_or_none(sep),
        "SECONDARY_WOO_nonlinear_M2_score_gap_full_minus_half_mm":
            median_or_none(gap),
        "SECONDARY_WOO_accuracy_full_minus_half":
            accuracy(rows, "WOO", "MID_PLUS_FULL")
            - accuracy(rows, "WOO", "MID_PLUS_HALF"),
        "DIAGNOSTIC_WOO_elastic_M2_recall_full_minus_half":
            accuracy(rows, "WOO", "MID_PLUS_FULL", "elastic", "M2")
            - accuracy(rows, "WOO", "MID_PLUS_HALF", "elastic", "M2"),
        "DIAGNOSTIC_OOO_accuracy_full_minus_half":
            accuracy(rows, "OOO", "MID_PLUS_FULL")
            - accuracy(rows, "OOO", "MID_PLUS_HALF"),
    }, {
        "n_nonlinear_m2": len(nm2),
        "n_elastic_m2": len(em2),
        "n_separation_pairs": len(sep),
        "n_score_gap_pairs": len(gap),
    }


def cluster_bootstrap(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_id"]].append(row)
    ids = sorted(groups)
    must(len(ids) == N_SCENES, "Scene cluster count mismatch")
    rng = np.random.default_rng(BOOT_SEED)
    distributions = defaultdict(list)
    for _ in range(BOOT_N):
        picked = rng.choice(ids, size=len(ids), replace=True)
        sample = [r for sid in picked for r in groups[int(sid)]]
        values, _ = effects(sample)
        for name, value in values.items():
            if value is not None:
                distributions[name].append(value)
    return {
        "cluster_unit": "base_scene_id", "n_clusters": N_SCENES,
        "seed": BOOT_SEED, "n_resamples": BOOT_N,
        "95CI": {name: [float(np.quantile(values, .025)),
                         float(np.quantile(values, .975))]
                 for name, values in distributions.items() if values},
        "valid_resamples": {name: len(values) for name, values in distributions.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("P4-M2D2B v2 — AMPLITUDE DOSE-RESPONSE / ABSTENTION-AWARE")
    print("=" * 72)
    print("Smoke:", args.smoke, "| E4 sealed | Frozen fitting")
    r88, _ = verify_sources()
    p80 = run_path(str(P80))
    p80["check_provenance"]()
    p74, p75, p77, p79, src, sim, api, commands = p80["initialize"]()
    must(tuple(p77["TRAIN"]) == ("e1", "e2", "weak", "strong")
         and p77["VALID"] == "mid", "Unexpected fit or validation actions")
    must(np.isclose(p77["SIGMA"], 0.005), "Noise sigma mismatch")
    high_command = np.asarray(commands["high"], dtype=float)
    must(high_command.ndim == 1 and len(high_command) >= 8,
         "Invalid HIGH command")

    n = 1 if args.smoke else N_SCENES
    seed = SCENE_SEED + (SMOKE_OFFSET if args.smoke else 0)
    fit_seed = FIT_NOISE_SEED + (SMOKE_OFFSET if args.smoke else 0)
    scenes = src["make_scenes"](sim, seed, n)
    independence = verify_new_scenes(p79, src, sim, scenes, r88, args.smoke)
    print("[PASS] Fresh scenes:", n, "| fingerprints:", independence["n_fingerprints"])
    print("[ACTION] HALF peak:", float(np.max(np.abs(high_command * 0.5))),
          "| FULL peak:", float(np.max(np.abs(high_command))))

    rows = []
    for sid, scene_set in enumerate(scenes):
        for band in ("original", "above_original_range"):
            for mode, conditions in CONDITIONS.items():
                for axis, level in conditions:
                    rows.append(evaluate(
                        p80, p74, p75, p77, sim, api, commands,
                        scene_set[band], sid, band, mode, axis, level,
                        fit_seed, high_command
                    ))
        print(f"Completed {sid + 1}/{n} base scenes")
    must(len(rows) == 16 * n, "Case matrix size mismatch")
    case_keys = [(r["scene_id"], r["band"], r["mode_PRIVATE"],
                  r["axis_PRIVATE"], r["level_PRIVATE"]) for r in rows]
    must(len(set(case_keys)) == len(rows), "Duplicate case key")

    metrics, diagnostic_n = effects(rows)
    print("\nPREDECLARED TESTS")
    for name, value in metrics.items():
        print(f"{name}: {value:+.6f}" if value is not None else f"{name}: UNAVAILABLE")

    print("\nCONFIGURATION COVERAGE (all-case denominator)")
    config_coverage = {}
    for config in CONFIGS:
        config_coverage[config] = {}
        for method in METHODS:
            cv = coverage(rows, config, method)
            correct_n = sum(correct(r, config, method) for r in rows)
            cv.update({"correct": correct_n,
                       "accuracy_all": correct_n / len(rows),
                       "accuracy_available": (correct_n / cv["available"]
                                              if cv["available"] else None)})
            config_coverage[config][method] = cv
            print(f"{config}/{method}: {correct_n}/{len(rows)} "
                  f"acc_all={cv['accuracy_all']:.4f} "
                  f"coverage={cv['coverage']:.4f}")

    primary_cov = coverage(rows, "WOO", "MID_PLUS_FULL", "nonlinear", "M2")
    half_cov = coverage(rows, "WOO", "MID_PLUS_HALF", "nonlinear", "M2")
    must(primary_cov == half_cov, "Paired HALF/FULL availability mismatch")
    primary_interpretable = primary_cov["coverage"] >= MIN_PRIMARY_COVERAGE
    print("[PRIMARY COVERAGE]", f"{primary_cov['available']}/{primary_cov['total']}",
          f"= {primary_cov['coverage']:.4f}")
    if not primary_interpretable:
        print("[WARNING] Primary coverage below v2 interpretability gate;"
              " do not claim a confirmed dose-response effect.")

    nested = Counter(law for row in rows
                     for law in row["nested_violations_PRIVATE"])
    confirmed_nested = Counter(law for row in rows
                               for law in row["both_converged_nested_violations_PRIVATE"])
    unavailable = Counter((d, law, reason)
                          for row in rows
                          for d, laws in row["fit_status_PRIVATE"].items()
                          for law, status in laws.items()
                          for reason in status["reasons"])
    failures = [{"scene_id": r["scene_id"], "band": r["band"],
                 "mode_PRIVATE": r["mode_PRIVATE"],
                 "axis_PRIVATE": r["axis_PRIVATE"],
                 "level_PRIVATE": r["level_PRIVATE"],
                 "failed_candidates": [
                     {"domain": d, "law": law, "reasons": status["reasons"],
                      "nfev": status["nfev"]}
                     for d, laws in r["fit_status_PRIVATE"].items()
                     for law, status in laws.items() if not status["available"]
                 ]}
                for r in rows if any(not item["available"]
                                     for laws in r["fit_status_PRIVATE"].values()
                                     for item in laws.values())]
    print("Nested solver gaps:", dict(nested))
    print("Both-converged nested gaps:", dict(confirmed_nested))
    print("Rows with >=1 failed candidate:", len(failures))
    print("Failure reasons:", {"/".join(k): v for k, v in unavailable.items()})
    bootstrap = None if args.smoke else cluster_bootstrap(rows)

    report = {
        "protocol": "P4-M2D2B v2 frozen-model amplitude dose response; abstention-aware",
        "revision": "v2_unavailable_candidates", "smoke": args.smoke,
        "scene_seed": seed, "fit_noise_seed": fit_seed,
        "action_noise_seed": ACTION_NOISE_SEED,
        "n_base_scenes": n, "n_cases": len(rows),
        "action_amplitudes": {"half": 0.5, "full": 1.0},
        "same_waveform": True, "paired_noise": True,
        "new_action_observations": True,
        "source_provenance": {
            P80.name: digest(P80), P88.name: digest(P88),
            R88.name: digest(R88), R89.name: digest(R89),
            Path(__file__).name: digest(Path(__file__)),
        },
        "scene_independence": independence,
        "predeclared_effects": metrics,
        "diagnostic_pair_counts": diagnostic_n,
        "bootstrap": bootstrap,
        "configuration_coverage": config_coverage,
        "primary_coverage": primary_cov,
        "primary_interpretable_under_v2_gate": primary_interpretable,
        "v2_interpretability_gate": MIN_PRIMARY_COVERAGE,
        "unavailable_is_abstention": True,
        "rows_with_failed_candidates": len(failures),
        "fit_failures_PRIVATE": failures,
        "fit_failure_reasons": {"/".join(k): v for k, v in unavailable.items()},
        "nested_solver_violations": dict(nested),
        "both_converged_nested_solver_violations": dict(confirmed_nested),
        "E4_accessed": False, "action_used_for_fit": False,
        "threshold_tuning": False,
        "limitations": [
            "Same simulator family, new scenes only",
            "Half/full differ in actuation amplitude",
            "Unavailable fits abstain; counted incorrect in all-case accuracy",
            "Coverage threshold added for v2 reporting after first failed run; not original preregistration",
            "Median separation/score gaps computed on available paired candidates only",
            "Candidate separation is not accuracy or proof of structural identifiability",
            "Finite-start SLSQP remains uncorrected",
            "Truth used only for offline diagnostics",
            "No E4",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "v2_smoke" if args.smoke else "v2_fresh"
    report_path = OUT / f"p4m2d2b_{suffix}_report.json"
    audit_path = OUT / f"p4m2d2b_{suffix}_audit_PRIVATE.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                      allow_nan=False), encoding="utf-8")
    audit_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding="utf-8")
    print("Saved:", report_path)
    print("Saved:", audit_path)
    print("P4-M2D2B v2 COMPLETE")


if __name__ == "__main__":
    main()
