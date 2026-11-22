
"""#110 P4-B3C: fresh-scene equal hard model-evaluation-cap confirmation.

TRF_CAP vs SLSQP_CAP.
24 independent base scenes x 2 bands x 8 cases = 384 cases.
100/200 total hard caps, split across the frozen 2+3 starts.
Every 204-vector residual call, including numerical Jacobian calls,
is counted by the exact #108 implementation.

Primary: TRF-SLSQP, cap=200, WOO, MID_PLUS_FULL, all 384 cases.
Secondary: cap=100; MID_PLUS_HALF; budget and dose diagnostics.
Unavailable decisions remain errors in full denominators.
24-scene paired cluster bootstrap, 2000 resamples.
E4 is never accessed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
import platform
from pathlib import Path
from runpy import run_path
import sys

import numpy as np
import scipy


SELF = Path(__file__).resolve()
HERE = SELF.parent
BASE = HERE / "results"
OUT = BASE / "p4b3c"

P98 = HERE / "98_p4g1a_independent_physics_generator_feasibility.py"
P99 = HERE / "99_p4g1b_independent_generator_identification.py"
P103 = HERE / "103_p4g2c_fresh_frozen_vs_refit_confirmation.py"
P104 = HERE / "104_p4g2d_fresh_confirmation_closure_audit.py"
P105 = HERE / "105_p4b2a_independent_identification_baseline_pilot.py"
P106 = HERE / "106_p4b2b_fresh_classical_identification_comparison.py"
P108 = HERE / "108_p4b3a_equal_model_evaluation_budget_pilot.py"
P109 = HERE / "109_p4b3b_equal_budget_closure_audit.py"

R98 = BASE / "p4g1a" / "p4g1a_feasibility_report.json"
S98 = BASE / "p4g1a" / "p4g1a_smoke_report.json"
R99 = BASE / "p4g1b" / "p4g1b_fresh_report.json"
R103 = BASE / "p4g2c" / "p4g2c_fresh_report.json"
A103 = BASE / "p4g2c" / "p4g2c_fresh_audit_PRIVATE.json"
M103 = BASE / "p4g2c" / "p4g2c_source_manifest.json"
R104 = BASE / "p4g2d" / "p4g2d_closure_report.json"
R106 = BASE / "p4b2b" / "p4b2b_fresh_report.json"
M106 = BASE / "p4b2b" / "p4b2b_source_manifest.json"
R107 = BASE / "p4b2c" / "p4b2c_closure_report.json"
R108 = BASE / "p4b3a" / "p4b3a_pilot_report.json"
A108 = BASE / "p4b3a" / "p4b3a_pilot_audit_PRIVATE.json"
R109 = BASE / "p4b3b" / "p4b3b_closure_report.json"
A109 = BASE / "p4b3b" / "p4b3b_closure_audit_PRIVATE.json"


# All values are fixed before formal outcomes.
SCENE_SEED = 20300323
SMOKE_SCENE_SEED = 20310323
NOISE_SEED = 2030032301
SMOKE_NOISE_SEED = 2031032301
BOOT_SEED = 2030032302

N_SCENES = 24
N_CASES = 384
N_BOOT = 2000
N_RESIDUALS = 204
MIN_COVERAGE = 0.90

BUDGETS = (100, 200)
SMOKE_BUDGET = 50

TRAIN = ("e1", "e2", "weak", "strong")
LAWS = ("elastic", "deadzone", "cubic")
METHODS = ("MID_PLUS_HALF", "MID_PLUS_FULL")
SOLVERS = ("TRF_CAP", "SLSQP_CAP")
DOMAINS = ("wide", "outer_box")

DOMAIN_BY_LAW = {
    "elastic": "wide",
    "deadzone": "outer_box",
    "cubic": "outer_box",
}

PRIMARY_CONFIG = "WOO"
PRIMARY_BUDGET = 200
PRIMARY_METHOD = "MID_PLUS_FULL"


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    need(path.is_file(), f"Missing frozen evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    need(path.is_file(), f"Missing source: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(
        json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def write_once(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        obj, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"

    if path.exists():
        need(
            path.read_text(encoding="utf-8") == content,
            f"Immutable output differs; refuse overwrite: {path}",
        )
        return

    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def write_smoke(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old = read(path)
        need(
            old.get("source_sha256") == obj.get("source_sha256"),
            "Smoke source changed; archive old smoke before rerun",
        )

    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def preflight():
    # Reuse the exact frozen #108 optimizer implementation.
    p108 = run_path(str(P108))

    (
        p103, p99, gen, p77, p22,
        commands, starts, rmap, lineage,
    ) = p108["preflight"]()

    r108, a108 = read(R108), read(A108)
    r109, a109 = read(R109), read(A109)

    # Verify #108 source and historical feasibility result.
    need(
        r108["smoke"] is False
        and r108["n_cases"] == 3
        and r108["budgets"] == [100, 200]
        and r108["per_start_caps"] == [20, 40]
        and r108["training_residual_count"] == 204
        and r108["methods"] == list(SOLVERS)
        and r108["new_scenes"] is False
        and r108["E4_accessed"] is False,
        "#108 pilot protocol changed",
    )

    need(
        a108["source_sha256"] == r108["source_sha256"]
        and len(a108["rows_PRIVATE"]) == 3,
        "#108 private audit mismatch",
    )

    need(
        r108["source_sha256"]["SELF"] == lineage["SELF"],
        "#108 source hash mismatch",
    )

    # Verify #109 closure and exact #108 provenance.
    need(
        r109["status"]
        == "PASS_READ_ONLY_CLOSURE_NO_CONFIRMATORY_INFERENCE"
        and r109["n_historical_rows"] == 3
        and r109["n_base_scenes"] == 1
        and r109["optimizer_start_records"] == 180
        and r109["E4_accessed"] is False
        and r109["fitting_performed"] is False
        and r109["simulation_performed"] is False,
        "#109 closure status changed",
    )

    need(
        r109["source_hashes_verified"]["P108"] == lineage["SELF"]
        and a109["source_sha256"]["SELF"] == lineage["SELF"]
        and a109["closure_script_sha256"] == sha(P109),
        "#109/#108 source provenance mismatch",
    )

    for key in ("P105", "P103", "R106", "M106", "R107"):
        need(
            r109["source_hashes_verified"][key] == lineage[key]
            and a109["source_sha256"][key] == lineage[key],
            f"#109 upstream source mismatch: {key}",
        )

    need(
        all(x["status"] == "PASS"
            for x in r109["checkpoints_verified"]),
        "#109 checkpoint verification failed",
    )

    # Independently record the full lineage used by #110.
    paths = {
        "P98": P98,
        "P99": P99,
        "P103": P103,
        "P104": P104,
        "P105": P105,
        "P106": P106,
        "P108": P108,
        "R98": R98,
        "S98": S98,
        "R99": R99,
        "R103": R103,
        "A103": A103,
        "M103": M103,
        "R104": R104,
        "R106": R106,
        "M106": M106,
        "R107": R107,
        "R108": R108,
        "A108": A108,
        "R109": R109,
        "A109": A109,
    }

    hashes = {key: sha(path) for key, path in paths.items()}
    hashes["SELF"] = sha(SELF)

    need(
        hashes["P108"] == lineage["SELF"],
        "Local #108 differs from the source verified by #109",
    )

    p106 = run_path(str(P106))

    # The #108 preflight already validates P103/P105/P106 closure.
    need(
        tuple(p108["TRAIN"]) == TRAIN
        and tuple(p108["LAWS"]) == LAWS
        and tuple(p108["BUDGETS"]) == BUDGETS
        and tuple(p108["SOLVERS"]) == SOLVERS
        and p108["N"] == N_RESIDUALS,
        "Frozen #108 interface changed",
    )

    start_points = {}
    for law in LAWS:
        vectors = [
            *p77["initial_points"](law),
            *starts[law],
        ]
        need(len(vectors) == 5, f"Start count changed: {law}")
        start_points[law] = [
            np.asarray(x, dtype=float).tolist() for x in vectors
        ]

    return {
        "p108": p108,
        "p103": p103,
        "p99": p99,
        "gen": gen,
        "p77": p77,
        "p22": p22,
        "commands": commands,
        "starts": starts,
        "start_points": start_points,
        "p106": p106,
        "hashes": hashes,
    }


def scene_fingerprints(gen, seed, n_scenes):
    rng = np.random.default_rng(seed)
    out = []

    for _ in range(n_scenes):
        for band in gen["BANDS"]:
            scene = gen["sample_scene"](rng, band)
            out.append(gen["digest_object"](scene))

    return out


def collect_fingerprints(doc):
    values = []

    for key in ("scene_fingerprints_PRIVATE", "scene_fingerprints"):
        found = doc.get(key)
        if isinstance(found, list):
            values.extend(str(x) for x in found)

    return values


def generate_scenes(ctx, smoke):
    gen = ctx["gen"]
    p99 = ctx["p99"]
    p103 = ctx["p103"]
    p106 = ctx["p106"]

    seed = SMOKE_SCENE_SEED if smoke else SCENE_SEED
    n = 1 if smoke else N_SCENES

    rng = np.random.default_rng(seed)
    scenes = [
        {band: gen["sample_scene"](rng, band) for band in gen["BANDS"]}
        for _ in range(n)
    ]

    fingerprints = [
        gen["digest_object"](pair[band])
        for pair in scenes
        for band in gen["BANDS"]
    ]

    need(
        len(fingerprints) == len(set(fingerprints)) == 2 * n,
        "Duplicate fresh base-scene fingerprint",
    )

    # Historical formal cohorts and manifests.
    previous = set()

    for path in (R98, S98, R99, M103, R106, M106):
        previous.update(collect_fingerprints(read(path)))

    m103 = read(M103)
    m106 = read(M106)

    need(
        len(m103["scene_fingerprints_PRIVATE"]) == 48
        and len(m106["scene_fingerprints_PRIVATE"]) == 48,
        "Historical scene manifest incomplete",
    )

    # Explicitly exclude earlier pilot/smoke seeds.
    previous_seeds = {
        int(p99["SMOKE_SCENE_SEED"]),
        int(p103["SCENE_SEED"]),
        int(p103["SMOKE_SCENE_SEED"]),
        int(p106["SCENE_SEED"]),
        int(p106["SMOKE_SCENE_SEED"]),
        20300107,
    }

    for old_seed in previous_seeds:
        previous.update(scene_fingerprints(gen, old_seed, 1))

    # The #110 smoke and formal cohorts must also be disjoint.
    if smoke:
        previous.update(scene_fingerprints(gen, SCENE_SEED, N_SCENES))
    else:
        previous.update(
            scene_fingerprints(gen, SMOKE_SCENE_SEED, 1)
        )

    need(
        not (previous & set(fingerprints)),
        "Fresh cohort overlaps historical or smoke scenes",
    )

    return scenes, fingerprints


def observation_digest(target):
    actions = (*TRAIN, "mid", "HALF", "FULL")

    payload = {
        action: np.asarray(target[action].y_b, dtype=float).tolist()
        for action in actions
    }

    return digest(payload)


def scoring_fit_pack(fits):
    domains = {
        domain: {law: None for law in LAWS}
        for domain in DOMAINS
    }

    for law, domain in DOMAIN_BY_LAW.items():
        domains[domain][law] = fits[domain][law]

    return {"MATCHED5": domains}


def run_case(sid, band, mode, axis, level, base, ctx):
    p108 = ctx["p108"]
    p103 = ctx["p103"]
    p99 = ctx["p99"]
    gen = ctx["gen"]
    p77 = ctx["p77"]
    p22 = ctx["p22"]
    commands = ctx["commands"]
    starts = ctx["starts"]

    scene = gen["apply_condition"](base, mode, axis, level)

    observations, clean, fingerprints = p99["sample_noisy_probes"](
        gen,
        p22["RevisionProbe"],
        scene,
        mode,
        commands,
        sid,
        band,
        axis,
        level,
        NOISE_SEED,
    )

    target = observations["SHIFTED_ODE"]

    need(
        all(np.asarray(target[a].y_b).shape == (51,) for a in TRAIN)
        and sum(len(target[a].y_b) for a in TRAIN) == N_RESIDUALS,
        "Training residual count differs from 204",
    )

    fits = {}
    logs = {}

    for cap in BUDGETS:
        cap_key = str(cap)
        fits[cap_key] = {}
        logs[cap_key] = {}

        for solver in SOLVERS:
            fits[cap_key][solver] = {
                "wide": {},
                "outer_box": {},
            }
            logs[cap_key][solver] = {
                "wide": {},
                "outer_box": {},
            }

            for law in LAWS:
                domain = DOMAIN_BY_LAW[law]

                # Exact #108 implementation: no reimplementation.
                best, info = p108["fit_capped"](
                    target,
                    p77,
                    law,
                    domain,
                    starts,
                    solver,
                    cap,
                )

                fits[cap_key][solver][domain][law] = best
                logs[cap_key][solver][domain][law] = info

    # Verify 100-call traces are exact prefixes of 200-call traces.
    for solver in SOLVERS:
        for law in LAWS:
            domain = DOMAIN_BY_LAW[law]

            p108["prefix_audit"](
                logs["100"][solver][domain][law],
                logs["200"][solver][domain][law],
                f"{solver}/{law}/{mode}/{axis}/{level}",
            )

    # Heldout scoring occurs only after all fitting is finished.
    decisions = {}

    for cap in BUDGETS:
        cap_key = str(cap)
        decisions[cap_key] = {}

        for solver in SOLVERS:
            fitted = scoring_fit_pack(fits[cap_key][solver])

            decisions[cap_key][solver] = p103["scored_decision"](
                fitted,
                target,
                PRIMARY_CONFIG,
            )

    # Truth is accessed only after fitting and model selection.
    truth = p99["TRUTH"][mode]

    return {
        "key_PRIVATE": [sid, band, mode, axis, level],
        "scene_id": sid,
        "band_PRIVATE": band,
        "mode_PRIVATE": mode,
        "condition_PRIVATE": level,
        "true_law_PRIVATE": truth,
        "physical_scene_sha256_PRIVATE": gen["digest_object"](scene),
        "clean_fingerprints_PRIVATE": fingerprints,
        "target_observation_sha256_PRIVATE": observation_digest(target),
        "fits_PRIVATE": fits,
        "fit_logs_PRIVATE": logs,
        "decisions_PRIVATE": decisions,
        "prefix_audit_passed": True,
    }


def decision_correct(decision, truth):
    return int(
        bool(decision["available"])
        and decision["winner"] == truth
    )


def paired_contrast(rows, spec_a, spec_b, draws):
    """
    spec = (budget, solver, method).
    Positive effect means A is more accurate than B.
    All 384 cases stay in the denominator.
    """
    cap_a, solver_a, method_a = spec_a
    cap_b, solver_b, method_b = spec_b

    per_scene = defaultdict(list)
    status = Counter()

    for row in rows:
        truth = row["true_law_PRIVATE"]

        da = row["decisions_PRIVATE"][str(cap_a)][solver_a][method_a]
        db = row["decisions_PRIVATE"][str(cap_b)][solver_b][method_b]

        a = decision_correct(da, truth)
        b = decision_correct(db, truth)

        per_scene[row["scene_id"]].append(a - b)
        status[(a, b)] += 1

    need(
        len(per_scene) == N_SCENES
        and all(len(per_scene[s]) == 16 for s in range(N_SCENES)),
        "Paired scene-cluster denominator changed",
    )

    scene_values = np.asarray(
        [np.mean(per_scene[s]) for s in range(N_SCENES)],
        dtype=float,
    )

    ci = np.quantile(
        scene_values[draws].mean(axis=1),
        [0.025, 0.975],
    )

    both = status[(1, 1)]
    a_only = status[(1, 0)]
    b_only = status[(0, 1)]
    both_wrong = status[(0, 0)]

    return {
        "n": len(rows),
        "A": {
            "budget": cap_a,
            "solver": solver_a,
            "method": method_a,
        },
        "B": {
            "budget": cap_b,
            "solver": solver_b,
            "method": method_b,
        },
        "a_correct": both + a_only,
        "b_correct": both + b_only,
        "both_correct": both,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": both_wrong,
        "net": a_only - b_only,
        "effect": float(scene_values.mean()),
        "scene_cluster_CI95": [float(x) for x in ci],
    }


def summarize(rows):
    draws = np.random.default_rng(BOOT_SEED).integers(
        0, N_SCENES, size=(N_BOOT, N_SCENES)
    )

    stats = {}
    coverage = {}
    compute = {}

    for cap in BUDGETS:
        cap_key = str(cap)
        stats[cap_key] = {}
        coverage[cap_key] = {}
        compute[cap_key] = {}

        for solver in SOLVERS:
            stats[cap_key][solver] = {}
            coverage[cap_key][solver] = {}

            for method in METHODS:
                decisions = [
                    r["decisions_PRIVATE"][cap_key][solver][method]
                    for r in rows
                ]

                available = sum(int(d["available"]) for d in decisions)
                correct = sum(
                    decision_correct(d, r["true_law_PRIVATE"])
                    for d, r in zip(decisions, rows)
                )

                by_law = {}

                for law in LAWS:
                    subset = [
                        (r, d)
                        for r, d in zip(rows, decisions)
                        if r["true_law_PRIVATE"] == law
                    ]

                    by_law[law] = {
                        "n": len(subset),
                        "available": sum(
                            int(d["available"]) for _, d in subset
                        ),
                        "correct": sum(
                            decision_correct(d, r["true_law_PRIVATE"])
                            for r, d in subset
                        ),
                    }

                rate = available / len(rows)

                stats[cap_key][solver][method] = {
                    "n": len(rows),
                    "available": available,
                    "correct": correct,
                    "accuracy_all": correct / len(rows),
                    "availability_rate": rate,
                    "by_law": by_law,
                }

                coverage[cap_key][solver][method] = {
                    "available": available,
                    "denominator": len(rows),
                    "rate": rate,
                    "passes_90pct_guard": rate >= MIN_COVERAGE,
                }

            fit_logs = [
                r["fit_logs_PRIVATE"][cap_key][solver][
                    DOMAIN_BY_LAW[law]
                ][law]
                for r in rows
                for law in LAWS
            ]

            attempts = [
                a for info in fit_logs for a in info["attempts"]
            ]

            stop_counts = Counter(a["status"] for a in attempts)

            compute[cap_key][solver] = {
                "fit_units": len(fit_logs),
                "optimizer_start_records": len(attempts),
                "per_start_cap": cap // 5,
                "max_calls_total": sum(x["budget"] for x in fit_logs),
                "actual_204vector_calls": sum(
                    x["calls"] for x in fit_logs
                ),
                "unused_calls": sum(
                    x["unused_calls"] for x in fit_logs
                ),
                "unavailable_fits": sum(
                    not x["best_available"] for x in fit_logs
                ),
                "invalid_residual_calls": sum(
                    a["invalid_residual_calls"] for a in attempts
                ),
                "start_stop_status_counts": dict(stop_counts),
                "wall_seconds_descriptive_only": sum(
                    x["wall_seconds"] for x in fit_logs
                ),
            }

    pairs = {
        # One and only one primary endpoint.
        "PRIMARY_TRF_MINUS_SLSQP_CAP200_FULL": paired_contrast(
            rows,
            (200, "TRF_CAP", "MID_PLUS_FULL"),
            (200, "SLSQP_CAP", "MID_PLUS_FULL"),
            draws,
        ),

        "SECONDARY_TRF_MINUS_SLSQP_CAP100_FULL": paired_contrast(
            rows,
            (100, "TRF_CAP", "MID_PLUS_FULL"),
            (100, "SLSQP_CAP", "MID_PLUS_FULL"),
            draws,
        ),

        "SECONDARY_TRF_MINUS_SLSQP_CAP200_HALF": paired_contrast(
            rows,
            (200, "TRF_CAP", "MID_PLUS_HALF"),
            (200, "SLSQP_CAP", "MID_PLUS_HALF"),
            draws,
        ),

        "SECONDARY_TRF_MINUS_SLSQP_CAP100_HALF": paired_contrast(
            rows,
            (100, "TRF_CAP", "MID_PLUS_HALF"),
            (100, "SLSQP_CAP", "MID_PLUS_HALF"),
            draws,
        ),

        "DOSE_TRF_CAP100_FULL_MINUS_HALF": paired_contrast(
            rows,
            (100, "TRF_CAP", "MID_PLUS_FULL"),
            (100, "TRF_CAP", "MID_PLUS_HALF"),
            draws,
        ),

        "DOSE_TRF_CAP200_FULL_MINUS_HALF": paired_contrast(
            rows,
            (200, "TRF_CAP", "MID_PLUS_FULL"),
            (200, "TRF_CAP", "MID_PLUS_HALF"),
            draws,
        ),

        "DOSE_SLSQP_CAP100_FULL_MINUS_HALF": paired_contrast(
            rows,
            (100, "SLSQP_CAP", "MID_PLUS_FULL"),
            (100, "SLSQP_CAP", "MID_PLUS_HALF"),
            draws,
        ),

        "DOSE_SLSQP_CAP200_FULL_MINUS_HALF": paired_contrast(
            rows,
            (200, "SLSQP_CAP", "MID_PLUS_FULL"),
            (200, "SLSQP_CAP", "MID_PLUS_HALF"),
            draws,
        ),

        "BUDGET_TRF_CAP200_MINUS_CAP100_FULL": paired_contrast(
            rows,
            (200, "TRF_CAP", "MID_PLUS_FULL"),
            (100, "TRF_CAP", "MID_PLUS_FULL"),
            draws,
        ),

        "BUDGET_SLSQP_CAP200_MINUS_CAP100_FULL": paired_contrast(
            rows,
            (200, "SLSQP_CAP", "MID_PLUS_FULL"),
            (100, "SLSQP_CAP", "MID_PLUS_FULL"),
            draws,
        ),
    }

    return stats, coverage, compute, pairs


def run_smoke(ctx, scenes, fingerprints):
    p108 = ctx["p108"]
    p99 = ctx["p99"]
    gen = ctx["gen"]
    p77 = ctx["p77"]
    p22 = ctx["p22"]
    commands = ctx["commands"]
    starts = ctx["starts"]

    mode, axis, level = "elastic", "matched", "M0"
    band = "original"

    scene = gen["apply_condition"](
        scenes[0][band], mode, axis, level
    )

    observations, clean, fps = p99["sample_noisy_probes"](
        gen,
        p22["RevisionProbe"],
        scene,
        mode,
        commands,
        0,
        band,
        axis,
        level,
        SMOKE_NOISE_SEED,
    )

    target = observations["SHIFTED_ODE"]

    need(
        sum(len(target[a].y_b) for a in TRAIN) == N_RESIDUALS,
        "Smoke training residual count changed",
    )

    smoke_fits = {}

    for solver in SOLVERS:
        best, info = p108["fit_capped"](
            target,
            p77,
            "elastic",
            "wide",
            starts,
            solver,
            SMOKE_BUDGET,
        )

        need(
            info["calls"] <= SMOKE_BUDGET
            and info["per_start_cap"] == 10
            and len(info["attempts"]) == 5
            and sum(a["calls"] for a in info["attempts"])
            == info["calls"],
            f"Smoke hard-cap violation: {solver}",
        )

        smoke_fits[solver] = {
            "best_available": best is not None,
            "calls": info["calls"],
            "per_start_cap": info["per_start_cap"],
            "attempts": info["attempts"],
        }

    report = {
        "protocol": "P4-B3C independent fresh-scene interface smoke",
        "smoke": True,
        "n_scenes": 1,
        "n_cases": 1,
        "source_sha256": ctx["hashes"],
        "scene_seed": SMOKE_SCENE_SEED,
        "noise_seed": SMOKE_NOISE_SEED,
        "base_scene_fingerprints_PRIVATE": fingerprints,
        "conditioned_scene_sha256_PRIVATE": gen["digest_object"](scene),
        "training_residual_count": N_RESIDUALS,
        "budget": SMOKE_BUDGET,
        "per_start_cap": 10,
        "solver_interface": {
            s: {
                "calls": smoke_fits[s]["calls"],
                "best_available": smoke_fits[s]["best_available"],
            }
            for s in SOLVERS
        },
        "new_scientific_results": 0,
        "truth_used": False,
        "E4_accessed": False,
    }

    audit = {
        "source_sha256": ctx["hashes"],
        "clean_fingerprints_PRIVATE": fps,
        "target_observation_sha256_PRIVATE": observation_digest(target),
        "fits_PRIVATE": smoke_fits,
    }

    write_smoke(OUT / "p4b3c_smoke_report.json", report)
    write_smoke(OUT / "p4b3c_smoke_audit_PRIVATE.json", audit)

    print("[PASS] Fresh smoke scene is disjoint from historical cohorts")
    print("[PASS] 204 training residuals")
    print("[PASS] TRF_CAP and SLSQP_CAP hard 50-call smoke")
    print("[PASS] Five starts, 10 calls/start maximum")
    print("[NOTE] Smoke is not an accuracy estimate")
    print("P4-B3C SMOKE COMPLETE")


def run_formal(ctx, scenes, fingerprints):
    p108 = ctx["p108"]
    p103 = ctx["p103"]
    p99 = ctx["p99"]
    gen = ctx["gen"]
    p77 = ctx["p77"]
    starts = ctx["starts"]
    start_points = ctx["start_points"]

    protocol = {
        "name": "P4-B3C independent fresh-scene equal hard model-evaluation caps",
        "n_scenes": N_SCENES,
        "n_cases": N_CASES,
        "cases_per_scene": 16,
        "bands": list(gen["BANDS"]),
        "budgets": list(BUDGETS),
        "per_start_caps": [20, 40],
        "smoke_budget": SMOKE_BUDGET,
        "solvers": list(SOLVERS),
        "training_actions": list(TRAIN),
        "training_residual_count": N_RESIDUALS,
        "candidate_laws": list(LAWS),
        "WOO_domain_by_law": DOMAIN_BY_LAW,
        "heldout_actions": ["mid", "HALF", "FULL"],
        "primary_config": PRIMARY_CONFIG,
        "primary_budget": PRIMARY_BUDGET,
        "primary_method": PRIMARY_METHOD,
        "bootstrap_clusters": N_SCENES,
        "bootstrap_resamples": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "scene_seed": SCENE_SEED,
        "smoke_scene_seed": SMOKE_SCENE_SEED,
        "noise_seed": NOISE_SEED,
        "smoke_noise_seed": SMOKE_NOISE_SEED,
        "protocol_fixed_before_formal_outcomes": True,
        "E4_accessed": False,
    }

    manifest = {
        "protocol": protocol,
        "source_sha256": ctx["hashes"],
        "scene_fingerprints_PRIVATE": fingerprints,
        "start_points_PRIVATE": start_points,
        "start_points_sha256": digest(start_points),
    }

    manifest_path = OUT / "p4b3c_source_manifest.json"
    write_once(manifest_path, manifest)
    manifest_hash = digest(manifest)

    rows = []

    for sid, base_pair in enumerate(scenes):
        checkpoint = (
            OUT / "checkpoints" / f"scene_{sid:02d}.json"
        )

        base_fps = fingerprints[2 * sid: 2 * sid + 2]

        if checkpoint.is_file():
            pack = read(checkpoint)

            need(
                pack["scene_id"] == sid
                and pack["manifest_sha256"] == manifest_hash
                and pack["base_fingerprints_PRIVATE"] == base_fps
                and len(pack["rows"]) == 16
                and pack["rows_sha256"] == digest(pack["rows"]),
                f"Checkpoint mismatch: scene {sid}",
            )

            block = pack["rows"]
            print(f"[RESUME] scene {sid + 1}/24", flush=True)

        else:
            block = []

            for band in gen["BANDS"]:
                for mode in p99["MODES"]:
                    for axis, level in p99["CONDITIONS"][mode]:
                        block.append(
                            run_case(
                                sid,
                                band,
                                mode,
                                axis,
                                level,
                                base_pair[band],
                                ctx,
                            )
                        )

            need(
                len(block) == 16,
                f"Wrong case count in scene {sid}",
            )

            write_once(
                checkpoint,
                {
                    "scene_id": sid,
                    "manifest_sha256": manifest_hash,
                    "source_sha256": ctx["hashes"],
                    "base_fingerprints_PRIVATE": base_fps,
                    "rows_sha256": digest(block),
                    "rows": block,
                },
            )

            print(
                f"[DONE] scene {sid + 1}/24: 16 cases",
                flush=True,
            )

        rows.extend(block)

    need(
        len(rows) == N_CASES
        and len({tuple(r["key_PRIVATE"]) for r in rows}) == N_CASES,
        "Formal cohort incomplete or duplicated",
    )

    for sid in range(N_SCENES):
        need(
            sum(r["scene_id"] == sid for r in rows) == 16,
            f"Incomplete scene cluster: {sid}",
        )

    # Final hard-cap and trace checks.
    for row in rows:
        for cap in BUDGETS:
            for solver in SOLVERS:
                for law in LAWS:
                    domain = DOMAIN_BY_LAW[law]
                    info = row["fit_logs_PRIVATE"][str(cap)][solver][domain][law]

                    need(
                        info["calls"] <= cap
                        and info["per_start_cap"] == cap // 5
                        and len(info["attempts"]) == 5
                        and sum(a["calls"] for a in info["attempts"])
                        == info["calls"],
                        f"Hard budget violation: {row['key_PRIVATE']}/{solver}/{law}/{cap}",
                    )

    stats, coverage, compute, pairs = summarize(rows)

    primary = pairs["PRIMARY_TRF_MINUS_SLSQP_CAP200_FULL"]

    report = {
        "protocol": protocol["name"],
        "smoke": False,
        "protocol_fixed_before_formal_outcomes": True,
        "n_base_scenes": N_SCENES,
        "n_base_scene_bands": 48,
        "n_cases": N_CASES,
        "n_cases_per_scene": 16,
        "n_fit_units_per_solver_per_budget": N_CASES * len(LAWS),
        "n_fit_units_total": N_CASES * len(LAWS) * len(SOLVERS) * len(BUDGETS),
        "n_optimizer_start_records": (
            N_CASES * len(LAWS) * len(SOLVERS) * len(BUDGETS) * 5
        ),
        "prefix_audits": N_CASES * len(LAWS) * len(SOLVERS),
        "source_sha256": ctx["hashes"],
        "source_manifest_sha256": manifest_hash,
        "scene_seed": SCENE_SEED,
        "noise_seed": NOISE_SEED,
        "scene_fingerprints_PRIVATE": fingerprints,
        "training_actions": list(TRAIN),
        "training_residual_count": N_RESIDUALS,
        "candidate_laws": list(LAWS),
        "domain_by_law_WOO": DOMAIN_BY_LAW,
        "budgets": list(BUDGETS),
        "per_start_caps": [20, 40],
        "solvers": list(SOLVERS),
        "primary_endpoint": {
            "contrast": "TRF_CAP minus SLSQP_CAP",
            "budget": 200,
            "config": "WOO",
            "method": "MID_PLUS_FULL",
            "denominator": 384,
            "result": primary,
        },
        "secondary_and_diagnostic_contrasts": {
            k: v for k, v in pairs.items() if k !=
            "PRIMARY_TRF_MINUS_SLSQP_CAP200_FULL"
        },
        "aggregate_classification": stats,
        "full_denominator_coverage": coverage,
        "compute_cost": compute,
        "bootstrap": {
            "unit": "base_scene",
            "n_clusters": N_SCENES,
            "resamples": N_BOOT,
            "seed": BOOT_SEED,
            "method": "paired scene-cluster percentile 95% CI",
        },
        "truth_used_for_fit_or_selection": False,
        "heldout_used_for_fit": False,
        "adaptive_tuning": False,
        "E4_accessed": False,
        "limitations": [
            "Equal hard maximum residual-vector calls, NOT equal realized calls or wall time",
            "Every counted call evaluates the full 204-position training residual vector",
            "Finite-difference Jacobian residual calls are included",
            "Each budget is split across five starts; per-start caps are 20 and 40",
            "Best feasible evaluated point is retained even without optimizer success",
            "Unavailable decisions remain errors in the full 384-case denominator",
            "Optimizer success is not equivalent to correct mechanism identification",
            "24 base scenes are the independent bootstrap clusters; 384 rows are not IID",
            "The physical generator is one RK4 implementation, not hardware validation",
            "E4 remains sealed",
        ],
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }

    audit = {
        "rows_PRIVATE": rows,
        "source_sha256": ctx["hashes"],
        "source_manifest_sha256": manifest_hash,
    }

    write_once(OUT / "p4b3c_fresh_report.json", report)
    write_once(OUT / "p4b3c_fresh_audit_PRIVATE.json", audit)

    print("[PRIMARY]", primary)
    print("[COVERAGE]", coverage)
    print("[COMPUTE]", compute)
    print("[E4] SEALED")
    print("P4-B3C COMPLETE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    print("=" * 76)
    print("P4-B3C — FRESH-SCENE EQUAL HARD MODEL-EVALUATION CAP CONFIRMATION")
    print("=" * 76)
    print("Smoke:", args.smoke, "| E4 sealed")

    ctx = preflight()
    print("[PASS] #108 source, #109 closure, upstream lineage")

    scenes, fingerprints = generate_scenes(ctx, args.smoke)
    print("[PASS] Independent fresh scene fingerprints")
    print("[PASS] Historical and smoke cohort exclusion")

    if args.smoke:
        run_smoke(ctx, scenes, fingerprints)
    else:
        run_formal(ctx, scenes, fingerprints)


if __name__ == "__main__":
    main()
