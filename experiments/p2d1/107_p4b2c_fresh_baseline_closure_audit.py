
r"""#107 / P4-B2C — read-only closure audit of P4-B2B (#106).

Run:
  python 107_p4b2c_fresh_baseline_closure_audit.py --smoke
  python 107_p4b2c_fresh_baseline_closure_audit.py

No simulations, no fits, no E4 access.
Historical files are read-only. Output is written only to results/p4b2c.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import os
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "results"
IN = BASE / "p4b2b"
OUT = BASE / "p4b2c"

R106 = IN / "p4b2b_fresh_report.json"
A106 = IN / "p4b2b_fresh_audit_PRIVATE.json"
M106 = IN / "p4b2b_source_manifest.json"
S106 = IN / "p4b2b_smoke_report.json"
SCRIPT106 = HERE / "106_p4b2b_fresh_classical_identification_comparison.py"

SOURCE_PATHS = {
    "P105": HERE / "105_p4b2a_independent_identification_baseline_pilot.py",
    "R105": BASE / "p4b2a" / "p4b2a_pilot_report.json",
    "A105": BASE / "p4b2a" / "p4b2a_pilot_audit_PRIVATE.json",
    "R103": BASE / "p4g2c" / "p4g2c_fresh_report.json",
    "M103": BASE / "p4g2c" / "p4g2c_source_manifest.json",
    "R99": BASE / "p4g1b" / "p4g1b_fresh_report.json",
    "R98": BASE / "p4g1a" / "p4g1a_feasibility_report.json",
    "S98": BASE / "p4g1a" / "p4g1a_smoke_report.json",
}

ARMS = ("EQERR_SG", "TRF_OUT", "SLSQP_MATCHED5_REFERENCE")
CFGS = ("WWW", "OOO", "WOO")
METHODS = ("MID_PLUS_HALF", "MID_PLUS_FULL")
LAWS = ("elastic", "deadzone", "cubic")
DOMAINS = ("wide", "outer_box")

ENDPOINTS = {
    "PRIMARY_EQERR_MINUS_SLSQP_WOO_FULL":
        ("EQERR_SG", "MID_PLUS_FULL",
         "SLSQP_MATCHED5_REFERENCE", "MID_PLUS_FULL", False),

    "SECONDARY_TRF_MINUS_SLSQP_WOO_FULL":
        ("TRF_OUT", "MID_PLUS_FULL",
         "SLSQP_MATCHED5_REFERENCE", "MID_PLUS_FULL", False),

    "SECONDARY_EQERR_MINUS_TRF_WOO_FULL":
        ("EQERR_SG", "MID_PLUS_FULL",
         "TRF_OUT", "MID_PLUS_FULL", False),

    "DIAGNOSTIC_EQERR_SG_NM2_FULL_MINUS_HALF":
        ("EQERR_SG", "MID_PLUS_FULL",
         "EQERR_SG", "MID_PLUS_HALF", True),

    "DIAGNOSTIC_TRF_OUT_NM2_FULL_MINUS_HALF":
        ("TRF_OUT", "MID_PLUS_FULL",
         "TRF_OUT", "MID_PLUS_HALF", True),

    "DIAGNOSTIC_SLSQP_MATCHED5_REFERENCE_NM2_FULL_MINUS_HALF":
        ("SLSQP_MATCHED5_REFERENCE", "MID_PLUS_FULL",
         "SLSQP_MATCHED5_REFERENCE", "MID_PLUS_HALF", True),
}


def ensure(ok, message):
    if not ok:
        raise RuntimeError("AUDIT FAILED: " + str(message))


def load(path):
    ensure(path.is_file(), f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(obj):
    data = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def same(a, b, what):
    ensure(a == b, what)


def close(a, b, what, tol=1e-10):
    ensure(
        math.isfinite(float(a)) and math.isfinite(float(b)),
        what + ": nonfinite",
    )
    ensure(
        math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol),
        f"{what}: {a} differs from {b}",
    )


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        obj, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"

    if path.is_file():
        same(
            path.read_text(encoding="utf-8"),
            data,
            f"existing output differs; do not overwrite: {path}",
        )
        return

    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def get_checkpoint(sid, archive=None):
    name = f"scene_{sid:02d}.json"

    if archive is None:
        return load(IN / "checkpoints" / name)

    with zipfile.ZipFile(archive) as z:
        matches = [
            x for x in z.namelist()
            if x.endswith("/" + name)
        ]
        ensure(
            len(matches) == 1,
            f"checkpoint {sid} missing or ambiguous in ZIP",
        )
        return json.loads(z.read(matches[0]))


def preflight():
    report, audit, manifest, smoke = map(
        load, (R106, A106, M106, S106)
    )

    same(
        report["source_sha256"],
        manifest["source_sha256"],
        "report/manifest source hashes",
    )
    same(
        report["source_sha256"],
        smoke["source_sha256"],
        "report/smoke source hashes",
    )
    same(report["scene_seed"], manifest["scene_seed"], "scene seed")
    same(report["noise_seed"], manifest["noise_seed"], "noise seed")
    same(
        report["bootstrap"]["seed"],
        manifest["bootstrap_seed"],
        "bootstrap seed",
    )
    same(report["n_cases"], 384, "total cases")
    same(report["n_base_scenes"], 24, "base scenes")
    same(report["n_training_B_positions"], 204, "training samples")
    same(report["generator"], "SHIFTED_ODE", "generator")

    ensure(
        report["protocol_frozen_before_outcome"]
        and not report["new_e4_accessed"]
        and not report["heldout_used_for_fit"]
        and not report["truth_used_during_fit"]
        and not report["adaptive_tuning"]
        and not manifest["E4_accessed"]
        and not smoke["E4_accessed"],
        "protocol, holdout or E4 flags inconsistent",
    )

    rows = audit["rows_PRIVATE"]

    ensure(
        len(rows) == 384
        and len({tuple(x["key_PRIVATE"]) for x in rows}) == 384,
        "duplicate or missing cases",
    )
    ensure(
        len(manifest["scene_fingerprints_PRIVATE"]) == 48,
        "expected 48 physical scene fingerprints",
    )

    warnings = []
    source = {}

    for tag, path in {**SOURCE_PATHS, "SELF": SCRIPT106}.items():
        if path.is_file():
            observed = sha(path)
            expected = report["source_sha256"][tag]

            source[tag] = {
                "present": True,
                "match_executed_hash": observed == expected,
                "observed_sha256": observed,
                "executed_sha256": expected,
            }

            if observed != expected:
                warnings.append(f"SOURCE_HASH_MISMATCH:{tag}")
        else:
            source[tag] = {
                "present": False,
                "executed_sha256": report["source_sha256"][tag],
            }
            warnings.append(f"SOURCE_NOT_LOCALLY_AVAILABLE:{tag}")

    return report, rows, manifest, source, warnings


def valid_decision(row, arm, cfg, method):
    decision = row["decisions"][arm][cfg][method]
    scores = decision["scores_mm_PRIVATE"]

    same(set(scores), set(LAWS), "candidate score keys")

    ensure(
        all(math.isfinite(float(scores[l])) for l in LAWS),
        "nonfinite candidate score",
    )
    same(bool(decision["available"]), True, "unexpected abstention")
    same(
        decision["winner"],
        min(LAWS, key=lambda l: scores[l]),
        "winner is not lowest score",
    )

    return int(decision["winner"] == row["true_law_PRIVATE"])


def verify_rows(rows, report, smoke):
    classes = {}
    differences = {}
    checked_scores = 0

    for arm in ARMS:
        classes[arm] = {}

        for cfg in CFGS:
            classes[arm][cfg] = {}

            for meth in METHODS:
                per_law = {}

                for law in LAWS:
                    part = [
                        r for r in rows
                        if r["true_law_PRIVATE"] == law
                    ]
                    per_law[law] = {
                        "n": len(part),
                        "correct": sum(
                            valid_decision(r, arm, cfg, meth)
                            for r in part
                        ),
                    }

                stat = {
                    "n": len(rows),
                    "correct": sum(
                        x["correct"] for x in per_law.values()
                    ),
                    "available": len(rows),
                    "by_truth": per_law,
                }

                classes[arm][cfg][meth] = stat

                if not smoke:
                    same(
                        stat,
                        report["classification"][arm][cfg][meth],
                        f"{arm}/{cfg}/{meth} aggregated counts",
                    )

                checked_scores += len(rows) * len(LAWS)

    for cfg in CFGS:
        differences[cfg] = {}
        for meth in METHODS:
            differences[cfg][meth] = sum(
                r["decisions"]["TRF_OUT"][cfg][meth]["winner"]
                != r["decisions"]["SLSQP_MATCHED5_REFERENCE"][cfg][meth]["winner"]
                for r in rows
            )

    return classes, differences, checked_scores


def verify_costs_and_nested(rows, report):
    costs = {}
    nested = {}

    for arm in ARMS:
        nested[arm] = {}

        for law in LAWS:
            count = 0

            for row in rows:
                pair = row["fits_PRIVATE"][arm]
                wide = pair["wide"][law]
                outer = pair["outer_box"][law]
                gap = float(wide["mse"]) - float(outer["mse"])
                flag = bool(gap > 1e-6)

                stored = row["nested_PRIVATE"][arm][law]

                close(gap, stored["gap"], "nested fit gap", 1e-8)
                same(flag, stored["violation"], "nested flag")
                count += flag

            nested[arm][law] = count

        if arm == "EQERR_SG":
            calls = sum(
                int(row["fit_logs_PRIVATE"][arm][d][law]["attempts"])
                for row in rows
                for d in DOMAINS
                for law in LAWS
            )
            cost_key = "total_regression_grid_evaluations"

        else:
            field = (
                "residual_calls_ALL"
                if arm == "TRF_OUT"
                else "nfev_total"
            )
            calls = sum(
                int(row["fit_logs_PRIVATE"][arm][d][law][field])
                for row in rows
                for d in DOMAINS
                for law in LAWS
            )
            cost_key = "total_optimizer_objective_calls"

        if arm == "SLSQP_MATCHED5_REFERENCE":
            wall = sum(
                float(row["SLSQP_fit_plus_internal_scoring_wall_seconds"])
                for row in rows
            )
            wall_key = "total_fit_plus_internal_scoring_wall_seconds"
        else:
            wall = sum(
                float(row["fit_logs_PRIVATE"][arm][d][law]["elapsed_s"])
                for row in rows
                for d in DOMAINS
                for law in LAWS
            )
            wall_key = "total_fit_wall_seconds"

        costs[arm] = {
            "calls_or_grid_evaluations": calls,
            "reported_cost_unit": report["compute_cost"][arm]["cost_unit"],
            "reported_wall_seconds": wall,
            "note": "Timing definitions differ across methods",
        }

        if len(rows) == 384:
            same(
                calls,
                report["compute_cost"][arm][cost_key],
                "fit call count",
            )
            close(
                wall,
                report["compute_cost"][arm][wall_key],
                "wall time",
                1e-6,
            )
            same(
                {k: v for k, v in nested[arm].items() if v},
                report["nested_violations"][arm],
                "nested violations",
            )

    return costs, nested


def verify_pairs(rows, report):
    rng = np.random.default_rng(report["bootstrap"]["seed"])
    draw = rng.integers(0, 24, size=(2000, 24))
    result = {}

    for name, (aa, ma, bb, mb, nm2_only) in ENDPOINTS.items():
        subset = [
            r for r in rows
            if (
                not nm2_only
                or (
                    r["true_law_PRIVATE"] == "cubic"
                    and r["condition_PRIVATE"] == "M2"
                )
            )
        ]

        per = defaultdict(list)
        cnt = Counter()

        for row in subset:
            a = valid_decision(row, aa, "WOO", ma)
            b = valid_decision(row, bb, "WOO", mb)

            per[row["scene_id"]].append(a - b)
            cnt[(a, b)] += 1

        same(
            set(per),
            set(range(24)),
            name + ": cluster coverage",
        )
        same(
            set(map(len, per.values())),
            {4 if nm2_only else 16},
            name + ": cluster size",
        )

        vals = np.array([
            np.mean(per[i]) for i in range(24)
        ])

        ci = np.quantile(
            vals[draw].mean(axis=1),
            [.025, .975],
        )

        item = {
            "n": len(subset),
            "a_correct": cnt[(1, 1)] + cnt[(1, 0)],
            "b_correct": cnt[(1, 1)] + cnt[(0, 1)],
            "both_correct": cnt[(1, 1)],
            "a_only": cnt[(1, 0)],
            "b_only": cnt[(0, 1)],
            "both_wrong": cnt[(0, 0)],
            "net": cnt[(1, 0)] - cnt[(0, 1)],
            "effect": float(vals.mean()),
            "scene_cluster_CI95": [
                float(ci[0]),
                float(ci[1]),
            ],
        }

        old = report["primary_and_secondary"][name]

        for k in (
            "n", "a_correct", "b_correct", "both_correct",
            "a_only", "b_only", "both_wrong", "net"
        ):
            same(item[k], old[k], f"{name}/{k}")

        close(item["effect"], old["effect"], name + "/effect")

        for j in (0, 1):
            close(
                item["scene_cluster_CI95"][j],
                old["scene_cluster_CI95"][j],
                name + "/CI",
            )

        result[name] = item

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--checkpoint-zip",
        type=Path,
        help="Optional ZIP containing the 24 original checkpoints",
    )
    args = parser.parse_args()

    print("=" * 76)
    print("P4-B2C — READ-ONLY #106 CLASSICAL IDENTIFICATION CLOSURE")
    print("=" * 76)
    print(
        f"Smoke: {args.smoke} | no simulations/fits | E4 sealed"
    )

    report, all_rows, manifest, source, warnings = preflight()

    count = 16 if args.smoke else 384
    rows = all_rows[:count]

    if args.smoke:
        same(
            [r["scene_id"] for r in rows],
            [0] * 16,
            "smoke scene selection",
        )
    else:
        same(
            [
                sum(r["scene_id"] == i for r in rows)
                for i in range(24)
            ],
            [16] * 24,
            "scene coverage",
        )

    seen = 0

    for sid in range(1 if args.smoke else 24):
        pack = get_checkpoint(sid, args.checkpoint_zip)

        block = [
            r for r in rows if r["scene_id"] == sid
        ]

        same(pack["scene_id"], sid, "checkpoint scene ID")
        same(
            pack["source_sha256"],
            report["source_sha256"],
            "checkpoint source hashes",
        )
        same(
            pack["base_fingerprints"],
            manifest["scene_fingerprints_PRIVATE"][2*sid:2*sid+2],
            "checkpoint base fingerprints",
        )
        same(
            pack["rows_sha256"],
            digest(pack["rows"]),
            "checkpoint row digest",
        )
        same(
            pack["rows"],
            block,
            "checkpoint content vs formal audit",
        )

        seen += len(block)

    print(
        f"[PASS] {seen} exact checkpoint rows; "
        "source/manifest/smoke internal lineage"
    )

    classes, divergence, scores = verify_rows(
        rows, report, args.smoke
    )

    print(
        f"[PASS] {scores} checked candidate score values "
        "and winners"
    )

    if args.smoke:
        doc = {
            "protocol": "P4-B2C interface smoke only",
            "smoke": True,
            "n_rows_checked": 16,
            "score_values_checked": scores,
            "checkpoint_rows_checked": seen,
            "source_provenance_warnings": warnings,
            "E4_accessed": False,
            "new_fits": 0,
            "new_simulations": 0,
        }
        file = OUT / "p4b2c_smoke_report.json"
        save(file, doc)
        print("[NOTE] Smoke has no scientific interpretation")
        print("Saved:", file)
        return

    costs, nested = verify_costs_and_nested(rows, report)
    pairs = verify_pairs(rows, report)

    print(
        "[PASS] 18 aggregate cells, all six paired effects "
        "and 2000 cluster draws"
    )
    print(
        "[PASS] Fit costs and nested flags reconstructed"
    )
    print(
        "[PRIMARY]",
        pairs["PRIMARY_EQERR_MINUS_SLSQP_WOO_FULL"],
    )
    print(
        "[SOLVER WINNER DISAGREEMENTS]",
        divergence,
    )
    print(
        "[COMPUTE]",
        {
            k: v["calls_or_grid_evaluations"]
            for k, v in costs.items()
        },
    )

    if warnings:
        print("[PROVENANCE CAUTION]", warnings)
        print(
            "[NOTE] Archive the exact executed source; "
            "do not change historical results"
        )

    status = (
        "PASS_WITH_SOURCE_ARCHIVE_WARNINGS"
        if warnings
        else "PASS_WITH_COMPUTE_AND_MODEL_LIMITATIONS"
    )

    doc = {
        "protocol": "P4-B2C read-only closure of #106",
        "status": status,
        "n_cases": 384,
        "n_scene_clusters": 24,
        "checkpoint_rows_verified": seen,
        "score_values_verified": scores,
        "source_provenance": source,
        "source_provenance_warnings": warnings,
        "classification": classes,
        "paired_effects": pairs,
        "TRF_vs_SLSQP_winner_disagreements": divergence,
        "compute_cost_recomputed": costs,
        "nested_violations": nested,
        "interpretation": [
            "Predeclared EQERR-minus-SLSQP contrast",
            "TRF and SLSQP match WOO decisions but not all WWW decisions",
            "Equal data access is not equal compute",
            "EQERR equation-error cost differs from trajectory residual cost",
            "Nested optimization violations persist",
            "One RK4 generator family, not hardware generalization",
            "24 independent scene clusters, not 384 IID cases",
        ],
        "E4_accessed": False,
        "new_fits": 0,
        "new_simulations": 0,
    }

    audit = {
        "case_keys_PRIVATE": [
            r["key_PRIVATE"] for r in rows
        ],
        "row_digest_SHA256": digest(rows),
        "source_hash_expected": report["source_sha256"],
        "source_verification": source,
        "winner_comparisons_by_config": divergence,
        "all_pairs_recomputed": pairs,
        "checkpoint_count": 24,
        "E4_accessed": False,
    }

    rp = OUT / "p4b2c_closure_report.json"
    ap = OUT / "p4b2c_closure_audit_PRIVATE.json"

    save(rp, doc)
    save(ap, audit)

    print("[SCIENTIFIC STATUS]", status)
    print("Saved:", rp)
    print("Saved:", ap)
    print("P4-B2C COMPLETE")


if __name__ == "__main__":
    main()
