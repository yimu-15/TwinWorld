
"""P4-B3B — READ-ONLY #108 equal-budget pilot closure.

No fitting, no simulation, no MuJoCo, no ODE generation, no E4.
Verifies source lineage, checkpoint hashes, all 3 historical rows,
all 100/200 budgets, 45 starts per solver/budget, best-fit reconstruction,
100-to-200 trace prefixes, decisions, and smoke parity.

Historical audit only. No confirmatory scientific inference.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
BASE = HERE / "results"
OUT = BASE / "p4b3b"

R108 = BASE / "p4b3a" / "p4b3a_pilot_report.json"
A108 = BASE / "p4b3a" / "p4b3a_pilot_audit_PRIVATE.json"
RS = BASE / "p4b3a" / "p4b3a_smoke_report.json"
AS = BASE / "p4b3a" / "p4b3a_smoke_audit_PRIVATE.json"

CHECKPOINTS = BASE / "p4b3a" / "checkpoints"

P105 = HERE / "105_p4b2a_independent_identification_baseline_pilot.py"
P103 = HERE / "103_p4g2c_fresh_frozen_vs_refit_confirmation.py"
P108 = HERE / "108_p4b3a_equal_model_evaluation_budget_pilot.py"

R106 = BASE / "p4b2b" / "p4b2b_fresh_report.json"
M106 = BASE / "p4b2b" / "p4b2b_source_manifest.json"
R107 = BASE / "p4b2c" / "p4b2c_closure_report.json"

LAWS = ("elastic", "deadzone", "cubic")
DOMAIN = {
    "elastic": "wide",
    "deadzone": "outer_box",
    "cubic": "outer_box",
}
SOLVERS = ("TRF_CAP", "SLSQP_CAP")
FORMAL_BUDGETS = (100, 200)
SMOKE_BUDGET = 50
N_STARTS = 5
N_RESIDUALS = 204

EXPECTED_KEYS = [
    [0, "original", "elastic", "matched", "M0"],
    [0, "original", "deadzone", "matched", "M0"],
    [0, "original", "nonlinear", "matched", "M0"],
]


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    need(path.is_file(), f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    need(path.is_file(), f"Missing source: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def close(a, b, atol=1e-10, rtol=1e-10):
    return math.isclose(
        float(a), float(b),
        abs_tol=atol, rel_tol=rtol,
    )


def vector_close(a, b, tol=1e-11):
    return (
        len(a) == len(b)
        and all(
            math.isclose(float(x), float(y),
                         rel_tol=tol, abs_tol=tol)
            for x, y in zip(a, b)
        )
    )


def verify_sources(report):
    src = report["source_sha256"]

    paths = {
        "P105": P105,
        "P103": P103,
        "P108": P108,
        "R106": R106,
        "M106": M106,
        "R107": R107,
    }

    for key, path in paths.items():
        expected_key = "SELF" if key == "P108" else key
        need(
            src[expected_key] == sha(path),
            f"Source SHA mismatch: {expected_key}",
        )

    need(
        src["protocol_budget_caps"] == [100, 200]
        and src["smoke_budget"] == 50
        and src["same_five_starts"] is True
        and src["domain_by_law_WOO"] == DOMAIN,
        "Frozen budget/protocol definition changed",
    )

    return {
        key: sha(path)
        for key, path in paths.items()
    }


def verify_metadata(report, smoke=False):
    need(
        report["protocol"]
        == "P4-B3A historical WOO-only maximum model-evaluations feasibility",
        "Protocol name mismatch",
    )

    need(
        report["smoke"] is smoke,
        "Smoke/formal flag mismatch",
    )

    need(
        report["training_actions"]
        == ["e1", "e2", "weak", "strong"]
        and report["training_residual_count"] == N_RESIDUALS
        and report["methods"] == list(SOLVERS)
        and report["domain_by_law_WOO"] == DOMAIN,
        "Frozen fitting interface changed",
    )

    need(
        report["new_scenes"] is False
        and report["E4_accessed"] is False
        and report["no_confirmatory_inference"] is True
        and report["truth_not_used_for_fitting"] is True
        and report["heldout_not_used_for_fitting"] is True
        and report["scoring_not_used_to_select_budget"] is True,
        "Scientific isolation guard failed",
    )

    if smoke:
        need(
            report["n_cases"] == 1
            and report["budgets"] == [50]
            and report["per_start_caps"] == [10],
            "Smoke protocol changed",
        )
    else:
        need(
            report["n_cases"] == 3
            and report["n_base_scenes"] == 1
            and report["budgets"] == [100, 200]
            and report["per_start_caps"] == [20, 40],
            "Formal protocol changed",
        )


def laws_for(smoke):
    return ("elastic",) if smoke else LAWS


def domain_for(law, smoke):
    return "wide" if smoke else DOMAIN[law]


def verify_trace(trace, law):
    expected_dim = 2 if law == "elastic" else 3

    for i, item in enumerate(trace):
        q = item["q"]
        mse = item["mse"]

        need(
            len(q) == expected_dim,
            f"Parameter dimension mismatch at call {i + 1}/{law}",
        )
        need(
            all(math.isfinite(float(x)) for x in q)
            and math.isfinite(float(mse))
            and mse >= 0,
            f"Non-finite trace at call {i + 1}/{law}",
        )
        need(
            isinstance(item["admissible"], bool),
            "Missing admissibility flag",
        )


def reconstruct_best(attempts, solver):
    candidates = []

    for i, attempt in enumerate(attempts):
        for j, item in enumerate(attempt["trace_PRIVATE"]):
            if item["admissible"]:
                candidates.append((
                    float(item["mse"]),
                    i,
                    j,
                    item,
                    attempt,
                ))

    need(candidates, "No admissible evaluated parameter found")

    # Stable order reproduces the original strict-less-than selection.
    mse, i, j, item, attempt = min(
        candidates, key=lambda x: x[0]
    )

    return {
        "mse": mse,
        "params_PRIVATE": item["q"],
        "at_call": j + 1,
        "origin": f"{solver}_start_{i}",
        "converged": bool(attempt["solver_success"]),
    }


def verify_attempts(info, solver, law, budget):
    per = budget // N_STARTS

    need(
        info["budget"] == budget
        and info["per_start_cap"] == per
        and len(info["attempts"]) == N_STARTS,
        "Per-start budget configuration mismatch",
    )

    total = 0

    for attempt in info["attempts"]:
        calls = attempt["calls"]
        trace = attempt["trace_PRIVATE"]

        need(
            calls <= per
            and calls == len(trace),
            "Per-start hard cap or trace length mismatch",
        )

        need(
            attempt["cap"] == per
            and attempt["invalid_residual_calls"] == 0,
            "Invalid call or cap metadata",
        )

        need(
            attempt["status"] in (
                "budget_cap",
                "optimizer_success",
                "optimizer_non_success",
            ),
            "Unknown optimizer termination status",
        )

        if attempt["status"] == "budget_cap":
            need(
                calls == per and not attempt["solver_success"],
                "Budget-cap termination inconsistency",
            )

        if attempt["status"] == "optimizer_success":
            need(
                attempt["solver_success"],
                "Optimizer success flag mismatch",
            )

        verify_trace(trace, law)

        if trace:
            local = min(
                (x for x in trace if x["admissible"]),
                key=lambda x: x["mse"],
                default=None,
            )

            need(local is not None, "No local feasible candidate")

            need(
                close(local["mse"], attempt["best_mse"])
                and vector_close(
                    local["q"],
                    attempt["best_params_PRIVATE"],
                ),
                "Per-start best parameter cannot be reconstructed",
            )

        total += calls

    need(
        total == info["calls"]
        and info["unused_calls"] == budget - total
        and total <= budget,
        f"{solver} total budget mismatch",
    )

    need(
        info["best_available"] is True,
        "Unavailable fit in formal pilot",
    )


def verify_fit(row, cap, solver, law, smoke=False):
    domain = domain_for(law, smoke)
    info = row["fit_logs_PRIVATE"][str(cap)][solver][domain][law]
    fit = row["fits_PRIVATE"][str(cap)][solver][domain][law]

    verify_attempts(info, solver, law, cap)

    rebuilt = reconstruct_best(info["attempts"], solver)

    need(
        close(fit["mse"], rebuilt["mse"])
        and vector_close(
            fit["params_PRIVATE"],
            rebuilt["params_PRIVATE"],
        )
        and fit["at_call"] == rebuilt["at_call"]
        and fit["origin"] == rebuilt["origin"]
        and fit["converged"] == rebuilt["converged"],
        f"Global best fit reconstruction failed: {solver}/{law}/{cap}",
    )

    return info


def verify_prefix(row):
    """Check exact deterministic 100-cap trajectory prefix of 200-cap run."""

    for solver in SOLVERS:
        for law in LAWS:
            domain = DOMAIN[law]

            low = row["fit_logs_PRIVATE"]["100"][solver][domain][law]
            high = row["fit_logs_PRIVATE"]["200"][solver][domain][law]

            for i, (a, b) in enumerate(zip(
                low["attempts"], high["attempts"]
            )):
                ta = a["trace_PRIVATE"]
                tb = b["trace_PRIVATE"]

                need(
                    len(tb) >= len(ta),
                    f"200-budget trace shorter than 100-budget trace: {solver}/{law}/{i}",
                )

                for j, x in enumerate(ta):
                    y = tb[j]

                    need(
                        x["admissible"] == y["admissible"]
                        and vector_close(x["q"], y["q"])
                        and close(
                            x["mse"], y["mse"],
                            atol=1e-9, rtol=1e-10,
                        ),
                        f"Prefix mismatch: {solver}/{law}/start{i}/call{j}",
                    )


def verify_decisions(row):
    for cap in FORMAL_BUDGETS:
        for solver in SOLVERS:
            decisions = row["decisions_PRIVATE"][str(cap)][solver]

            need(
                set(decisions) == {"MID_PLUS_HALF", "MID_PLUS_FULL"},
                "Missing heldout decision cell",
            )

            for method, result in decisions.items():
                scores = result["scores_mm_PRIVATE"]

                need(
                    result["available"] is True
                    and set(scores) == set(LAWS)
                    and all(
                        math.isfinite(float(x))
                        for x in scores.values()
                    ),
                    "Invalid candidate score cell",
                )

                expected = min(scores, key=scores.get)

                need(
                    result["winner"] == expected,
                    f"Winner-score mismatch: {cap}/{solver}/{method}",
                )


def aggregate_rows(rows, cap, smoke=False):
    laws = laws_for(smoke)
    result = {}

    for solver in SOLVERS:
        fit_units = 0
        max_calls = 0
        actual_calls = 0
        unused = 0
        unavailable = 0
        wall = 0.0
        stops = Counter()

        for row in rows:
            for law in laws:
                domain = domain_for(law, smoke)
                info = row["fit_logs_PRIVATE"][str(cap)][solver][domain][law]

                fit_units += 1
                max_calls += cap
                actual_calls += info["calls"]
                unused += info["unused_calls"]
                unavailable += int(not info["best_available"])
                wall += info["wall_seconds"]

                stops.update(
                    a["status"] for a in info["attempts"]
                )

        result[solver] = {
            "fit_units": fit_units,
            "max_calls_total": max_calls,
            "actual_204vector_calls": actual_calls,
            "unused_calls": unused,
            "start_stop_status_counts": dict(stops),
            "unavailable_fits": unavailable,
            "wall_seconds_descriptive_only": wall,
        }

    return result


def verify_aggregate(report, rows, smoke=False):
    caps = report["budgets"]

    for cap in caps:
        expected = aggregate_rows(rows, cap, smoke)

        actual = report["aggregate"][str(cap)]

        for solver in SOLVERS:
            a = actual[solver]
            e = expected[solver]

            for key in (
                "fit_units",
                "max_calls_total",
                "actual_204vector_calls",
                "unused_calls",
                "start_stop_status_counts",
                "unavailable_fits",
            ):
                need(
                    a[key] == e[key],
                    f"Aggregate mismatch: {cap}/{solver}/{key}",
                )

            need(
                close(
                    a["wall_seconds_descriptive_only"],
                    e["wall_seconds_descriptive_only"],
                    atol=1e-8,
                    rtol=1e-10,
                ),
                "Wall-time aggregate reconstruction mismatch",
            )

    return {
        str(cap): aggregate_rows(rows, cap, smoke)
        for cap in caps
    }


def verify_checkpoints(report, rows):
    need(
        CHECKPOINTS.is_dir(),
        f"Missing checkpoint directory: {CHECKPOINTS}",
    )

    expected_names = {
        "original_elastic_M0.json",
        "original_deadzone_M0.json",
        "original_nonlinear_M0.json",
    }

    files = list(CHECKPOINTS.glob("*.json"))

    need(
        {p.name for p in files} == expected_names,
        "Checkpoint file set mismatch",
    )

    by_key = {
        canonical(row["key_PRIVATE"]): row
        for row in rows
    }

    verified = []

    for path in files:
        pack = read(path)
        row = pack["row"]

        need(
            pack["source_sha256"] == report["source_sha256"],
            f"Checkpoint source lineage mismatch: {path.name}",
        )

        need(
            digest(row) == pack["row_sha256"],
            f"Checkpoint row SHA mismatch: {path.name}",
        )

        key = canonical(row["key_PRIVATE"])

        need(
            key in by_key
            and digest(row) == digest(by_key[key]),
            f"Checkpoint/audit row mismatch: {path.name}",
        )

        verified.append({
            "filename": path.name,
            "row_sha256": pack["row_sha256"],
            "status": "PASS",
        })

    return verified


def verify_smoke(formal_source):
    report = read(RS)
    audit = read(AS)

    verify_metadata(report, smoke=True)

    need(
        report["source_sha256"] == formal_source,
        "Smoke/formal source lineage differs",
    )

    rows = audit["rows_PRIVATE"]

    need(
        len(rows) == 1
        and rows[0]["key_PRIVATE"]
        == [0, "original", "elastic", "matched", "M0"],
        "Smoke case identity mismatch",
    )

    for solver in SOLVERS:
        info = rows[0]["fit_logs_PRIVATE"]["50"][solver]["wide"]["elastic"]

        verify_attempts(info, solver, "elastic", 50)

        need(
            info["calls"] == 50
            and info["unused_calls"] == 0
            and all(
                x["status"] == "budget_cap"
                for x in info["attempts"]
            ),
            "Smoke budget behavior mismatch",
        )

    agg = verify_aggregate(report, rows, smoke=True)

    return {
        "status": "PASS",
        "n_cases": 1,
        "aggregate": agg,
    }


def retrospective_diagnostic(rows):
    """Historical labels are inspected only after fitting; never used for fitting."""

    out = {}

    for cap in FORMAL_BUDGETS:
        out[str(cap)] = {}

        for solver in SOLVERS:
            out[str(cap)][solver] = {}

            for method in ("MID_PLUS_HALF", "MID_PLUS_FULL"):
                correct = 0

                for row in rows:
                    truth = row["label_for_retrospective_audit_ONLY"]
                    pred = row["decisions_PRIVATE"][str(cap)][solver][method]["winner"]
                    correct += int(pred == truth)

                out[str(cap)][solver][method] = {
                    "correct": correct,
                    "n": len(rows),
                    "accuracy_descriptive_only": correct / len(rows),
                }

    return out


def save_immutable(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        doc,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"

    if path.exists():
        need(
            path.read_text(encoding="utf-8") == content,
            f"Refusing to overwrite different closure result: {path}",
        )
        return

    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def main():
    print("=" * 76)
    print("P4-B3B — READ-ONLY #108 EQUAL-BUDGET CLOSURE AUDIT")
    print("=" * 76)
    print("No fits | No simulations | No MuJoCo | No ODE | E4 sealed")

    report = read(R108)
    audit = read(A108)

    verify_metadata(report, smoke=False)
    hashes = verify_sources(report)

    rows = audit["rows_PRIVATE"]

    need(
        len(rows) == 3
        and [r["key_PRIVATE"] for r in rows] == EXPECTED_KEYS,
        "Formal case identity/order mismatch",
    )

    for row in rows:
        for cap in FORMAL_BUDGETS:
            for solver in SOLVERS:
                for law in LAWS:
                    verify_fit(row, cap, solver, law)

        verify_prefix(row)
        verify_decisions(row)

        print(
            "[PASS] case:",
            row["key_PRIVATE"][2],
            "| fits=18 | trace prefixes | decisions",
            flush=True,
        )

    aggregate = verify_aggregate(report, rows)
    checkpoint_results = verify_checkpoints(report, rows)
    smoke_result = verify_smoke(report["source_sha256"])

    diagnostic = retrospective_diagnostic(rows)

    total_cells = 0
    for row in rows:
        for cap in FORMAL_BUDGETS:
            for solver in SOLVERS:
                for law in LAWS:
                    info = row["fit_logs_PRIVATE"][str(cap)][solver][DOMAIN[law]][law]
                    total_cells += len(info["attempts"])

    need(
        total_cells == 3 * 2 * 2 * 3 * 5,
        "Expected 180 optimizer-start records",
    )

    closure = {
        "protocol": "P4-B3B READ-ONLY #108 CLOSURE",
        "status": "PASS_READ_ONLY_CLOSURE_NO_CONFIRMATORY_INFERENCE",
        "n_historical_rows": len(rows),
        "n_base_scenes": 1,
        "fit_units_per_solver_per_budget": 9,
        "optimizer_start_records": total_cells,
        "candidate_law_fit_dimensions": {
            "elastic": 2,
            "deadzone": 3,
            "cubic": 3,
        },
        "source_hashes_verified": hashes,
        "checkpoints_verified": checkpoint_results,
        "smoke_closure": smoke_result,
        "aggregate_reconstructed": aggregate,
        "retrospective_diagnostic_ONLY": diagnostic,
        "E4_accessed": False,
        "fitting_performed": False,
        "simulation_performed": False,
        "limitations": [
            "One historical base scene only",
            "Three mechanism conditions are not an independent sample",
            "Equal maximum evaluation cap does not imply equal realized evaluations",
            "Optimizer success does not imply correct physical identification",
            "Historical correctness is descriptive only",
            "No confirmatory statistical inference",
        ],
        "next_stage": (
            "P4-B3C independent-scene equal-budget confirmation; "
            "freeze 100/200 caps and 2+3 starts before outcomes"
        ),
    }

    private = {
        "formal_rows_PRIVATE": rows,
        "smoke_rows_PRIVATE": read(AS)["rows_PRIVATE"],
        "source_sha256": report["source_sha256"],
        "checkpoint_verification": checkpoint_results,
        "closure_script_sha256": sha(Path(__file__)),
    }

    report_path = OUT / "p4b3b_closure_report.json"
    audit_path = OUT / "p4b3b_closure_audit_PRIVATE.json"

    save_immutable(report_path, closure)
    save_immutable(audit_path, private)

    print("[PASS] 3 formal rows / 18 fits / 180 optimizer-start records")
    print("[PASS] 100-to-200 nested trace prefixes")
    print("[PASS] Best training fits reconstructed from evaluated traces")
    print("[PASS] All candidate winners match stored scores")
    print("[PASS] 3 checkpoint row hashes and source lineage")
    print("[PASS] Smoke/formal lineage and aggregate reconstruction")
    print("[DIAGNOSTIC ONLY]", diagnostic)
    print("[SCIENTIFIC STATUS] READ-ONLY CLOSURE PASS; NO INFERENCE")
    print("Saved:", report_path)
    print("Saved:", audit_path)
    print("P4-B3B COMPLETE")


if __name__ == "__main__":
    main()
