
"""
P3-A7B: Held-out Predictive Separation Gate

Four strategies:
  weak       : E1/E2 + Low Weak only
  full       : always execute Reference Strong
  a7a        : original BIC evidence gate
  a7b        : BIC + E1/E2-to-Weak predictive separation

Calibration and evaluation use NEW scene seeds.
No historical evaluation records are read.

No E4.
No online access to truth or Strong before gate.
No evaluation threshold tuning.
All physical parameters are fitted independently
for each candidate law.

The 5 mm separation guard is predeclared.
This is a falsifiable pilot, not a risk guarantee.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a7b"

CAL_SEED = 20270310
TEST_SEED = 20270311

CAL_NOISE = 2027031001
TEST_NOISE = 2027031101

N_CAL = 30
N_TEST = 36
NFEV = 60

BIC_GAP = 10.0
SEPARATION_MIN_M = 0.005
Q = 0.95

BANDS = ("original", "above_original_range")

KNOWN = (
    "independent", "rigid", "elastic",
    "deadzone", "nonlinear",
)
MODES = KNOWN + ("saturating",)

TRUTH_TO_LAW = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}


def hash_file(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_modules():
    base = run_path(str(
        HERE / "55_p3a7a_evidence_gated_escalation.py"
    ))

    src, policy, sim, api = base["init_modules"]()
    commands = base["get_commands"](sim)

    return src, policy, sim, api, commands


def run_episode(
    src, sim, api, commands,
    scene, sid, mode, noise_seed,
    thresholds=None,
):
    model = sim["make_model"](scene)
    mode_index = src["MODES"].index(mode)

    base_seed = (
        noise_seed + sid * 1000
        + mode_index * 20
    )

    probes = {}
    works = {}

    # Only previously observed experiments
    # are used to decide escalation.
    for action_index, action in enumerate(
        ("e1", "e2", "weak")
    ):
        probe, work = src["get_probe"](
            sim, api, model, scene, mode,
            commands[action],
            base_seed + action_index,
        )
        probes[action] = probe
        works[action] = float(work)

    early = [probes["e1"], probes["e2"]]
    train = early + [probes["weak"]]

    # Step 1: fit all laws on E1/E2 ONLY.
    _, early_fits = api["fit_library"](
        early, src["LAWS"], NFEV
    )

    weak_errors = {
        law: float(
            api["prediction_rms"](
                probes["weak"], early_fits[law]
            )
        )
        for law in src["LAWS"]
    }

    weak_order = sorted(
        src["LAWS"],
        key=lambda law: weak_errors[law],
    )

    weak_winner = weak_order[0]
    separation = (
        weak_errors[weak_order[1]]
        - weak_errors[weak_winner]
    )

    # Step 2: original A7A BIC anchor,
    # fitted on E1/E2/Weak.
    _, full_fits = api["fit_library"](
        train, src["LAWS"], NFEV
    )

    bic_order = sorted(
        src["LAWS"],
        key=lambda law: full_fits[law]["bic"],
    )

    anchor = bic_order[0]

    bic_gap = float(
        full_fits[bic_order[1]]["bic"]
        - full_fits[anchor]["bic"]
    )

    anchor_weak_error = weak_errors[anchor]

    metrics = (
        list(weak_errors.values())
        + [bic_gap, separation]
    )

    if not np.isfinite(metrics).all():
        raise RuntimeError(
            "Nonfinite early evidence"
        )

    # Decisions are frozen BEFORE Strong
    # is generated. No truth label is used.
    old_stop = None
    new_stop = None

    if thresholds is not None:
        old_stop = (
            bic_gap >= BIC_GAP
            and anchor_weak_error
            <= thresholds["tau_weak_m"]
        )

        new_stop = (
            old_stop
            and weak_winner == anchor
            and separation >= SEPARATION_MIN_M
        )

        if new_stop and not old_stop:
            raise RuntimeError(
                "New gate violates subset property"
            )

    # Strong is now simulated for offline
    # paired policy evaluation only.
    strong, strong_work = src["get_probe"](
        sim, api, model, scene, mode,
        commands["strong"],
        base_seed + 3,
    )

    works["strong"] = float(strong_work)

    strong_errors = {
        law: float(
            api["prediction_rms"](
                strong, full_fits[law]
            )
        )
        for law in src["LAWS"]
    }

    strong_winner = min(
        strong_errors,
        key=strong_errors.get,
    )

    full_record = {
        "selected_by_BIC_before_strong":
            anchor,
        "selected_by_strong":
            strong_winner,
        "minimum_RMS_m":
            strong_errors[strong_winner],
        "BIC_chosen_RMS_m":
            strong_errors[anchor],
        "all_law_RMS_m":
            strong_errors,
    }

    if not np.isfinite(
        list(strong_errors.values())
    ).all():
        raise RuntimeError(
            "Nonfinite Strong predictions"
        )

    return {
        "scene_id": sid,
        "truth_PRIVATE": mode,
        "anchor": anchor,
        "weak_winner": weak_winner,
        "bic_gap": bic_gap,
        "weak_anchor_RMS_m":
            anchor_weak_error,
        "weak_best_RMS_m":
            weak_errors[weak_winner],
        "weak_separation_m":
            separation,
        "old_stop": old_stop,
        "new_stop": new_stop,
        "full_record": full_record,
        "work": works,
    }


def calibrate(records):
    weak_values = np.array([
        r["weak_anchor_RMS_m"]
        for r in records
    ])

    strong_values = np.array([
        r["full_record"]["minimum_RMS_m"]
        for r in records
    ])

    correct_anchors = [
        r for r in records
        if r["anchor"]
        == TRUTH_TO_LAW[r["truth_PRIVATE"]]
    ]

    if not correct_anchors:
        raise RuntimeError(
            "No correct calibration anchors"
        )

    gains = np.array([
        max(
            0.0,
            r["full_record"]["BIC_chosen_RMS_m"]
            - r["full_record"]["minimum_RMS_m"],
        )
        for r in correct_anchors
    ])

    return {
        "tau_weak_m":
            float(np.quantile(weak_values, Q)),
        "tau_absolute_m":
            float(np.quantile(strong_values, Q)),
        "tau_switch_m":
            float(np.quantile(gains, Q)),
        "bic_gap_min": BIC_GAP,
        "weak_separation_min_m":
            SEPARATION_MIN_M,
        "calibration_n": len(records),
        "correct_anchor_n":
            len(correct_anchors),
    }


def evaluate(records, thresholds, policy):
    result = []

    for r in records:
        old_stop = (
            r["bic_gap"] >= BIC_GAP
            and r["weak_anchor_RMS_m"]
            <= thresholds["tau_weak_m"]
        )

        new_stop = (
            old_stop
            and r["weak_winner"] == r["anchor"]
            and r["weak_separation_m"]
            >= SEPARATION_MIN_M
        )

        if (
            r["old_stop"] != old_stop
            or r["new_stop"] != new_stop
        ):
            raise RuntimeError(
                "Early gate replay mismatch"
            )

        weak_decision = (
            r["anchor"]
            if r["weak_anchor_RMS_m"]
            <= thresholds["tau_weak_m"]
            else "unresolved"
        )

        full_decision, full_reason = (
            policy["decide"](
                r["full_record"], thresholds
            )
        )

        decisions = {
            "weak": weak_decision,
            "full": full_decision,
            "a7a": (
                r["anchor"]
                if old_stop else full_decision
            ),
            "a7b": (
                r["anchor"]
                if new_stop else full_decision
            ),
        }

        truth = r["truth_PRIVATE"]

        outcomes = {
            name: policy["verdict"](
                truth, decision
            )
            for name, decision
            in decisions.items()
        }

        work = r["work"]

        common_work = sum(
            work[action]
            for action in ("e1", "e2", "weak")
        )

        strong_work = work["strong"]

        costs = {
            "weak": common_work,
            "full": common_work + strong_work,
            "a7a": (
                common_work
                + (0 if old_stop else strong_work)
            ),
            "a7b": (
                common_work
                + (0 if new_stop else strong_work)
            ),
        }

        result.append({
            "scene_id": r["scene_id"],
            "truth_PRIVATE": truth,
            "anchor": r["anchor"],
            "weak_winner": r["weak_winner"],
            "bic_gap": r["bic_gap"],
            "weak_anchor_RMS_mm":
                1000 * r["weak_anchor_RMS_m"],
            "weak_separation_mm":
                1000 * r["weak_separation_m"],
            "a7a_escalate": not old_stop,
            "a7b_escalate": not new_stop,
            "decisions": decisions,
            "outcomes_PRIVATE": outcomes,
            "costs_J": costs,
            "full_reason": full_reason,
        })

    return result


def summarize(rows):
    methods = ("weak", "full", "a7a", "a7b")
    result = {}

    for mode in MODES:
        subset = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        outcome_counts = {
            method: dict(Counter(
                r["outcomes_PRIVATE"][method]
                for r in subset
            ))
            for method in methods
        }

        result[mode] = {
            "n": len(subset),
            "outcomes": outcome_counts,
            "a7a_strong": sum(
                r["a7a_escalate"] for r in subset
            ),
            "a7b_strong": sum(
                r["a7b_escalate"] for r in subset
            ),
        }

        print("\nMODE:", mode)
        print(
            "  Strong escalation A7A -> A7B:",
            result[mode]["a7a_strong"],
            "->",
            result[mode]["a7b_strong"],
            "/", len(subset)
        )

        for method in methods:
            print(
                " ", method, ":",
                outcome_counts[method]
            )

    known_rows = [
        r for r in rows
        if r["truth_PRIVATE"] in KNOWN
    ]

    unknown_rows = [
        r for r in rows
        if r["truth_PRIVATE"] == "saturating"
    ]

    full_cost = sum(
        r["costs_J"]["full"] for r in rows
    )

    overall = {
        "known_n": len(known_rows),
        "unknown_n": len(unknown_rows),
        "known_correct": {},
        "unknown_correct_reject": {},
        "strong_count": {},
        "total_cost_J": {},
        "work_reduction_vs_full": {},
    }

    for method in methods:
        overall["known_correct"][method] = sum(
            r["outcomes_PRIVATE"][method]
            == "correct"
            for r in known_rows
        )

        overall[
            "unknown_correct_reject"
        ][method] = sum(
            r["outcomes_PRIVATE"][method]
            == "correct_reject"
            for r in unknown_rows
        )

        if method in ("a7a", "a7b"):
            strong_count = sum(
                r[f"{method}_escalate"]
                for r in rows
            )
        elif method == "full":
            strong_count = len(rows)
        else:
            strong_count = 0

        overall["strong_count"][method] = (
            strong_count
        )

        total = sum(
            r["costs_J"][method] for r in rows
        )

        overall["total_cost_J"][method] = (
            float(total)
        )

        overall[
            "work_reduction_vs_full"
        ][method] = float(
            (full_cost - total)
            / max(full_cost, 1e-12)
        )

    result["_overall"] = overall

    print("\n" + "=" * 68)
    print("OVERALL")
    print("=" * 68)

    for method in methods:
        print(
            method,
            "known correct",
            overall["known_correct"][method],
            "/", len(known_rows),
            "| unknown rejected",
            overall["unknown_correct_reject"][method],
            "/", len(unknown_rows),
            "| strong",
            overall["strong_count"][method],
            "/", len(rows),
            "| work reduction",
            round(
                100 * overall[
                    "work_reduction_vs_full"
                ][method],
                3,
            ),
            "%"
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true"
    )
    args = parser.parse_args()

    src, policy, sim, api, commands = (
        load_modules()
    )

    offset = 100000 if args.smoke else 0

    n_cal = 2 if args.smoke else N_CAL
    n_test = 2 if args.smoke else N_TEST

    cal_scenes = src["make_scenes"](
        sim, CAL_SEED + offset, n_cal
    )

    calibration = []

    for sid, scene_set in enumerate(cal_scenes):
        for mode in KNOWN:
            calibration.append(
                run_episode(
                    src, sim, api, commands,
                    scene_set["original"],
                    sid, mode,
                    CAL_NOISE + offset,
                )
            )

    thresholds = calibrate(calibration)

    print(
        "\nP3-A7B HELDOUT-SEPARATION GATE"
    )

    print("Calibration:", len(calibration))
    print(
        "Weak RMS cutoff:",
        round(
            1000 * thresholds["tau_weak_m"], 4
        ),
        "mm"
    )
    print(
        "Strong RMS cutoff:",
        round(
            1000 * thresholds["tau_absolute_m"], 4
        ),
        "mm"
    )
    print(
        "Switch cutoff:",
        round(
            1000 * thresholds["tau_switch_m"], 4
        ),
        "mm"
    )
    print(
        "Weak separation guard:",
        1000 * SEPARATION_MIN_M,
        "mm"
    )

    test_scenes = src["make_scenes"](
        sim, TEST_SEED + offset, n_test
    )

    summaries = {}
    audits = {}

    for band in BANDS:
        records = []

        for sid, scene_set in enumerate(test_scenes):
            for mode in MODES:
                records.append(
                    run_episode(
                        src, sim, api, commands,
                        scene_set[band],
                        sid, mode,
                        TEST_NOISE + offset,
                        thresholds,
                    )
                )

        evaluated = evaluate(
            records, thresholds, policy
        )

        print("\n" + "=" * 68)
        print("BAND:", band)
        print("=" * 68)

        summaries[band] = summarize(evaluated)
        audits[band] = evaluated

    OUT.mkdir(parents=True, exist_ok=True)

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report_path = OUT / (
        f"p3a7b_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a7b_{suffix}_audit_PRIVATE.json"
    )

    files = (
        "16_p2d1b_probe_diversity.py",
        "22_p3a1b_structural_revision.py",
        "27_p3a3b_evidence_vs_structure.py",
        "47_p3a6h_six_mechanism_openworld_audit.py",
        "48_p3a6i_calibrated_revision.py",
        "55_p3a7a_evidence_gated_escalation.py",
    )

    report = {
        "protocol":
            "P3-A7B prospective separation gate",

        "smoke": args.smoke,
        "calibration_seed": CAL_SEED + offset,
        "test_seed": TEST_SEED + offset,
        "n_cal_scenes": n_cal,
        "n_test_scenes": n_test,
        "thresholds": thresholds,
        "summary": summaries,

        "E4_accessed": False,
        "test_threshold_tuning": False,
        "online_uses_truth": False,
        "online_uses_strong_before_gate": False,

        "limitations": [
            "MuJoCo synthetic environments only",
            "One specific unknown Saturating law",
            "No safety or open-set coverage guarantee",
            "Weak margin selects among E1/E2 fits",
            "BIC anchor itself was selected using Weak",
            "The guard may increase Strong usage",
            "Strong counterfactuals used for audit only",
            "Fixed Strong action, not action optimization",
            "No RGB visual perception yet",
            "Predictions condition on observed A",
            "Gap bands share base scenes",
        ],

        "source_sha256": {
            name: hash_file(HERE / name)
            for name in files
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            audits,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A7B PROSPECTIVE AUDIT COMPLETE"
    )
    print(
        "No E4, no oracle online, "
        "no test threshold tuning."
    )
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
