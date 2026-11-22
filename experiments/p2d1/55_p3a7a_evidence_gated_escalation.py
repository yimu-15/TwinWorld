
"""
P3-A7A: Prospective Evidence-Gated Escalation Pilot

Compare:
1. Weak-only
2. Always-Strong
3. Evidence-Gated Strong escalation

All rules calibrated on independent known-law
scenes. Saturating is excluded from calibration.

No E4.
No privileged online variables.
No evaluation-set threshold tuning.

Strong is simulated for offline counterfactual
comparison, but is NOT supplied to the online
gate before its decision.

This is a pilot, not a proven safe-stop policy.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a7a"

CAL_SEED = 20270221
TEST_SEED = 20270222

CAL_NOISE = 2027022101
TEST_NOISE = 2027022201

N_CAL = 30
N_TEST = 36

NFEV = 60
Q = 0.95

BIC_GAP_MIN = 10.0

BANDS = (
    "original",
    "above_original_range",
)

KNOWN = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear",
)

MODES = KNOWN + ("saturating",)


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def init_modules():
    src = run_path(str(
        HERE /
        "47_p3a6h_six_mechanism_openworld_audit.py"
    ))

    policy = run_path(str(
        HERE /
        "48_p3a6i_calibrated_revision.py"
    ))

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    extension = run_path(str(
        HERE /
        "27_p3a3b_evidence_vs_structure.py"
    ))

    extension["install_saturating"](sim)

    return src, policy, sim, api


def get_commands(sim):
    pulse = sim["make_profiles"]()["pulse"]

    return {
        "e1": pulse * 0.25,
        "e2": pulse * 0.50,
        "weak": pulse * 0.75,
        "strong": pulse * 2.25,
    }


def get_episode(
    src, sim, api, scene, sid, mode,
    commands, noise_seed, thresholds=None,
):
    model = sim["make_model"](scene)

    mode_idx = src["MODES"].index(mode)

    probes = {}
    works = {}

    # The first three actions are available
    # to the online decision-making process.
    for action_idx, action in enumerate(
        ("e1", "e2", "weak")
    ):
        p, work = src["get_probe"](
            sim, api, model, scene, mode,
            commands[action],
            noise_seed
            + sid * 1000
            + mode_idx * 20
            + action_idx,
        )

        probes[action] = p
        works[action] = float(work)

    early = [probes["e1"], probes["e2"]]
    train = early + [probes["weak"]]

    _, fits = api["fit_library"](
        train,
        src["LAWS"],
        NFEV,
    )

    order = sorted(
        src["LAWS"],
        key=lambda law: fits[law]["bic"]
    )

    anchor = order[0]

    bic_gap = float(
        fits[order[1]]["bic"]
        - fits[anchor]["bic"]
    )

    # Fit anchor on E1/E2 only, then predict Weak.
    # The anchor was selected after looking at Weak:
    # this is NOT a selection-free heldout estimate.
    early_anchor, _ = api["fit_library"](
        early,
        (anchor,),
        NFEV,
    )

    weak_cv = float(
        api["prediction_rms"](
            probes["weak"],
            early_anchor,
        )
    )

    evidence = {
        "anchor": anchor,
        "bic_gap": bic_gap,
        "weak_cv_m": weak_cv,
    }

    # Crucial: make the escalation decision
    # BEFORE generating Strong observations.
    escalate = None

    if thresholds is not None:
        escalate = not (
            bic_gap >= BIC_GAP_MIN
            and weak_cv <= thresholds["tau_weak_m"]
        )

    # Strong is generated offline for a paired
    # always-Strong baseline.
    # The online gate above cannot access it.
    strong, strong_work = src["get_probe"](
        sim, api, model, scene, mode,
        commands["strong"],
        noise_seed
        + sid * 1000
        + mode_idx * 20
        + 3,
    )

    works["strong"] = float(strong_work)

    strong_errors = {
        law: float(
            api["prediction_rms"](
                strong, fits[law]
            )
        )
        for law in src["LAWS"]
    }

    winner = min(
        src["LAWS"],
        key=lambda law: strong_errors[law]
    )

    full_record = {
        "selected_by_BIC_before_strong":
            anchor,
        "selected_by_strong":
            winner,
        "minimum_RMS_m":
            strong_errors[winner],
        "BIC_chosen_RMS_m":
            strong_errors[anchor],
        "all_law_RMS_m":
            strong_errors,
    }

    if not all(
        np.isfinite(v) and v >= 0
        for v in strong_errors.values()
    ):
        raise RuntimeError("Invalid Strong RMS")

    return {
        "scene_id": sid,
        "truth_PRIVATE": mode,
        "early_evidence": evidence,
        "escalate": escalate,
        "full_record": full_record,
        "works": works,
    }


def calibrate(records):
    # All calibration records are known-law
    # scenes. Never include Saturating.
    weak_rms = [
        r["early_evidence"]["weak_cv_m"]
        for r in records
    ]

    abs_rms = [
        r["full_record"]["minimum_RMS_m"]
        for r in records
    ]

    truth_to_law = {
        "independent": "independent",
        "rigid": "rigid",
        "elastic": "elastic",
        "deadzone": "deadzone",
        "nonlinear": "cubic",
    }

    correct_anchors = [
        r for r in records
        if (
            r["early_evidence"]["anchor"]
            == truth_to_law[r["truth_PRIVATE"]]
        )
    ]

    if not correct_anchors:
        raise RuntimeError(
            "No correct calibration anchors"
        )

    gains = [
        max(
            0.0,
            r["full_record"]["BIC_chosen_RMS_m"]
            - r["full_record"]["minimum_RMS_m"],
        )
        for r in correct_anchors
    ]

    return {
        "tau_weak_m":
            float(np.quantile(weak_rms, Q)),
        "tau_absolute_m":
            float(np.quantile(abs_rms, Q)),
        "tau_switch_m":
            float(np.quantile(gains, Q)),
        "bic_gap_min":
            BIC_GAP_MIN,
        "calibration_n": len(records),
        "correct_anchor_n":
            len(correct_anchors),
    }


def evaluate(records, thresholds, policy):
    output = []

    for r in records:
        evidence = r["early_evidence"]
        anchor = evidence["anchor"]

        # A Weak-only baseline:
        # no unseen Strong evidence is used.
        weak_decision = (
            anchor
            if evidence["weak_cv_m"]
            <= thresholds["tau_weak_m"]
            else "unresolved"
        )

        # Full evidence baseline and escalation
        # branch use the same frozen A6I logic.
        full_decision, _ = policy["decide"](
            r["full_record"],
            thresholds,
        )

        escalate = not (
            evidence["bic_gap"]
            >= thresholds["bic_gap_min"]
            and evidence["weak_cv_m"]
            <= thresholds["tau_weak_m"]
        )

        if r["escalate"] is not None:
            if r["escalate"] != escalate:
                raise RuntimeError(
                    "Online gate integrity failure"
                )

        gated_decision = (
            full_decision
            if escalate
            else anchor
        )

        truth = r["truth_PRIVATE"]

        weak_outcome = policy["verdict"](
            truth, weak_decision
        )

        full_outcome = policy["verdict"](
            truth, full_decision
        )

        gated_outcome = policy["verdict"](
            truth, gated_decision
        )

        works = r["works"]

        common_work = sum(
            works[action]
            for action in ("e1", "e2", "weak")
        )

        full_work = (
            common_work + works["strong"]
        )

        gated_work = (
            common_work
            + (works["strong"] if escalate else 0)
        )

        output.append({
            "scene_id": r["scene_id"],
            "truth_PRIVATE": truth,
            "anchor": anchor,
            "bic_gap": evidence["bic_gap"],
            "weak_cv_mm":
                1000 * evidence["weak_cv_m"],
            "escalate": escalate,
            "weak_decision": weak_decision,
            "full_decision": full_decision,
            "gated_decision": gated_decision,
            "weak_outcome_PRIVATE": weak_outcome,
            "full_outcome_PRIVATE": full_outcome,
            "gated_outcome_PRIVATE": gated_outcome,
            "common_work_J": common_work,
            "full_work_J": full_work,
            "gated_work_J": gated_work,
        })

    return output


def summarize(rows):
    result = {}

    for mode in MODES:
        group = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        print("\nMODE:", mode)

        item = {
            "n": len(group),
            "strong_escalations": sum(
                r["escalate"] for r in group
            ),
            "weak_outcomes": dict(Counter(
                r["weak_outcome_PRIVATE"]
                for r in group
            )),
            "full_outcomes": dict(Counter(
                r["full_outcome_PRIVATE"]
                for r in group
            )),
            "gated_outcomes": dict(Counter(
                r["gated_outcome_PRIVATE"]
                for r in group
            )),
        }

        result[mode] = item

        print(
            "  Strong escalations:",
            item["strong_escalations"],
            "/", len(group)
        )
        print(
            "  Weak-only:",
            item["weak_outcomes"]
        )
        print(
            "  Always-Strong:",
            item["full_outcomes"]
        )
        print(
            "  Evidence-Gated:",
            item["gated_outcomes"]
        )

    full_cost = sum(
        r["full_work_J"] for r in rows
    )

    gated_cost = sum(
        r["gated_work_J"] for r in rows
    )

    known = [
        r for r in rows
        if r["truth_PRIVATE"] in KNOWN
    ]

    known_results = {}

    for method in ("weak", "full", "gated"):
        known_results[method] = sum(
            r[f"{method}_outcome_PRIVATE"]
            == "correct"
            for r in known
        )

    unknown = [
        r for r in rows
        if r["truth_PRIVATE"] == "saturating"
    ]

    rejected = {}

    for method in ("weak", "full", "gated"):
        rejected[method] = sum(
            r[f"{method}_outcome_PRIVATE"]
            == "correct_reject"
            for r in unknown
        )

    overall = {
        "known_n": len(known),
        "known_correct": known_results,
        "unknown_n": len(unknown),
        "unknown_rejected": rejected,
        "strong_escalation_count": sum(
            r["escalate"] for r in rows
        ),
        "total_episodes": len(rows),
        "full_work_proxy_J": full_cost,
        "gated_work_proxy_J": gated_cost,
        "relative_work_reduction":
            (full_cost - gated_cost)
            / max(full_cost, 1e-12),
    }

    print("\nOVERALL")
    print(
        "  Known correct:",
        known_results,
        "/", len(known)
    )
    print(
        "  Unknown rejected:",
        rejected,
        "/", len(unknown)
    )
    print(
        "  Strong used:",
        overall["strong_escalation_count"],
        "/", len(rows)
    )
    print(
        "  Counterfactual work reduction:",
        round(
            100 * overall["relative_work_reduction"],
            3,
        ),
        "%"
    )

    result["_overall"] = overall
    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    src, policy, sim, api = init_modules()

    commands = get_commands(sim)

    offset = 100000 if args.smoke else 0

    n_cal = 2 if args.smoke else N_CAL
    n_test = 2 if args.smoke else N_TEST

    cal_scenes = src["make_scenes"](
        sim, CAL_SEED + offset, n_cal
    )

    test_scenes = src["make_scenes"](
        sim, TEST_SEED + offset, n_test
    )

    calibration = []

    for sid, scene_set in enumerate(cal_scenes):
        scene = scene_set["original"]

        for mode in KNOWN:
            calibration.append(
                get_episode(
                    src, sim, api,
                    scene, sid, mode,
                    commands,
                    CAL_NOISE + offset,
                )
            )

    thresholds = calibrate(calibration)

    print(
        "\nP3-A7A ONLINE EVIDENCE GATE PILOT"
    )
    print("Calibration episodes:", len(calibration))
    print(
        "Weak threshold:",
        round(1000 * thresholds["tau_weak_m"], 4),
        "mm"
    )
    print(
        "Strong absolute threshold:",
        round(1000 * thresholds["tau_absolute_m"], 4),
        "mm"
    )
    print(
        "Strong switch threshold:",
        round(1000 * thresholds["tau_switch_m"], 4),
        "mm"
    )
    print(
        "BIC gap minimum:",
        BIC_GAP_MIN
    )

    results = {}
    audits = {}

    for band in BANDS:
        episodes = []

        for sid, scene_set in enumerate(test_scenes):
            scene = scene_set[band]

            for mode in MODES:
                episodes.append(
                    get_episode(
                        src, sim, api,
                        scene, sid, mode,
                        commands,
                        TEST_NOISE + offset,
                        thresholds,
                    )
                )

        evaluated = evaluate(
            episodes, thresholds, policy
        )

        print("\n" + "=" * 70)
        print("BAND:", band)
        print("=" * 70)

        results[band] = summarize(evaluated)
        audits[band] = evaluated

    OUT.mkdir(parents=True, exist_ok=True)

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report_path = OUT / (
        f"p3a7a_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a7a_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P3-A7A prospective gate pilot",
        "smoke": args.smoke,
        "calibration_seed": CAL_SEED + offset,
        "evaluation_seed": TEST_SEED + offset,
        "n_cal_scenes": n_cal,
        "n_test_scenes": n_test,
        "calibration_modes": list(KNOWN),
        "evaluation_modes": list(MODES),
        "thresholds": thresholds,
        "summary": results,
        "E4_accessed": False,
        "test_threshold_tuning": False,
        "online_uses_truth": False,
        "online_uses_strong_before_decision": False,
        "limitations": [
            "Synthetic MuJoCo only",
            "Strong counterfactuals simulated for audit",
            "No real hardware budget saving demonstrated",
            "Weak model selection reuses Weak information",
            "BIC gap is not an identifiability certificate",
            "Only one unknown law",
            "Conditioned on observed A",
            "Separate gap bands share base scenes",
            "Empirical calibration without risk guarantee",
            "One predefined gate, not optimized action selection",
        ],
        "source_sha256": {
            filename: sha256(HERE / filename)
            for filename in (
                "16_p2d1b_probe_diversity.py",
                "22_p3a1b_structural_revision.py",
                "27_p3a3b_evidence_vs_structure.py",
                "47_p3a6h_six_mechanism_openworld_audit.py",
                "48_p3a6i_calibrated_revision.py",
            )
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
        "\nP3-A7A PROSPECTIVE PILOT COMPLETE"
    )
    print(
        "No E4, no oracle online, "
        "no test threshold tuning."
    )
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
