
"""
P3-A7E0: Sequential Reverse-Probe Value Audit

Strategies:
  mid_only       : Vmid only
  always_reverse : Vmid + Vreverse
  adaptive       : Vreverse only if Vmid set size > 1

Calibration:
  Per-law/per-band independent scenes
  alpha_mid = alpha_reverse = 0.05

Same frozen law-fitting pipeline as P3-A7D.
No E4. No oracle online. No test tuning.

This is adaptive continuation, NOT
optimization over intervention candidates.
"""

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
from collections import Counter
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a7e0"

OLD = (
    HERE / "results" / "p3a7d"
    / "p3a7d_fresh_report.json"
)

CAL_SEED = 20270512
TEST_SEED = 20270513

CAL_NOISE = 2027051201
TEST_NOISE = 2027051301

N_CAL = 39
N_TEST = 36

ALPHA_MID = 0.05
ALPHA_REVERSE = 0.05

BANDS = ("original", "above_original_range")

KNOWN = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear",
)

MODES = KNOWN + ("saturating",)

TRUE_LAW = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}


def read(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def cutoff(values, alpha):
    ordered = sorted(float(v) for v in values)
    n = len(ordered)

    k = ceil((n + 1) * (1 - alpha))

    if k > n:
        raise RuntimeError(
            "Insufficient calibration size"
        )

    return ordered[k - 1]


def allowed(scores, thresholds):
    return sorted(
        h for h, error in scores.items()
        if error <= thresholds[h]
    )


def classify(group, truth):
    n = len(group)

    if truth == "saturating":
        if n == 0:
            return "unknown_empty"
        if n == 1:
            return "unknown_false_singleton"
        return "unknown_ambiguous"

    true_law = TRUE_LAW[truth]

    if n == 0:
        return "known_empty"

    if n == 1:
        return (
            "correct_singleton"
            if group[0] == true_law
            else "wrong_singleton"
        )

    return (
        "ambiguous_contains_true"
        if true_law in group
        else "ambiguous_excludes_true"
    )


def calibrate(core, src, sim, api, commands,
              scene_sets, noise_seed):

    observations = {
        band: {
            law: {"mid": [], "reverse": []}
            for law in src["LAWS"]
        }
        for band in BANDS
    }

    for sid, scene_set in enumerate(scene_sets):
        for band in BANDS:
            for mode in KNOWN:
                r = core["run_episode"](
                    src, sim, api, commands,
                    scene_set[band], sid, mode,
                    band, noise_seed,
                    cutoffs=None,
                    do_reverse=True,
                )

                law = TRUE_LAW[mode]

                observations[band][law]["mid"].append(
                    r["vmid_scores_m"][law]
                )

                observations[band][law]["reverse"].append(
                    r["reverse_scores_m_PRIVATE"][law]
                )

    result = {}

    for band in BANDS:
        result[band] = {
            "mid": {},
            "reverse": {},
        }

        print("\nCALIBRATION BAND:", band)

        for law in src["LAWS"]:
            a = cutoff(
                observations[band][law]["mid"],
                ALPHA_MID,
            )

            b = cutoff(
                observations[band][law]["reverse"],
                ALPHA_REVERSE,
            )

            result[band]["mid"][law] = a
            result[band]["reverse"][law] = b

            print(
                f"  {law:12s}",
                f"mid={a*1000:.4f} mm",
                f"reverse={b*1000:.4f} mm",
            )

    return result


def evaluate_case(r, thresholds):
    # The Vmid decision was computed by
    # P3-A7D BEFORE generating Vreverse.
    first = r["set"]

    if first != allowed(
        r["vmid_scores_m"],
        thresholds["mid"],
    ):
        raise RuntimeError(
            "Vmid decision integrity failure"
        )

    reverse_compatible = allowed(
        r["reverse_scores_m_PRIVATE"],
        thresholds["reverse"],
    )

    second = sorted(
        set(first).intersection(
            reverse_compatible
        )
    )

    # Adaptive gate uses FIRST set only.
    need_reverse = len(first) > 1

    groups = {
        "mid_only": first,
        "always_reverse": second,
        "adaptive": (
            second if need_reverse else first
        ),
    }

    truth = r["truth_PRIVATE"]

    outcomes = {
        method: classify(group, truth)
        for method, group in groups.items()
    }

    base_cost = float(
        r["online_work_J"]
    )

    reverse_cost = float(
        r["reverse_work_J_PRIVATE"]
    )

    costs = {
        "mid_only": base_cost,
        "always_reverse":
            base_cost + reverse_cost,
        "adaptive":
            base_cost
            + (reverse_cost if need_reverse else 0),
    }

    return {
        "scene_id": r["scene_id"],
        "band": r["band"],
        "truth_PRIVATE": truth,
        "first_set": first,
        "reverse_compatible": reverse_compatible,
        "strategies": groups,
        "outcomes_PRIVATE": outcomes,
        "perform_reverse_adaptive": need_reverse,
        "costs_J": costs,
    }


def summarize(rows):
    methods = (
        "mid_only",
        "always_reverse",
        "adaptive",
    )

    summary = {}

    for mode in MODES:
        subset = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        print("\nMODE:", mode)
        item = {}

        for method in methods:
            counts = dict(Counter(
                r["outcomes_PRIVATE"][method]
                for r in subset
            ))

            avg_size = float(np.mean([
                len(r["strategies"][method])
                for r in subset
            ]))

            item[method] = {
                "outcomes": counts,
                "mean_set_size": avg_size,
            }

            print(
                f"  {method:14s}",
                counts,
                "mean size=",
                round(avg_size, 3),
            )

        summary[mode] = item

    overall = {
        "n": len(rows),
        "known_n": sum(
            r["truth_PRIVATE"] in KNOWN
            for r in rows
        ),
        "unknown_n": sum(
            r["truth_PRIVATE"] == "saturating"
            for r in rows
        ),
        "policies": {},
    }

    always_cost = sum(
        r["costs_J"]["always_reverse"]
        for r in rows
    )

    for method in methods:
        known = [
            r for r in rows
            if r["truth_PRIVATE"] in KNOWN
        ]

        unknown = [
            r for r in rows
            if r["truth_PRIVATE"] == "saturating"
        ]

        cost = sum(
            r["costs_J"][method]
            for r in rows
        )

        attempts = (
            0 if method == "mid_only"
            else len(rows)
            if method == "always_reverse"
            else sum(
                r["perform_reverse_adaptive"]
                for r in rows
            )
        )

        overall["policies"][method] = {
            "known_correct_unique": sum(
                r["outcomes_PRIVATE"][method]
                == "correct_singleton"
                for r in known
            ),
            "known_wrong_unique": sum(
                r["outcomes_PRIVATE"][method]
                == "wrong_singleton"
                for r in known
            ),
            "known_contains_true": sum(
                TRUE_LAW[r["truth_PRIVATE"]]
                in r["strategies"][method]
                for r in known
            ),
            "known_ambiguous": sum(
                len(r["strategies"][method]) > 1
                for r in known
            ),
            "unknown_false_singleton": sum(
                r["outcomes_PRIVATE"][method]
                == "unknown_false_singleton"
                for r in unknown
            ),
            "unknown_empty": sum(
                r["outcomes_PRIVATE"][method]
                == "unknown_empty"
                for r in unknown
            ),
            "unknown_ambiguous": sum(
                r["outcomes_PRIVATE"][method]
                == "unknown_ambiguous"
                for r in unknown
            ),
            "reverse_executions": attempts,
            "cost_J": float(cost),
            "cost_reduction_vs_always": float(
                (always_cost - cost)
                / max(always_cost, 1e-12)
            ),
        }

    print("\nOVERALL")

    for method, s in overall["policies"].items():
        print("\n ", method)
        print(
            "  known correct unique:",
            s["known_correct_unique"],
            "/", overall["known_n"],
        )
        print(
            "  known wrong unique:",
            s["known_wrong_unique"],
        )
        print(
            "  known contains true:",
            s["known_contains_true"],
        )
        print(
            "  known ambiguous:",
            s["known_ambiguous"],
        )
        print(
            "  unknown false singleton:",
            s["unknown_false_singleton"],
            "/", overall["unknown_n"],
        )
        print(
            "  unknown empty:",
            s["unknown_empty"],
        )
        print(
            "  reverse executions:",
            s["reverse_executions"],
            "/", len(rows),
        )
        print(
            "  work reduction vs always:",
            round(
                s["cost_reduction_vs_always"]
                * 100, 3,
            ),
            "%",
        )

    summary["_overall"] = overall
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    args = parser.parse_args()

    historical = read(OLD)

    if historical["smoke"]:
        raise RuntimeError(
            "Expected formal P3-A7D provenance"
        )

    for name, expected in (
        historical["source_sha256"].items()
    ):
        if digest(HERE / name) != expected:
            raise RuntimeError(
                "Historical source changed: " + name
            )

    core = run_path(str(
        HERE /
        "58_p3a7d_hypothesis_set_validation.py"
    ))

    src, sim, api, commands = core["init"]()

    offset = 100000 if args.smoke else 0

    n_cal = 19 if args.smoke else N_CAL
    n_test = 2 if args.smoke else N_TEST

    cal_scenes = src["make_scenes"](
        sim, CAL_SEED + offset, n_cal
    )

    test_scenes = src["make_scenes"](
        sim, TEST_SEED + offset, n_test
    )

    print(
        "\nP3-A7E0 SEQUENTIAL REVERSE AUDIT"
    )

    print(
        "Calibration scenes:", n_cal,
        "| Test scenes:", n_test,
    )

    print(
        "Alpha allocation:",
        ALPHA_MID, "+", ALPHA_REVERSE,
    )

    cutoffs = calibrate(
        core, src, sim, api, commands,
        cal_scenes, CAL_NOISE + offset,
    )

    results = {}
    audits = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(test_scenes):
            for mode in MODES:
                r = core["run_episode"](
                    src, sim, api, commands,
                    scene_set[band],
                    sid, mode, band,
                    TEST_NOISE + offset,
                    cutoffs=cutoffs[band]["mid"],
                    do_reverse=True,
                )

                rows.append(
                    evaluate_case(
                        r, cutoffs[band]
                    )
                )

        if len(rows) != (
            n_test * len(MODES)
        ):
            raise RuntimeError(
                "Incomplete test matrix"
            )

        print("\n" + "=" * 70)
        print("BAND:", band)
        print("=" * 70)

        results[band] = summarize(rows)
        audits[band] = rows

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report = {
        "protocol":
            "P3-A7E0 adaptive reverse continuation",
        "smoke": args.smoke,
        "calibration_seed": CAL_SEED + offset,
        "test_seed": TEST_SEED + offset,
        "n_cal_scenes": n_cal,
        "n_test_scenes": n_test,
        "alpha_mid": ALPHA_MID,
        "alpha_reverse": ALPHA_REVERSE,
        "cutoffs_m": cutoffs,
        "summary": results,
        "E4_accessed": False,
        "online_uses_truth": False,
        "reverse_used_for_initial_gate": False,
        "thresholds_tuned_on_test": False,
        "limitations": [
            "Synthetic MuJoCo experiment",
            "Same fitted laws used for both challenges",
            "Vreverse is fixed, not optimized",
            "All reverse counterfactuals are simulated",
            "Only executed actions charged to each policy",
            "Only one held-out Saturating law",
            "No open-world coverage guarantee",
            "Conditional set coverage requires exchangeability",
            "Gap bands share scene parameters",
            "No visual input or E4",
        ],
    }

    report_path = OUT / (
        f"p3a7e0_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a7e0_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            audits, indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A7E0 SEQUENTIAL "
        "REVERSE AUDIT COMPLETE"
    )
    print(
        "No E4, no oracle online, "
        "no test threshold tuning."
    )
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
