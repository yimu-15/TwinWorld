
"""
P3-A6H: Six-Mechanism Open-World Audit

Five represented models:
  independent, rigid, elastic, deadzone, cubic

Six true physical mechanisms:
  independent, rigid, elastic, deadzone,
  nonlinear, saturating

Calibration:
  Independent scenes, five represented laws only.
  Strong residual threshold from separate
  calibration scenes, Q95.

Evaluation:
  All six true modes on fresh scenes.
  Saturating NEVER enters calibration.

Train: E1 + E2 + Weak
Confirmation: Strong (not used for parameter fit)

No E4.
No adaptive policy.
No threshold tuning on evaluation.
"""

from pathlib import Path
from collections import Counter
from runpy import run_path
import argparse
import hashlib
import json

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a6h"

CAL_SEED = 20261221
TEST_SEED = 20261222

CAL_NOISE_SEED = 2026122101
TEST_NOISE_SEED = 2026122201

CAL_N = 24
TEST_N = 36

NFEV = 60
Q = 0.95

LAWS = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "cubic",
)

MODES = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear",
    "saturating",
)

REPRESENTED = MODES[:-1]

TRUTH_TO_LAW = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

ACTIONS = ("e1", "e2", "weak", "strong")
BANDS = ("original", "above_original_range")


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def make_commands(sim):
    p = sim["make_profiles"]()

    return {
        "e1": p["pulse"] * 0.25,
        "e2": p["pulse"] * 0.50,
        "weak": p["pulse"] * 1.50,
        "strong": p["pulse"] * 2.25,
    }


def make_scenes(sim, seed, n):
    rng = np.random.default_rng(seed)
    result = []

    for _ in range(n):
        base = sim["sample_scene"](rng)

        wide = dict(base)
        wide["gap"] = float(
            rng.uniform(0.12, 0.18)
        )

        result.append({
            "original": dict(base),
            "above_original_range": wide,
        })

    return result


def get_probe(
    sim, api, model, scene, mode,
    command, noise_seed,
):
    passive, active, work, _ = sim[
        "simulate"
    ](
        model, scene, mode, command
    )

    rng = np.random.default_rng(noise_seed)

    p, a = api["noisy_observation"](
        passive, active, rng
    )

    return api["RevisionProbe"](p, a), float(work)


def run_episode(
    sim, api, scene, sid, mode,
    commands, noise_seed
):
    model = sim["make_model"](scene)

    probes = {}
    works = {}

    mode_idx = MODES.index(mode)

    for action_idx, action in enumerate(ACTIONS):
        probe, work = get_probe(
            sim, api, model, scene, mode,
            commands[action],
            noise_seed
            + sid * 1000
            + mode_idx * 20
            + action_idx,
        )

        probes[action] = probe
        works[action] = work

    # All five candidate models receive exactly
    # the same three training observations.
    train = [
        probes["e1"],
        probes["e2"],
        probes["weak"],
    ]

    _, fits = api["fit_library"](
        train, LAWS, NFEV
    )

    if set(fits) != set(LAWS):
        raise RuntimeError(
            "Candidate law library mismatch"
        )

    # BIC choice is made BEFORE Strong is examined.
    bic_choice = min(
        LAWS,
        key=lambda law: fits[law]["bic"],
    )

    # Strong is a confirmation observation.
    # No model is refitted on Strong.
    strong = probes["strong"]

    errors = {
        law: float(
            api["prediction_rms"](
                strong, fits[law]
            )
        )
        for law in LAWS
    }

    if not all(
        np.isfinite(v) and v >= 0
        for v in errors.values()
    ):
        raise RuntimeError(
            "Invalid heldout prediction RMS"
        )

    best_law = min(
        LAWS,
        key=lambda law: errors[law],
    )

    return {
        "scene_id": sid,
        "truth_PRIVATE": mode,

        "selected_by_BIC_before_strong":
            bic_choice,

        "selected_by_strong":
            best_law,

        "minimum_RMS_m":
            min(errors.values()),

        "BIC_chosen_RMS_m":
            errors[bic_choice],

        "all_law_RMS_m":
            errors,

        "BIC_by_law": {
            law: float(fits[law]["bic"])
            for law in LAWS
        },

        "optimizer_success": {
            law: bool(
                fits[law]["optimizer_success"]
            )
            for law in LAWS
        },

        "weak_work_proxy_J":
            works["weak"],

        "strong_work_proxy_J":
            works["strong"],
    }


def collect(
    sim, api, scenes, commands,
    noise_seed, modes, bands,
):
    records = {band: [] for band in bands}

    for band in bands:
        for sid, scene_set in enumerate(scenes):
            scene = scene_set[band]

            for mode in modes:
                records[band].append(
                    run_episode(
                        sim, api, scene, sid,
                        mode, commands,
                        noise_seed,
                    )
                )

    return records


def calibrate(rows):
    """
    The calibration pool contains ONLY the
    five represented mechanisms.

    Saturating never enters threshold selection.

    This is a marginal empirical Q95 cutoff,
    not a finite-sample coverage guarantee.
    """
    values = np.asarray([
        r["minimum_RMS_m"]
        for r in rows
    ])

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            "Invalid calibration values"
        )

    return float(np.quantile(values, Q))


def evaluate(rows, cutoff):
    result = []
    for r in rows:
        item = dict(r)

        if r["minimum_RMS_m"] > cutoff:
            item["decision"] = "unresolved"
        else:
            item["decision"] = (
                r["selected_by_strong"]
            )

        # Evaluation labels remain private.
        if r["truth_PRIVATE"] == "saturating":
            item["evaluation_PRIVATE"] = (
                "correct_reject"
                if item["decision"] == "unresolved"
                else "unknown_false_accept"
            )
        else:
            target = TRUTH_TO_LAW[
                r["truth_PRIVATE"]
            ]

            if item["decision"] == target:
                outcome = "correct"
            elif item["decision"] == "unresolved":
                outcome = "represented_false_reject"
            else:
                outcome = "wrong_law"

            item["evaluation_PRIVATE"] = outcome

        result.append(item)

    return result


def summarize(band, records):
    print("\n" + "=" * 72)
    print("BAND:", band)
    print("=" * 72)

    result = {}

    for mode in MODES:
        rows = [
            r for r in records
            if r["truth_PRIVATE"] == mode
        ]

        outcomes = Counter(
            r["evaluation_PRIVATE"]
            for r in rows
        )

        decisions = Counter(
            r["decision"]
            for r in rows
        )

        success = sum(
            all(r["optimizer_success"].values())
            for r in rows
        )

        item = {
            "n": len(rows),

            "minimum_RMS_median_mm":
                float(
                    1000 * np.median([
                        r["minimum_RMS_m"]
                        for r in rows
                    ])
                ),

            "outcomes":
                dict(outcomes),

            "decisions":
                dict(decisions),

            "all_optimizers_success":
                success,
        }

        result[mode] = item

        print(
            f"\n{mode:12s}",
            f"Rmin_median={item['minimum_RMS_median_mm']:.3f}mm"
        )

        print("  outcomes :", dict(outcomes))
        print("  decisions:", dict(decisions))
        print(
            "  all fits converged:",
            f"{success}/{len(rows)}"
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    helper27_path = (
        HERE /
        "27_p3a3b_evidence_vs_structure.py"
    )

    helper27 = run_path(str(helper27_path))

    # Reuse the established Saturating extension.
    # Do not silently substitute another law.
    if "install_saturating" not in helper27:
        raise RuntimeError(
            "Missing established Saturating extension"
        )

    helper27["install_saturating"](sim)

    commands = make_commands(sim)

    # Smoke uses different random seeds and
    # cannot enter the formal calibration/test.
    if args.smoke:
        cal_seed = CAL_SEED + 100000
        test_seed = TEST_SEED + 100000

        cal_noise = CAL_NOISE_SEED + 100000
        test_noise = TEST_NOISE_SEED + 100000

        ncal, ntest = 2, 2
    else:
        cal_seed = CAL_SEED
        test_seed = TEST_SEED

        cal_noise = CAL_NOISE_SEED
        test_noise = TEST_NOISE_SEED

        ncal, ntest = CAL_N, TEST_N

    cal_scenes = make_scenes(
        sim, cal_seed, ncal
    )

    test_scenes = make_scenes(
        sim, test_seed, ntest
    )

    cal_records = collect(
        sim, api, cal_scenes, commands,
        cal_noise, REPRESENTED,
        ("original",),
    )["original"]

    cutoff = calibrate(cal_records)

    print(
        "\nP3-A6H SIX-MECHANISM "
        "OPEN-WORLD AUDIT"
    )

    print(
        "Calibration scenes:", ncal,
        "| Evaluation scenes:", ntest
    )

    print(
        "Frozen represented-family Q95:",
        round(cutoff * 1000, 4),
        "mm"
    )

    print(
        "Calibration library:",
        LAWS
    )

    print(
        "Saturating excluded from calibration."
    )

    evaluations = collect(
        sim, api, test_scenes, commands,
        test_noise, MODES, BANDS
    )

    all_records = {}
    all_summaries = {}

    for band in BANDS:
        records = evaluate(
            evaluations[band], cutoff
        )

        all_records[band] = records
        all_summaries[band] = summarize(
            band, records
        )

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report_path = OUT / (
        f"p3a6h_{suffix}_report.json"
    )

    private_path = OUT / (
        f"p3a6h_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P3-A6H five-library six-truth audit",

        "calibration_seed": cal_seed,
        "evaluation_seed": test_seed,

        "n_calibration_scenes": ncal,
        "n_evaluation_scenes": ntest,

        "calibrated_cutoff_m": cutoff,
        "quantile": Q,

        "calibrated_on": list(
            REPRESENTED
        ),

        "unknown_law":
            "saturating",

        "candidate_laws": list(LAWS),

        "summary": all_summaries,

        "source_sha256": {
            name: sha256(HERE / name)
            for name in (
                "16_p2d1b_probe_diversity.py",
                "22_p3a1b_structural_revision.py",
                "27_p3a3b_evidence_vs_structure.py",
            )
        },

        "limitations": [
            "Synthetic MuJoCo environments only.",
            "Five modeled laws, one held-out law.",
            "Weak and Strong are fixed experiments.",
            "No adaptive action selection.",
            "Strong data selects the final law but does not refit it.",
            "The minimum among five heldout residuals is selection-optimistic.",
            "Calibration uses the same selection rule on separate scenes.",
            "Calibration Q95 does not guarantee uniform per-mode coverage.",
            "B prediction is conditional on observed A.",
            "Two gap bands share base scenes.",
            "No E4 information is accessed.",
            "No test threshold is optimized.",
        ],
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    private_path.write_text(
        json.dumps(
            all_records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A6H SIX-MECHANISM "
        "AUDIT COMPLETE"
    )

    print(
        "No E4, no adaptive policy, "
        "no test threshold tuning."
    )

    print("Saved:", report_path)
    print("Saved:", private_path)


if __name__ == "__main__":
    main()
