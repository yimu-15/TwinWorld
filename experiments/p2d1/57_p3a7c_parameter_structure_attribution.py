
"""
P3-A7C: Structure-vs-Parameter Attribution

New physical scene seeds.

Fit E1/E2/Low Weak -> frozen BIC anchor.
Refit SAME law with Reference Strong added.

Evaluate BOTH models on:
  Vmid     = +1.75 * pulse
  Vreverse = -1.75 * pulse

These two challenges are never used for fitting.

Oracle correct-law comparison:
  PRIVATE diagnostic only.

No E4 access.
No online truth access.
No threshold tuning.
No new active policy claimed.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a7c"

HISTORIC = (
    HERE / "results" / "p3a7b"
    / "p3a7b_fresh_report.json"
)

SEED = 20270325
NOISE = 2027032501

N_FULL = 36
NFEV = 60

MODES = (
    "elastic",
    "deadzone",
    "nonlinear",
    "saturating",
)

BANDS = (
    "original",
    "above_original_range",
)

TRUE_LAW = {
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}


def read(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def median(values):
    return float(np.median(values))


def probe(
    src, sim, api, model, scene,
    mode, command, seed,
):
    result, work = src["get_probe"](
        sim, api, model, scene,
        mode, command, seed
    )
    return result, float(work)


def fit_model(api, probes, laws):
    chosen, fitted = api["fit_library"](
        probes, laws, NFEV
    )
    return chosen, fitted


def error_mm(api, probe_obj, fit):
    value = float(
        api["prediction_rms"](
            probe_obj, fit
        )
    )
    if not np.isfinite(value):
        raise RuntimeError(
            "Nonfinite prediction error"
        )
    return 1000.0 * value


def run_case(
    src, sim, api, commands,
    scene, sid, mode, band,
):
    model = sim["make_model"](scene)

    mode_idx = src["MODES"].index(mode)
    band_idx = BANDS.index(band)

    seed_base = (
        NOISE
        + sid * 1000
        + mode_idx * 20
        + band_idx * 100000
    )

    observations = {}
    work = {}

    # All actions have independent
    # measurement-noise realizations.
    actions = (
        "e1", "e2", "weak",
        "strong", "vmid", "vreverse",
    )

    for index, action in enumerate(actions):
        observations[action], work[action] = (
            probe(
                src, sim, api, model,
                scene, mode,
                commands[action],
                seed_base + index,
            )
        )

    initial = [
        observations[name]
        for name in ("e1", "e2", "weak")
    ]

    expanded = initial + [
        observations["strong"]
    ]

    # Fit the whole candidate library BEFORE
    # seeing Strong or validation probes.
    _, fits = fit_model(
        api, initial, src["LAWS"]
    )

    anchor = min(
        src["LAWS"],
        key=lambda h: fits[h]["bic"]
    )

    pre_fit = fits[anchor]

    # Keep the structural law frozen.
    # Only its parameters are reestimated.
    post_fit, _ = fit_model(
        api, expanded, (anchor,)
    )

    # Oracle comparison: fit the true law
    # using the same expanded data.
    # NEVER used to choose an online model.
    correct_law = TRUE_LAW.get(mode)

    oracle_fit = None

    if correct_law is not None:
        oracle_fit, _ = fit_model(
            api, expanded, (correct_law,)
        )

    result = {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "anchor": anchor,
        "anchor_matches_truth_PRIVATE":
            (
                anchor == correct_law
                if correct_law is not None
                else None
            ),

        "strong_frozen_RMS_mm":
            error_mm(
                api,
                observations["strong"],
                pre_fit,
            ),

        # In-sample after refitting.
        # NOT independent validation.
        "strong_refit_RMS_mm":
            error_mm(
                api,
                observations["strong"],
                post_fit,
            ),

        "initial_work_J": sum(
            work[name]
            for name in ("e1", "e2", "weak")
        ),

        "strong_work_J": work["strong"],
    }

    for action in ("vmid", "vreverse"):
        before = error_mm(
            api, observations[action], pre_fit
        )

        after = error_mm(
            api, observations[action], post_fit
        )

        result[
            f"{action}_frozen_RMS_mm"
        ] = before

        result[
            f"{action}_refit_RMS_mm"
        ] = after

        result[
            f"{action}_gain_mm"
        ] = before - after

        result[
            f"{action}_work_J"
        ] = work[action]

        if oracle_fit is not None:
            result[
                f"{action}_oracle_RMS_mm_PRIVATE"
            ] = error_mm(
                api,
                observations[action],
                oracle_fit,
            )
        else:
            result[
                f"{action}_oracle_RMS_mm_PRIVATE"
            ] = None

    return result


def print_summary(records):
    summary = {}

    for band in BANDS:
        print("\n" + "=" * 70)
        print("BAND:", band)
        print("=" * 70)

        summary[band] = {}

        for mode in MODES:
            rows = [
                row for row in records
                if (
                    row["band"] == band
                    and row["truth_PRIVATE"] == mode
                )
            ]

            if not rows:
                continue

            item = {
                "n": len(rows),

                "anchor_distribution": {
                    law: sum(
                        r["anchor"] == law
                        for r in rows
                    )
                    for law in (
                        "independent", "rigid",
                        "elastic", "deadzone",
                        "cubic",
                    )
                },
            }

            if mode in TRUE_LAW:
                item["correct_anchor"] = sum(
                    r["anchor_matches_truth_PRIVATE"]
                    for r in rows
                )

            for action in ("vmid", "vreverse"):
                gains = [
                    r[f"{action}_gain_mm"]
                    for r in rows
                ]

                item[
                    f"{action}_median_gain_mm"
                ] = median(gains)

                item[
                    f"{action}_improved_count"
                ] = sum(g > 0 for g in gains)

                item[
                    f"{action}_worsened_count"
                ] = sum(g < 0 for g in gains)

            summary[band][mode] = item

            print("\nMODE:", mode)
            print(
                "  Anchors:",
                item["anchor_distribution"]
            )

            if mode in TRUE_LAW:
                print(
                    "  Correct anchor:",
                    item["correct_anchor"],
                    "/", len(rows)
                )

            for action in ("vmid", "vreverse"):
                print(
                    " ", action,
                    "median refit gain:",
                    round(
                        item[
                            f"{action}_median_gain_mm"
                        ], 4
                    ),
                    "mm",
                    "| improved:",
                    item[
                        f"{action}_improved_count"
                    ],
                    "| worsened:",
                    item[
                        f"{action}_worsened_count"
                    ],
                )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    historic = read(HISTORIC)

    if historic["smoke"]:
        raise RuntimeError(
            "Expected formal historical report"
        )

    # Preserve the underlying physics.
    for name, expected in (
        historic["source_sha256"].items()
    ):
        if sha256(HERE / name) != expected:
            raise RuntimeError(
                "Physics or fitting source changed: "
                + name
            )

    base = run_path(str(
        HERE /
        "55_p3a7a_evidence_gated_escalation.py"
    ))

    src, _, sim, api = base[
        "init_modules"
    ]()

    pulse = sim["make_profiles"]()["pulse"]

    commands = {
        "e1": pulse * 0.25,
        "e2": pulse * 0.50,
        "weak": pulse * 0.75,
        "strong": pulse * 2.25,

        # Independently measured validation.
        "vmid": pulse * 1.75,
        "vreverse": pulse * -1.75,
    }

    offset = 100000 if args.smoke else 0
    n_scenes = 2 if args.smoke else N_FULL

    scenes = src["make_scenes"](
        sim, SEED + offset, n_scenes
    )

    results = []

    print(
        "\nP3-A7C PARAMETER-STRUCTURE "
        "ATTRIBUTION"
    )
    print("Scenes:", n_scenes)
    print("Modes:", MODES)
    print("No E4, no test tuning.")

    for sid, scene_set in enumerate(scenes):
        for band in BANDS:
            for mode in MODES:
                results.append(
                    run_case(
                        src, sim, api, commands,
                        scene_set[band],
                        sid, mode, band,
                    )
                )

    expected = (
        n_scenes * len(BANDS) * len(MODES)
    )

    if len(results) != expected:
        raise RuntimeError(
            "Incomplete experimental matrix"
        )

    summary = print_summary(results)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    audit_path = OUT / (
        f"p3a7c_{suffix}_audit_PRIVATE.csv"
    )

    with audit_path.open(
        "w", newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    report = {
        "protocol":
            "P3-A7C parameter-structure attribution",

        "smoke": args.smoke,
        "scene_seed": SEED + offset,
        "n_base_scenes": n_scenes,
        "n_cases": len(results),
        "summary": summary,

        "frozen_candidate_structure": True,
        "parameter_refit_after_strong": True,
        "validation_probes_used_for_fit": False,
        "E4_accessed": False,
        "online_uses_truth": False,
        "thresholds_tuned": False,

        "limitations": [
            "MuJoCo simulation only",
            "Post-Strong parameter refit",
            "Validation probes share scene physics",
            "Validation varies pulse amplitude/sign",
            "No active policy is tested",
            "Oracle comparison is PRIVATE only",
            "No identifiability guarantee",
            "Both gap bands share base scenes",
            "Results become development evidence",
        ],
    }

    report_path = OUT / (
        f"p3a7c_{suffix}_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A7C PARAMETER-STRUCTURE "
        "AUDIT COMPLETE"
    )
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
