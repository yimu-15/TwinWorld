
"""
P4-M2B1 — Band-Blind Physical Prior Audit

Reproduce P4-M2B0 development observations.

Only change:
  Deadzone gap parameter domain becomes
  [0.015, 0.18] for BOTH evaluation bands.

The fitting algorithm is otherwise identical
to the P4-M2B0 physical fit.

Runtime-only function override;
no historical scripts are edited.

No E4. No new test data.
No structural threshold tuning.
"""

import argparse
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p4m2b1"

P74 = HERE / "74_p4m2a_generator_inference_fidelity.py"
P75 = HERE / "75_p4m2b0_physical_domain_fit_pilot.py"

REF_DIR = HERE / "results" / "p4m2b0"

BANDS = ("original", "above_original_range")

MODES = (
    "independent", "rigid", "elastic",
    "deadzone", "nonlinear",
)

TRUE_LAW = {
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

SEED = 20290317
NOISE_SEED = 2029031701


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values):
    return float(np.median(np.asarray(values, dtype=float)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    suffix = "smoke" if args.smoke else "dev"
    n = 1 if args.smoke else 12
    offset = 100000 if args.smoke else 0

    old_report = read(
        REF_DIR / f"p4m2b0_{suffix}_report.json"
    )
    old_audit = read(
        REF_DIR / f"p4m2b0_{suffix}_audit_PRIVATE.json"
    )

    if bool(old_report["smoke"]) != args.smoke:
        raise RuntimeError("Reference mode mismatch")

    # Verify unchanged historical implementation.
    for name, expected in old_report["source_sha256"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError(
                "Historical source changed: " + name
            )

    p74 = run_path(str(P74))
    p75 = run_path(str(P75))

    src, sim, api, commands = p74["initialize"]()

    # run_path functions retain a shared private
    # module-global namespace. Change only the
    # gap bound function in the current process.
    runtime = p75["fit_physical"].__globals__

    if runtime is not p75["run_case"].__globals__:
        raise RuntimeError("Unexpected module namespace")

    old_gap_bounds = runtime["gap_bounds"]

    if not np.allclose(
        old_gap_bounds("original"),
        (0.015, 0.12)
    ):
        raise RuntimeError("Unexpected original prior")

    if not np.allclose(
        old_gap_bounds("above_original_range"),
        (0.12, 0.18)
    ):
        raise RuntimeError("Unexpected extended prior")

    runtime["gap_bounds"] = (
        lambda band: (0.015, 0.18)
    )

    reference = {
        (band, row["scene_id"], row["truth_PRIVATE"]): row
        for band, rows in old_audit.items()
        for row in rows
    }

    scenes = src["make_scenes"](
        sim, SEED + offset, n
    )

    print("\nP4-M2B1 BAND-BLIND PRIOR AUDIT")
    print("Scenes:", n)
    print("Deadzone prior: [0.015, 0.18] m")
    print("Same development observations as P4-M2B0")
    print("No E4. No test threshold tuning.")

    audits = {}
    summaries = {}

    for band in BANDS:
        rows = []

        for sid, scene_set in enumerate(scenes):
            scene = scene_set[band]

            for mode in MODES:
                probes = p75["create_noisy_probes"](
                    sim, api, commands,
                    scene, sid, band, mode,
                    NOISE_SEED + offset
                )

                training = [
                    probes[name]
                    for name in p75["TRAIN"]
                ]

                validation = probes["mid"]

                # Parity check: reproduce the
                # original unconstrained fit.
                legacy = api["fit_one"](
                    training, "deadzone", 40
                )

                legacy_rms = p75["rms_mm"](
                    validation.predict(
                        "deadzone", legacy["params"]
                    ),
                    validation.y_b,
                )

                baseline = reference[
                    (band, sid, mode)
                ]

                expected_rms = baseline[
                    "fits"
                ]["deadzone"]["legacy_rms_mm"]

                if not np.isclose(
                    legacy_rms, expected_rms,
                    atol=1e-5, rtol=1e-9
                ):
                    raise RuntimeError(
                        f"Observation parity mismatch: "
                        f"{band}/{sid}/{mode}"
                    )

                # Band-blind constrained fit.
                physical = p75["fit_physical"](
                    training, "deadzone", band
                )

                if not physical["optimizer_success"]:
                    raise RuntimeError(
                        "Physical solver did not converge"
                    )

                if not p75["exactly_feasible"](
                    "deadzone", physical["params"], band
                ):
                    raise RuntimeError(
                        "Invalid band-blind parameters"
                    )

                blind_rms = p75["rms_mm"](
                    validation.predict(
                        "deadzone", physical["params"]
                    ),
                    validation.y_b
                )

                aware_rms = baseline[
                    "fits"
                ]["deadzone"]["physical_rms_mm"]

                # Other physical fits are unchanged
                # by design; only Deadzone is rerun.
                aware_scores = {
                    law: entry["physical_rms_mm"]
                    for law, entry in baseline["fits"].items()
                }

                blind_scores = dict(aware_scores)
                blind_scores["deadzone"] = blind_rms

                true_law = TRUE_LAW.get(mode)

                row = {
                    "scene_id": sid,
                    "band": band,
                    "truth_PRIVATE": mode,
                    "legacy_deadzone_mm": legacy_rms,
                    "band_aware_deadzone_mm": aware_rms,
                    "band_blind_deadzone_mm": blind_rms,
                    "blind_minus_aware_mm":
                        blind_rms - aware_rms,
                    "blind_params_PRIVATE":
                        physical["params"],
                    "band_aware_scores_PRIVATE":
                        aware_scores,
                    "band_blind_scores_PRIVATE":
                        blind_scores,
                    "true_winner_aware_PRIVATE": (
                        min(aware_scores, key=aware_scores.get)
                        == true_law
                        if true_law else None
                    ),
                    "true_winner_blind_PRIVATE": (
                        min(blind_scores, key=blind_scores.get)
                        == true_law
                        if true_law else None
                    ),
                }

                rows.append(row)

        if len(rows) != n * len(MODES):
            raise RuntimeError("Incomplete audit")

        audits[band] = rows
        summaries[band] = {}

        print("\nBAND:", band)

        for mode in MODES:
            subset = [
                row for row in rows
                if row["truth_PRIVATE"] == mode
            ]

            item = {
                "n": len(subset),
                "legacy_median_mm": median([
                    r["legacy_deadzone_mm"]
                    for r in subset
                ]),
                "aware_median_mm": median([
                    r["band_aware_deadzone_mm"]
                    for r in subset
                ]),
                "blind_median_mm": median([
                    r["band_blind_deadzone_mm"]
                    for r in subset
                ]),
                "median_paired_delta_mm": median([
                    r["blind_minus_aware_mm"]
                    for r in subset
                ]),
            }

            if mode in TRUE_LAW:
                item["true_winner_aware"] = sum(
                    r["true_winner_aware_PRIVATE"]
                    for r in subset
                )
                item["true_winner_blind"] = sum(
                    r["true_winner_blind_PRIVATE"]
                    for r in subset
                )

            summaries[band][mode] = item

            print(
                f"  {mode:12s}",
                f"aware={item['aware_median_mm']:.3f}",
                f"blind={item['blind_median_mm']:.3f}",
                f"paired_delta="
                f"{item['median_paired_delta_mm']:+.3f} mm"
            )

            if mode in TRUE_LAW:
                print(
                    "    true winner:",
                    item["true_winner_aware"],
                    "->",
                    item["true_winner_blind"],
                    "/",
                    n,
                )

    OUT.mkdir(parents=True, exist_ok=True)

    report_path = (
        OUT / f"p4m2b1_{suffix}_report.json"
    )
    audit_path = (
        OUT / f"p4m2b1_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol": "P4-M2B1 band-blind prior audit",
        "smoke": args.smoke,
        "n_scenes": n,
        "scene_seed": SEED + offset,
        "noise_seed": NOISE_SEED + offset,
        "deadzone_gap_prior": [0.015, 0.18],
        "reference": "P4-M2B0 paired development",
        "summary": summaries,
        "E4_accessed": False,
        "test_threshold_tuning": False,
        "independent_evaluation": False,
        "source_sha256": {
            P74.name: sha(P74),
            P75.name: sha(P75),
            Path(__file__).name: sha(Path(__file__)),
        },
        "limitations": [
            "Previously used development scenes",
            "Only the gap prior changed",
            "No structural threshold calibration",
            "Three flexible candidates in ranking audit",
            "Generator-domain assumptions still used",
            "No prior-misspecification stress test",
            "No independent test conclusions",
        ],
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(audits, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nP4-M2B1 COMPLETE")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
