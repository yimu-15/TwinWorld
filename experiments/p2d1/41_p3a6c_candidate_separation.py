
"""
P3-A6C: Frozen Candidate-Set Separation Audit

Inputs:
  - P3-A6B saved parameter profiles
  - reconstructed noisy E1/E2 observations
  - planned E3 commands

No true future A or B enters the prediction.
True labels are used only to group audit results.

Reports min, Q25, median and max separation
between retained Deadzone models and Independent.

This is not a calibrated statistical test or
a validated online action-selection policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"

P3A4D = (
    ROOT / "p3a4d"
    / "p3a4d_stress_audit_PRIVATE.json"
)

REPORT = (
    ROOT / "p3a6b"
    / "p3a6b_development_report.json"
)

SAVED = (
    ROOT / "p3a6b"
    / "p3a6b_development_audit_PRIVATE.json"
)

OUTPUT = ROOT / "p3a6c"

BANDS = (
    "original",
    "high_in_range",
    "above_original_range",
)

ACTIONS = (
    "pulse_1p5",
    "pulse_2p25",
    "sweep_2p25",
)

SIGMA_M = 0.005
NFEV = 40


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def verify_sources(report):
    for name, expected in (
        report["source_sha256"].items()
    ):
        if sha256(HERE / name) != expected:
            raise RuntimeError(
                "Source changed since P3-A6B: "
                + name
            )


def predict_vector(probe, law, params):
    value = np.asarray(
        probe.predict(
            law,
            np.asarray(params, dtype=float)
        ),
        dtype=float
    ).reshape(-1)

    if (
        value.shape != (51,)
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(
            f"Invalid {law} prediction: "
            f"{value.shape}"
        )

    return value


def rms(a, b):
    return float(np.sqrt(
        np.mean((a - b) ** 2)
    ))


def action_metrics(independent, candidates):
    differences = np.asarray([
        rms(x, independent)
        for x in candidates
    ])

    if len(differences) == 0:
        raise ValueError(
            "Empty candidate set"
        )

    return {
        "min_mm": float(
            1000 * np.min(differences)
        ),

        "q25_mm": float(
            1000 * np.quantile(
                differences, 0.25
            )
        ),

        "median_mm": float(
            1000 * np.median(differences)
        ),

        "max_mm": float(
            1000 * np.max(differences)
        ),

        "separable_count_at_5mm": int(
            np.sum(differences > SIGMA_M)
        ),

        "candidate_count": int(
            len(differences)
        ),

        "fraction_over_5mm": float(
            np.mean(differences > SIGMA_M)
        ),
    }


def evaluate_band(
    scenes,
    archived,
    old,
    sim,
    api,
    helper35,
    helper40,
    commands,
):
    index = {
        (
            int(x["scene_id"]),
            x["truth_PRIVATE"]
        ): x
        for x in archived
    }

    old_index = {
        (
            int(x["scene_id"]),
            x["truth_PRIVATE_EVAL_ONLY"]
        ): x
        for x in old["trajectories"]
    }

    observations = helper40[
        "reconstruct_early"
    ](
        scenes,
        old_index,
        sim,
        api,
        commands,
    )

    records = []

    for obs in observations:
        sid = int(obs["scene_id"])
        mode = obs["mode"]

        saved = index[(sid, mode)]

        if (
            saved["known_kind"]
            != obs["known_fit"]["kind"]
        ):
            raise RuntimeError(
                "Early fit mismatch"
            )

        if not np.isclose(
            saved["known_rms_m"],
            obs["known_fit"]["fit_rms_m"],
            atol=1e-8,
            rtol=0,
        ):
            raise RuntimeError(
                "Early RMS mismatch"
            )

        # Frozen candidate profiles.
        # No re-optimization occurs here.
        retained = [
            p for p in saved["profiles"]
            if p["rms_m"]
            <= saved["frozen_cutoff_m"]
        ]

        if len(retained) != int(
            saved["retained_count"]
        ):
            raise RuntimeError(
                "Candidate set mismatch"
            )

        if not retained:
            raise RuntimeError(
                "Empty candidate set"
            )

        # Forecast future A from E1/E2 only.
        future_A, gain = helper35[
            "forecast_A"
        ](
            obs["pairs"],
            commands,
            sim,
        )

        passive = np.mean(
            [
                pair[0]
                for pair in obs["pairs"]
            ],
            axis=0,
        )

        placeholder_B = np.full(
            51,
            float(passive[-1, 1]),
        )

        # Fit independent reference using
        # early data only.
        independent_fit, _ = api[
            "fit_library"
        ](
            obs["probes"],
            ("independent",),
            NFEV,
        )

        action_results = {}

        for action in ACTIONS:
            probe = api[
                "RevisionProbe"
            ](
                passive,
                np.column_stack([
                    future_A[action],
                    placeholder_B,
                ]),
            )

            independent_B = predict_vector(
                probe,
                "independent",
                independent_fit["params"],
            )

            candidate_B = [
                predict_vector(
                    probe,
                    "deadzone",
                    [
                        p["k"],
                        p["c"],
                        p["gap_m"],
                    ],
                )
                for p in retained
            ]

            action_results[action] = (
                action_metrics(
                    independent_B,
                    candidate_B,
                )
            )

        # Historical labels and stop decisions
        # are joined only for retrospective audit.
        old_stopped = (
            old_index[(sid, mode)]
            ["policies"]
            ["old_stop_or_strong"]
            ["action"] == "none"
        )

        records.append({
            "scene_id": sid,
            "mode_PRIVATE": mode,
            "old_stopped_PRIVATE": bool(
                old_stopped
            ),
            "retained_count": len(retained),
            "A_forecast_gain": float(gain),
            "action_separation":
                action_results,
        })

    if len(records) != 2 * len(scenes):
        raise RuntimeError(
            "Unexpected episode count"
        )

    return records


def print_group(label, rows):
    if not rows:
        return

    print(
        "\nGROUP:", label,
        "| N =", len(rows),
    )

    print(
        "Action             Min    Q25"
        "   Median    Max    All>5mm"
    )

    for action in ACTIONS:
        stats = [
            r["action_separation"][action]
            for r in rows
        ]

        def med(key):
            return float(np.median([
                x[key] for x in stats
            ]))

        all_separated = sum(
            x["min_mm"] > 5.0
            for x in stats
        )

        print(
            f"{action:16s}"
            f"{med('min_mm'):7.3f}"
            f"{med('q25_mm'):7.3f}"
            f"{med('median_mm'):9.3f}"
            f"{med('max_mm'):8.3f}"
            f"  {all_separated}/{len(rows)}"
        )

    print(
        "A forecast gain median:",
        round(float(np.median([
            r["A_forecast_gain"]
            for r in rows
        ])), 4),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    for path in (
        P3A4D, REPORT, SAVED
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    old = json.loads(
        P3A4D.read_text(encoding="utf-8")
    )

    report = json.loads(
        REPORT.read_text(encoding="utf-8")
    )

    saved = json.loads(
        SAVED.read_text(encoding="utf-8")
    )

    verify_sources(report)

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    helper35 = run_path(str(
        HERE /
        "35_p3a5b_preaction_model_disagreement.py"
    ))

    helper40 = run_path(str(
        HERE /
        "40_p3a6b_observation_compatible_set.py"
    ))

    commands = helper35[
        "command_bank"
    ](
        sim["make_profiles"]()
    )

    n = 2 if args.smoke else 24

    output = {
        "protocol":
            "P3-A6C frozen candidate separation",

        "prediction_inputs":
            "E1/E2 and planned commands only",

        "noise_reference_m": SIGMA_M,

        "not_statistical_detection_test": True,
        "not_action_policy": True,

        "limitations": [
            "Candidate sets come from reused development scenes.",
            "Only Independent and Deadzone are modeled.",
            "Future A is forecast, not observed.",
            "Pointwise 5mm comparison is descriptive.",
            "No calibrated confidence or worst-case detection guarantee.",
            "True labels are used only for reporting.",
        ],

        "bands": {},
    }

    for band in BANDS:
        scenes = old[band][
            "scene_parameters_PRIVATE"
        ][:n]

        records = evaluate_band(
            scenes,
            [
                r for r in saved[band]
                if int(r["scene_id"]) < n
            ],
            old[band],
            sim,
            api,
            helper35,
            helper40,
            commands,
        )

        output["bands"][band] = records

        print("\n" + "=" * 68)
        print(
            "BAND:", band,
            "| SCENES:", n,
        )
        print("=" * 68)

        print_group(
            "Independent",
            [
                r for r in records
                if r["mode_PRIVATE"]
                == "independent"
            ],
        )

        print_group(
            "Deadzone",
            [
                r for r in records
                if r["mode_PRIVATE"]
                == "deadzone"
            ],
        )

        print_group(
            "Old-stopped Deadzone",
            [
                r for r in records
                if (
                    r["mode_PRIVATE"]
                    == "deadzone"
                    and r["old_stopped_PRIVATE"]
                )
            ],
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        "p3a6c_smoke_audit_PRIVATE.json"
        if args.smoke
        else "p3a6c_development_audit_PRIVATE.json"
    )

    path = OUTPUT / filename

    path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A6C PREACTION CANDIDATE "
        "SEPARATION AUDIT PASS"
    )

    print(
        "No future E3/E4 observations, "
        "no candidate refitting, no policy tuning."
    )

    print("Saved:", path)


if __name__ == "__main__":
    main()
