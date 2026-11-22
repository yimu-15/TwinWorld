
"""
P3-A6L Stage 2: Clean-trajectory attribution.

Replays original P3-A6K Saturating episodes.
Compares:
  1. True Saturating vs true Elastic clean trajectories
  2. True Saturating vs fitted Elastic, clean A-conditioned
  3. True Saturating vs fitted Elastic, noisy A-conditioned

The fitted Elastic model uses E1/E2/Weak ONLY.
Strong never participates in parameter estimation.

The 17 known failures are used for attribution,
not for fitting a new decision threshold.

No E4. No policy tuning. No Git changes.
"""

from pathlib import Path
from runpy import run_path
from collections import Counter
import argparse
import csv
import hashlib
import json

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"

A6K = ROOT / "p3a6k"
OUT = ROOT / "p3a6l"

REGIMES = {
    "reference": (1.50, 2.25),
    "moderate": (1.25, 1.75),
    "low": (0.75, 1.10),
}

ACTIONS = ("e1", "e2", "weak", "strong")

MODE = "saturating"


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def rms(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise RuntimeError("RMS shape mismatch")

    value = np.sqrt(
        np.mean((x - y) ** 2)
    )

    if not np.isfinite(value):
        raise RuntimeError("Nonfinite RMS")

    return float(value)


def same(actual, expected, label):
    if not np.isclose(
        actual,
        expected,
        atol=1e-8,
        rtol=0,
    ):
        raise RuntimeError(
            f"REPLAY FAILURE: {label}, "
            f"{actual} != {expected}"
        )


def get_index(rows):
    return {
        (
            int(row["scene_id"]),
            row["truth_PRIVATE"],
        ): row
        for row in rows
    }


def median(values):
    return float(
        np.median(np.asarray(values, float))
    )


def group_summary(records, group, regime):
    rows = [
        r for r in records
        if (
            r["group"] == group
            and r["regime"] == regime
        )
    ]

    if not rows:
        return {}

    fields = (
        "true_vs_same_param_elastic_mm",
        "fitted_elastic_clean_RMS_mm",
        "fitted_elastic_noisy_RMS_mm",
        "strong_B_noise_RMS_mm",
        "max_relative_excursion_mm",
    )

    output = {
        "n": len(rows)
    }

    print(
        "\nGROUP:", group,
        "REGIME:", regime,
        "N:", len(rows)
    )

    for field in fields:
        values = [
            float(r[field])
            for r in rows
        ]

        output[field] = {
            "median": median(values),
            "min": float(min(values)),
            "max": float(max(values)),
        }

        print(
            " ", field,
            "median=",
            round(median(values), 4),
            "range=",
            (
                round(min(values), 4),
                round(max(values), 4),
            ),
        )

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    args = parser.parse_args()

    report = load(
        A6K / "p3a6k_fresh_report.json"
    )

    audit = load(
        A6K /
        "p3a6k_fresh_audit_PRIVATE.json"
    )

    if report["smoke"]:
        raise RuntimeError(
            "Expected formal P3-A6K report"
        )

    # Fail-fast if historical source files
    # differ from the exact A6K experiment.
    for filename, expected_hash in (
        report["source_sha256"].items()
    ):
        path = HERE / filename

        actual_hash = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        if actual_hash != expected_hash:
            raise RuntimeError(
                "Historical source changed: "
                + filename
            )

    src = run_path(str(
        HERE /
        "47_p3a6h_six_mechanism_openworld_audit.py"
    ))

    sim = run_path(str(
        HERE /
        "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE /
        "22_p3a1b_structural_revision.py"
    ))

    helper = run_path(str(
        HERE /
        "27_p3a3b_evidence_vs_structure.py"
    ))

    policy = run_path(str(
        HERE /
        "48_p3a6i_calibrated_revision.py"
    ))

    helper["install_saturating"](sim)

    scenes = src["make_scenes"](
        sim,
        int(report["scene_seed"]),
        int(report["n_base_scenes"]),
    )

    if len(scenes) != 48:
        raise RuntimeError(
            "Expected 48 formal scenes"
        )

    indexes = {
        regime: get_index(
            audit[regime]["original"]
        )
        for regime in REGIMES
    }

    # Saturating ignores the Deadzone gap.
    # Verify the two historical gap-band
    # records agree before deduplicating.
    for regime in REGIMES:
        other = get_index(
            audit[regime][
                "above_original_range"
            ]
        )

        for sid in range(48):
            key = (sid, MODE)

            a = indexes[regime][key]
            b = other[key]

            same(
                a["minimum_RMS_mm"],
                b["minimum_RMS_mm"],
                "Saturating cross-band consistency",
            )

            if (
                a["candidate_decision"]
                != b["candidate_decision"]
            ):
                raise RuntimeError(
                    "Saturating band decision mismatch"
                )

    failed_ids = {
        sid
        for sid in range(48)
        if indexes["low"][
            (sid, MODE)
        ]["candidate_outcome_PRIVATE"]
        == "unknown_false_accept"
    }

    if len(failed_ids) != 17:
        raise RuntimeError(
            f"Expected 17 failed scenes, "
            f"got {len(failed_ids)}"
        )

    ids = (
        sorted(failed_ids)[:2]
        if args.smoke
        else list(range(48))
    )

    print(
        "P3-A6L STAGE-2 CLEAN "
        "TRAJECTORY ATTRIBUTION"
    )
    print(
        "Formal Low failures:",
        len(failed_ids), "/48"
    )
    print(
        "Currently replayed scenes:",
        len(ids)
    )

    profiles = sim["make_profiles"]()
    base_commands = src[
        "make_commands"
    ](sim)

    records = []

    for regime, (
        weak_gain,
        strong_gain,
    ) in REGIMES.items():

        commands = dict(base_commands)

        commands["weak"] = (
            profiles["pulse"] * weak_gain
        )
        commands["strong"] = (
            profiles["pulse"] * strong_gain
        )

        for sid in ids:
            scene = scenes[sid]["original"]
            model = sim["make_model"](scene)

            # First replay the complete original
            # five-candidate inference outcome.
            replay = src["run_episode"](
                sim,
                api,
                scene,
                sid,
                MODE,
                commands,
                int(report["noise_seed"]),
            )

            archived = indexes[
                regime
            ][(sid, MODE)]

            same(
                replay["minimum_RMS_m"]
                * 1000.0,
                archived["minimum_RMS_mm"],
                f"{regime}/{sid} minimum RMS",
            )

            decision, reason = policy[
                "decide"
            ](
                replay,
                report["frozen_thresholds"],
            )

            if (
                decision
                != archived["candidate_decision"]
                or reason
                != archived["decision_reason"]
            ):
                raise RuntimeError(
                    f"Decision replay mismatch: "
                    f"{regime}/{sid}"
                )

            # Reconstruct exactly the same
            # noisy E1/E2/Weak observations.
            noisy = {}

            mode_index = src[
                "MODES"
            ].index(MODE)

            for action_index, action in enumerate(
                ACTIONS
            ):
                noise_seed = (
                    int(report["noise_seed"])
                    + sid * 1000
                    + mode_index * 20
                    + action_index
                )

                probe, _ = src["get_probe"](
                    sim,
                    api,
                    model,
                    scene,
                    MODE,
                    commands[action],
                    noise_seed,
                )

                noisy[action] = probe

            train = [
                noisy["e1"],
                noisy["e2"],
                noisy["weak"],
            ]

            elastic_fit, _ = api[
                "fit_library"
            ](
                train,
                ("elastic",),
                src["NFEV"],
            )

            # Exact same Elastic fit as
            # the original candidate library.
            noisy_rms = float(
                api["prediction_rms"](
                    noisy["strong"],
                    elastic_fit,
                )
            )

            same(
                noisy_rms,
                replay[
                    "all_law_RMS_m"
                ]["elastic"],
                f"{regime}/{sid} elastic fit",
            )

            # Clean Saturating world.
            sat_passive, sat_active, _, _ = (
                sim["simulate"](
                    model,
                    scene,
                    MODE,
                    commands["strong"],
                )
            )

            # Clean Elastic counterfactual:
            # same scene parameters, not a
            # best-fitting Elastic model.
            _, elastic_active, _, _ = (
                sim["simulate"](
                    model,
                    scene,
                    "elastic",
                    commands["strong"],
                )
            )

            clean_probe = api[
                "RevisionProbe"
            ](
                sat_passive,
                sat_active,
            )

            clean_prediction = (
                clean_probe.predict(
                    "elastic",
                    elastic_fit["params"],
                )
            )

            if (
                len(clean_prediction)
                != len(sat_active)
            ):
                raise RuntimeError(
                    "Clean trajectory mismatch"
                )

            # Physical responses with fixed
            # original simulator parameters.
            physical_difference = rms(
                sat_active[:, 1],
                elastic_active[:, 1],
            )

            # Frozen inferred Elastic law,
            # conditioned on clean observed A.
            fitted_clean_error = rms(
                clean_prediction,
                sat_active[:, 1],
            )

            # Noisy conditional prediction
            # with original noise replay.
            noisy_error = noisy_rms

            # Measurement noise on B only.
            # A noise also affects the
            # predictor's conditional input.
            b_observation_noise = rms(
                noisy["strong"].y_b,
                sat_active[:, 1],
            )

            relative = (
                (sat_active[:, 1]
                 - sat_active[0, 1])
                -
                (sat_active[:, 0]
                 - sat_active[0, 0])
            )

            record = {
                "scene_id": sid,
                "regime": regime,

                "group": (
                    "low_false_accept"
                    if sid in failed_ids
                    else "low_correct_reject"
                ),

                "historical_decision":
                    archived["candidate_decision"],

                "true_vs_same_param_elastic_mm":
                    1000 * physical_difference,

                "fitted_elastic_clean_RMS_mm":
                    1000 * fitted_clean_error,

                "fitted_elastic_noisy_RMS_mm":
                    1000 * noisy_error,

                "strong_B_noise_RMS_mm":
                    1000 * b_observation_noise,

                "max_relative_excursion_mm":
                    1000 * float(
                        np.max(np.abs(relative))
                    ),

                "elastic_alpha":
                    float(
                        elastic_fit["params"][0]
                    ),

                "elastic_beta":
                    float(
                        elastic_fit["params"][1]
                    ),

                "true_mass_b_PRIVATE":
                    float(scene["mass_b"]),

                "true_k_PRIVATE":
                    float(scene["k"]),

                "true_c_PRIVATE":
                    float(scene["c"]),
            }

            records.append(record)

    summary = {}

    for group in (
        "low_false_accept",
        "low_correct_reject",
    ):
        summary[group] = {}

        for regime in REGIMES:
            summary[group][regime] = (
                group_summary(
                    records,
                    group,
                    regime,
                )
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke
        else "formal"
    )

    csv_path = (
        OUT /
        f"p3a6l_stage2_{suffix}_PRIVATE.csv"
    )

    if records:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(records[0]),
            )
            writer.writeheader()
            writer.writerows(records)

    json_path = (
        OUT /
        f"p3a6l_stage2_{suffix}_report.json"
    )

    json_path.write_text(
        json.dumps(
            {
                "protocol":
                    "P3-A6L Stage-2",
                "smoke": args.smoke,
                "replayed_scene_ids": ids,
                "low_failed_scene_ids":
                    sorted(failed_ids),
                "thresholds_changed": False,
                "E4_accessed": False,
                "summary": summary,
                "limitations": [
                    "Known failure cases",
                    "Not new heldout validation",
                    "Conditional on observed A",
                    "Same-parameter counterfactual "
                    "is not a best-fit law comparison",
                    "No force-level replay",
                    "Cannot prove global "
                    "structural equivalence",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A6L STAGE-2 CLEAN "
        "TRAJECTORY AUDIT PASS"
    )
    print("No thresholds changed. No E4.")
    print("Saved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
