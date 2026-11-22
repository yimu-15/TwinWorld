
"""
P3-A6D: Heldout Candidate Elimination Audit

Freeze P3-A6B's E1/E2-fitted candidate set.
Evaluate Weak and Strong as ALTERNATIVE E3 actions.

Action-specific thresholds are cross-fitted using
Independent-only scenes.

Evaluation:
  - Independent reference survives?
  - Deadzone candidate set shrinks?
  - True gap's nearest grid cell survives?
  - Does Strong improve on Weak?

No candidate refitting with E3.
No E4 observations.
No action policy or threshold tuning.
Exploratory development scenes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"

A4D = (
    ROOT / "p3a4d"
    / "p3a4d_stress_audit_PRIVATE.json"
)

A6B = (
    ROOT / "p3a6b"
    / "p3a6b_development_audit_PRIVATE.json"
)

OUT = ROOT / "p3a6d"

SEED = 20261128
FOLD_SEED = 20261129

SIGMA = 0.005
NFEV = 40
QUANTILE = 0.95
N_FOLDS = 4

BANDS = (
    "original",
    "high_in_range",
    "above_original_range",
)

MODES = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear",
    "saturating",
)

EVALUATED = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
)

FOCUS = (
    "independent",
    "deadzone",
)

ACTIONS = (
    "pulse_1p5",
    "pulse_2p25",
)

ACTION_ORDER = (
    "e1",
    "e2",
    "pulse_1p5",
    "pulse_2p25",
    "sweep_2p25",
    "pulse_2p75",
    "oscillation_2p25",
)


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def skip_noise(rng):
    rng.normal(
        0.0, SIGMA, (51, 2)
    )
    rng.normal(
        0.0, SIGMA, (51, 2)
    )


def rms_prediction(probe, observed, kind, params):
    predicted = np.asarray(
        probe.predict(
            kind,
            np.asarray(params, dtype=float),
        ),
        dtype=float,
    ).reshape(-1)

    target = np.asarray(
        observed[:, 1],
        dtype=float,
    ).reshape(-1)

    if predicted.shape != target.shape:
        raise RuntimeError(
            "Prediction/observation shape mismatch"
        )

    if not np.all(np.isfinite(predicted)):
        raise RuntimeError(
            "Nonfinite model prediction"
        )

    return float(np.sqrt(
        np.mean(
            (predicted - target) ** 2
        )
    ))


def replay_band(
    band,
    scenes,
    old,
    saved,
    sim,
    api,
    commands,
):
    """
    Reproduce original observation noise order.

    For every scenario and every mode, all actions
    consume their original random-number slots.

    Only E1/E2/Weak/Strong for selected modes
    are actually simulated and inspected.
    """
    rng = np.random.default_rng(
        SEED + 222 + 100000
    )

    old_index = {
        (
            int(r["scene_id"]),
            r["truth_PRIVATE_EVAL_ONLY"],
        ): r
        for r in old[band]["trajectories"]
    }

    saved_index = {
        (
            int(r["scene_id"]),
            r["truth_PRIVATE"],
        ): r
        for r in saved[band]
    }

    records = []
    calibration_records = []

    for sid, scene in enumerate(scenes):
        model = sim["make_model"](scene)

        for mode in MODES:
            observations = {}

            for action in ACTION_ORDER:

                if (
                    mode in EVALUATED
                    and action in (
                        "e1", "e2", *ACTIONS
                    )
                ):
                    passive, active, _, _ = (
                        sim["simulate"](
                            model,
                            scene,
                            mode,
                            commands[action],
                        )
                    )

                    p, a = api[
                        "noisy_observation"
                    ](
                        passive,
                        active,
                        rng,
                    )

                    observations[action] = (
                        p, a
                    )

                else:
                    skip_noise(rng)

            if mode not in EVALUATED:
                continue

            early = [
                api["RevisionProbe"](
                    *observations[action]
                )
                for action in ("e1", "e2")
            ]

            known, _ = api[
                "fit_library"
            ](
                early,
                ("independent", "rigid", "elastic"),
                NFEV,
            )

            edited, _ = api[
                "fit_library"
            ](
                early,
                ("deadzone", "cubic"),
                NFEV,
            )

            archived = old_index[(sid, mode)]
            evidence = archived["early_evidence"]

            margin = float(
                known["bic"] - edited["bic"]
            )

            if (
                known["kind"]
                != evidence["known_kind"]
                or not np.isclose(
                    margin,
                    evidence["bic_advantage"],
                    atol=1e-3,
                    rtol=0.0,
                )
                or not np.isclose(
                    known["fit_rms_m"],
                    evidence["known_fit_rms_m"],
                    atol=1e-6,
                    rtol=0.0,
                )
            ):
                raise RuntimeError(
                    "Early replay mismatch: "
                    f"{band} scene={sid} mode={mode}"
                )

            # Known-only diagnostic: not used as
            # the targeted Independent-null cutoff.
            if mode in (
                "independent", "rigid", "elastic"
            ):
                item = {
                    "scene_id": sid,
                    "mode": mode,
                }

                for action in ACTIONS:
                    p, a = observations[action]
                    probe = api[
                        "RevisionProbe"
                    ](p, a)

                    item[action] = float(
                        api["prediction_rms"](
                            probe, known
                        )
                    )

                calibration_records.append(item)

            if mode not in FOCUS:
                continue

            frozen = saved_index[(sid, mode)]

            if not np.isclose(
                frozen["known_rms_m"],
                known["fit_rms_m"],
                atol=1e-8,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "P3-A6B frozen-fit mismatch"
                )

            candidates = [
                c for c in frozen["profiles"]
                if (
                    c["rms_m"]
                    <= frozen["frozen_cutoff_m"]
                )
            ]

            if (
                len(candidates)
                != frozen["retained_count"]
            ):
                raise RuntimeError(
                    "P3-A6B candidate mismatch"
                )

            # Independent reference is fitted
            # using E1/E2 ONLY.
            null_fit, _ = api[
                "fit_library"
            ](
                early,
                ("independent",),
                NFEV,
            )

            action_results = {}

            for action in ACTIONS:
                passive, active = (
                    observations[action]
                )

                probe = api[
                    "RevisionProbe"
                ](
                    passive,
                    active,
                )

                null_error = rms_prediction(
                    probe,
                    active,
                    "independent",
                    null_fit["params"],
                )

                candidate_errors = [
                    rms_prediction(
                        probe,
                        active,
                        "deadzone",
                        [
                            c["k"],
                            c["c"],
                            c["gap_m"],
                        ],
                    )
                    for c in candidates
                ]

                action_results[action] = {
                    "independent_RMS_m":
                        null_error,

                    "candidate_RMS_m":
                        candidate_errors,
                }

            records.append({
                "scene_id": sid,

                "mode_PRIVATE": mode,

                "pre_candidate_count":
                    len(candidates),

                "candidate_gaps_m": [
                    float(c["gap_m"])
                    for c in candidates
                ],

                "nearest_true_gap_PRIVATE": (
                    frozen.get(
                        "true_gap_nearest_grid_m_PRIVATE"
                    )
                ),

                "actions": action_results,
            })

    expected_n = 2 * len(scenes)

    if len(records) != expected_n:
        raise RuntimeError(
            "Unexpected focus episode count"
        )

    return records, calibration_records


def scene_folds(scene_ids):
    rng = np.random.default_rng(
        FOLD_SEED
    )

    order = rng.permutation(
        sorted(scene_ids)
    )

    return {
        int(sid): int(i % N_FOLDS)
        for i, sid in enumerate(order)
    }


def fit_null_thresholds(records, folds):
    """
    Only Independent truth on OTHER scenes
    calibrates the action-specific null residual.

    A threshold calibrated for Independent cannot
    be claimed to guarantee Deadzone coverage.
    """
    independent = [
        r for r in records
        if r["mode_PRIVATE"] == "independent"
    ]

    result = {}

    for fold in range(N_FOLDS):
        train = [
            r for r in independent
            if folds[r["scene_id"]] != fold
        ]

        if len(train) < 10:
            raise RuntimeError(
                "Insufficient crossfit calibration"
            )

        result[fold] = {
            action: float(np.quantile(
                [
                    r["actions"][action][
                        "independent_RMS_m"
                    ]
                    for r in train
                ],
                QUANTILE,
            ))
            for action in ACTIONS
        }

    return result


def apply_thresholds(rows, folds, thresholds):
    evaluated = []

    for row in rows:
        item = {
            "scene_id": row["scene_id"],
            "mode_PRIVATE":
                row["mode_PRIVATE"],
            "fold": folds[row["scene_id"]],
            "pre_candidate_count":
                row["pre_candidate_count"],
            "actions": {},
        }

        fold = item["fold"]

        for action in ACTIONS:
            threshold = thresholds[fold][action]
            data = row["actions"][action]

            retained_gaps = [
                gap
                for gap, error in zip(
                    row["candidate_gaps_m"],
                    data["candidate_RMS_m"],
                )
                if error <= threshold
            ]

            null_ok = (
                data["independent_RMS_m"]
                <= threshold
            )

            candidates_ok = (
                len(retained_gaps) > 0
            )

            if null_ok and candidates_ok:
                state = "ambiguous"
            elif null_ok:
                state = "independent_only"
            elif candidates_ok:
                state = "deadzone_only"
            else:
                state = "unresolved"

            nearest = row[
                "nearest_true_gap_PRIVATE"
            ]

            true_grid_kept = (
                None if nearest is None
                else bool(any(
                    np.isclose(
                        gap, nearest,
                        atol=1e-10,
                        rtol=0.0,
                    )
                    for gap in retained_gaps
                ))
            )

            item["actions"][action] = {
                "threshold_m": threshold,

                "independent_RMS_m":
                    data["independent_RMS_m"],

                "null_accepted": bool(
                    null_ok
                ),

                "pre_n":
                    row["pre_candidate_count"],

                "post_n":
                    len(retained_gaps),

                "retained_gaps_m":
                    retained_gaps,

                "state": state,

                "nearest_true_grid_retained_PRIVATE":
                    true_grid_kept,
            }

        evaluated.append(item)

    return evaluated


def summarize(band, evaluated):
    print("\n" + "=" * 72)
    print("BAND:", band)
    print("=" * 72)

    results = {}

    for mode in FOCUS:
        subset = [
            r for r in evaluated
            if r["mode_PRIVATE"] == mode
        ]

        mode_results = {}

        print(
            "\nTRUE MODE:", mode,
            "| N =", len(subset)
        )

        for action in ACTIONS:
            data = [
                r["actions"][action]
                for r in subset
            ]

            pre = np.asarray([
                r["pre_n"] for r in data
            ])

            post = np.asarray([
                r["post_n"] for r in data
            ])

            states = dict(Counter(
                r["state"] for r in data
            ))

            item = {
                "median_pre_n":
                    float(np.median(pre)),

                "median_post_n":
                    float(np.median(post)),

                "full_elimination": int(
                    np.sum(post == 0)
                ),

                "null_accepted": sum(
                    r["null_accepted"]
                    for r in data
                ),

                "states": states,
            }

            if mode == "deadzone":
                item["true_nearest_grid_kept"] = (
                    sum(
                        r[
                            "nearest_true_grid_retained_PRIVATE"
                        ] is True
                        for r in data
                    )
                )

            mode_results[action] = item

            print(
                f"  {action:14s}",
                "pre_median=",
                item["median_pre_n"],
                "post_median=",
                item["median_post_n"],
                "eliminated_all=",
                item["full_elimination"],
                "null_accepted=",
                item["null_accepted"],
            )

            print(
                "    states:", states
            )

            if mode == "deadzone":
                print(
                    "    nearest true-gap grid retained:",
                    f"{item['true_nearest_grid_kept']}"
                    f"/{len(data)}",
                )

        weak = [
            r["actions"]["pulse_1p5"]["post_n"]
            for r in subset
        ]

        strong = [
            r["actions"]["pulse_2p25"]["post_n"]
            for r in subset
        ]

        mode_results["paired_comparison"] = {
            "strong_fewer_survivors":
                sum(
                    s < w
                    for w, s in zip(weak, strong)
                ),

            "equal_survivors":
                sum(
                    s == w
                    for w, s in zip(weak, strong)
                ),

            "strong_more_survivors":
                sum(
                    s > w
                    for w, s in zip(weak, strong)
                ),
        }

        print(
            "  Strong vs Weak survivor count:",
            mode_results["paired_comparison"],
        )

        results[mode] = mode_results

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    old = json.loads(
        A4D.read_text(encoding="utf-8")
    )

    saved = json.loads(
        A6B.read_text(encoding="utf-8")
    )

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

    commands = helper35[
        "command_bank"
    ](
        sim["make_profiles"]()
    )

    n = 2 if args.smoke else 24

    all_rows = {}
    known_calibration = {}

    for band in BANDS:
        scenes = old[band][
            "scene_parameters_PRIVATE"
        ][:n]

        rows, known_rows = replay_band(
            band,
            scenes,
            old,
            saved,
            sim,
            api,
            commands,
        )

        all_rows[band] = rows
        known_calibration[band] = known_rows

        print(
            "Early replay PASS:",
            band,
            "focus episodes =", len(rows),
            "known-only records =", len(known_rows),
        )

    if args.smoke:
        print(
            "\nP3-A6D SMOKE REPLAY PASS"
        )
        print(
            "No thresholds evaluated in smoke mode."
        )
        return

    folds = scene_folds(
        range(n)
    )

    # Freeze calibration from the original band,
    # based exclusively on Independent.
    thresholds = fit_null_thresholds(
        all_rows["original"],
        folds,
    )

    print("\nFrozen crossfit null thresholds:")

    for fold, values in thresholds.items():
        print(
            "  fold=", fold,
            "weak_mm=",
            round(
                1000 * values["pulse_1p5"],
                4,
            ),
            "strong_mm=",
            round(
                1000 * values["pulse_2p25"],
                4,
            ),
        )

    output = {
        "protocol":
            "P3-A6D heldout candidate elimination",

        "calibration":
            "original-band Independent-only, "
            "scene-disjoint 4-fold Q95",

        "calibration_quantile": QUANTILE,

        "calibration_thresholds_m":
            thresholds,

        "frozen_candidates_from":
            str(A6B),

        "warnings": [
            "Existing previously inspected development scenes.",
            "Weak and Strong are alternative E3 experiments.",
            "E3 actual A is observed for conditional B prediction.",
            "Null calibration does not guarantee true Deadzone coverage.",
            "No statistical power or coverage guarantee.",
            "No online action policy has been tested.",
            "No E4 information is used.",
        ],

        "summary": {},
        "audit_PRIVATE": {},
    }

    for band in BANDS:
        evaluated = apply_thresholds(
            all_rows[band],
            folds,
            thresholds,
        )

        output["audit_PRIVATE"][band] = evaluated

        output["summary"][band] = summarize(
            band,
            evaluated,
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        OUT /
        "p3a6d_development_report.json"
    )

    audit_path = (
        OUT /
        "p3a6d_development_audit_PRIVATE.json"
    )

    report = {
        key: value
        for key, value in output.items()
        if key != "audit_PRIVATE"
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            output["audit_PRIVATE"],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A6D HELDOUT ELIMINATION AUDIT PASS"
    )

    print(
        "No E3 refitting, no E4 leakage, "
        "no policy tuning."
    )

    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
