
"""
P3-A6B: Observation-Compatible Physical Hypothesis Sets.

Goal:
  Construct a discrete set of Deadzone candidates
  using only noisy E1/E2 observations.

Target mechanisms:
  independent and deadzone

Candidate profile:
  For each fixed gap, optimize stiffness k and
  damping c without access to true parameters.

Frozen acceptance cutoff:
  P3-A4C known-only early-fit RMS q95.

Private audit:
  Compare selected candidates with actual gap
  and exact Independent equivalence envelope.

Limitations:
  - Exploratory reanalysis of existing scenes.
  - This is not a confidence set with guaranteed
    parameter coverage.
  - Physical parameter bounds are explicit priors.
  - No online E3 action selection.
  - No E4 information enters candidate fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent

SOURCE = (
    HERE / "results" / "p3a4d"
    / "p3a4d_stress_audit_PRIVATE.json"
)

CALIBRATION = (
    HERE / "results" / "p3a4c"
    / "p3a4c_prospective_report.json"
)

OUT = HERE / "results" / "p3a6b"

SEED = 20261128
NOISE_SIGMA_M = 0.005
NFEV = 40

MODES = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear",
    "saturating",
)

FOCUS = (
    "independent",
    "deadzone",
)

KNOWN = (
    "independent",
    "rigid",
    "elastic",
)

BANDS = (
    "original",
    "high_in_range",
    "above_original_range",
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

# This is a computational grid, not a true
# parameter confidence interval.
GAP_MIN = 0.015
GAP_MAX = 0.180
FULL_GRID_POINTS = 23

# Explicit nonzero-coupling prior.
# Prevents k=0 from trivially turning every
# Deadzone candidate into Independent.
K_BOUNDS = (2.0, 100.0)
C_BOUNDS = (0.0, 10.0)

GAP_MARGIN = 1e-6


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def grid_values(smoke):
    return np.linspace(
        GAP_MIN,
        GAP_MAX,
        7 if smoke else FULL_GRID_POINTS,
    )


def draw_unused_observation_noise(rng):
    """
    Match the two noise arrays consumed by
    api.noisy_observation in source 37.
    """
    rng.normal(
        0.0, NOISE_SIGMA_M, (51, 2)
    )
    rng.normal(
        0.0, NOISE_SIGMA_M, (51, 2)
    )


def reconstruct_early(
    scenes, old_rows, sim, api, commands
):
    """
    Reconstruct exact P3-A4D E1/E2 observations.

    All other mechanisms and actions consume the
    original random stream without fitting.

    Physical scene parameters enter only the
    simulator that GENERATES observations.

    The model fitter receives observed probes,
    not scene parameters or true mechanisms.
    """
    rng = np.random.default_rng(
        SEED + 222 + 100000
    )

    for sid, scene in enumerate(scenes):
        model = sim["make_model"](scene)

        for mode in MODES:
            pairs = []

            for action in ACTION_ORDER:
                if (
                    mode in FOCUS
                    and action in ("e1", "e2")
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

                    pairs.append((p, a))

                else:
                    draw_unused_observation_noise(
                        rng
                    )

            if mode not in FOCUS:
                continue

            probes = [
                api["RevisionProbe"](p, a)
                for p, a in pairs
            ]

            known, _ = api["fit_library"](
                probes,
                KNOWN,
                NFEV,
            )

            edit, edit_fits = api[
                "fit_library"
            ](
                probes,
                ("deadzone", "cubic"),
                NFEV,
            )

            archived = old_rows[(sid, mode)]
            evidence = archived[
                "early_evidence"
            ]

            margin = float(
                known["bic"] - edit["bic"]
            )

            # Replay integrity is mandatory.
            if (
                known["kind"]
                != evidence["known_kind"]
                or abs(
                    margin
                    - evidence["bic_advantage"]
                ) > 1e-3
                or abs(
                    known["fit_rms_m"]
                    - evidence["known_fit_rms_m"]
                ) > 1e-6
            ):
                raise RuntimeError(
                    "Early observation mismatch: "
                    f"scene={sid}, mode={mode}"
                )

            yield {
                "scene_id": sid,
                "mode": mode,
                "pairs": pairs,
                "probes": probes,
                "known_fit": known,
                "edit_fit": edit,
                "deadzone_fit":
                    edit_fits["deadzone"],
            }


def probe_rms(probes, model, api):
    errors = [
        float(
            api["prediction_rms"](
                probe, model
            )
        )
        for probe in probes
    ]

    return float(np.sqrt(
        np.mean(
            np.square(errors)
        )
    ))


def profile_one_gap(
    probes,
    pairs,
    initial_params,
    gap,
    max_nfev,
):
    """
    Profile over k,c while holding gap fixed.

    Uses only measured E1/E2 probes and their B
    targets. Does not use any true scene parameter.

    The nonzero lower bound on k is an explicit
    modeling choice, not an inferred fact.
    """
    initial = np.asarray(
        initial_params,
        dtype=float,
    )

    if initial.size < 3:
        raise ValueError(
            "Expected Deadzone params [k,c,gap]"
        )

    low = np.asarray([
        K_BOUNDS[0],
        C_BOUNDS[0],
    ])

    high = np.asarray([
        K_BOUNDS[1],
        C_BOUNDS[1],
    ])

    starts = [
        np.clip(
            initial[:2],
            low + 1e-6,
            high - 1e-6,
        ),
        np.asarray([10.0, 1.0]),
    ]

    def residual(x):
        params = np.asarray([
            float(x[0]),
            float(x[1]),
            float(gap),
        ])

        parts = []

        for probe, (_, active) in zip(
            probes, pairs
        ):
            predicted = np.asarray(
                probe.predict(
                    "deadzone", params
                ),
                dtype=float,
            ).reshape(-1)

            observed = np.asarray(
                active[:, 1],
                dtype=float,
            ).reshape(-1)

            if predicted.shape != observed.shape:
                raise RuntimeError(
                    "Unexpected prediction shape"
                )

            parts.append(
                predicted - observed
            )

        return np.concatenate(parts)

    best = None

    for start in starts:
        fit = least_squares(
            residual,
            x0=np.clip(
                start,
                low + 1e-6,
                high - 1e-6,
            ),
            bounds=(low, high),
            max_nfev=max_nfev,
        )

        error = residual(fit.x)

        rms = float(np.sqrt(
            np.mean(error ** 2)
        ))

        if (
            best is None
            or rms < best["rms_m"]
        ):
            best = {
                "gap_m": float(gap),
                "k": float(fit.x[0]),
                "c": float(fit.x[1]),
                "rms_m": rms,
                "optimizer_success":
                    bool(fit.success),
                "nfev": int(fit.nfev),
            }

    return best


def build_candidate_set(
    observation,
    cutoff_m,
    gaps,
    api,
    max_nfev,
):
    """
    This is the complete nonprivileged
    candidate construction step.

    No true law, true gap, oracle envelope,
    E3 outcome or E4 error is accessed here.
    """
    probes = observation["probes"]
    pairs = observation["pairs"]

    known = observation["known_fit"]

    reference_rms = probe_rms(
        probes, known, api
    )

    # Ensure calibration RMS and candidate RMS
    # are measured on a compatible scale.
    scale_difference = abs(
        reference_rms
        - float(known["fit_rms_m"])
    )

    if scale_difference > 1e-4:
        raise RuntimeError(
            "RMS definition mismatch. "
            "Inspect fit_library and prediction_rms "
            "before applying the frozen cutoff. "
            f"Difference={scale_difference:.8f} m"
        )

    initial = observation[
        "deadzone_fit"
    ]["params"]

    profiles = [
        profile_one_gap(
            probes,
            pairs,
            initial,
            gap,
            max_nfev,
        )
        for gap in gaps
    ]

    retained = [
        p for p in profiles
        if (
            np.isfinite(p["rms_m"])
            and p["rms_m"] <= cutoff_m
        )
    ]

    return {
        "scene_id":
            observation["scene_id"],

        "known_kind":
            known["kind"],

        "known_rms_m":
            float(known["fit_rms_m"]),

        "frozen_cutoff_m":
            float(cutoff_m),

        "best_profile_rms_m":
            min(
                p["rms_m"]
                for p in profiles
            ),

        "retained_count":
            len(retained),

        "retained_gaps_m": [
            p["gap_m"] for p in retained
        ],

        "profiles": profiles,
    }


def add_private_evaluation(
    result,
    mode,
    scene,
    original_scene,
    archived,
    oracle_early_M,
    gaps,
):
    """
    Called ONLY after the candidate set has
    been constructed.

    Private physical parameters and labels
    are not available to build_candidate_set.
    """
    result = dict(result)

    result["truth_PRIVATE"] = mode

    result["oracle_early_M_m_PRIVATE"] = (
        float(oracle_early_M)
    )

    retained = result["retained_gaps_m"]

    # Check which retained candidates lie in
    # an oracle exact-equivalence region.
    result[
        "retained_oracle_equivalent_count_PRIVATE"
    ] = sum(
        gap > oracle_early_M + GAP_MARGIN
        for gap in retained
    )

    if mode == "deadzone":
        true_gap = float(scene["gap"])

        nearest_grid = float(
            min(
                gaps,
                key=lambda gap:
                    abs(gap - true_gap),
            )
        )

        result["true_gap_PRIVATE"] = true_gap

        result[
            "true_gap_nearest_grid_m_PRIVATE"
        ] = nearest_grid

        result[
            "true_gap_grid_cell_retained_PRIVATE"
        ] = bool(
            any(
                np.isclose(
                    p,
                    nearest_grid,
                    rtol=0.0,
                    atol=1e-10,
                )
                for p in retained
            )
        )

        result[
            "true_gap_oracle_inactive_PRIVATE"
        ] = bool(
            true_gap
            > oracle_early_M + GAP_MARGIN
        )

        archived_inactive = not bool(
            archived[
                "activation_by_scene_PRIVATE"
            ][str(result["scene_id"])]
            ["early_activated_PRIVATE"]
        )

        if (
            result[
                "true_gap_oracle_inactive_PRIVATE"
            ] != archived_inactive
        ):
            raise RuntimeError(
                "Oracle activation classification "
                "does not match P3-A4D"
            )

    return result


def summarize(rows, band):
    print("\n" + "=" * 70)
    print("BAND:", band)
    print("EPISODES:", len(rows))
    print("=" * 70)

    summary = {}

    for mode in FOCUS:
        selected = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        counts = [
            r["retained_count"]
            for r in selected
        ]

        oracle_equivalent = [
            r[
                "retained_oracle_equivalent_count_PRIVATE"
            ]
            for r in selected
        ]

        item = {
            "episodes": len(selected),

            "any_candidate_retained": sum(
                x > 0 for x in counts
            ),

            "multiple_candidates_retained": sum(
                x > 1 for x in counts
            ),

            "median_retained_count":
                float(np.median(counts)),

            "has_oracle_equivalent_candidate": sum(
                x > 0 for x in oracle_equivalent
            ),

            "median_best_profile_RMS_mm":
                float(np.median([
                    r["best_profile_rms_m"]
                    * 1000.0
                    for r in selected
                ])),
        }

        if mode == "deadzone":
            item[
                "true_gap_nearest_grid_retained"
            ] = sum(
                r[
                    "true_gap_grid_cell_retained_PRIVATE"
                ]
                for r in selected
            )

            item[
                "truth_exactly_inactive_early"
            ] = sum(
                r[
                    "true_gap_oracle_inactive_PRIVATE"
                ]
                for r in selected
            )

        summary[mode] = item

        print(
            f"{mode:12s}",
            "any_retained=",
            f"{item['any_candidate_retained']}"
            f"/{len(selected)}",
            "multi_retained=",
            item["multiple_candidates_retained"],
            "median_count=",
            item["median_retained_count"],
            "oracle_equiv_retained=",
            item[
                "has_oracle_equivalent_candidate"
            ],
            "best_RMS_mm=",
            round(
                item[
                    "median_best_profile_RMS_mm"
                ],
                4,
            ),
        )

        if mode == "deadzone":
            print(
                "  True-gap nearest grid retained:",
                item[
                    "true_gap_nearest_grid_retained"
                ],
                "/", len(selected),
            )

            print(
                "  Oracle early inactive:",
                item[
                    "truth_exactly_inactive_early"
                ],
                "/", len(selected),
            )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    audit = json.loads(
        SOURCE.read_text(encoding="utf-8")
    )

    reference = json.loads(
        CALIBRATION.read_text(
            encoding="utf-8"
        )
    )

    cutoff_m = float(
        reference["calibration_known_only"]
        ["early"]["known_fit_rms_m"]
    )

    if not (
        np.isfinite(cutoff_m)
        and cutoff_m > 0.0
    ):
        raise RuntimeError(
            "Invalid frozen calibration cutoff"
        )

    sim = run_path(str(
        HERE /
        "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE /
        "22_p3a1b_structural_revision.py"
    ))

    helper35 = run_path(str(
        HERE /
        "35_p3a5b_preaction_model_disagreement.py"
    ))

    helper39 = run_path(str(
        HERE /
        "39_p3a6a_exact_identifiability.py"
    ))

    commands = helper35[
        "command_bank"
    ](
        sim["make_profiles"]()
    )

    gaps = grid_values(args.smoke)

    n = 2 if args.smoke else 24

    # Oracle envelopes are computed for the
    # evaluation stage ONLY.
    # They are never passed to fitting functions.
    original_scenes = audit[
        "original"
    ]["scene_parameters_PRIVATE"][:n]

    oracle_envelopes = {}

    for sid, scene in enumerate(
        original_scenes
    ):
        model = sim["make_model"](scene)

        envelope = helper39[
            "compute_envelope"
        ](
            sim,
            model,
            scene,
            helper39["make_commands"](sim),
        )

        oracle_envelopes[sid] = max(
            envelope["e1"],
            envelope["e2"],
        )

    all_private = {}
    all_summary = {}

    print(
        "P3-A6B OBSERVATION-COMPATIBLE "
        "HYPOTHESIS SET AUDIT"
    )

    print(
        "Frozen known-only cutoff:",
        round(cutoff_m * 1000, 4),
        "mm",
    )

    print(
        "Grid:",
        len(gaps), "gaps,",
        round(float(gaps[0]), 4), "to",
        round(float(gaps[-1]), 4), "m",
    )

    for band in BANDS:
        group = audit[band]

        scenes = group[
            "scene_parameters_PRIVATE"
        ][:n]

        old_rows = {
            (
                int(r["scene_id"]),
                r["truth_PRIVATE_EVAL_ONLY"],
            ): r
            for r in group["trajectories"]
        }

        # Replay exact early noisy observations.
        reconstructed = reconstruct_early(
            scenes,
            old_rows,
            sim,
            api,
            commands,
        )

        records = []

        for obs in reconstructed:
            sid = obs["scene_id"]
            mode = obs["mode"]

            # Nonprivileged candidate construction.
            estimated = build_candidate_set(
                obs,
                cutoff_m,
                gaps,
                api,
                25 if args.smoke else 40,
            )

            # Private truth enters only here.
            evaluated = add_private_evaluation(
                estimated,
                mode,
                scenes[sid],
                original_scenes[sid],
                group,
                oracle_envelopes[sid],
                gaps,
            )

            records.append(evaluated)

        all_private[band] = records
        all_summary[band] = summarize(
            records, band
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke
        else "development"
    )

    report_path = OUT / (
        f"p3a6b_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a6b_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "study":
            "P3-A6B observation-compatible sets",

        "status":
            "Exploratory targeted candidate-set audit",

        "n_scenes_per_band": n,

        "candidate_gap_grid_m":
            gaps.tolist(),

        "stiffness_bounds":
            list(K_BOUNDS),

        "damping_bounds":
            list(C_BOUNDS),

        "frozen_cutoff_m":
            cutoff_m,

        "calibration_source":
            str(CALIBRATION),

        "summary": all_summary,

        "source_sha256": {
            name: digest(HERE / name)
            for name in (
                "16_p2d1b_probe_diversity.py",
                "22_p3a1b_structural_revision.py",
                "35_p3a5b_preaction_model_disagreement.py",
                "39_p3a6a_exact_identifiability.py",
            )
        },

        "limitations": [
            "Existing scenes already examined in earlier audits.",
            "Only Independent and Deadzone are fitted.",
            "Candidate sets use discrete gap grids.",
            "Stiffness is constrained strictly above zero.",
            "The RMS cutoff is known-only calibrated.",
            "There is no formal parameter coverage guarantee.",
            "Oracle envelopes and true gaps are evaluation-only.",
            "No E3 action is selected by this script.",
            "No E4 observations enter fitting.",
        ],
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
            all_private,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A6B CANDIDATE-SET AUDIT COMPLETE"
    )

    print(
        "No historical files changed, "
        "no policy training."
    )

    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
