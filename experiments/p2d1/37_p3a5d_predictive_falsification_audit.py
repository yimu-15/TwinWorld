
"""
P3-A5D: Sequential Predictive Falsification Audit

E1/E2:
    Fit known and editable structures.

Weak E3:
    Evaluate frozen early models on NEW observations
    before adding E3 to parameter fitting.

Strong E3 and E4:
    Accessed only for retrospective evaluation.

Uses the same paired P3-A4D scene/noise streams.

No policy training, no threshold fitting.
This is an exploratory diagnostic, not a new policy.
"""

from pathlib import Path
from runpy import run_path
from collections import defaultdict
import argparse
import json
import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results" / "p3a4d"
OUT = HERE / "results" / "p3a5d"

SEED = 20261128
SIGMA = 0.005
NFEV = 40

KNOWN = ("independent", "rigid", "elastic")
EDIT = ("deadzone", "cubic")

MODES = (
    "independent", "rigid", "elastic",
    "deadzone", "nonlinear", "saturating"
)

BANDS = (
    "original",
    "high_in_range",
    "above_original_range"
)

E4 = ("pulse_2p75", "oscillation_2p25")


def median(values):
    return (
        round(float(np.median(values)), 4)
        if values else None
    )


def rank_corr(x, y):
    if len(x) < 4:
        return None

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None

    value = float(spearmanr(x, y).statistic)

    return (
        round(value, 4)
        if np.isfinite(value) else None
    )


def e4_benefit_mm(old):
    weak = old["policies"]["always_weak"]["E4_m"]
    strong = old["policies"]["always_strong"]["E4_m"]

    return 500.0 * sum(
        float(weak[t]) - float(strong[t])
        for t in E4
    )


def replay_one_band(
    scenes, previous, sim, api, helper35
):
    commands = helper35["command_bank"](
        sim["make_profiles"]()
    )

    order = helper35["ALL_COMMANDS"]

    expected = {
        (
            int(r["scene_id"]),
            r["truth_PRIVATE_EVAL_ONLY"]
        ): r
        for r in previous["trajectories"]
    }

    noise_rng = np.random.default_rng(
        SEED + 222 + 100000
    )

    results = []

    for sid, scene in enumerate(scenes):
        model = sim["make_model"](scene)

        for mode in MODES:
            measured = {}

            # Reproduce the original random
            # observation order exactly.
            for action in order:
                if action in (
                    "e1", "e2", "pulse_1p5"
                ):
                    passive, active, _, _ = (
                        sim["simulate"](
                            model,
                            scene,
                            mode,
                            commands[action]
                        )
                    )

                    p, a = api["noisy_observation"](
                        passive, active, noise_rng
                    )

                    measured[action] = api[
                        "RevisionProbe"
                    ](p, a)

                else:
                    # Skip two observation-noise
                    # arrays for the unused action.
                    noise_rng.normal(
                        0, SIGMA, (51, 2)
                    )
                    noise_rng.normal(
                        0, SIGMA, (51, 2)
                    )

            early = [
                measured["e1"],
                measured["e2"]
            ]

            known, _ = api["fit_library"](
                early, KNOWN, NFEV
            )

            edit, edit_fits = api[
                "fit_library"
            ](
                early, EDIT, NFEV
            )

            old = expected[(sid, mode)]
            old_e = old["early_evidence"]

            # Essential replay integrity checks.
            margin = (
                known["bic"] - edit["bic"]
            )

            if (
                known["kind"]
                != old_e["known_kind"]
                or abs(
                    margin - old_e["bic_advantage"]
                ) > 1e-3
                or abs(
                    known["fit_rms_m"]
                    - old_e["known_fit_rms_m"]
                ) > 1e-6
            ):
                raise RuntimeError(
                    "Early replay mismatch: "
                    f"scene={sid}, mode={mode}"
                )

            # Actual Weak E3 is the predictive
            # holdout. No refitting on E3 here.
            weak = measured["pulse_1p5"]

            known_rms = api["prediction_rms"](
                weak, known
            )

            edit_rms = api["prediction_rms"](
                weak, edit
            )

            # An additional Deadzone-vs-known
            # audit, even if Cubic is preferred.
            deadzone_rms = api[
                "prediction_rms"
            ](
                weak,
                edit_fits["deadzone"]
            )

            results.append({
                "scene_id": sid,
                "truth_PRIVATE": mode,

                "early_known_kind":
                    known["kind"],

                "early_edit_kind":
                    edit["kind"],

                "early_BIC_advantage":
                    float(margin),

                "weak_known_RMS_mm":
                    1000.0 * known_rms,

                "weak_edit_RMS_mm":
                    1000.0 * edit_rms,

                "weak_deadzone_RMS_mm":
                    1000.0 * deadzone_rms,

                "weak_edit_gain_mm":
                    1000.0 * (
                        known_rms - edit_rms
                    ),

                "weak_deadzone_gain_mm":
                    1000.0 * (
                        known_rms - deadzone_rms
                    ),

                "actual_strong_E4_benefit_mm":
                    e4_benefit_mm(old),

                "weak_final_decision_PRIVATE":
                    old["policies"][
                        "always_weak"
                    ]["decision"],

                "weak_final_kind_PRIVATE":
                    old["policies"][
                        "always_weak"
                    ]["kind"],

                "strong_final_decision_PRIVATE":
                    old["policies"][
                        "always_strong"
                    ]["decision"],

                "strong_final_kind_PRIVATE":
                    old["policies"][
                        "always_strong"
                    ]["kind"],
            })

    return results


def summarize(rows):
    output = {}

    for mode in MODES:
        subset = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        gains = [
            r["weak_edit_gain_mm"]
            for r in subset
        ]

        benefits = [
            r["actual_strong_E4_benefit_mm"]
            for r in subset
        ]

        output[mode] = {
            "n": len(subset),

            "known_RMS_median_mm": median([
                r["weak_known_RMS_mm"]
                for r in subset
            ]),

            "edit_RMS_median_mm": median([
                r["weak_edit_RMS_mm"]
                for r in subset
            ]),

            "edit_gain_median_mm":
                median(gains),

            "gain_positive_count": sum(
                x > 0 for x in gains
            ),

            "gain_vs_strong_benefit_spearman":
                rank_corr(gains, benefits),

            "strong_benefit_median_mm":
                median(benefits),

            "early_known_kinds": dict(
                (name, sum(
                    r["early_known_kind"] == name
                    for r in subset
                ))
                for name in KNOWN
            ),

            "early_edit_kinds": {
                name: sum(
                    r["early_edit_kind"] == name
                    for r in subset
                )
                for name in EDIT
            }
        }

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke", action="store_true"
    )

    args = parser.parse_args()

    path = (
        SOURCE /
        "p3a4d_stress_audit_PRIVATE.json"
    )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    helper27 = run_path(str(
        HERE / "27_p3a3b_evidence_vs_structure.py"
    ))

    helper35 = run_path(str(
        HERE / "35_p3a5b_preaction_model_disagreement.py"
    ))

    helper27["install_saturating"](sim)

    n = 2 if args.smoke else 24

    results = {
        "protocol":
            "P3-A5D weak-probe predictive falsification",

        "retrospective": True,

        "known_model_and_edit_fitted_on":
            "E1/E2 only",

        "predictive_validation":
            "Weak E3 before refitting",

        "limitations": [
            "Weak E3 observed A conditions B prediction.",
            "Weak E3 is not a pre-action score.",
            "E4 benefit is evaluation-only.",
            "Known and edited models use point estimates.",
            "Physical scenes repeat across gap bands.",
            "No new adaptive policy is validated.",
            "No threshold tuning is performed."
        ],

        "bands": {}
    }

    for band in BANDS:
        records = replay_one_band(
            data[band][
                "scene_parameters_PRIVATE"
            ][:n],
            data[band],
            sim,
            api,
            helper35
        )

        stats = summarize(records)

        results["bands"][band] = {
            "summary": stats,
            "audit_PRIVATE": records
        }

        print("\n" + "=" * 70)
        print("P3-A5D BAND:", band)
        print("SCENES:", n)
        print("=" * 70)

        for mode in MODES:
            s = stats[mode]

            print(
                f"{mode:12s}",
                "known_RMS=",
                s["known_RMS_median_mm"],
                "edit_RMS=",
                s["edit_RMS_median_mm"],
                "positive_gain=",
                f"{s['gain_positive_count']}/{s['n']}",
                "gain_median=",
                s["edit_gain_median_mm"],
                "rho_gain_benefit=",
                s[
                    "gain_vs_strong_benefit_spearman"
                ]
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "development"
    )

    output_path = (
        OUT /
        f"p3a5d_{suffix}_audit_PRIVATE.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "\nP3-A5D PREDICTIVE "
        "FALSIFICATION AUDIT PASS"
    )

    print(
        "No threshold tuning. "
        "No adaptive policy trained."
    )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
