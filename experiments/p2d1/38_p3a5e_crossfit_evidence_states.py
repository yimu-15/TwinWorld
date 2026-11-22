
"""
P3-A5E: Cross-fitted Three-State Evidence Audit

Input: P3-A5D existing development JSON only.

Calibrate two thresholds using known mechanisms only:
  R_min = min(known weak predictive RMS,
              edited weak predictive RMS)
  G = known RMS - edited RMS

Output:
  keep_support
  edit_support
  unresolved

Four-fold scene-disjoint exploratory evaluation.

The three gap bands share the same base scenes.
Never treat them as independent samples.

No simulation, no refitting, no policy training.
No E4 information in evidence-state decisions.
"""

from pathlib import Path
from collections import Counter
import json
import numpy as np


HERE = Path(__file__).resolve().parent

SOURCE = (
    HERE / "results" / "p3a5d"
    / "p3a5d_development_audit_PRIVATE.json"
)

SEED = 20261129
N_FOLDS = 4
Q = 0.95

BANDS = (
    "original",
    "high_in_range",
    "above_original_range",
)

KNOWN = (
    "independent",
    "rigid",
    "elastic",
)

MODES = KNOWN + (
    "deadzone",
    "nonlinear",
    "saturating",
)

STATES = (
    "keep_support",
    "edit_support",
    "unresolved",
)


def read_source():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    obj = json.loads(
        SOURCE.read_text(encoding="utf-8")
    )

    data = {}

    for band in BANDS:
        rows = obj["bands"][band]["audit_PRIVATE"]

        keys = {
            (int(r["scene_id"]), r["truth_PRIVATE"])
            for r in rows
        }

        if len(keys) != len(rows):
            raise ValueError(
                f"Duplicate episodes in {band}"
            )

        data[band] = rows

    return data


def residual(r):
    return float(min(
        r["weak_known_RMS_mm"],
        r["weak_edit_RMS_mm"],
    ))


def edit_gain(r):
    return float(
        r["weak_edit_gain_mm"]
    )


def make_scene_folds(rows):
    scenes = sorted({
        int(r["scene_id"]) for r in rows
    })

    rng = np.random.default_rng(SEED)
    order = rng.permutation(scenes)

    folds = {
        int(sid): int(i % N_FOLDS)
        for i, sid in enumerate(order)
    }

    return folds


def calibrate(data, folds):
    """
    Calibration uses ORIGINAL-band known-only
    episodes from different scenes.

    No hidden law enters the runtime scoring rule.
    No new or out-of-library cases calibrate.
    """
    thresholds = {}

    for fold in range(N_FOLDS):
        training = [
            r for r in data["original"]
            if (
                r["truth_PRIVATE"] in KNOWN
                and folds[int(r["scene_id"])]
                    != fold
            )
        ]

        if len(training) < 10:
            raise RuntimeError(
                "Insufficient calibration episodes"
            )

        thresholds[fold] = {
            "tau_R_mm": float(np.quantile(
                [residual(r) for r in training],
                Q,
            )),
            "tau_G_mm": float(np.quantile(
                [edit_gain(r) for r in training],
                Q,
            )),
            "calibration_N": len(training),
        }

    return thresholds


def evidence_state(r, threshold):
    """
    Only E1/E2 fitted models and observed Weak E3
    predictive errors affect this state.

    NEVER use:
      truth_PRIVATE
      actual_strong_E4_benefit_mm
      strong_final_kind_PRIVATE
    """
    rmin = residual(r)
    gain = edit_gain(r)

    if rmin > threshold["tau_R_mm"]:
        return "unresolved"

    if gain > threshold["tau_G_mm"]:
        return "edit_support"

    return "keep_support"


def route_rows(rows, folds, thresholds):
    output = []

    for r in rows:
        scene = int(r["scene_id"])
        fold = folds[scene]

        state = evidence_state(
            r, thresholds[fold]
        )

        output.append({
            "scene_id": scene,
            "fold": fold,
            "state": state,

            # Private labels and future outcomes
            # are attached AFTER state computation.
            "truth_PRIVATE": r["truth_PRIVATE"],
            "strong_benefit_PRIVATE_mm": float(
                r["actual_strong_E4_benefit_mm"]
            ),

            "weak_kind_PRIVATE":
                r["weak_final_kind_PRIVATE"],

            "strong_kind_PRIVATE":
                r["strong_final_kind_PRIVATE"],
        })

    return output


def validate_pairing(data):
    """
    Three bands share base scene IDs.

    For every non-Deadzone mechanism, gap does not
    affect the physical force law; its weak
    predictive scores should remain identical.
    """
    base = {
        (r["scene_id"], r["truth_PRIVATE"]): r
        for r in data["original"]
    }

    for band in BANDS[1:]:
        for r in data[band]:
            key = (
                r["scene_id"],
                r["truth_PRIVATE"],
            )

            other = base[key]

            if r["truth_PRIVATE"] == "deadzone":
                continue

            for metric in (
                "weak_known_RMS_mm",
                "weak_edit_RMS_mm",
                "weak_edit_gain_mm",
            ):
                if not np.isclose(
                    r[metric],
                    other[metric],
                    atol=1e-7,
                    rtol=0,
                ):
                    raise AssertionError(
                        f"Pairing mismatch: "
                        f"{band}, {key}, {metric}"
                    )


def print_summary(band, records):
    print("\n" + "=" * 74)
    print("BAND:", band)
    print("EPISODES:", len(records))
    print("=" * 74)

    print(
        f"{'True mode':14s}"
        f"{'Keep':>9s}"
        f"{'Edit':>9s}"
        f"{'Unresolved':>13s}"
    )

    for mode in MODES:
        subset = [
            r for r in records
            if r["truth_PRIVATE"] == mode
        ]

        counts = Counter(
            r["state"] for r in subset
        )

        print(
            f"{mode:14s}"
            f"{counts['keep_support']:9d}"
            f"{counts['edit_support']:9d}"
            f"{counts['unresolved']:13d}"
        )

    known = [
        r for r in records
        if r["truth_PRIVATE"] in KNOWN
    ]

    new = [
        r for r in records
        if r["truth_PRIVATE"] in (
            "deadzone", "nonlinear"
        )
    ]

    outside = [
        r for r in records
        if r["truth_PRIVATE"] == "saturating"
    ]

    known_alerts = sum(
        r["state"] != "keep_support"
        for r in known
    )

    new_keep = sum(
        r["state"] == "keep_support"
        for r in new
    )

    unknown_keep = sum(
        r["state"] == "keep_support"
        for r in outside
    )

    print(
        "\nKnown non-keep:",
        f"{known_alerts}/{len(known)}"
    )

    print(
        "New-law keep-support:",
        f"{new_keep}/{len(new)}"
    )

    print(
        "Saturating keep-support:",
        f"{unknown_keep}/{len(outside)}"
    )

    print("\nRetrospective Strong benefit by state:")

    for state in STATES:
        subset = [
            r for r in records
            if r["state"] == state
        ]

        if not subset:
            continue

        gain = np.asarray([
            r["strong_benefit_PRIVATE_mm"]
            for r in subset
        ])

        print(
            state,
            "N=", len(subset),
            "Strong>1mm=",
            int(np.sum(gain > 1.0)),
            "median_gain_mm=",
            round(float(np.median(gain)), 4),
        )

    correct_name = {
        "deadzone": "deadzone",
        "nonlinear": "cubic",
    }

    print("\nNovel structure corrections:")

    for mode in correct_name:
        subset = [
            r for r in new
            if r["truth_PRIVATE"] == mode
        ]

        corrected = [
            r for r in subset
            if (
                r["weak_kind_PRIVATE"]
                    != correct_name[mode]
                and r["strong_kind_PRIVATE"]
                    == correct_name[mode]
            )
        ]

        print(
            mode,
            "Strong corrects Weak:",
            len(corrected),
            "/",
            len(subset),
            "evidence_states=",
            dict(Counter(
                r["state"] for r in corrected
            )),
        )


def main():
    data = read_source()
    validate_pairing(data)

    folds = make_scene_folds(
        data["original"]
    )

    thresholds = calibrate(
        data, folds
    )

    print(
        "P3-A5E CROSS-FITTED "
        "EVIDENCE-STATE AUDIT"
    )

    print(
        "Calibration: known-only, 4 folds, "
        "separated by base scene"
    )

    for fold, threshold in thresholds.items():
        print(
            f"fold={fold}",
            "tau_R_mm=",
            round(
                threshold["tau_R_mm"], 4
            ),
            "tau_G_mm=",
            round(
                threshold["tau_G_mm"], 4
            ),
            "calibration_N=",
            threshold["calibration_N"],
        )

    all_results = {}

    for band in BANDS:
        records = route_rows(
            data[band],
            folds,
            thresholds,
        )

        all_results[band] = records
        print_summary(band, records)

    print(
        "\nP3-A5E CROSS-FITTED "
        "EVIDENCE AUDIT PASS"
    )

    print(
        "No MuJoCo, no fit, no new policy "
        "and no historical files modified."
    )

    print(
        "Three states are evidence diagnostics, "
        "NOT validated action decisions."
    )


if __name__ == "__main__":
    main()
