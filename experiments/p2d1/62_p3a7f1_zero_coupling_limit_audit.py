
"""
P3-A7F1 — Zero-Coupling Limit Audit

1. Capture original force and fitter definitions.
2. Record fitted parameters in Independent worlds.
3. Compare noiseless model predictions directly.
4. Evaluate formal zero-parameter extensions.

PRIVATE diagnostic only.
Zero parameters may lie OUTSIDE admissible bounds.
No E4, no threshold tuning, no online oracle.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a7f1"

HISTORY = (
    HERE / "results" / "p3a7e1"
    / "p3a7e1_fresh_report.json"
)

SEED = 20270821
NOISE = 2027082101

BANDS = ("original", "above_original_range")
ACTIONS = ("mid", "reverse", "high")
LAWS = ("elastic", "deadzone", "cubic")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_of(obj):
    if not callable(obj):
        return "UNAVAILABLE"
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return "SOURCE_NOT_RESOLVED"


def as_prediction(value):
    array = np.asarray(value, dtype=float).ravel()
    if not np.isfinite(array).all():
        raise RuntimeError("Nonfinite prediction")
    return array


def rms_mm(a, b):
    if a.shape != b.shape:
        raise RuntimeError("Prediction shape mismatch")
    return float(
        np.sqrt(np.mean((a - b) ** 2)) * 1000
    )


def get_sources(sim, api):
    wrapper = sim["simulate"].__globals__.get(
        "internal_force"
    )

    original = None

    if callable(wrapper):
        try:
            original = inspect.getclosurevars(
                wrapper
            ).nonlocals.get("original")
        except (TypeError, ValueError):
            pass

    fitter = api["fit_library"].__globals__.get(
        "fit_one"
    )

    return {
        "force_wrapper": source_of(wrapper),
        "original_force": source_of(original),
        "fit_one": source_of(fitter),
        "fit_library": source_of(api["fit_library"]),
        "prediction_rms": source_of(
            api["prediction_rms"]
        ),
    }


def median(values):
    return float(np.median(values))


def analyze(records):
    summary = {}

    for band in BANDS:
        summary[band] = {}

        print("\nBAND:", band)

        for action in ACTIONS:
            summary[band][action] = {}

            print("  ACTION:", action)

            for law in LAWS:
                group = [
                    r for r in records
                    if r["band"] == band
                    and r["action"] == action
                    and r["law"] == law
                ]

                fitted = [
                    r["fitted_prediction_gap_mm"]
                    for r in group
                ]

                zero = [
                    r["zero_limit_prediction_gap_mm"]
                    for r in group
                    if r["zero_limit_prediction_gap_mm"]
                    is not None
                ]

                result = {
                    "n": len(group),
                    "fitted_gap_median_mm": median(fitted),
                    "fitted_gap_max_mm": max(fitted),
                    "zero_test_success": len(zero),
                    "zero_gap_max_mm": (
                        max(zero) if zero else None
                    ),
                }

                summary[band][action][law] = result

                print(
                    f"    {law:10s}",
                    "fitted gap median=",
                    round(
                        result["fitted_gap_median_mm"],
                        6
                    ),
                    "mm | zero-limit max=",
                    result["zero_gap_max_mm"]
                )

    return summary


def param_summary(records):
    summary = {}

    for band in BANDS:
        summary[band] = {}

        for law in LAWS:
            group = [
                r["params"]
                for r in records
                if r["band"] == band
                and r["law"] == law
            ]

            if not group:
                continue

            arr = np.asarray(group, dtype=float)

            summary[band][law] = {
                "n": len(group),
                "parameter_min": (
                    arr.min(axis=0).tolist()
                ),
                "parameter_median": (
                    np.median(arr, axis=0).tolist()
                ),
                "parameter_max": (
                    arr.max(axis=0).tolist()
                ),
            }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true"
    )
    args = parser.parse_args()

    historic = json.loads(
        HISTORY.read_text(encoding="utf-8")
    )

    if historic["smoke"]:
        raise RuntimeError(
            "Formal provenance is required"
        )

    for filename, expected in (
        historic["source_sha256"].items()
    ):
        if digest(HERE / filename) != expected:
            raise RuntimeError(
                "Historical source changed: "
                + filename
            )

    core = run_path(str(
        HERE / "60_p3a7e1_action_family_pilot.py"
    ))

    src, sim, api, commands = (
        core["initialize"]()
    )

    sources = get_sources(sim, api)

    n_scenes = 2 if args.smoke else 24
    offset = 100000 if args.smoke else 0

    scenes = src["make_scenes"](
        sim, SEED + offset, n_scenes
    )

    records = []
    params_records = []

    print("\nP3-A7F1 ZERO-COUPLING AUDIT")
    print("Independent scenes:", n_scenes)
    print("No E4. No threshold tuning.")

    for band in BANDS:
        for sid, scene_set in enumerate(scenes):
            state = core["collect_first"](
                src, sim, api, commands,
                scene_set[band], sid,
                "independent", band,
                NOISE + offset
            )

            for law in LAWS:
                params = state["fits"][law]["params"]

                if params is None:
                    raise RuntimeError(
                        "Missing fitted parameters"
                    )

                params_records.append({
                    "band": band,
                    "scene_id": sid,
                    "law": law,
                    "params": [
                        float(x) for x in params
                    ],
                })

            for action, index in (
                ("mid", 4),
                ("reverse", 5),
                ("high", 6),
            ):
                observation, _ = core["probe"](
                    src, sim, api,
                    state["model"],
                    scene_set[band],
                    "independent",
                    commands[action],
                    state["seedbase"] + index,
                )

                independent_prediction = (
                    as_prediction(
                        observation.predict(
                            "independent", None
                        )
                    )
                )

                for law in LAWS:
                    fit = state["fits"][law]

                    fitted_prediction = (
                        as_prediction(
                            observation.predict(
                                law, fit["params"]
                            )
                        )
                    )

                    fitted_gap = rms_mm(
                        fitted_prediction,
                        independent_prediction
                    )

                    # Formal mathematical extension:
                    # zero parameters may be outside
                    # the allowed optimizer domain.
                    zero_params = [
                        0.0 for _ in fit["params"]
                    ]

                    zero_gap = None
                    zero_error = None

                    try:
                        zero_prediction = (
                            as_prediction(
                                observation.predict(
                                    law, zero_params
                                )
                            )
                        )

                        zero_gap = rms_mm(
                            zero_prediction,
                            independent_prediction
                        )

                    except Exception as exc:
                        zero_error = (
                            type(exc).__name__
                            + ": " + str(exc)
                        )

                    records.append({
                        "band": band,
                        "scene_id": sid,
                        "action": action,
                        "law": law,
                        "fitted_prediction_gap_mm":
                            fitted_gap,
                        "zero_limit_prediction_gap_mm":
                            zero_gap,
                        "zero_test_error":
                            zero_error,
                    })

    expected = (
        n_scenes * len(BANDS)
        * len(ACTIONS) * len(LAWS)
    )

    if len(records) != expected:
        raise RuntimeError(
            "Incomplete prediction matrix"
        )

    summary = analyze(records)
    parameter_results = param_summary(
        params_records
    )

    OUT.mkdir(parents=True, exist_ok=True)

    suffix = (
        "smoke" if args.smoke else "formal"
    )

    report = {
        "protocol":
            "P3-A7F1 zero-coupling limit audit",
        "smoke": args.smoke,
        "n_base_scenes": n_scenes,
        "n_prediction_records": len(records),
        "summary": summary,
        "parameter_summary_PRIVATE":
            parameter_results,
        "records_PRIVATE": records,
        "E4_accessed": False,
        "thresholds_tuned": False,
        "zero_parameters_guaranteed_admissible":
            False,
        "limitations": [
            "Independent worlds only",
            "Finite-action prediction comparison",
            "Zero-limit may be outside fit bounds",
            "No global equivalence proof",
            "Truth used only for PRIVATE diagnosis",
            "No online policy tested",
            "Gap bands share base scenes",
        ],
    }

    report_path = OUT / (
        f"p3a7f1_{suffix}_report_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    source_path = OUT / (
        f"p3a7f1_{suffix}_sources_PRIVATE.md"
    )

    source_path.write_text(
        "\n\n".join(
            "## " + key + "\n\n"
            "```python\n" + value + "\n```"
            for key, value in sources.items()
        ),
        encoding="utf-8"
    )

    print("\nP3-A7F1 AUDIT COMPLETE")
    print("Saved:", report_path)
    print("Saved:", source_path)


if __name__ == "__main__":
    main()
