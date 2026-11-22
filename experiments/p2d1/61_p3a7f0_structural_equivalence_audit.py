
"""
P3-A7F0: Structural Equivalence and
Candidate Parameter Audit

Purpose:
1. Inspect actual force-law and fitting source.
2. Examine fitted parameter representations.
3. Test empirical predictive equivalence on
   Independent worlds under multiple actions.

No E4.
No decision threshold changes.
No model selection using test labels.
No proof of global structural equivalence.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a7f0"

HISTORY = (
    HERE / "results" / "p3a7e1"
    / "p3a7e1_fresh_report.json"
)

SEED = 20270810
NOISE_SEED = 2027081001

LAWS = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "cubic",
)

ACTIONS = ("mid", "reverse", "high")


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def source_of(function):
    if not callable(function):
        return "UNAVAILABLE"

    try:
        return inspect.getsource(function)
    except (OSError, TypeError):
        return "SOURCE_NOT_RESOLVED"


def short_repr(value, limit=250):
    value = repr(value)
    if len(value) > limit:
        return value[:limit] + "...[truncated]"
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    history = json.loads(
        HISTORY.read_text(encoding="utf-8")
    )

    if history["smoke"]:
        raise RuntimeError(
            "Formal history is required"
        )

    for filename, expected in (
        history["source_sha256"].items()
    ):
        if sha256(HERE / filename) != expected:
            raise RuntimeError(
                "Source provenance mismatch: "
                + filename
            )

    core = run_path(str(
        HERE / "60_p3a7e1_action_family_pilot.py"
    ))

    src, sim, api, commands = (
        core["initialize"]()
    )

    n_scenes = 2 if args.smoke else 12
    scene_seed = SEED + (
        100000 if args.smoke else 0
    )

    scenes = src["make_scenes"](
        sim, scene_seed, n_scenes
    )

    print("\nP3-A7F0 STRUCTURAL EQUIVALENCE")
    print("Scenes:", n_scenes)
    print("Mode: independent")
    print("Source hashes: PASS")

    # Source-level evidence is captured for
    # later mathematical inspection.
    force_fn = sim["simulate"].__globals__.get(
        "internal_force"
    )

    source_sections = {
        "internal_force": source_of(force_fn),
        "fit_library": source_of(
            api.get("fit_library")
        ),
        "prediction_rms": source_of(
            api.get("prediction_rms")
        ),
    }

    scores = {
        band: {
            action: {law: [] for law in LAWS}
            for action in ACTIONS
        }
        for band in (
            "original", "above_original_range"
        )
    }

    fitted_examples = {}

    for band in scores:
        for sid, scene_set in enumerate(scenes):
            scene = scene_set[band]

            state = core["collect_first"](
                src, sim, api, commands,
                scene, sid, "independent",
                band, NOISE_SEED,
            )

            if sid == 0:
                fitted_examples[band] = {}

                for law in LAWS:
                    fitted = state["fits"][law]

                    if isinstance(fitted, dict):
                        fitted_examples[band][law] = {
                            key: short_repr(value)
                            for key, value in fitted.items()
                            if key not in (
                                "prediction",
                                "trajectory",
                                "residuals",
                            )
                        }
                    else:
                        fitted_examples[band][law] = (
                            short_repr(fitted)
                        )

            for index, action in enumerate(
                ACTIONS, start=5
            ):
                observation, _ = core["probe"](
                    src, sim, api,
                    state["model"],
                    scene,
                    "independent",
                    commands[action],
                    state["seedbase"] + index,
                )

                for law in LAWS:
                    error = float(
                        api["prediction_rms"](
                            observation,
                            state["fits"][law],
                        )
                    )

                    if not np.isfinite(error):
                        raise RuntimeError(
                            "Nonfinite RMS"
                        )

                    scores[band][action][law].append(
                        1000.0 * error
                    )

    summary = {}

    for band in scores:
        print("\nBAND:", band)
        summary[band] = {}

        for action in ACTIONS:
            reference = np.asarray(
                scores[band][action]["independent"]
            )

            summary[band][action] = {}

            print(" ACTION:", action)

            for law in LAWS:
                values = np.asarray(
                    scores[band][action][law]
                )

                # Residual comparison on identical
                # observed challenge trajectories.
                differences = values - reference

                item = {
                    "median_RMS_mm": float(
                        np.median(values)
                    ),
                    "median_delta_vs_independent_mm":
                        float(np.median(differences)),
                    "max_abs_delta_mm": float(
                        np.max(np.abs(differences))
                    ),
                }

                summary[band][action][law] = item

                print(
                    f"  {law:12s}",
                    "RMS=",
                    round(item["median_RMS_mm"], 5),
                    "delta=",
                    round(
                        item[
                            "median_delta_vs_independent_mm"
                        ], 5
                    ),
                    "mm",
                )

    OUT.mkdir(parents=True, exist_ok=True)

    suffix = (
        "smoke" if args.smoke else "formal"
    )

    report_path = OUT / (
        f"p3a7f0_{suffix}_report.json"
    )

    source_path = OUT / (
        f"p3a7f0_{suffix}_law_sources_PRIVATE.md"
    )

    report = {
        "protocol":
            "P3-A7F0 structural equivalence audit",
        "smoke": args.smoke,
        "n_scenes": n_scenes,
        "summary": summary,
        "fitted_examples_PRIVATE":
            fitted_examples,
        "E4_accessed": False,
        "thresholds_tuned": False,
        "limitations": [
            "Finite-action numerical comparison",
            "Does not prove global equivalence",
            "Parameters estimated from finite data",
            "Independent mechanism only",
            "Oracle truth for diagnostic selection",
            "Requires source-level interpretation",
        ],
    }

    report_path.write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8"
    )

    sections = []

    for name, source in source_sections.items():
        sections.append(
            f"## {name}\n\n"
            f"```python\n{source}\n```\n"
        )

    source_path.write_text(
        "\n".join(sections),
        encoding="utf-8"
    )

    print("\nP3-A7F0 AUDIT COMPLETE")
    print("No E4. No threshold tuning.")
    print("Saved:", report_path)
    print("Saved:", source_path)


if __name__ == "__main__":
    main()
