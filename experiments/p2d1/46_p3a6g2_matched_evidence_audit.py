
"""
P3-A6G2: Matched-Evidence Structural Fairness Audit

Replays the exact P3-A6G observations.

OLD:
  Independent trained on E1/E2
  Deadzone trained on E1/E2 + Weak

FAIR:
  Both trained on E1/E2 + Weak
  Both evaluated on heldout Strong

Reverse:
  Both trained on E1/E2 + Strong
  Both evaluated on heldout Weak

This is a retrospective fairness correction,
not an additional independent validation.

No new scenes, no thresholds, no E4.
Historical results remain unchanged.
"""

from pathlib import Path
from runpy import run_path
import argparse
import hashlib
import json
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a6g2"
SOURCE = HERE / "results" / "p3a6g"

BANDS = (
    "original",
    "high_in_range",
    "above_original_range",
)
MODES = ("independent", "deadzone")
ACTIONS = ("e1", "e2", "weak", "strong")


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def check_close(actual, expected, description):
    if not np.isclose(
        actual, expected, rtol=0.0, atol=1e-8
    ):
        raise RuntimeError(
            f"Historical replay mismatch: {description} "
            f"actual={actual}, expected={expected}"
        )


def fit(api, probes, law, nfev):
    result, _ = api["fit_library"](
        probes, (law,), nfev
    )
    return result


def error(api, probe, model):
    value = float(api["prediction_rms"](probe, model))
    if not np.isfinite(value):
        raise RuntimeError("Nonfinite prediction RMS")
    return value


def evaluate_band(
    band, ids, scenes, saved, src, sim, api, commands
):
    archive_index = {
        (int(r["scene_id"]), r["truth_PRIVATE"]): r
        for r in saved[band]
    }

    records = []

    for sid in ids:
        scene = scenes[sid][band]
        model = sim["make_model"](scene)

        for mode_index, mode in enumerate(MODES):
            probes = {}

            for action_index, action in enumerate(ACTIONS):
                probe, _ = src["observation"](
                    sim, api, model, scene,
                    mode, commands[action],
                    sid, mode_index, action_index
                )
                probes[action] = probe

            early = [probes["e1"], probes["e2"]]

            # Reproduce P3-A6G original comparisons.
            old_null = fit(api, early, "independent", 40)

            deadzone_weak = fit(
                api,
                early + [probes["weak"]],
                "deadzone",
                src["NFEV"],
            )

            deadzone_strong = fit(
                api,
                early + [probes["strong"]],
                "deadzone",
                src["NFEV"],
            )

            archived = archive_index[(sid, mode)]

            for label, value in {
                "null_to_strong_m": error(
                    api, probes["strong"], old_null
                ),
                "null_to_weak_m": error(
                    api, probes["weak"], old_null
                ),
                "weak_to_strong_m": error(
                    api, probes["strong"], deadzone_weak
                ),
                "strong_to_weak_m": error(
                    api, probes["weak"], deadzone_strong
                ),
            }.items():
                check_close(
                    value, float(archived[label]),
                    f"{band} scene={sid} mode={mode} {label}"
                )

            # FAIR COMPARISON:
            # Both models receive identical training probes.
            independent_weak = fit(
                api,
                early + [probes["weak"]],
                "independent",
                src["NFEV"],
            )

            independent_strong = fit(
                api,
                early + [probes["strong"]],
                "independent",
                src["NFEV"],
            )

            fair_null_strong = error(
                api, probes["strong"], independent_weak
            )
            fair_deadzone_strong = error(
                api, probes["strong"], deadzone_weak
            )

            fair_null_weak = error(
                api, probes["weak"], independent_strong
            )
            fair_deadzone_weak = error(
                api, probes["weak"], deadzone_strong
            )

            records.append({
                "scene_id": sid,
                "truth_PRIVATE": mode,

                "old_strong_margin_mm":
                    1000.0 * float(
                        archived["strong_margin_m"]
                    ),

                "fair_strong_margin_mm":
                    1000.0 * (
                        fair_null_strong
                        - fair_deadzone_strong
                    ),

                "fair_weak_margin_mm":
                    1000.0 * (
                        fair_null_weak
                        - fair_deadzone_weak
                    ),

                "fair_independent_to_strong_mm":
                    1000.0 * fair_null_strong,

                "fair_deadzone_to_strong_mm":
                    1000.0 * fair_deadzone_strong,

                "fair_independent_to_weak_mm":
                    1000.0 * fair_null_weak,

                "fair_deadzone_to_weak_mm":
                    1000.0 * fair_deadzone_weak,
            })

    if len(records) != 2 * len(ids):
        raise RuntimeError("Unexpected number of records")

    return records


def show_summary(band, rows):
    print("\n" + "=" * 72)
    print("BAND:", band, "EPISODES:", len(rows))
    print("=" * 72)

    grouped = {}

    for mode in MODES:
        subset = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        metrics = {}

        for field in (
            "old_strong_margin_mm",
            "fair_strong_margin_mm",
            "fair_weak_margin_mm",
        ):
            values = np.asarray([
                r[field] for r in subset
            ])

            metrics[field] = {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
                "positive": int(np.sum(values > 0)),
            }

        grouped[mode] = metrics

        print("\nMODE:", mode, "| N:", len(subset))

        for name, result in metrics.items():
            print(name, result)

    # This is a descriptive distribution check.
    # Do not use its gap to select a threshold.
    ind = grouped["independent"]
    dz = grouped["deadzone"]

    gaps = {}

    for name in (
        "fair_strong_margin_mm",
        "fair_weak_margin_mm",
    ):
        gap = (
            dz[name]["min"] - ind[name]["max"]
        )

        gaps[name] = float(gap)

        print(
            "\nEmpirical distribution gap",
            name,
            "=",
            round(gap, 4),
            "mm",
        )

    return {
        "by_mode": grouped,
        "empirical_gaps_mm": gaps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    suffix = "smoke" if args.smoke else "prospective"

    report = load(
        SOURCE / f"p3a6g_{suffix}_report.json"
    )
    archived = load(
        SOURCE / f"p3a6g_{suffix}_audit_PRIVATE.json"
    )

    # Verify the historical simulator and
    # inference implementation have not changed.
    for filename, old_hash in (
        report["source_sha256"].items()
    ):
        actual_hash = hashlib.sha256(
            (HERE / filename).read_bytes()
        ).hexdigest()

        if actual_hash != old_hash:
            raise RuntimeError(
                f"Historical source changed: {filename}"
            )

    src = run_path(str(
        HERE / "45_p3a6g_fresh_structure_validation.py"
    ))
    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))
    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    scenes = src["make_scenes"](sim)
    commands = src["get_commands"](sim)

    ids = list(map(int, report["base_scene_ids"]))

    expected_ids = (
        list(range(src["N_SMOKE"]))
        if args.smoke
        else list(range(
            src["N_SMOKE"],
            src["N_SMOKE"] + src["N_FULL"]
        ))
    )

    if ids != expected_ids:
        raise RuntimeError(
            "Smoke/full scene split mismatch"
        )

    result = {
        "protocol": "P3-A6G2 matched-evidence fairness",
        "retrospective": True,
        "threshold_tuning": False,
        "new_independent_validation": False,
        "E4_accessed": False,
        "training_weak_to_strong":
            "E1/E2+Weak for both mechanisms",
        "training_strong_to_weak":
            "E1/E2+Strong for both mechanisms",
        "bands": {},
        "summary": {},
    }

    for band in BANDS:
        rows = evaluate_band(
            band, ids, scenes,
            archived, src, sim, api, commands
        )

        result["bands"][band] = rows
        result["summary"][band] = show_summary(
            band, rows
        )

    OUT.mkdir(parents=True, exist_ok=True)

    path = OUT / (
        f"p3a6g2_{suffix}_audit_PRIVATE.json"
    )

    path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "\nP3-A6G2 MATCHED-EVIDENCE AUDIT PASS"
    )
    print("Historical P3-A6G results reproduced.")
    print("No threshold tuning, no E4, no new policy.")
    print("Saved:", path)


if __name__ == "__main__":
    main()
