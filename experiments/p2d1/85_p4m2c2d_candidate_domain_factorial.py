"""
P4-M2C2D — Candidate-Domain Factorial Audit

Eight fixed assignments of WIDE / OUTER_BOX
across elastic, deadzone, and cubic candidates.

Inputs:
  P4-M2C0 frozen candidate fits
  P4-M2C2C solver-corrected decision audit

Read-only historical analysis:
  - No simulation
  - No optimization
  - No new scene generation
  - No threshold selection
  - No E4

All configurations are evaluated as a factorial
diagnostic, not as a trained or selected policy.
"""

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

C0_REPORT = (
    RESULTS / "p4m2c0" / "p4m2c0_dev_report.json"
)
C0_AUDIT = (
    RESULTS / "p4m2c0" /
    "p4m2c0_dev_audit_PRIVATE.json"
)

C2C_REPORT = (
    RESULTS / "p4m2c2c" / "p4m2c2c_dev_report.json"
)
C2C_AUDIT = (
    RESULTS / "p4m2c2c" /
    "p4m2c2c_dev_audit_PRIVATE.json"
)

P84 = (
    HERE / "84_p4m2c2c_solver_corrected_decision_audit.py"
)

OUT = RESULTS / "p4m2c2d"

LAWS = ("elastic", "deadzone", "cubic")

TRUE_LAW = {
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

CONFIGS = (
    "WWW",
    "OWW",
    "WOW",
    "WWO",
    "OOW",
    "OWO",
    "WOO",
    "OOO",
)

N_BOOT = 2000
BOOT_SEED = 2029052101
TOL = 1e-7


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(a, b):
    return bool(np.isclose(
        float(a), float(b),
        rtol=TOL,
        atol=TOL,
    ))


def key_c0(row):
    return (
        row["scene_id"],
        row["band"],
        row["truth_PRIVATE"],
        row["stress_axis"],
        row["level"],
    )


def key_c2c(row):
    return (
        row["scene_id"],
        row["band"],
        row["mode_PRIVATE"],
        row["axis_PRIVATE"],
        row["level_PRIVATE"],
    )


def make_index(rows, key_func):
    indexed = {}

    for row in rows:
        key = key_func(row)

        check(
            key not in indexed,
            f"Duplicate key: {key}",
        )
        indexed[key] = row

    return indexed


def verify_inputs():
    c0_report = read_json(C0_REPORT)
    c2c_report = read_json(C2C_REPORT)

    for report in (c0_report, c2c_report):
        check(report["smoke"] is False,
              "Formal reports required")
        check(report["development_only"] is True,
              "Development-only check failed")
        check(report["E4_accessed"] is False,
              "E4 flag detected")

    check(
        c0_report["truth_used_in_fitting"] is False,
        "Truth-assisted fit flag",
    )

    check(
        c2c_report["new_fits"] == 0,
        "Unexpected refitting",
    )

    check(
        c2c_report["n_rows"] == 432
        and c2c_report["n_target_repairs"] == 21,
        "Unexpected C2C coverage",
    )

    expected_hashes = c2c_report["source_sha256"]

    for alias, path in (
        ("c0_report", C0_REPORT),
        ("c0_audit", C0_AUDIT),
    ):
        check(
            sha(path) == expected_hashes[alias],
            f"Historical hash mismatch: {alias}",
        )

    check(
        P84.is_file()
        and sha(P84) == c2c_report["code_sha256"],
        "P84 script hash mismatch",
    )

    c0 = make_index(
        read_json(C0_AUDIT), key_c0
    )
    c2c = make_index(
        read_json(C2C_AUDIT), key_c2c
    )

    check(
        len(c0) == len(c2c) == 432
        and set(c0) == set(c2c),
        "432-row key alignment failed",
    )

    check(
        sum(bool(x["repaired"]) for x in c2c.values())
        == 21,
        "Expected 21 repaired rows",
    )

    check(
        set(x[0] for x in c0) == set(range(12)),
        "Expected 12 base scene IDs",
    )

    print("[PASS] Frozen source identities")
    print("[PASS] 432/432 exact key alignment")
    print("[PASS] 21 repaired records")

    return c0, c2c


def winner(values):
    return min(
        LAWS,
        key=lambda law: values[law],
    )


def candidate_values(source, corrected):
    result = {
        domain: {
            law: float(
                source["fits"][domain][law][
                    "validation_rms_mm"
                ]
            )
            for law in LAWS
        }
        for domain in ("wide", "outer_box")
    }

    original_deadzone = result["wide"]["deadzone"]

    check(
        close(
            original_deadzone,
            corrected["deadzone_wide_mid_before_mm"],
        ),
        "Historical Deadzone RMS mismatch",
    )

    repaired_deadzone = float(
        corrected["deadzone_wide_mid_after_mm"]
    )

    if corrected["repaired"]:
        result["wide"]["deadzone"] = repaired_deadzone
    else:
        check(
            close(original_deadzone, repaired_deadzone),
            "Unexpected change in unrepaired row",
        )

    for domain in result:
        for law in LAWS:
            value = result[domain][law]
            check(
                np.isfinite(value) and value >= 0,
                "Invalid candidate RMS",
            )

    return result


def policy_values(values, config):
    return {
        law: values[
            "outer_box" if config[i] == "O"
            else "wide"
        ][law]
        for i, law in enumerate(LAWS)
    }


def true_margin(values, truth):
    competitors = [
        values[law]
        for law in LAWS
        if law != truth
    ]

    return float(
        min(competitors) - values[truth]
    )


def analyze_case(key, source, corrected):
    values = candidate_values(
        source, corrected
    )

    old_wide = {
        law: float(
            source["fits"]["wide"][law][
                "validation_rms_mm"
            ]
        )
        for law in LAWS
    }

    check(
        winner(old_wide)
        == corrected["winner_wide_before"],
        f"Old Wide replay mismatch: {key}",
    )

    check(
        winner(values["wide"])
        == corrected["winner_wide_after"],
        f"Corrected Wide mismatch: {key}",
    )

    check(
        winner(values["outer_box"])
        == corrected["winner_outer"],
        f"Outer replay mismatch: {key}",
    )

    truth = TRUE_LAW[key[2]]

    policies = {}

    for config in CONFIGS:
        chosen = policy_values(values, config)

        pred = winner(chosen)

        policies[config] = {
            "winner": pred,
            "correct_PRIVATE": pred == truth,
            "margin_mm_PRIVATE":
                true_margin(chosen, truth),
        }

    check(
        policies["WWW"]["correct_PRIVATE"]
        == corrected["wide_correct_after_PRIVATE"],
        "WWW correctness mismatch",
    )

    check(
        policies["OOO"]["correct_PRIVATE"]
        == corrected["outer_correct_PRIVATE"],
        "OOO correctness mismatch",
    )

    true_w = values["wide"][truth]
    true_o = values["outer_box"][truth]

    rival_w = min(
        values["wide"][law]
        for law in LAWS if law != truth
    )

    rival_o = min(
        values["outer_box"][law]
        for law in LAWS if law != truth
    )

    return {
        "scene_id": key[0],
        "band": key[1],
        "mode_PRIVATE": key[2],
        "axis_PRIVATE": key[3],
        "level_PRIVATE": key[4],
        "repaired": corrected["repaired"],

        "true_law_PRIVATE": truth,
        "policies": policies,

        "competition_PRIVATE": {
            "true_rms_change_outer_minus_wide_mm":
                true_o - true_w,
            "rival_rms_change_outer_minus_wide_mm":
                rival_o - rival_w,
            "margin_change_outer_minus_wide_mm":
                (rival_o - true_o) -
                (rival_w - true_w),
        },
    }


def summarize(rows):
    summary = {}

    for config in CONFIGS:
        successes = [
            int(
                row["policies"][config][
                    "correct_PRIVATE"
                ]
            )
            for row in rows
        ]

        per_mode = {}
        confusion = {}

        for mode, truth in TRUE_LAW.items():

            subset = [
                row for row in rows
                if row["mode_PRIVATE"] == mode
            ]

            count = sum(
                row["policies"][config][
                    "correct_PRIVATE"
                ]
                for row in subset
            )

            per_mode[mode] = {
                "n": len(subset),
                "correct": int(count),
                "accuracy": (
                    count / len(subset)
                    if subset else None
                ),
            }

            confusion[truth] = {
                law: sum(
                    row["policies"][config][
                        "winner"
                    ] == law
                    for row in subset
                )
                for law in LAWS
            }

        class_rates = [
            item["accuracy"]
            for item in per_mode.values()
            if item["accuracy"] is not None
        ]

        summary[config] = {
            "n": len(rows),
            "correct": int(sum(successes)),
            "accuracy": float(np.mean(successes)),
            "macro_accuracy": (
                float(np.mean(class_rates))
                if class_rates else None
            ),
            "by_mode": per_mode,
            "confusion_PRIVATE": confusion,
        }

    return summary


def main_effects(summary):
    effects = {}

    for position, law in enumerate(LAWS):
        differences = []

        for config in CONFIGS:
            if config[position] != "W":
                continue

            outer_config = (
                config[:position]
                + "O"
                + config[position + 1:]
            )

            differences.append(
                summary[outer_config]["accuracy"]
                - summary[config]["accuracy"]
            )

        check(
            len(differences) == 4,
            "Incomplete factorial design",
        )

        effects[law] = {
            "average_accuracy_effect":
                float(np.mean(differences)),
            "four_context_effects": differences,
        }

    return effects


def scene_bootstrap(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[int(row["scene_id"])].append(row)

    ids = sorted(grouped)

    if len(ids) < 10:
        return None

    check(
        len(ids) == 12,
        "Expected 12 scene clusters",
    )

    rng = np.random.default_rng(BOOT_SEED)

    policy_samples = {
        name: [] for name in CONFIGS
    }

    effect_samples = {
        law: [] for law in LAWS
    }

    for _ in range(N_BOOT):
        chosen = rng.choice(
            ids,
            size=len(ids),
            replace=True,
        )

        sample = [
            row
            for sid in chosen
            for row in grouped[int(sid)]
        ]

        rates = {
            config: float(np.mean([
                row["policies"][config][
                    "correct_PRIVATE"
                ]
                for row in sample
            ]))
            for config in CONFIGS
        }

        for config in CONFIGS:
            policy_samples[config].append(
                rates[config]
            )

        for position, law in enumerate(LAWS):
            contrasts = []

            for config in CONFIGS:
                if config[position] == "W":
                    alternative = (
                        config[:position]
                        + "O"
                        + config[position + 1:]
                    )

                    contrasts.append(
                        rates[alternative]
                        - rates[config]
                    )

            effect_samples[law].append(
                float(np.mean(contrasts))
            )

    def interval(values):
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    return {
        "cluster_unit": "base_scene_id",
        "n_clusters": len(ids),
        "n_bootstrap": N_BOOT,
        "seed": BOOT_SEED,
        "policy_accuracy_95CI": {
            config: interval(values)
            for config, values
            in policy_samples.items()
        },
        "main_effect_95CI": {
            law: interval(values)
            for law, values
            in effect_samples.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("P4-M2C2D CANDIDATE-DOMAIN FACTORIAL AUDIT")
    print("=" * 72)

    print("Smoke:", args.smoke)
    print("Read-only. No fitting. No E4.")

    c0, c2c = verify_inputs()

    keys = sorted(c0)

    if args.smoke:
        keys = [
            key for key in keys
            if key[0] == 0
        ]

    analyzed = [
        analyze_case(
            key,
            c0[key],
            c2c[key],
        )
        for key in keys
    ]

    expected_n = 36 if args.smoke else 432

    check(
        len(analyzed) == expected_n,
        "Case count mismatch",
    )

    summary = summarize(analyzed)
    effects = main_effects(summary)

    if not args.smoke:
        check(
            summary["WWW"]["correct"] == 269,
            "Corrected Wide replay mismatch",
        )
        check(
            summary["OOO"]["correct"] == 333,
            "Outer replay mismatch",
        )

    bootstrap = (
        None if args.smoke
        else scene_bootstrap(analyzed)
    )

    print("\nFACTOrIAL CONFIGURATIONS")
    print("Order: Elastic / Deadzone / Cubic")

    for config in CONFIGS:
        item = summary[config]

        print(
            f"{config}: "
            f"{item['correct']}/{item['n']} "
            f"Acc={item['accuracy']:.4f} "
            f"Macro={item['macro_accuracy']:.4f}"
        )

    print("\nCANDIDATE-DOMAIN MAIN EFFECTS")

    for law in LAWS:
        value = effects[law][
            "average_accuracy_effect"
        ]

        print(
            f"{law}: {value:+.6f}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = "smoke" if args.smoke else "dev"

    report_path = OUT / (
        f"p4m2c2d_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2c2d_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P4-M2C2D candidate-domain factorial audit",

        "smoke": args.smoke,
        "development_only": True,

        "n_rows": len(analyzed),

        "candidate_laws": list(LAWS),
        "configurations": list(CONFIGS),

        "summary": summary,
        "main_effects": effects,
        "bootstrap": bootstrap,

        "E4_accessed": False,
        "new_simulations": 0,
        "new_fits": 0,
        "thresholds_selected": False,

        "source_sha256": {
            C0_REPORT.name: sha(C0_REPORT),
            C0_AUDIT.name: sha(C0_AUDIT),
            C2C_REPORT.name: sha(C2C_REPORT),
            C2C_AUDIT.name: sha(C2C_AUDIT),
            P84.name: sha(P84),
        },

        "limitations": [
            "All eight policies are retrospective",
            "No policy was independently selected",
            "Only twelve independent base scenes",
            "Mid RMS is used for original model selection",
            "Factorial effects are not deployment effects",
            "No independent prior-mismatch confirmation",
            "No global optimizer certificate",
            "No E4 access",
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
            analyzed,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nP4-M2C2D COMPLETE")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()