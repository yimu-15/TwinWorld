
"""
P4-M2C2E — Fresh-Scene Candidate-Domain Replication

Primary:
    OOO versus WWW overall accuracy.

Secondary:
    WOO versus OOO macro recall.

Diagnostic:
    Elastic/Cubic confusion and nested solver violations.

24 fresh scenes, two bands, eight fixed conditions
per scene/band.

Frozen P74/P75/P77/P79/P80 fitting pipeline.
No E4. No threshold tuning. No adaptive policy.
"""

import argparse
import hashlib
import json

from collections import Counter, defaultdict
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "p4m2c2e"

P80 = HERE / "80_p4m2c0_prior_mismatch_stress.py"

P80_REPORT = (
    RESULTS / "p4m2c0" /
    "p4m2c0_dev_report.json"
)

P79_REPORT = (
    RESULTS / "p4m2b2b" /
    "p4m2b2b_fresh_report.json"
)

P85_REPORT = (
    RESULTS / "p4m2c2d" /
    "p4m2c2d_dev_report.json"
)

SCENE_SEED = 20290617
NOISE_SEED = 2029061701
BOOT_SEED = 2029061702

N_SCENES = 24
N_BOOT = 2000
SMOKE_OFFSET = 100000
MIN_COVERAGE = 0.90

LAWS = ("elastic", "deadzone", "cubic")

CONFIGS = (
    "WWW", "OWW", "WOW", "WWO",
    "OOW", "OWO", "WOO", "OOO",
)

TRUE_LAW = {
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

CONDITIONS = {
    "elastic": (
        ("matched", "M0"),
        ("alpha_high", "M2"),
    ),

    "deadzone": (
        ("matched", "M0"),
        ("alpha_high", "M2"),
        ("gap_high", "M2"),
    ),

    "nonlinear": (
        ("matched", "M0"),
        ("alpha_high", "M2"),
        ("gamma_high", "M2"),
    ),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    require(
        path.is_file(),
        f"Missing frozen input: {path}",
    )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def frozen_provenance():

    old80 = read_json(P80_REPORT)
    old79 = read_json(P79_REPORT)
    old85 = read_json(P85_REPORT)

    require(
        all(
            r.get("smoke") is False
            for r in (old80, old79, old85)
        ),
        "Historical formal reports required",
    )

    require(
        old80["scene_seed"] == 20290517
        and old80["n_base_scenes"] == 12,
        "Unexpected P4-M2C0 provenance",
    )

    require(
        old79["scene_seed"] == 20290417
        and old79["n_scenes"] == 72,
        "Unexpected P4-M2B2B provenance",
    )

    require(
        old85["n_rows"] == 432
        and old85["new_fits"] == 0,
        "Unexpected P4-M2C2D report",
    )

    require(
        old85["source_sha256"][
            P80_REPORT.name
        ] == digest(P80_REPORT),
        "P4-M2C2D to C0 SHA mismatch",
    )

    require(
        old85["summary"]["WWW"]["correct"] == 269
        and old85["summary"]["WOO"]["correct"] == 316
        and old85["summary"]["OOO"]["correct"] == 333,
        "Frozen 85 conclusions mismatch",
    )

    require(
        all(
            r.get("E4_accessed") is False
            for r in (old80, old79, old85)
        ),
        "E4 flag found in historical evidence",
    )

    for name, expected in old80[
        "source_sha256"
    ].items():

        path = HERE / name

        require(
            path.is_file()
            and digest(path) == expected,
            f"Frozen source mismatch: {name}",
        )

    return old79, old80, old85


def scene_audit(
    p79, src, sim, scenes,
    hist79, hist80, smoke
):

    audit = p79[
        "verify_scene_independence"
    ](src, sim, scenes)

    fresh = set(
        audit["new_scene_fingerprints"]
    )

    historical = {
        "P4-M2B2B independent evaluation":
            set(
                hist79[
                    "scene_independence_audit"
                ]["new_scene_fingerprints"]
            ),

        "P4-M2C0 mismatch development":
            set(
                hist80["scene_audit"][
                    "new_fingerprints"
                ]
            ),
    }

    for label, old in historical.items():

        require(
            not fresh.intersection(old),
            f"Historical overlap: {label}",
        )

        print(
            f"[PASS] No exact scene overlap: {label}"
        )

    if not smoke:

        prior_smoke = src["make_scenes"](
            sim,
            SCENE_SEED + SMOKE_OFFSET,
            1,
        )

        smoke_fps = set(
            p79["fingerprint_scenes"](
                prior_smoke
            )
        )

        require(
            not fresh.intersection(smoke_fps),
            "Smoke/formal scene overlap",
        )

    require(
        len(fresh) == 2 * len(scenes),
        "Duplicate scene fingerprint",
    )

    return {
        "new_scene_fingerprints": sorted(fresh),
        "n_fingerprints": len(fresh),
        "historical_overlap": 0,
        "historical_audit": audit,
    }


def candidate_selection(fits, config):

    candidate = {}

    for i, law in enumerate(LAWS):

        domain = (
            "outer_box"
            if config[i] == "O"
            else "wide"
        )

        fit = fits[domain][law]

        if not (
            fit["valid"] and fit["converged"]
        ):
            return None, None

        value = fit["validation_rms_mm"]

        if (
            value is None
            or not np.isfinite(float(value))
        ):
            return None, None

        candidate[law] = float(value)

    decision = min(
        LAWS,
        key=lambda law: candidate[law],
    )

    return decision, candidate


def analyze_case(
    p80, fits,
    sid, band, mode, axis, level, effective
):

    truth = TRUE_LAW[mode]

    nested_violations = []

    for law in LAWS:

        wide = fits["wide"][law]
        outer = fits["outer_box"][law]

        if (
            wide["valid"]
            and outer["valid"]
            and wide["train_mse"] is not None
            and outer["train_mse"] is not None
            and float(outer["train_mse"])
            < float(wide["train_mse"]) - 1e-6
        ):
            nested_violations.append(law)

    policies = {}

    for config in CONFIGS:

        predicted, values = candidate_selection(
            fits, config
        )

        margin = None

        if values is not None:

            margin = (
                min(
                    values[law]
                    for law in LAWS
                    if law != truth
                )
                - values[truth]
            )

        policies[config] = {
            "winner": predicted,
            "available": predicted is not None,
            "correct_PRIVATE": predicted == truth,
            "margin_mm_PRIVATE": margin,
        }

    require(
        policies["WWW"]["winner"]
        == p80["winner"](fits, "wide"),
        "WWW replay mismatch",
    )

    require(
        policies["OOO"]["winner"]
        == p80["winner"](fits, "outer_box"),
        "OOO replay mismatch",
    )

    return {
        "scene_id": sid,
        "band": band,
        "mode_PRIVATE": mode,
        "axis_PRIVATE": axis,
        "level_PRIVATE": level,

        "true_law_PRIVATE": truth,
        "nested_solver_violations": nested_violations,

        "true_effective_PRIVATE":
            np.asarray(
                effective, dtype=float
            ).tolist(),

        "policies": policies,
        "fits_PRIVATE": fits,
    }


def accuracy(rows, config):

    return float(
        np.mean([
            r["policies"][config][
                "correct_PRIVATE"
            ]
            for r in rows
        ])
    )


def class_recall(rows, config, mode):

    subset = [
        r for r in rows
        if r["mode_PRIVATE"] == mode
    ]

    return accuracy(subset, config)


def macro(rows, config):

    return float(
        np.mean([
            class_recall(rows, config, mode)
            for mode in TRUE_LAW
        ])
    )


def summary(rows):

    result = {}

    for config in CONFIGS:

        covered = [
            r for r in rows
            if r["policies"][config]["available"]
        ]

        correct = sum(
            r["policies"][config]["correct_PRIVATE"]
            for r in rows
        )

        result[config] = {
            "n": len(rows),

            "coverage":
                len(covered) / len(rows),

            "n_available": len(covered),
            "correct": int(correct),

            "accuracy_all":
                correct / len(rows),

            "accuracy_available": (
                correct / len(covered)
                if covered else None
            ),

            "macro_all": macro(rows, config),

            "by_mode": {
                mode: {
                    "n": sum(
                        r["mode_PRIVATE"] == mode
                        for r in rows
                    ),

                    "recall":
                        class_recall(
                            rows, config, mode
                        ),
                }
                for mode in TRUE_LAW
            },
        }

    return result


def paired_counts(rows, a, b):

    counts = Counter(
        (
            r["policies"][a]["correct_PRIVATE"],
            r["policies"][b]["correct_PRIVATE"],
        )
        for r in rows
    )

    return {
        "both_correct":
            counts[(True, True)],

        "only_first_correct":
            counts[(True, False)],

        "only_second_correct":
            counts[(False, True)],

        "both_incorrect_or_unavailable":
            counts[(False, False)],
    }


def bootstrap(rows):

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            row["scene_id"]
        ].append(row)

    ids = sorted(grouped)

    require(
        len(ids) == N_SCENES,
        "Unexpected bootstrap cluster count",
    )

    rng = np.random.default_rng(
        BOOT_SEED
    )

    metrics = {
        "primary_OOO_minus_WWW_overall": [],

        "secondary_WOO_minus_OOO_macro": [],

        "diagnostic_WOO_minus_OOO_elastic_recall": [],

        "diagnostic_OOO_minus_WOO_nonlinear_recall": [],
    }

    for _ in range(N_BOOT):

        picked = rng.choice(
            ids,
            size=len(ids),
            replace=True,
        )

        sample = [
            row
            for sid in picked
            for row in grouped[int(sid)]
        ]

        metrics[
            "primary_OOO_minus_WWW_overall"
        ].append(
            accuracy(sample, "OOO")
            - accuracy(sample, "WWW")
        )

        metrics[
            "secondary_WOO_minus_OOO_macro"
        ].append(
            macro(sample, "WOO")
            - macro(sample, "OOO")
        )

        metrics[
            "diagnostic_WOO_minus_OOO_elastic_recall"
        ].append(
            class_recall(
                sample, "WOO", "elastic"
            )
            - class_recall(
                sample, "OOO", "elastic"
            )
        )

        metrics[
            "diagnostic_OOO_minus_WOO_nonlinear_recall"
        ].append(
            class_recall(
                sample, "OOO", "nonlinear"
            )
            - class_recall(
                sample, "WOO", "nonlinear"
            )
        )

    return {
        "unit": "base_scene_id",
        "n_clusters": N_SCENES,
        "replications": N_BOOT,
        "seed": BOOT_SEED,

        "paired_95CI": {
            name: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for name, values
            in metrics.items()
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
    print(
        "P4-M2C2E FRESH-SCENE "
        "FACTORIAL REPLICATION"
    )
    print("=" * 72)

    print("Smoke:", args.smoke)
    print("No E4. No thresholds. No adaptive policy.")

    print("\n[1] Frozen provenance")

    hist79, hist80, hist85 = frozen_provenance()

    p80 = run_path(str(P80))

    p80["check_provenance"]()

    (
        p74, p75, p77, p79,
        src, sim, api, commands
    ) = p80["initialize"]()

    require(
        tuple(p80["LAWS"]) == LAWS,
        "Candidate law mismatch",
    )

    require(
        tuple(p77["TRAIN"]) ==
        ("e1", "e2", "weak", "strong")
        and p77["VALID"] == "mid",
        "Observation protocol mismatch",
    )

    require(
        p77["MAXITER"] == 150
        and len(
            p77["initial_points"]("deadzone")
        ) == 2,
        "Solver protocol mismatch",
    )

    require(
        tuple(p80["BANDS"]) ==
        ("original", "above_original_range"),
        "Band mismatch",
    )

    print("[PASS] Frozen fitting and source chain")

    print("\n[2] Fresh scenes")

    n = 1 if args.smoke else N_SCENES

    seed = SCENE_SEED + (
        SMOKE_OFFSET if args.smoke else 0
    )

    noise_seed = NOISE_SEED + (
        SMOKE_OFFSET if args.smoke else 0
    )

    scenes = src["make_scenes"](
        sim, seed, n
    )

    require(
        len(scenes) == n,
        "Fresh scene count mismatch",
    )

    independence = scene_audit(
        p79,
        src,
        sim,
        scenes,
        hist79,
        hist80,
        args.smoke,
    )

    print(
        f"[PASS] Scene independence: "
        f"{n} base clusters"
    )

    print("\n[3] Frozen candidate fits")

    rows = []

    for sid, scene_set in enumerate(scenes):

        for band in p80["BANDS"]:

            for mode, conditions in CONDITIONS.items():

                for axis, level in conditions:

                    if level == "M0":

                        require(
                            axis == "matched",
                            "Matched condition mismatch",
                        )

                        scene = scene_set[band]

                        effective = p80[
                            "true_effective"
                        ](
                            p74, mode, scene
                        )

                    else:

                        scene, effective = p80[
                            "make_stress_scene"
                        ](
                            p74,
                            p77,
                            scene_set[band],
                            mode,
                            axis,
                            level,
                        )

                    fits = p80["evaluate_case"](
                        p75,
                        p77,
                        sim,
                        api,
                        commands,
                        scene,
                        sid,
                        band,
                        mode,
                        noise_seed,
                    )

                    rows.append(
                        analyze_case(
                            p80,
                            fits,
                            sid,
                            band,
                            mode,
                            axis,
                            level,
                            effective,
                        )
                    )

        print(
            f"Completed fresh base scenes: "
            f"{sid + 1}/{n}"
        )

    require(
        len(rows) == n * 2 * 8,
        "Case matrix size mismatch",
    )

    keys = [
        (
            row["scene_id"],
            row["band"],
            row["mode_PRIVATE"],
            row["axis_PRIVATE"],
            row["level_PRIVATE"],
        )
        for row in rows
    ]

    require(
        len(set(keys)) == len(keys),
        "Duplicate experiment cases",
    )

    print("\n[4] Factorial results")

    by_config = summary(rows)

    coverage_min = min(
        value["coverage"]
        for value in by_config.values()
    )

    nested_counts = dict(
        Counter(
            law
            for row in rows
            for law in row[
                "nested_solver_violations"
            ]
        )
    )

    checks = {
        "coverage_ge_90pct":
            coverage_min >= MIN_COVERAGE
    }

    by_regime = {
        regime: summary([
            row
            for row in rows
            if (
                row["level_PRIVATE"] == "M0"
            ) == (
                regime == "M0"
            )
        ])
        for regime in ("M0", "M2")
    }

    comparisons = {
        "primary_OOO_minus_WWW_overall":
            accuracy(rows, "OOO")
            - accuracy(rows, "WWW"),

        "secondary_WOO_minus_OOO_macro":
            macro(rows, "WOO")
            - macro(rows, "OOO"),

        "diagnostic_WOO_minus_OOO_elastic_recall":
            class_recall(
                rows, "WOO", "elastic"
            )
            - class_recall(
                rows, "OOO", "elastic"
            ),

        "diagnostic_OOO_minus_WOO_nonlinear_recall":
            class_recall(
                rows, "OOO", "nonlinear"
            )
            - class_recall(
                rows, "WOO", "nonlinear"
            ),
    }

    intervals = (
        None if args.smoke
        else bootstrap(rows)
    )

    for config in CONFIGS:

        item = by_config[config]

        print(
            f"{config}: "
            f"{item['correct']}/{item['n']} "
            f"acc={item['accuracy_all']:.4f} "
            f"macro={item['macro_all']:.4f} "
            f"coverage={item['coverage']:.3f}"
        )

    print("\n[5] Pre-registered comparisons")

    for name, value in comparisons.items():

        ci = (
            intervals["paired_95CI"][name]
            if intervals else "SMOKE_SKIP"
        )

        print(
            name,
            f"{value:+.6f}",
            "95CI=", ci,
        )

    print(
        "Nested solver violations:",
        nested_counts,
    )

    print(
        "Coverage >= 90%:",
        checks["coverage_ge_90pct"],
    )

    print("\n[6] Saving reports")

    report = {
        "protocol":
            "P4-M2C2E fresh-scene "
            "factorial replication",

        "smoke": args.smoke,
        "development_independent_seed": True,

        "scene_seed": seed,
        "noise_seed": noise_seed,

        "n_base_scenes": n,
        "n_cases": len(rows),

        "regime":
            "M0 plus fixed M2 mismatch "
            "per applicable axis",

        "conditions": {
            name: list(values)
            for name, values in CONDITIONS.items()
        },

        "config_order": list(CONFIGS),

        "summary": by_config,
        "by_regime": by_regime,

        "nested_solver_violations":
            nested_counts,

        "comparisons": comparisons,
        "bootstrap": intervals,

        "paired_counts": {
            "OOO_vs_WWW":
                paired_counts(
                    rows, "OOO", "WWW"
                ),

            "WOO_vs_OOO":
                paired_counts(
                    rows, "WOO", "OOO"
                ),
        },

        "checks": checks,

        "scene_independence":
            independence,

        "E4_accessed": False,
        "thresholds_tuned": False,
        "new_algorithm": False,

        "source_sha256": {
            P80.name: digest(P80),

            P79_REPORT.name:
                digest(P79_REPORT),

            P80_REPORT.name:
                digest(P80_REPORT),

            P85_REPORT.name:
                digest(P85_REPORT),

            Path(__file__).name:
                digest(Path(__file__)),
        },

        "limitations": [
            "Fresh seed but same simulator distribution",
            "M2-only mismatch, not all-stress generalization",
            "No adaptive policy",
            "Domain configurations were fixed in advance",
            "Only 24 independent base-scene clusters",
            "Mid action used for model selection",
            "Truth used only for generation and offline audit",
            "No global SLSQP optimality certificate",
            "No E4 or test threshold tuning",
        ],
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke else "fresh"
    )

    report_path = OUT / (
        f"p4m2c2e_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2c2e_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\nP4-M2C2E COMPLETE")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
