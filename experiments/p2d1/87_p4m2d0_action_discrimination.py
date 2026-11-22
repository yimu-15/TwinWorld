
"""
P4-M2D0 — Matched-Budget Action Discrimination Audit

Frozen input: P4-M2C2E 384 fitted cases.

No refitting, no parameter adaptation.
New held-out action observations only.

Primary prospective diagnostic:
OOO Elastic-M2 recall:
MID + BIPHASIC_HIGH versus MID + HIGH.

All fitted parameters are reused unchanged.
Truth labels are used only for offline scoring.

No E4. No thresholds. No optimizer calls.
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

OUT = RESULTS / "p4m2d0"

P80 = HERE / "80_p4m2c0_prior_mismatch_stress.py"
P86 = HERE / "86_p4m2c2e_fresh_scene_factorial.py"

REFERENCE_REPORT = (
    RESULTS / "p4m2c2e" /
    "p4m2c2e_fresh_report.json"
)

REFERENCE_AUDIT = (
    RESULTS / "p4m2c2e" /
    "p4m2c2e_fresh_audit_PRIVATE.json"
)

ACTIONS = (
    "high",
    "reverse_high",
    "biphasic_high",
    "shifted_high",
)

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

METHODS = (
    "MID",
    "MID_PLUS_HIGH",
    "MID_PLUS_REVERSE_HIGH",
    "MID_PLUS_BIPHASIC_HIGH",
    "MID_PLUS_SHIFTED_HIGH",
    "HIGH_ONLY",
    "REVERSE_HIGH_ONLY",
    "BIPHASIC_HIGH_ONLY",
    "SHIFTED_HIGH_ONLY",
)

SCENE_SEED = 20290617
ACTION_NOISE_SEED = 2029071701

N_SCENES = 24
N_BOOT = 2000
BOOT_SEED = 2029071702


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def read_json(path):
    require(
        path.is_file(),
        f"Missing input: {path}",
    )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def case_key(row):
    return (
        int(row["scene_id"]),
        row["band"],
        row["mode_PRIVATE"],
        row["axis_PRIVATE"],
        row["level_PRIVATE"],
    )


def verify_inputs():

    report = read_json(REFERENCE_REPORT)
    audit = read_json(REFERENCE_AUDIT)

    require(
        report["smoke"] is False,
        "Formal 86 report required",
    )

    require(
        report["scene_seed"] == SCENE_SEED,
        "Unexpected scene seed",
    )

    require(
        report["n_base_scenes"] == N_SCENES
        and report["n_cases"] == 384,
        "Unexpected case count",
    )

    require(
        report["E4_accessed"] is False
        and report["thresholds_tuned"] is False,
        "Frozen protocol violation",
    )

    require(
        sha(P80) ==
        report["source_sha256"][P80.name],
        "Frozen P80 SHA mismatch",
    )

    require(
        sha(P86) ==
        report["source_sha256"][P86.name],
        "Frozen P86 SHA mismatch",
    )

    require(
        len(audit) == 384,
        "Missing fitted cases",
    )

    keys = [case_key(row) for row in audit]

    require(
        len(set(keys)) == 384,
        "Duplicate case keys",
    )

    counts = Counter(
        row["scene_id"] for row in audit
    )

    require(
        set(counts) == set(range(N_SCENES))
        and all(n == 16 for n in counts.values()),
        "Scene condition matrix mismatch",
    )

    for row in audit:
        for domain in ("wide", "outer_box"):
            for law in LAWS:

                fit = row["fits_PRIVATE"][domain][law]

                require(
                    fit["valid"]
                    and fit["converged"]
                    and fit["params_PRIVATE"] is not None
                    and fit["validation_rms_mm"] is not None,
                    "Incomplete frozen fit",
                )

    print("[PASS] 384 frozen cases")
    print("[PASS] 24 scene clusters")
    print("[PASS] Frozen source identities")

    return report, audit


def verify_actions(commands):

    reference = np.asarray(
        commands["high"], dtype=float
    )

    require(
        reference.ndim == 1
        and len(reference) >= 8,
        "Invalid reference command",
    )

    budget = np.sum(np.abs(reference))
    peak = np.max(np.abs(reference))

    for name in ACTIONS:

        require(
            name in commands,
            f"Missing action: {name}",
        )

        cmd = np.asarray(
            commands[name], dtype=float
        )

        require(
            cmd.shape == reference.shape,
            "Action shape mismatch",
        )

        require(
            np.isclose(
                np.sum(np.abs(cmd)),
                budget,
                rtol=1e-10,
                atol=1e-12,
            ),
            f"Command L1 mismatch: {name}",
        )

        require(
            np.isclose(
                np.max(np.abs(cmd)),
                peak,
                rtol=1e-10,
                atol=1e-12,
            ),
            f"Command peak mismatch: {name}",
        )

    return {
        "command_L1": float(budget),
        "command_peak": float(peak),
    }


def reconstruct(row, scenes, p80, p74, p77):

    sid, band, mode, axis, level = case_key(row)

    original = scenes[sid][band]

    if level == "M0":

        require(
            axis == "matched",
            "Unexpected M0 condition",
        )

        scene = original

        effective = p80["true_effective"](
            p74, mode, scene
        )

    else:

        scene, effective = p80[
            "make_stress_scene"
        ](
            p74,
            p77,
            original,
            mode,
            axis,
            level,
        )

    # Private identity check only.
    require(
        np.allclose(
            effective,
            row["true_effective_PRIVATE"],
            rtol=1e-9,
            atol=1e-9,
        ),
        "Reconstructed physics mismatch",
    )

    return scene


def noise_seed(row, action):

    # Omit axis and level deliberately:
    # stress conditions share paired action noise.

    sid, band, mode, _, _ = case_key(row)

    identity = (
        f"{ACTION_NOISE_SEED}|"
        f"{sid}|{band}|{mode}|{action}"
    )

    value = hashlib.sha256(
        identity.encode("utf-8")
    ).digest()

    return int.from_bytes(
        value[:8], "little"
    )


def make_probe(
    row, scene, action,
    sim, api, commands, sigma
):

    model = sim["make_model"](scene)

    passive, active, _, _ = sim["simulate"](
        model,
        scene,
        row["mode_PRIVATE"],
        commands[action],
    )

    passive = np.asarray(
        passive, dtype=float
    )

    active = np.asarray(
        active, dtype=float
    )

    require(
        passive.shape == active.shape
        and passive.shape == (51, 2),
        "Unexpected observation shape",
    )

    rng = np.random.default_rng(
        noise_seed(row, action)
    )

    noisy_passive = (
        passive +
        rng.normal(0, sigma, passive.shape)
    )

    noisy_active = (
        active +
        rng.normal(0, sigma, active.shape)
    )

    return api["RevisionProbe"](
        noisy_passive,
        noisy_active,
    )


def candidate_scores(row, probes, p75):

    old = row["fits_PRIVATE"]

    scores = {
        "MID": {
            domain: {
                law: float(
                    old[domain][law][
                        "validation_rms_mm"
                    ]
                )
                for law in LAWS
            }
            for domain in ("wide", "outer_box")
        }
    }

    for action, probe in probes.items():

        only = {
            domain: {}
            for domain in ("wide", "outer_box")
        }

        pooled = {
            domain: {}
            for domain in ("wide", "outer_box")
        }

        for domain in only:
            for law in LAWS:

                params = old[domain][law][
                    "params_PRIVATE"
                ]

                predicted = probe.predict(
                    law, params
                )

                rms = float(
                    p75["rms_mm"](
                        predicted,
                        probe.y_b,
                    )
                )

                require(
                    np.isfinite(rms) and rms >= 0,
                    "Invalid challenge RMS",
                )

                old_rms = scores["MID"][domain][law]

                # Both actions have the same
                # observation shape and length.
                pooled_rms = float(np.sqrt(
                    (old_rms**2 + rms**2) / 2
                ))

                only[domain][law] = rms
                pooled[domain][law] = pooled_rms

        name = action.upper()

        scores[name + "_ONLY"] = only
        scores["MID_PLUS_" + name] = pooled

    require(
        set(scores) == set(METHODS),
        "Incomplete method matrix",
    )

    return scores


def choose(scores, config):

    values = {}

    for i, law in enumerate(LAWS):

        domain = (
            "outer_box"
            if config[i] == "O"
            else "wide"
        )

        values[law] = scores[domain][law]

    return min(
        LAWS,
        key=lambda law: values[law],
    )


def evaluate_row(
    row, scenes,
    p80, p74, p75, p77,
    sim, api, commands
):

    scene = reconstruct(
        row, scenes, p80, p74, p77
    )

    probes = {
        action: make_probe(
            row,
            scene,
            action,
            sim,
            api,
            commands,
            float(p77["SIGMA"]),
        )
        for action in ACTIONS
    }

    scores = candidate_scores(
        row, probes, p75
    )

    truth = TRUE_LAW[row["mode_PRIVATE"]]

    decisions = {}

    for method in METHODS:

        decisions[method] = {}

        for config in CONFIGS:

            predicted = choose(
                scores[method],
                config,
            )

            decisions[method][config] = {
                "winner": predicted,
                "correct_PRIVATE":
                    predicted == truth,
            }

            if method == "MID":

                require(
                    predicted ==
                    row["policies"][config]["winner"],
                    "Historical Mid decision mismatch",
                )

    return {
        "scene_id": row["scene_id"],
        "band": row["band"],
        "mode_PRIVATE": row["mode_PRIVATE"],
        "axis_PRIVATE": row["axis_PRIVATE"],
        "level_PRIVATE": row["level_PRIVATE"],

        "nested_solver_violations":
            row["nested_solver_violations"],

        "decisions": decisions,

        "scores_PRIVATE": scores,
    }


def correct(row, method, config):

    return int(
        row["decisions"][method][config][
            "correct_PRIVATE"
        ]
    )


def accuracy(rows, method, config):

    require(bool(rows), "Empty accuracy group")

    return float(np.mean([
        correct(row, method, config)
        for row in rows
    ]))


def recall(rows, method, config, mode, level=None):

    subset = [
        row for row in rows
        if row["mode_PRIVATE"] == mode
        and (
            level is None
            or row["level_PRIVATE"] == level
        )
    ]

    return accuracy(
        subset, method, config
    )


def macro(rows, method, config):

    return float(np.mean([
        recall(rows, method, config, mode)
        for mode in TRUE_LAW
    ]))


def summarize(rows):

    output = {}

    for method in METHODS:

        output[method] = {}

        for config in CONFIGS:

            output[method][config] = {
                "n": len(rows),

                "accuracy":
                    accuracy(rows, method, config),

                "macro_recall":
                    macro(rows, method, config),

                "elastic_M2_recall":
                    recall(
                        rows, method, config,
                        "elastic", "M2",
                    ),

                "nonlinear_M2_recall":
                    recall(
                        rows, method, config,
                        "nonlinear", "M2",
                    ),

                "by_mode": {
                    mode: recall(
                        rows, method, config, mode
                    )
                    for mode in TRUE_LAW
                },
            }

    return output


def contrasts(rows):

    high = "MID_PLUS_HIGH"
    biphasic = "MID_PLUS_BIPHASIC_HIGH"

    return {
        "primary_OOO_elastic_M2_biphasic_minus_high":
            recall(
                rows, biphasic, "OOO",
                "elastic", "M2",
            )
            -
            recall(
                rows, high, "OOO",
                "elastic", "M2",
            ),

        "secondary_OOO_nonlinear_M2_biphasic_minus_high":
            recall(
                rows, biphasic, "OOO",
                "nonlinear", "M2",
            )
            -
            recall(
                rows, high, "OOO",
                "nonlinear", "M2",
            ),

        "secondary_OOO_macro_biphasic_minus_high":
            macro(rows, biphasic, "OOO")
            -
            macro(rows, high, "OOO"),

        "secondary_WOO_macro_biphasic_minus_high":
            macro(rows, biphasic, "WOO")
            -
            macro(rows, high, "WOO"),

        "diagnostic_OOO_accuracy_biphasic_minus_mid":
            accuracy(rows, biphasic, "OOO")
            -
            accuracy(rows, "MID", "OOO"),
    }


def bootstrap(rows):

    groups = defaultdict(list)

    for row in rows:
        groups[row["scene_id"]].append(row)

    ids = sorted(groups)

    require(
        len(ids) == N_SCENES,
        "Expected 24 scene clusters",
    )

    rng = np.random.default_rng(
        BOOT_SEED
    )

    distributions = defaultdict(list)

    for _ in range(N_BOOT):

        selected = rng.choice(
            ids,
            size=len(ids),
            replace=True,
        )

        sample = [
            row
            for sid in selected
            for row in groups[int(sid)]
        ]

        result = contrasts(sample)

        for name, value in result.items():
            distributions[name].append(value)

    return {
        "cluster_unit": "base_scene_id",
        "n_clusters": len(ids),
        "replications": N_BOOT,

        "paired_95CI": {
            name: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for name, values in distributions.items()
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
    print("P4-M2D0 — ACTION DISCRIMINATION AUDIT")
    print("=" * 72)

    print("Smoke:", args.smoke)
    print("No fitting. No E4.")

    print("\n[1] Frozen P4-M2C2E provenance")

    reference, frozen_rows = verify_inputs()

    p80 = run_path(str(P80))

    p80["check_provenance"]()

    (
        p74, p75, p77, p79,
        src, sim, api, commands
    ) = p80["initialize"]()

    budgets = verify_actions(commands)

    require(
        np.isclose(p77["SIGMA"], 0.005),
        "Observation noise mismatch",
    )

    print("[PASS] Matched command budgets")
    print("[PASS] Frozen model implementations")

    print("\n[2] Reconstructing frozen scenes")

    scenes = src["make_scenes"](
        sim,
        SCENE_SEED,
        N_SCENES,
    )

    require(
        len(scenes) == N_SCENES,
        "Scene count mismatch",
    )

    if args.smoke:

        selected = [
            row for row in frozen_rows
            if row["scene_id"] == 0
        ]

    else:
        selected = frozen_rows

    print("\n[3] New held-out action observations")

    records = []

    for i, row in enumerate(selected, 1):

        records.append(
            evaluate_row(
                row,
                scenes,
                p80,
                p74,
                p75,
                p77,
                sim,
                api,
                commands,
            )
        )

        if i % 16 == 0:

            print(
                f"Completed {i}/{len(selected)} cases"
            )

    require(
        len(records) ==
        (16 if args.smoke else 384),
        "Unexpected record count",
    )

    print("\n[4] Results")

    summary = summarize(records)

    compared = contrasts(records)

    for name, value in compared.items():

        print(
            name,
            f"{value:+.6f}",
        )

    print("\nKey configurations")

    for method in (
        "MID",
        "MID_PLUS_HIGH",
        "MID_PLUS_BIPHASIC_HIGH",
    ):

        for config in ("WWW", "WOO", "OOO"):

            item = summary[method][config]

            print(
                f"{method} / {config}: "
                f"Acc={item['accuracy']:.4f} "
                f"Macro={item['macro_recall']:.4f} "
                f"Elastic_M2={item['elastic_M2_recall']:.4f}"
            )

    ci = (
        None if args.smoke
        else bootstrap(records)
    )

    print("\n[5] Saving")

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke else "fresh_action"
    )

    report_path = OUT / (
        f"p4m2d0_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2d0_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P4-M2D0 matched-budget "
            "action discrimination audit",

        "smoke": args.smoke,

        "n_cases": len(records),
        "same_frozen_scenes": True,
        "new_action_observations": True,

        "actions": ACTIONS,
        "command_budgets": budgets,

        "summary": summary,
        "contrasts": compared,
        "bootstrap": ci,

        "E4_accessed": False,
        "new_fits": 0,
        "thresholds_tuned": False,
        "truth_used_for_selection": False,

        "source_sha256": {
            P80.name: sha(P80),
            P86.name: sha(P86),
            REFERENCE_REPORT.name:
                sha(REFERENCE_REPORT),
            REFERENCE_AUDIT.name:
                sha(REFERENCE_AUDIT),
            Path(__file__).name:
                sha(Path(__file__)),
        },

        "limitations": [
            "Same 24 physical scenes as P4-M2C2E",
            "New observations, not independent scenes",
            "Fixed fitted parameters",
            "Original Mid action reused",
            "Command budget is not mechanical work",
            "No adaptive action selection",
            "No global optimizer certificate",
            "No E4",
        ],
    }

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
            records,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\nP4-M2D0 COMPLETE")
    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()
