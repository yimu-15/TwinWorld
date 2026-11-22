
"""
P4-M2D2B — Frozen-model amplitude dose-response.

24 new base scenes; 384 cases.

All laws are fitted on the original four training
actions only. Two NEW held-out probes use the same
HIGH waveform at 0.5x and 1.0x amplitude.

The probes have equal sample counts, equal noise
variance and common random noise. Neither is used
for fitting, tuning or parameter selection.

PRIMARY:
WOO nonlinear-M2 recall under MID+FULL versus
MID+HALF, paired over new scenes.

SECONDARY:
Elastic-Cubic predictive separation and score
margin under the two amplitude levels.

No E4. No adaptive policy. No threshold tuning.
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
OUT = RESULTS / "p4m2d2b"

P80 = HERE / "80_p4m2c0_prior_mismatch_stress.py"
P88 = HERE / "88_p4m2d1_fresh_action_prior_interaction.py"

R88 = (
    RESULTS / "p4m2d1" /
    "p4m2d1_fresh_report.json"
)

R89 = (
    RESULTS / "p4m2d2a" /
    "p4m2d2a_fresh_report.json"
)

LAWS = ("elastic", "deadzone", "cubic")
CONFIGS = ("WWW", "OOO", "WOO")

TRUTH = {
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

METHODS = (
    "MID",
    "HALF_ONLY",
    "FULL_ONLY",
    "MID_PLUS_HALF",
    "MID_PLUS_FULL",
)

SCENE_SEED = 20291117
FIT_NOISE_SEED = 2029111701
ACTION_NOISE_SEED = 2029111702
BOOT_SEED = 2029111703

N_SCENES = 24
SMOKE_OFFSET = 100000
BOOT_N = 2000


def must(flag, message):
    if not flag:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def read(path):
    must(
        path.is_file(),
        f"Missing required input: {path}"
    )
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def stable_seed(sid, band, mode):
    message = (
        f"{ACTION_NOISE_SEED}|{sid}|{band}|"
        f"{mode}|amplitude_pair"
    )
    raw = hashlib.sha256(
        message.encode("utf-8")
    ).digest()

    return int.from_bytes(raw[:8], "little")


def verify_sources():
    r88 = read(R88)
    r89 = read(R89)

    must(
        r88["smoke"] is False
        and r89["smoke"] is False,
        "Formal reports required"
    )
    must(
        r88["n_cases"] == 384
        and r89["n_cases"] == 384,
        "History case-count mismatch"
    )
    must(
        r88["E4_accessed"] is False
        and r89["E4_accessed"] is False,
        "Historical E4 flag"
    )
    must(
        r89["new_fits"] == 0
        and r89["new_simulations"] == 0,
        "P89 should be read-only"
    )

    must(
        digest(R88) ==
        r89["source_sha256"][R88.name],
        "R88 hash mismatch"
    )
    must(
        digest(P88) ==
        r89["source_sha256"][P88.name],
        "P88 hash mismatch"
    )
    must(
        digest(P80) ==
        r88["source_sha256"][P80.name],
        "P80 hash mismatch"
    )

    return r88, r89


def verify_new_scenes(
    p79, src, sim, scenes, old, smoke
):
    fingerprints = p79[
        "fingerprint_scenes"
    ](scenes)

    must(
        len(fingerprints) == 2 * len(scenes)
        and len(set(fingerprints)) == len(fingerprints),
        "Duplicate scene fingerprints"
    )

    old_fps = set(
        old["scene_independence"][
            "new_scene_fingerprints"
        ]
    )

    must(
        not set(fingerprints) & old_fps,
        "Overlap with P88 scenes"
    )

    p79["verify_scene_independence"](
        src, sim, scenes
    )

    if not smoke:
        smoke_scenes = src["make_scenes"](
            sim,
            SCENE_SEED + SMOKE_OFFSET,
            1,
        )
        smoke_fps = set(
            p79["fingerprint_scenes"](
                smoke_scenes
            )
        )
        must(
            not set(fingerprints) & smoke_fps,
            "Smoke/formal overlap"
        )

    return {
        "n_fingerprints": len(fingerprints),
        "vs_p88_exact_overlap": 0,
        "fingerprints": fingerprints,
    }


def pair_probes(
    scene, sid, band, mode,
    sim, api, command, sigma
):
    clean = {}

    for dose, multiplier in (
        ("HALF", 0.5),
        ("FULL", 1.0),
    ):
        model = sim["make_model"](scene)

        passive, active, _, _ = sim[
            "simulate"
        ](
            model, scene, mode,
            command * multiplier
        )

        pa = np.asarray(
            passive, dtype=float
        )
        ac = np.asarray(
            active, dtype=float
        )

        must(
            pa.shape == ac.shape == (51, 2),
            f"Unexpected observation shape: {dose}"
        )

        clean[dose] = (pa, ac)

    # Common random numbers across amplitudes.
    rng = np.random.default_rng(
        stable_seed(sid, band, mode)
    )

    pa_noise = rng.normal(
        0, sigma, (51, 2)
    )
    ac_noise = rng.normal(
        0, sigma, (51, 2)
    )

    return {
        dose: api["RevisionProbe"](
            pair[0] + pa_noise,
            pair[1] + ac_noise,
        )
        for dose, pair in clean.items()
    }


def domain(config, i):
    return (
        "outer_box"
        if config[i] == "O"
        else "wide"
    )


def method_rms(fits, held, config, method):
    vals = {}

    for i, law in enumerate(LAWS):
        d = domain(config, i)

        mid = float(
            fits[d][law][
                "validation_rms_mm"
            ]
        )

        if method == "MID":
            value = mid

        elif method == "HALF_ONLY":
            value = held["HALF"][d][law]

        elif method == "FULL_ONLY":
            value = held["FULL"][d][law]

        elif method == "MID_PLUS_HALF":
            value = float(np.sqrt(
                (
                    mid**2 +
                    held["HALF"][d][law]**2
                ) / 2.0
            ))

        elif method == "MID_PLUS_FULL":
            value = float(np.sqrt(
                (
                    mid**2 +
                    held["FULL"][d][law]**2
                ) / 2.0
            ))

        else:
            raise ValueError(method)

        vals[law] = float(value)

    return vals


def evaluate(
    p80, p74, p75, p77,
    sim, api, commands,
    base, sid, band, mode,
    axis, level, fit_seed, high_command
):
    if level == "M0":
        must(
            axis == "matched",
            "M0 axis mismatch"
        )
        scene = base
    else:
        scene, _ = p80[
            "make_stress_scene"
        ](
            p74, p77, base,
            mode, axis, level
        )

    # Original training protocol, performed once.
    fits = p80["evaluate_case"](
        p75, p77, sim, api, commands,
        scene, sid, band, mode, fit_seed
    )

    for d in ("wide", "outer_box"):
        for law in LAWS:
            f = fits[d][law]
            if (
                not f.get("valid", False)
                or not f.get("converged", False)
                or f.get("params_PRIVATE") is None
                or f.get("validation_rms_mm") is None
            ):
                print(
                    "\n[FIT DIAGNOSTIC]",
                    f"scene={sid}",
                    f"band={band}",
                    f"mode={mode}",
                    f"axis={axis}",
                    f"level={level}",
                    f"domain={d}",
                    f"law={law}",
                    flush=True,
                )
                print(
                    "valid =", f.get("valid"),
                    "converged =", f.get("converged"),
                    "train_mse =", f.get("train_mse"),
                    "validation_rms_mm =",
                    f.get("validation_rms_mm"),
                    "nfev =", f.get("nfev"),
                    "params =", f.get("params_PRIVATE"),
                    flush=True,
                )
            must(
                f["valid"]
                and f["converged"]
                and f["params_PRIVATE"] is not None
                and np.isfinite(
                    f["validation_rms_mm"]
                ),
                (
                    f"Incomplete fit: "
                    f"{sid}/{band}/{mode}/"
                    f"{axis}/{law}/{d}"
                )
            )

    # No held-out action contributes to fitting.
    probes = pair_probes(
        scene, sid, band, mode,
        sim, api, high_command,
        float(p77["SIGMA"])
    )

    held = {}
    predicted = {}

    for dose, probe in probes.items():
        held[dose] = {
            "wide": {},
            "outer_box": {},
        }
        predicted[dose] = {
            "wide": {},
            "outer_box": {},
        }

        for d in ("wide", "outer_box"):
            for law in LAWS:
                params = fits[d][law][
                    "params_PRIVATE"
                ]

                pred = np.asarray(
                    probe.predict(law, params),
                    dtype=float
                )

                must(
                    pred.shape == probe.y_b.shape,
                    "Prediction shape mismatch"
                )

                held[dose][d][law] = float(
                    p75["rms_mm"](
                        pred, probe.y_b
                    )
                )

                predicted[dose][d][law] = pred

    separation = {}
    decisions = {}
    score_gap = {}

    true_law = TRUTH[mode]

    for config in CONFIGS:
        ed = domain(config, 0)
        cd = domain(config, 2)

        separation[config] = {
            dose: float(
                p75["rms_mm"](
                    predicted[dose][ed]["elastic"],
                    predicted[dose][cd]["cubic"],
                )
            )
            for dose in ("HALF", "FULL")
        }

        decisions[config] = {}
        score_gap[config] = {}

        for method in METHODS:
            vals = method_rms(
                fits, held, config, method
            )

            selection = min(
                LAWS,
                key=lambda law: vals[law]
            )

            decisions[config][method] = {
                "winner": selection,
                "correct_PRIVATE":
                    selection == true_law,
            }

            score_gap[config][method] = float(
                vals["elastic"] - vals["cubic"]
            )

    must(
        decisions["WWW"]["MID"]["winner"]
        == p80["winner"](fits, "wide")
        and decisions["OOO"]["MID"]["winner"]
        == p80["winner"](fits, "outer_box"),
        "Frozen Mid decision mismatch"
    )

    nested = [
        law
        for law in LAWS
        if fits["outer_box"][law]["train_mse"]
        <
        fits["wide"][law]["train_mse"] - 1e-6
    ]

    return {
        "scene_id": sid,
        "band": band,
        "mode_PRIVATE": mode,
        "axis_PRIVATE": axis,
        "level_PRIVATE": level,

        "nested_violations_PRIVATE": nested,

        "decisions": decisions,
        "separation_mm": separation,
        "score_gap_mm_PRIVATE": score_gap,
        "heldout_rms_PRIVATE": held,
    }


def correct(row, config, method):
    return int(
        row["decisions"][config][method][
            "correct_PRIVATE"
        ]
    )


def accuracy(
    rows, config, method,
    mode=None, level=None
):
    selected = [
        row for row in rows
        if (
            mode is None
            or row["mode_PRIVATE"] == mode
        )
        and (
            level is None
            or row["level_PRIVATE"] == level
        )
    ]

    must(
        len(selected) > 0,
        "Empty evaluation group"
    )

    return float(np.mean([
        correct(row, config, method)
        for row in selected
    ]))


def effects(rows):
    nonlinear_m2 = [
        row for row in rows
        if row["mode_PRIVATE"] == "nonlinear"
        and row["level_PRIVATE"] == "M2"
    ]

    elastic_m2 = [
        row for row in rows
        if row["mode_PRIVATE"] == "elastic"
        and row["level_PRIVATE"] == "M2"
    ]

    must(
        nonlinear_m2 and elastic_m2,
        "Missing diagnostic strata"
    )

    separation_difference = [
        row["separation_mm"]["WOO"]["FULL"]
        -
        row["separation_mm"]["WOO"]["HALF"]
        for row in nonlinear_m2
    ]

    return {
        "PRIMARY_WOO_nonlinear_M2_recall_full_minus_half":
            accuracy(
                rows, "WOO",
                "MID_PLUS_FULL", "nonlinear", "M2"
            )
            -
            accuracy(
                rows, "WOO",
                "MID_PLUS_HALF", "nonlinear", "M2"
            ),

        "SECONDARY_WOO_nonlinear_M2_pairwise_sep_full_minus_half_mm":
            float(np.median(
                separation_difference
            )),

        "SECONDARY_WOO_nonlinear_M2_score_gap_full_minus_half_mm":
            float(np.median([
                row["score_gap_mm_PRIVATE"][
                    "WOO"
                ]["MID_PLUS_FULL"]
                -
                row["score_gap_mm_PRIVATE"][
                    "WOO"
                ]["MID_PLUS_HALF"]
                for row in nonlinear_m2
            ])),

        "SECONDARY_WOO_accuracy_full_minus_half":
            accuracy(
                rows, "WOO", "MID_PLUS_FULL"
            )
            -
            accuracy(
                rows, "WOO", "MID_PLUS_HALF"
            ),

        "DIAGNOSTIC_WOO_elastic_M2_recall_full_minus_half":
            accuracy(
                rows, "WOO",
                "MID_PLUS_FULL", "elastic", "M2"
            )
            -
            accuracy(
                rows, "WOO",
                "MID_PLUS_HALF", "elastic", "M2"
            ),

        "DIAGNOSTIC_OOO_accuracy_full_minus_half":
            accuracy(
                rows, "OOO", "MID_PLUS_FULL"
            )
            -
            accuracy(
                rows, "OOO", "MID_PLUS_HALF"
            ),
    }


def cluster_bootstrap(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[
            row["scene_id"]
        ].append(row)

    ids = sorted(groups)

    must(
        len(ids) == N_SCENES,
        "Scene cluster count mismatch"
    )

    rng = np.random.default_rng(
        BOOT_SEED
    )

    distributions = defaultdict(list)

    for _ in range(BOOT_N):
        picked = rng.choice(
            ids,
            size=len(ids),
            replace=True
        )

        sample = [
            row
            for sid in picked
            for row in groups[int(sid)]
        ]

        for name, value in effects(
            sample
        ).items():
            distributions[name].append(value)

    return {
        "cluster_unit": "base_scene_id",
        "n_clusters": N_SCENES,
        "seed": BOOT_SEED,
        "n_resamples": BOOT_N,

        "95CI": {
            key: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for key, values
            in distributions.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke",
        action="store_true"
    )
    args = ap.parse_args()

    print("=" * 72)
    print(
        "P4-M2D2B — AMPLITUDE DOSE-RESPONSE AUDIT"
    )
    print("=" * 72)

    print(
        "Smoke:", args.smoke,
        "| E4 sealed | Frozen fitting"
    )

    r88, r89 = verify_sources()

    p80 = run_path(str(P80))
    p80["check_provenance"]()

    (
        p74, p75, p77, p79,
        src, sim, api, commands
    ) = p80["initialize"]()

    must(
        tuple(p77["TRAIN"])
        == ("e1", "e2", "weak", "strong")
        and p77["VALID"] == "mid",
        "Unexpected fit or validation actions"
    )

    must(
        np.isclose(p77["SIGMA"], 0.005),
        "Noise sigma mismatch"
    )

    high_command = np.asarray(
        commands["high"],
        dtype=float
    )

    must(
        high_command.ndim == 1
        and len(high_command) >= 8,
        "Invalid HIGH command"
    )

    n = (
        1 if args.smoke
        else N_SCENES
    )

    seed = SCENE_SEED + (
        SMOKE_OFFSET if args.smoke else 0
    )

    fit_seed = FIT_NOISE_SEED + (
        SMOKE_OFFSET if args.smoke else 0
    )

    scenes = src["make_scenes"](
        sim, seed, n
    )

    independence = verify_new_scenes(
        p79, src, sim, scenes,
        r88, args.smoke
    )

    print(
        "[PASS] Fresh scenes:", n,
        "| fingerprints:",
        independence["n_fingerprints"]
    )

    print(
        "[ACTION] HALF peak:",
        float(np.max(np.abs(
            high_command * 0.5
        ))),
        "| FULL peak:",
        float(np.max(np.abs(
            high_command
        )))
    )

    rows = []

    for sid, scene_set in enumerate(scenes):
        for band in (
            "original",
            "above_original_range",
        ):
            for mode, conditions in CONDITIONS.items():
                for axis, level in conditions:
                    rows.append(
                        evaluate(
                            p80, p74, p75, p77,
                            sim, api, commands,
                            scene_set[band],
                            sid, band, mode,
                            axis, level,
                            fit_seed, high_command
                        )
                    )

        print(
            f"Completed {sid + 1}/{n} "
            "base scenes"
        )

    must(
        len(rows) == 16 * n,
        "Case matrix size mismatch"
    )

    metrics = effects(rows)

    print("\nPREDECLARED TESTS")

    for name, value in metrics.items():
        print(
            f"{name}: {value:+.6f}"
        )

    print("\nCONFIGURATION CHECKS")

    for cfg in CONFIGS:
        for method in (
            "MID",
            "MID_PLUS_HALF",
            "MID_PLUS_FULL",
        ):
            hits = sum(
                correct(row, cfg, method)
                for row in rows
            )

            print(
                f"{cfg}/{method}: "
                f"{hits}/{len(rows)} "
                f"acc={accuracy(rows, cfg, method):.4f}"
            )

    nested = dict(Counter(
        law
        for row in rows
        for law in row[
            "nested_violations_PRIVATE"
        ]
    ))

    print(
        "Nested solver gaps:", nested
    )

    bootstrap = (
        None if args.smoke
        else cluster_bootstrap(rows)
    )

    report = {
        "protocol":
            "P4-M2D2B frozen-model "
            "amplitude dose response",

        "smoke": args.smoke,

        "scene_seed": seed,
        "fit_noise_seed": fit_seed,
        "action_noise_seed":
            ACTION_NOISE_SEED,

        "n_base_scenes": n,
        "n_cases": len(rows),

        "action_amplitudes": {
            "half": 0.5,
            "full": 1.0,
        },

        "same_waveform": True,
        "paired_noise": True,
        "new_action_observations": True,

        "source_provenance": {
            P80.name: digest(P80),
            P88.name: digest(P88),
            R88.name: digest(R88),
            R89.name: digest(R89),
            Path(__file__).name:
                digest(Path(__file__)),
        },

        "scene_independence":
            independence,

        "predeclared_effects": metrics,
        "bootstrap": bootstrap,

        "nested_solver_violations":
            nested,

        "E4_accessed": False,
        "action_used_for_fit": False,
        "threshold_tuning": False,

        "limitations": [
            "Same simulator family, new scenes only",
            "Half/full differ in actuation amplitude",
            "Candidate separation is not accuracy",
            "No structural identifiability proof",
            "Finite-start SLSQP remains uncorrected",
            "Truth used only for offline diagnostics",
            "No E4",
        ],
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report_path = OUT / (
        f"p4m2d2b_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2d2b_{suffix}_audit_PRIVATE.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8"
    )

    audit_path.write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8"
    )

    print("Saved:", report_path)
    print("Saved:", audit_path)
    print("P4-M2D2B COMPLETE")


if __name__ == "__main__":
    main()
