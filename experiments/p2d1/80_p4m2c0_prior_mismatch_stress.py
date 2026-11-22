"""
P4-M2C0 — Physical-Prior Mismatch Stress Test

Scientific question:
Does a physically constrained fitting domain remain
useful when the actual system violates that prior?

Compare:
    WIDE
    OUTER_BOX

Stress axes:
    alpha_high : stiffness-to-mass mismatch
    gap_high   : deadzone threshold mismatch
    gamma_high : nonlinear coefficient mismatch

Matched control:
    M0, original generated physical parameters

Mismatch levels:
    M1 / M2 / M3, fixed in advance

Protocol:
    12 new development scenes
    Two physical bands
    Same four training actions
    Same Mid validation action
    Same SLSQP objective, starts and iteration limit
    Paired noise within each scene/mode/band
    Truth used only for generation and offline audit

No E4.
No new optimizer design.
No prior-bound adaptation.
No validation-threshold tuning.
No modification of historical source files.
No independent generalization claim.
"""

import argparse
import hashlib
import json
from pathlib import Path
from runpy import run_path
from collections import defaultdict

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p4m2c0"

P74 = HERE / "74_p4m2a_generator_inference_fidelity.py"
P75 = HERE / "75_p4m2b0_physical_domain_fit_pilot.py"
P77 = HERE / "77_p4m2b2a_matched_domain_fit.py"
P79 = HERE / "79_p4m2b2b_independent_scene_confirmation.py"

REFERENCE = (
    HERE / "results" / "p4m2b2b"
    / "p4m2b2b_fresh_report.json"
)

SCENE_SEED = 20290517
NOISE_SEED = 2029051701
N_DEV = 12

BANDS = (
    "original",
    "above_original_range",
)

MODES = (
    "elastic",
    "deadzone",
    "nonlinear",
)

LAWS = (
    "elastic",
    "deadzone",
    "cubic",
)

DOMAINS = (
    "wide",
    "outer_box",
)

TRUTH = {
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

AXES = {
    "elastic": ("alpha_high",),
    "deadzone": ("alpha_high", "gap_high"),
    "nonlinear": ("alpha_high", "gamma_high"),
}

# These stress levels are fixed before running.
# They are not selected using validation outcomes.

LEVELS = {
    "alpha_high": {
        "M1": 0.02,
        "M2": 0.10,
        "M3": 0.25,
    },
    "gap_high": {
        "M1": 0.01,
        "M2": 0.05,
        "M3": 0.10,
    },
    "gamma_high": {
        "M1": 0.02,
        "M2": 0.10,
        "M3": 0.25,
    },
}

BOOTSTRAP_SEED = 2029051702
N_BOOTSTRAP = 2000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def check_provenance():
    """
    The previous formal evaluation is used only
    to verify source identity and scene separation.

    Its scores are never used for fitting.
    """

    report = read_json(REFERENCE)

    if report.get("smoke") is not False:
        raise RuntimeError(
            "Formal P4-M2B2B report required"
        )

    if report.get("scene_seed") != 20290417:
        raise RuntimeError(
            "Unexpected historical scene seed"
        )

    if report.get("n_scenes") != 72:
        raise RuntimeError(
            "Unexpected historical scene count"
        )

    for name in (
        P74.name,
        P75.name,
        P77.name,
        P79.name,
    ):
        expected = report["source_sha256"].get(name)

        if expected is None:
            raise RuntimeError(
                "Missing historical source hash: " + name
            )

        actual = sha(HERE / name)

        if actual != expected:
            raise RuntimeError(
                "Historical source mismatch: " + name
            )

    return report


def initialize():
    p74 = run_path(str(P74))
    p75 = run_path(str(P75))
    p77 = run_path(str(P77))
    p79 = run_path(str(P79))

    src, sim, api, commands = p74["initialize"]()

    if tuple(p77["LAWS"]) != LAWS:
        raise RuntimeError("Candidate-law mismatch")

    if tuple(p77["TRAIN"]) != (
        "e1", "e2", "weak", "strong"
    ):
        raise RuntimeError("Training-action mismatch")

    if p77["VALID"] != "mid":
        raise RuntimeError("Validation-action mismatch")

    if p77["MAXITER"] != 150:
        raise RuntimeError("Solver-budget mismatch")

    if len(p77["initial_points"]("elastic")) != 2:
        raise RuntimeError("Optimizer-start mismatch")

    if not np.isclose(p77["SIGMA"], 0.005):
        raise RuntimeError("Noise mismatch")

    return p74, p75, p77, p79, src, sim, api, commands


def check_scenes(src, sim, p79, scenes, report):
    """
    Exact fingerprint overlap check.
    This cannot prove distributional independence,
    but detects reuse of identical scene parameters.
    """

    fingerprints = p79["fingerprint_scenes"](scenes)

    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("Duplicate development scenes")

    old = set(
        report["scene_independence_audit"][
            "new_scene_fingerprints"
        ]
    )

    if old.intersection(fingerprints):
        raise RuntimeError(
            "Overlap with P4-M2B2B evaluation"
        )

    historical = p79["verify_scene_independence"](
        src, sim, scenes
    )

    return {
        "new_fingerprints": fingerprints,
        "overlap_with_p4m2b2b": 0,
        "historical_audit": historical,
    }


def true_effective(p74, mode, scene):
    """
    Private oracle parameters for audit only.
    Never passed to fit_domain.
    """
    item = p74["truth_fit"](mode, scene)

    if item["kind"] != TRUTH[mode]:
        raise RuntimeError(
            "Unexpected true law"
        )

    return np.asarray(
        item["params"], dtype=float
    )


def within_domain(p77, mode, params, domain):

    law = TRUTH[mode]

    low, high = p77["specifications"](
        law, domain
    )

    return bool(
        np.all(params >= low - 1e-9)
        and np.all(params <= high + 1e-9)
    )


def make_stress_scene(
    p74, p77, scene, mode, axis, level
):
    """
    Construct an out-of-prior physical scene.

    The shifted true effective coefficient must:
      1. violate OUTER_BOX;
      2. remain inside WIDE;
      3. be finite and physically positive.

    The original scene is never modified.
    """

    shifted = dict(scene)

    excess = LEVELS[axis][level]

    mass_b = float(scene["mass_b"])

    if not 0.8 <= mass_b <= 1.6:
        raise RuntimeError(
            "Unexpected generated mass"
        )

    if axis == "alpha_high":

        # alpha = k / mass_B
        # Outer Box alpha upper bound = 20.

        shifted["k"] = (
            20.0 * mass_b * (1.0 + excess)
        )

        index = 0
        target = 20.0 * (1.0 + excess)

    elif axis == "gamma_high":

        if mode != "nonlinear":
            raise RuntimeError(
                "Gamma mismatch requires nonlinear mode"
            )

        # gamma = k3 / mass_B
        # Outer Box gamma upper bound = 1062.5.

        shifted["k3"] = (
            1062.5 * mass_b * (1.0 + excess)
        )

        index = 2
        target = 1062.5 * (1.0 + excess)

    elif axis == "gap_high":

        if mode != "deadzone":
            raise RuntimeError(
                "Gap mismatch requires deadzone mode"
            )

        # Outer Box gap upper bound = 0.18 m.
        # All levels remain inside Wide max = 0.20 m.

        shifted["gap"] = (
            0.18 * (1.0 + excess)
        )

        index = 2
        target = 0.18 * (1.0 + excess)

    else:
        raise ValueError(axis)

    effective = true_effective(
        p74, mode, shifted
    )

    if not np.isfinite(effective).all():
        raise RuntimeError(
            "Nonfinite true effective parameters"
        )

    if not np.isclose(
        effective[index],
        target,
        atol=1e-8,
        rtol=1e-8,
    ):
        raise RuntimeError(
            "Stress parameter construction mismatch"
        )

    if within_domain(
        p77, mode, effective, "outer_box"
    ):
        raise RuntimeError(
            "Stress did not violate Outer Box"
        )

    if not within_domain(
        p77, mode, effective, "wide"
    ):
        raise RuntimeError(
            "Stress also violates Wide"
        )

    return shifted, effective


def evaluate_case(
    p75, p77, sim, api, commands,
    scene, sid, band, mode, noise_seed
):
    """
    Both methods see exactly the same probes.
    No true parameters enter either optimizer.
    """

    probes = p75["create_noisy_probes"](
        sim,
        api,
        commands,
        scene,
        sid,
        band,
        mode,
        noise_seed,
    )

    training = [
        probes[name]
        for name in p77["TRAIN"]
    ]

    validation = probes[p77["VALID"]]

    fits = {}

    for domain in DOMAINS:

        fits[domain] = {}

        for law in LAWS:

            result = p77["fit_domain"](
                training, law, domain
            )

            valid = bool(result["valid"])
            converged = bool(result["converged"])

            rms = None

            if valid:

                prediction = validation.predict(
                    law,
                    result["params_PRIVATE"],
                )

                rms = float(
                    p75["rms_mm"](
                        prediction,
                        validation.y_b,
                    )
                )

                if not np.isfinite(rms):
                    raise RuntimeError(
                        "Nonfinite validation RMS"
                    )

            fits[domain][law] = {
                "valid": valid,
                "converged": converged,
                "train_mse": result["train_mse"],
                "validation_rms_mm": rms,
                "nfev": int(result["nfev"]),
                "params_PRIVATE":
                    result["params_PRIVATE"],
            }

    return fits


def winner(fits, domain):
    """
    Strict convergence is required.
    Missing/nonconverged candidates cause
    an unavailable decision, not silent removal.
    """

    for law in LAWS:

        item = fits[domain][law]

        if (
            not item["valid"]
            or not item["converged"]
            or item["validation_rms_mm"] is None
        ):
            return None

    return min(
        LAWS,
        key=lambda law:
            fits[domain][law]["validation_rms_mm"],
    )


def record(
    sid, band, mode, axis, level,
    effective, fits
):

    outcomes = {}

    for domain in DOMAINS:

        selected = winner(fits, domain)

        outcomes[domain] = {
            "winner": selected,
            "correct_PRIVATE": (
                selected == TRUTH[mode]
                if selected is not None else None
            ),
            "true_law_validation_rms_mm": (
                fits[domain][TRUTH[mode]][
                    "validation_rms_mm"
                ]
            ),
            "complete_convergence":
                selected is not None,
        }

    return {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "stress_axis": axis,
        "level": level,
        "true_effective_PRIVATE":
            effective.tolist(),
        "fits": fits,
        "outcomes_PRIVATE": outcomes,
    }


def summarize_group(rows):

    result = {
        "n": len(rows),
    }

    for domain in DOMAINS:

        available = [
            row for row in rows
            if row["outcomes_PRIVATE"][domain][
                "complete_convergence"
            ]
        ]

        correct = sum(
            bool(row["outcomes_PRIVATE"][domain][
                "correct_PRIVATE"
            ])
            for row in available
        )

        true_rms = [
            row["outcomes_PRIVATE"][domain][
                "true_law_validation_rms_mm"
            ]
            for row in rows
            if row["outcomes_PRIVATE"][domain][
                "true_law_validation_rms_mm"
            ] is not None
        ]

        result[domain] = {
            "n_available": len(available),
            "convergence_coverage": (
                len(available) / len(rows)
                if rows else None
            ),
            "correct_count": correct,
            "accuracy_all_cases": (
                correct / len(rows)
                if rows else None
            ),
            "accuracy_conditional": (
                correct / len(available)
                if available else None
            ),
            "true_law_median_rms_mm": (
                float(np.median(true_rms))
                if true_rms else None
            ),
        }

    a = result["outer_box"]["accuracy_all_cases"]
    b = result["wide"]["accuracy_all_cases"]

    result["outer_minus_wide"] = (
        a - b
        if a is not None and b is not None
        else None
    )

    return result


def paired_differences(rows):

    values = []

    for row in rows:

        o = row["outcomes_PRIVATE"]["outer_box"]
        w = row["outcomes_PRIVATE"]["wide"]

        # Unavailable decisions count as failures
        # for this all-cases accuracy analysis.

        outer_correct = int(
            o["correct_PRIVATE"] is True
        )
        wide_correct = int(
            w["correct_PRIVATE"] is True
        )

        values.append({
            "scene_id": row["scene_id"],
            "difference":
                outer_correct - wide_correct,
        })

    return values


def scene_bootstrap(rows, smoke):

    values = paired_differences(rows)

    if smoke or not values:
        return None

    groups = defaultdict(list)

    for item in values:
        groups[item["scene_id"]].append(
            item["difference"]
        )

    ids = sorted(groups)

    if len(ids) < 10:
        return {
            "status": "insufficient_clusters",
            "n_clusters": len(ids),
        }

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    samples = []

    for _ in range(N_BOOTSTRAP):

        selected = rng.choice(
            ids,
            size=len(ids),
            replace=True,
        )

        sample = []

        for sid in selected:
            sample.extend(groups[int(sid)])

        samples.append(
            float(np.mean(sample))
        )

    return {
        "status": "ok",
        "n_clusters": len(ids),
        "n_bootstrap": N_BOOTSTRAP,
        "cluster_unit": "base_scene_id",
        "delta_95CI": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("P4-M2C0 PRIOR-MISMATCH STRESS TEST")
    print("=" * 72)

    print("Smoke:", args.smoke)
    print("No E4.")
    print("Frozen SLSQP fitting.")
    print("Development-only evidence.")

    print("\n[1] Provenance verification")

    previous = check_provenance()

    (
        p74, p75, p77, p79,
        src, sim, api, commands
    ) = initialize()

    n = 1 if args.smoke else N_DEV

    offset = 100000 if args.smoke else 0

    scene_seed = SCENE_SEED + offset
    noise_seed = NOISE_SEED + offset

    print("\n[2] Generating development scenes")

    scenes = src["make_scenes"](
        sim, scene_seed, n
    )

    if len(scenes) != n:
        raise RuntimeError(
            "Scene count mismatch"
        )

    print("\n[3] Scene independence audit")

    scene_audit = check_scenes(
        src, sim, p79, scenes, previous
    )

    print("\n[4] Matched and mismatched fitting")

    audit = []
    matched = {}

    for sid, scene_set in enumerate(scenes):

        for band in BANDS:

            base_scene = scene_set[band]

            for mode in MODES:

                base_effective = true_effective(
                    p74, mode, base_scene
                )

                if not within_domain(
                    p77, mode, base_effective,
                    "outer_box"
                ):
                    raise RuntimeError(
                        "Matched world violates prior"
                    )

                base_fits = evaluate_case(
                    p75, p77, sim, api, commands,
                    base_scene, sid, band, mode,
                    noise_seed,
                )

                base_row = record(
                    sid, band, mode,
                    "matched", "M0",
                    base_effective, base_fits
                )

                audit.append(base_row)

                matched[
                    (sid, band, mode)
                ] = base_row

                for axis in AXES[mode]:

                    for level in ("M1", "M2", "M3"):

                        scene, effective = (
                            make_stress_scene(
                                p74, p77,
                                base_scene,
                                mode,
                                axis,
                                level,
                            )
                        )

                        fits = evaluate_case(
                            p75, p77, sim, api,
                            commands,
                            scene, sid, band, mode,
                            noise_seed,
                        )

                        audit.append(
                            record(
                                sid, band, mode,
                                axis, level,
                                effective, fits
                            )
                        )

        print(
            f"Completed base scenes: {sid+1}/{n}"
        )

    print("\n[5] Summaries")

    summary = {}

    for band in BANDS:

        summary[band] = {}

        for mode in MODES:

            summary[band][mode] = {}

            baseline = [
                r for r in audit
                if (
                    r["band"] == band
                    and r["truth_PRIVATE"] == mode
                    and r["level"] == "M0"
                )
            ]

            baseline_result = summarize_group(
                baseline
            )

            summary[band][mode]["matched"] = (
                baseline_result
            )

            print(
                f"\n[{band}] [{mode}] M0 "
                f"delta={baseline_result['outer_minus_wide']}"
            )

            for axis in AXES[mode]:

                summary[band][mode][axis] = {}

                for level in ("M1", "M2", "M3"):

                    subset = [
                        r for r in audit
                        if (
                            r["band"] == band
                            and r["truth_PRIVATE"] == mode
                            and r["stress_axis"] == axis
                            and r["level"] == level
                        )
                    ]

                    item = summarize_group(subset)

                    item["bootstrap"] = (
                        scene_bootstrap(
                            subset, args.smoke
                        )
                    )

                    item["delta_change_vs_M0"] = (
                        item["outer_minus_wide"]
                        - baseline_result[
                            "outer_minus_wide"
                        ]
                    )

                    summary[band][mode][axis][level] = (
                        item
                    )

                    print(
                        f"  {axis:12s} {level}:",
                        "Wide=",
                        item["wide"]["correct_count"],
                        "/",
                        item["n"],
                        "Outer=",
                        item["outer_box"]["correct_count"],
                        "/",
                        item["n"],
                        "Delta=",
                        round(
                            item["outer_minus_wide"], 4
                        ),
                        "Change_vs_M0=",
                        round(
                            item["delta_change_vs_M0"], 4
                        ),
                    )

    print("\n[6] Saving reports")

    OUT.mkdir(parents=True, exist_ok=True)

    suffix = (
        "smoke" if args.smoke else "dev"
    )

    report_path = OUT / (
        f"p4m2c0_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2c0_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P4-M2C0 physical prior mismatch stress",
        "smoke": args.smoke,
        "development_only": True,
        "n_base_scenes": n,
        "scene_seed": scene_seed,
        "noise_seed": noise_seed,
        "bands": BANDS,
        "modes": MODES,
        "laws": LAWS,
        "domains": DOMAINS,
        "stress_levels": LEVELS,
        "scene_audit": scene_audit,
        "summary": summary,
        "solver": "SLSQP",
        "maxiter_per_start": 150,
        "starts_per_fit": 2,
        "truth_used_in_fitting": False,
        "E4_accessed": False,
        "historical_thresholds_changed": False,
        "source_sha256": {
            P74.name: sha(P74),
            P75.name: sha(P75),
            P77.name: sha(P77),
            P79.name: sha(P79),
            Path(__file__).name:
                sha(Path(__file__)),
        },
        "limitations": [
            "Synthetic MuJoCo only",
            "Development-set diagnostic",
            "Stress parameters constructed using private generator truth",
            "Only upper-bound mismatch tested",
            "Finite local optimization",
            "No global optimality certificate",
            "No adaptive prior-mismatch detector",
            "No independent confirmation of stress-test conclusions",
            "No E4",
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
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("P4-M2C0 COMPLETE")
    print("=" * 72)

    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()