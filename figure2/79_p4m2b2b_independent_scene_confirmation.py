"""
P4-M2B2B — Independent Scene Confirmation

Confirm the matched-solver parameter-domain
ablation on genuinely new physical scenes.

Domains:
    WIDE
    OUTER_BOX
    SHARED_MASS

Frozen from P4-M2B2A:
    Same objective
    Same initial points
    Same SLSQP solver
    Same optimizer budget
    Same physical parameter domains
    Same training observations
    Same validation observation

New independent evaluation:
    72 base scenes
    Original and above-original parameter bands
    Independent scene/noise seeds

Primary endpoint:
    True-law winner accuracy among
    Elastic / Deadzone / Nonlinear.

Secondary endpoints:
    Candidate validation RMS
    Optimizer convergence
    Feasibility
    Paired domain differences

Statistics:
    Paired scene-cluster bootstrap.

No E4.
No threshold tuning.
No future action.
No truth in fitting.
No modification of historical scripts.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np


# =========================================================
# 1. Paths and frozen protocol
# =========================================================

HERE = Path(__file__).resolve().parent

P68 = HERE / (
    "68_p3a8c1_temporal_action_feasibility.py"
)

P74 = HERE / (
    "74_p4m2a_generator_inference_fidelity.py"
)

P75 = HERE / (
    "75_p4m2b0_physical_domain_fit_pilot.py"
)

P76 = HERE / (
    "76_p4m2b1_band_blind_prior_audit.py"
)

P77 = HERE / (
    "77_p4m2b2a_matched_domain_fit.py"
)

OUT = HERE / "results" / "p4m2b2b"


SCENE_SEED = 20290417
NOISE_SEED = 2029041701

N_EVAL = 72

N_BOOTSTRAP = 2000

BOOTSTRAP_SEED = 2029041702

MIN_PRACTICAL_GAIN = 0.05

MIN_STRICT_COVERAGE = 0.90

BANDS = (
    "original",
    "above_original_range",
)


# Historical scene generators/seeds.
# These are checked for exact parameter-dictionary
# overlap before the new evaluation begins.

PRIOR_SCENE_SEEDS = (
    (20280521, 72, "P3-A8C2B"),
    (20290111, 24, "P4-M1A-development"),
    (20290112, 72, "P4-M1A-evaluation"),
    (20290201, 8, "P4-M1B0"),
    (20290317, 12, "P4-M2A-M2B-development"),
)


# =========================================================
# 2. Utilities and provenance
# =========================================================

def sha256_file(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def sha256_bytes(value):

    return hashlib.sha256(value).hexdigest()


def read_json(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"Required report missing: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def verify_report_sources(
    report_path,
    required_sources,
    label,
):

    report = read_json(report_path)

    if report.get("smoke") is not False:
        raise RuntimeError(
            f"{label}: formal historical report required"
        )

    recorded = report.get(
        "source_sha256", {}
    )

    for filename in required_sources:

        expected = recorded.get(filename)

        if expected is None:
            raise RuntimeError(
                f"{label}: missing SHA record for "
                f"{filename}"
            )

        source = HERE / filename

        if not source.is_file():
            raise FileNotFoundError(source)

        actual = sha256_file(source)

        if actual != expected:
            raise RuntimeError(
                f"{label}: source SHA mismatch: "
                f"{filename}"
            )

    print(
        f"[PROVENANCE PASS] {label}"
    )

    return report


def verify_historical_chain():

    r_m1b0 = verify_report_sources(
        HERE / "results" / "p4m1b0"
        / "p4m1b0_fresh_report.json",
        [
            P68.name,
        ],
        "P4-M1B0",
    )

    r_m2a = verify_report_sources(
        HERE / "results" / "p4m2a"
        / "p4m2a_dev_report.json",
        [
            P68.name,
            P74.name,
        ],
        "P4-M2A",
    )

    r_m2b0 = verify_report_sources(
        HERE / "results" / "p4m2b0"
        / "p4m2b0_dev_report.json",
        [
            P74.name,
            P75.name,
        ],
        "P4-M2B0",
    )

    r_m2b1 = verify_report_sources(
        HERE / "results" / "p4m2b1"
        / "p4m2b1_dev_report.json",
        [
            P74.name,
            P75.name,
            P76.name,
        ],
        "P4-M2B1",
    )

    r_m2b2a = verify_report_sources(
        HERE / "results" / "p4m2b2a"
        / "p4m2b2a_dev_report.json",
        [
            P74.name,
            P75.name,
            P77.name,
        ],
        "P4-M2B2A",
    )

    # Check that P4-M2B2A really used
    # the frozen configuration.

    expected = {
        "n_scenes": 12,
        "scene_seed": 20290317,
        "noise_seed": 2029031701,
        "solver": "SLSQP",
        "maxiter_per_start": 150,
        "starts_per_fit": 2,
    }

    for key, value in expected.items():

        if r_m2b2a.get(key) != value:
            raise RuntimeError(
                f"P4-M2B2A protocol mismatch: {key}"
            )

    if r_m2b2a.get("domains") != [
        "wide",
        "outer_box",
        "shared_mass",
    ]:
        raise RuntimeError(
            "Unexpected P4-M2B2A domains"
        )

    if not np.allclose(
        r_m2b2a["band_blind_gap_range"],
        [0.015, 0.18],
    ):
        raise RuntimeError(
            "P4-M2B2A gap range mismatch"
        )

    return {
        "P4-M1B0": r_m1b0,
        "P4-M2A": r_m2a,
        "P4-M2B0": r_m2b0,
        "P4-M2B1": r_m2b1,
        "P4-M2B2A": r_m2b2a,
    }


# =========================================================
# 3. Scene fingerprint and independence audit
# =========================================================

def json_default(value):

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    raise TypeError(
        f"Unsupported JSON type: {type(value)}"
    )


def scene_fingerprint(scene):

    payload = json.dumps(
        scene,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    )

    return sha256_bytes(
        payload.encode("utf-8")
    )


def fingerprint_scenes(scenes):

    hashes = []

    for scene_set in scenes:

        for band in BANDS:

            if band not in scene_set:
                raise RuntimeError(
                    f"Missing scene band: {band}"
                )

            hashes.append(
                scene_fingerprint(
                    scene_set[band]
                )
            )

    return hashes


def verify_scene_independence(
    src,
    sim,
    new_scenes,
):

    new_hashes = fingerprint_scenes(
        new_scenes
    )

    if len(set(new_hashes)) != len(new_hashes):
        raise RuntimeError(
            "Duplicate scenes detected in new evaluation"
        )

    previous_hashes = set()

    checks = []

    for seed, count, label in PRIOR_SCENE_SEEDS:

        previous = src["make_scenes"](
            sim, seed, count
        )

        old_hashes = fingerprint_scenes(
            previous
        )

        overlap = (
            set(old_hashes)
            & set(new_hashes)
        )

        if overlap:
            raise RuntimeError(
                f"Scene overlap detected with {label}: "
                f"{len(overlap)}"
            )

        previous_hashes.update(
            old_hashes
        )

        checks.append({
            "label": label,
            "seed": seed,
            "n_base_scenes": count,
            "n_band_fingerprints": len(old_hashes),
            "overlap": 0,
        })

        print(
            f"[SCENE INDEPENDENCE PASS] "
            f"{label}: 0 exact overlaps"
        )

    return {
        "new_scene_fingerprints": new_hashes,
        "new_fingerprint_count": len(new_hashes),
        "historical_checks": checks,
        "historical_unique_fingerprints":
            len(previous_hashes),
        "exact_overlap_count": 0,
    }


# =========================================================
# 4. Load frozen fitting implementation
# =========================================================

def initialize():

    p74 = run_path(str(P74))
    p75 = run_path(str(P75))
    p77 = run_path(str(P77))

    src, sim, api, commands = (
        p74["initialize"]()
    )

    # Check exact agreement between the
    # P4-M2B0 and P4-M2B2A protocols.

    if tuple(p75["TRAIN"]) != tuple(
        p77["TRAIN"]
    ):
        raise RuntimeError(
            "Training action mismatch"
        )

    if p75["VALID"] != p77["VALID"]:
        raise RuntimeError(
            "Validation action mismatch"
        )

    if not np.isclose(
        p75["SIGMA"], p77["SIGMA"]
    ):
        raise RuntimeError(
            "Noise sigma mismatch"
        )

    if tuple(p77["DOMAINS"]) != (
        "wide",
        "outer_box",
        "shared_mass",
    ):
        raise RuntimeError(
            "Unexpected fitting domains"
        )

    if p77["MAXITER"] != 150:
        raise RuntimeError(
            "Unexpected SLSQP iteration budget"
        )

    if len(p77["initial_points"]("deadzone")) != 2:
        raise RuntimeError(
            "Unexpected optimizer start count"
        )

    for action in (
        "e1",
        "e2",
        "weak",
        "strong",
        "mid",
    ):

        if action not in commands:
            raise RuntimeError(
                f"Missing command: {action}"
            )

    return p74, p75, p77, src, sim, api, commands


# =========================================================
# 5. Run one independent scene
# =========================================================

def run_case(
    p75,
    p77,
    sim,
    api,
    commands,
    scene,
    scene_id,
    band,
    mode,
    noise_seed,
):

    probes = p75["create_noisy_probes"](
        sim,
        api,
        commands,
        scene,
        scene_id,
        band,
        mode,
        noise_seed,
    )

    training = [
        probes[name]
        for name in p77["TRAIN"]
    ]

    validation = probes[
        p77["VALID"]
    ]

    laws = tuple(p77["LAWS"])
    domains = tuple(p77["DOMAINS"])

    results = {}

    # Truth is used only to generate the
    # physical world. It is never passed
    # to the fitting function.

    for domain in domains:

        results[domain] = {}

        for law in laws:

            fitted = p77["fit_domain"](
                training,
                law,
                domain,
            )

            valid = bool(
                fitted["valid"]
            )

            converged = bool(
                fitted["converged"]
            )

            validation_rms = None

            if valid:

                prediction = np.asarray(
                    validation.predict(
                        law,
                        fitted["params_PRIVATE"],
                    ),
                    dtype=float,
                )

                if not np.isfinite(
                    prediction
                ).all():
                    raise RuntimeError(
                        "Nonfinite validation prediction"
                    )

                validation_rms = float(
                    p75["rms_mm"](
                        prediction,
                        validation.y_b,
                    )
                )

                if not np.isfinite(
                    validation_rms
                ):
                    raise RuntimeError(
                        "Nonfinite validation RMS"
                    )

            results[domain][law] = {
                "valid": valid,
                "converged": converged,
                "train_mse": fitted["train_mse"],
                "nfev": fitted["nfev"],
                "params_PRIVATE":
                    fitted["params_PRIVATE"],
                "validation_rms_mm":
                    validation_rms,
            }

    return {
        "scene_id": int(scene_id),
        "band": band,
        "truth_PRIVATE": mode,
        "fits": results,
    }


# =========================================================
# 6. Model selection and summary
# =========================================================

def select_winner(
    row,
    domain,
    strict,
    laws,
):

    fits = row["fits"][domain]

    for law in laws:

        item = fits[law]

        if not item["valid"]:
            return None

        if item["validation_rms_mm"] is None:
            return None

        if strict and not item["converged"]:
            return None

    return min(
        laws,
        key=lambda law:
            fits[law]["validation_rms_mm"],
    )


def summarize(
    audits,
    modes,
    laws,
    domains,
    truth_to_law,
):

    output = {}

    for band, rows in audits.items():

        output[band] = {}

        for mode in modes:

            subset = [
                r for r in rows
                if r["truth_PRIVATE"] == mode
            ]

            output[band][mode] = {}

            true_law = truth_to_law.get(mode)

            for domain in domains:

                candidate_stats = {}

                for law in laws:

                    items = [
                        r["fits"][domain][law]
                        for r in subset
                    ]

                    finite = [
                        x["validation_rms_mm"]
                        for x in items
                        if (
                            x["valid"]
                            and x["validation_rms_mm"]
                            is not None
                        )
                    ]

                    converged = [
                        x["validation_rms_mm"]
                        for x in items
                        if (
                            x["valid"]
                            and x["converged"]
                            and x["validation_rms_mm"]
                            is not None
                        )
                    ]

                    candidate_stats[law] = {
                        "n_valid": len(finite),
                        "n_converged": len(converged),
                        "median_rms_mm": (
                            float(np.median(finite))
                            if finite else None
                        ),
                        "p90_rms_mm": (
                            float(np.quantile(
                                finite, 0.90
                            ))
                            if finite else None
                        ),
                        "median_converged_rms_mm": (
                            float(np.median(converged))
                            if converged else None
                        ),
                    }

                finite_winners = [
                    select_winner(
                        row, domain, False, laws
                    )
                    for row in subset
                ]

                strict_winners = [
                    select_winner(
                        row, domain, True, laws
                    )
                    for row in subset
                ]

                finite_valid = [
                    x for x in finite_winners
                    if x is not None
                ]

                strict_valid = [
                    x for x in strict_winners
                    if x is not None
                ]

                finite_correct = None
                strict_correct = None

                if true_law is not None:

                    finite_correct = sum(
                        x == true_law
                        for x in finite_valid
                    )

                    strict_correct = sum(
                        x == true_law
                        for x in strict_valid
                    )

                output[band][mode][domain] = {
                    "n_scenes": len(subset),
                    "candidate_stats":
                        candidate_stats,

                    "finite_fit_coverage": (
                        len(finite_valid)
                        / len(subset)
                        if subset else None
                    ),

                    "strict_convergence_coverage": (
                        len(strict_valid)
                        / len(subset)
                        if subset else None
                    ),

                    "finite_winner_count":
                        dict(Counter(finite_valid)),

                    "strict_winner_count":
                        dict(Counter(strict_valid)),

                    "true_law": true_law,

                    "finite_correct_winner_count":
                        finite_correct,

                    "finite_correct_winner_rate": (
                        finite_correct
                        / len(finite_valid)
                        if (
                            true_law is not None
                            and finite_valid
                        )
                        else None
                    ),

                    "strict_correct_winner_count":
                        strict_correct,

                    "strict_correct_winner_rate": (
                        strict_correct
                        / len(strict_valid)
                        if (
                            true_law is not None
                            and strict_valid
                        )
                        else None
                    ),
                }

    return output


# =========================================================
# 7. Paired scene-cluster bootstrap
# =========================================================

def correctness(
    row,
    domain,
    strict,
    laws,
    truth_to_law,
):

    true_law = truth_to_law.get(
        row["truth_PRIVATE"]
    )

    if true_law is None:
        return None

    winner = select_winner(
        row,
        domain,
        strict,
        laws,
    )

    if winner is None:
        return None

    return int(
        winner == true_law
    )


def paired_bootstrap(
    rows,
    domain_a,
    domain_b,
    laws,
    truth_to_law,
    expected_n,
    strict,
    seed,
    repetitions,
):

    groups = {}

    for row in rows:

        score_a = correctness(
            row,
            domain_a,
            strict,
            laws,
            truth_to_law,
        )

        score_b = correctness(
            row,
            domain_b,
            strict,
            laws,
            truth_to_law,
        )

        if score_a is None or score_b is None:
            continue

        groups.setdefault(
            row["scene_id"], []
        ).append(
            (score_a, score_b)
        )

    n_pairs = sum(
        len(group)
        for group in groups.values()
    )

    n_clusters = len(groups)

    coverage = (
        n_pairs / expected_n
        if expected_n else None
    )

    if n_clusters < 10:

        return {
            "status": "insufficient_clusters",
            "domain_a": domain_a,
            "domain_b": domain_b,
            "strict_convergence_required": strict,
            "n_pairs": n_pairs,
            "n_clusters": n_clusters,
            "coverage": coverage,
        }

    values_a = [
        a
        for group in groups.values()
        for a, b in group
    ]

    values_b = [
        b
        for group in groups.values()
        for a, b in group
    ]

    accuracy_a = float(
        np.mean(values_a)
    )

    accuracy_b = float(
        np.mean(values_b)
    )

    delta = accuracy_a - accuracy_b

    rng = np.random.default_rng(seed)

    cluster_ids = np.array(
        sorted(groups.keys())
    )

    bootstrap_deltas = []

    for _ in range(repetitions):

        sampled = rng.choice(
            cluster_ids,
            size=len(cluster_ids),
            replace=True,
        )

        sample_a = []
        sample_b = []

        for scene_id in sampled:

            for a, b in groups[int(scene_id)]:
                sample_a.append(a)
                sample_b.append(b)

        if sample_a:

            bootstrap_deltas.append(
                np.mean(sample_a)
                - np.mean(sample_b)
            )

    interval = [
        float(np.quantile(
            bootstrap_deltas, 0.025
        )),
        float(np.quantile(
            bootstrap_deltas, 0.975
        )),
    ]

    return {
        "status": "ok",
        "domain_a": domain_a,
        "domain_b": domain_b,
        "delta_definition":
            "accuracy(domain_a)-accuracy(domain_b)",

        "strict_convergence_required": strict,

        "n_pairs": n_pairs,
        "n_clusters": n_clusters,
        "expected_pairs": expected_n,
        "coverage": float(coverage),

        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "delta_accuracy": float(delta),

        "delta_95CI": interval,

        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "cluster_unit": "base_scene_id",
    }


def classify_comparison(result):

    if result.get("status") != "ok":
        return "INCONCLUSIVE_INSUFFICIENT_CLUSTERS"

    if result["coverage"] < MIN_STRICT_COVERAGE:
        return "INCONCLUSIVE_LOW_CONVERGENCE_COVERAGE"

    delta = result["delta_accuracy"]

    lower, upper = result["delta_95CI"]

    if (
        lower > 0
        and delta >= MIN_PRACTICAL_GAIN
    ):
        return "SUPPORTED_PRACTICAL_GAIN"

    if upper < 0:
        return "EVIDENCE_OF_HARM"

    if lower > 0:
        return "POSITIVE_BUT_BELOW_PRACTICAL_GAIN"

    return "NOT_CONFIRMED"


def run_comparisons(
    audits,
    n_scenes,
    laws,
    truth_to_law,
    smoke,
):

    all_rows = [
        row
        for band_rows in audits.values()
        for row in band_rows
    ]

    scopes = {
        "original": [
            r for r in all_rows
            if r["band"] == "original"
        ],
        "above_original_range": [
            r for r in all_rows
            if r["band"] == "above_original_range"
        ],
        "pooled": all_rows,
    }

    pairs = {
        "outer_box_minus_wide": (
            "outer_box",
            "wide",
        ),
        "shared_mass_minus_outer_box": (
            "shared_mass",
            "outer_box",
        ),
        "shared_mass_minus_wide": (
            "shared_mass",
            "wide",
        ),
    }

    output = {}

    repetitions = (
        0 if smoke else N_BOOTSTRAP
    )

    for scope_id, rows in scopes.items():

        output[scope_id] = {}

        expected = (
            n_scenes
            * len(truth_to_law)
            * (
                2 if scope_id == "pooled"
                else 1
            )
        )

        for pair_id, (
            domain_a,
            domain_b,
        ) in pairs.items():

            item = {}

            for strict in (True, False):

                mode = (
                    "strict"
                    if strict else "finite_feasible"
                )

                result = paired_bootstrap(
                    rows,
                    domain_a,
                    domain_b,
                    laws,
                    truth_to_law,
                    expected,
                    strict,
                    BOOTSTRAP_SEED
                    + len(output) * 100
                    + len(item) * 10
                    + len(pair_id),
                    repetitions,
                )

                item[mode] = result

            item["strict_decision"] = (
                classify_comparison(
                    item["strict"]
                )
            )

            output[scope_id][pair_id] = item

    return output


# =========================================================
# 8. Main experiment
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    print("=" * 76)
    print(
        "P4-M2B2B — INDEPENDENT SCENE CONFIRMATION"
    )
    print("=" * 76)

    print("\n[1] Verifying historical provenance")

    verify_historical_chain()

    print("\n[2] Loading frozen implementations")

    (
        p74,
        p75,
        p77,
        src,
        sim,
        api,
        commands,
    ) = initialize()

    modes = tuple(p75["MODES"])
    laws = tuple(p77["LAWS"])
    domains = tuple(p77["DOMAINS"])

    truth_to_law = dict(
        p77["TRUTH_TO_LAW"]
    )

    n_scenes = (
        1 if args.smoke
        else N_EVAL
    )

    offset = (
        100000 if args.smoke
        else 0
    )

    scene_seed = SCENE_SEED + offset
    noise_seed = NOISE_SEED + offset

    print("\n[3] Generating new scenes")

    scenes = src["make_scenes"](
        sim,
        scene_seed,
        n_scenes,
    )

    if len(scenes) != n_scenes:
        raise RuntimeError(
            "Unexpected generated scene count"
        )

    print(
        "New base scenes:",
        n_scenes,
    )

    print(
        "Scene seed:",
        scene_seed,
    )

    print(
        "Noise seed:",
        noise_seed,
    )

    print("\n[4] Exact scene fingerprint audit")

    scene_audit = verify_scene_independence(
        src,
        sim,
        scenes,
    )

    print("\n[5] Running matched-domain fits")

    audits = {
        band: []
        for band in BANDS
    }

    for sid, scene_set in enumerate(scenes):

        for band in BANDS:

            scene = scene_set[band]

            for mode in modes:

                row = run_case(
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

                audits[band].append(row)

        if (
            (sid + 1) % 12 == 0
            or sid + 1 == n_scenes
        ):

            print(
                f"Completed scenes: "
                f"{sid + 1}/{n_scenes}"
            )

    print("\n[6] Summarizing candidate selection")

    summaries = summarize(
        audits,
        modes,
        laws,
        domains,
        truth_to_law,
    )

    for band in BANDS:

        print("\nBAND:", band)

        for mode in modes:

            print("\n MODE:", mode)

            for domain in domains:

                item = summaries[
                    band
                ][mode][domain]

                print(
                    f"  {domain:12s}",
                    "coverage=",
                    round(
                        item["strict_convergence_coverage"],
                        4,
                    ),
                    "correct=",
                    item["strict_correct_winner_count"],
                    "/",
                    sum(
                        item["strict_convergence_coverage"]
                        is not None
                        for _ in [0]
                    ) * (
                        item["n_scenes"]
                    )
                    if item["strict_correct_winner_count"]
                    is not None
                    else "N/A",
                )

    print("\n[7] Paired scene-cluster bootstrap")

    comparisons = run_comparisons(
        audits,
        n_scenes,
        laws,
        truth_to_law,
        args.smoke,
    )

    # Bootstrap repetitions used in this run.
    repetitions = (
        0 if args.smoke else N_BOOTSTRAP
    )

    for scope, scope_results in comparisons.items():

        print("\nSCOPE:", scope)

        for name, item in scope_results.items():

            strict = item["strict"]

            print(
                name,
                "| decision:",
                item["strict_decision"],
            )

            if strict.get("status") == "ok":

                print(
                    "  delta:",
                    round(
                        strict["delta_accuracy"],
                        4,
                    ),
                    "| 95% CI:",
                    strict["delta_95CI"],
                    "| coverage:",
                    round(
                        strict["coverage"],
                        4,
                    ),
                )

    # =====================================================
    # Save reports
    # =====================================================

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke"
        if args.smoke
        else "fresh"
    )

    report_path = OUT / (
        f"p4m2b2b_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p4m2b2b_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P4-M2B2B independent scene confirmation",

        "smoke": args.smoke,

        "n_scenes": n_scenes,

        "scene_seed": scene_seed,
        "noise_seed": noise_seed,

        "bands": list(BANDS),
        "modes": list(modes),
        "candidate_laws": list(laws),
        "domains": list(domains),

        "training_actions": list(
            p77["TRAIN"]
        ),

        "validation_action": p77["VALID"],

        "noise_sigma_m": float(
            p77["SIGMA"]
        ),

        "solver": "SLSQP",

        "maxiter_per_start": int(
            p77["MAXITER"]
        ),

        "starts_per_fit": len(
            p77["initial_points"]("deadzone")
        ),

        "scene_independence_audit":
            scene_audit,

        "summary": summaries,

        "paired_comparisons": comparisons,

        "decision_rules": {
            "primary_endpoint":
                "strict-convergence true-law winner accuracy",

            "minimum_practical_gain":
                MIN_PRACTICAL_GAIN,

            "minimum_strict_coverage":
                MIN_STRICT_COVERAGE,

            "bootstrap_repetitions":
                repetitions,

            "cluster_unit":
                "base_scene_id",

            "primary_comparisons": [
                "outer_box_minus_wide",
                "shared_mass_minus_outer_box",
            ],
        },

        "independent_evaluation":
            not args.smoke,

        "truth_used_in_fitting": False,

        "test_threshold_tuning": False,

        "E4_accessed": False,

        "future_action_used": False,

        "historical_results_modified": False,

        "source_sha256": {
            P68.name: sha256_file(P68),
            P74.name: sha256_file(P74),
            P75.name: sha256_file(P75),
            P76.name: sha256_file(P76),
            P77.name: sha256_file(P77),
            Path(__file__).name:
                sha256_file(Path(__file__)),
        },

        "limitations": [
            "Synthetic MuJoCo evaluation",
            "Finite local SLSQP optimization",
            "Mid is the only validation action",
            "Candidate models are Elastic, Deadzone and Cubic",
            "Independent and Rigid are diagnostic non-candidate modes",
            "Physical priors remain dependent on generator assumptions",
            "No real robot hardware validation",
            "No E4 evaluation",
            "No global optimization certificate",
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
            audits,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("P4-M2B2B COMPLETE")
    print("=" * 76)

    print("Saved:", report_path)
    print("Saved:", audit_path)


if __name__ == "__main__":
    main()