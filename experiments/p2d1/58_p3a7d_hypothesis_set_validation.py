
"""
P3-A7D: Calibrated Structural Hypothesis Sets.

Calibration: new known-law scenes, separately
            for each law and gap band.
Test:        different new scenes, including
            unseen Saturating mechanism.

Online evidence:
    E1, E2, Weak, Strong -> parameter fitting
    Vmid -> construct hypothesis set

PRIVATE diagnostic:
    Vreverse -> independent predictive audit

No E4. No test tuning. No true-law input
to the online set decision.

This is a fixed-action set-valued pilot,
NOT an active or cost-saving policy.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path
from math import ceil

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
OUT = ROOT / "p3a7d"

HISTORIC = (
    ROOT / "p3a7b"
    / "p3a7b_fresh_report.json"
)

CAL_SEED = 20270410
TEST_SEED = 20270411
CAL_NOISE = 2027041001
TEST_NOISE = 2027041101

N_CAL = 24
N_TEST = 36
NFEV = 60
ALPHA = 0.10

BANDS = (
    "original",
    "above_original_range",
)

KNOWN = (
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear",
)

MODES = KNOWN + ("saturating",)

TRUTH_TO_LAW = {
    "independent": "independent",
    "rigid": "rigid",
    "elastic": "elastic",
    "deadzone": "deadzone",
    "nonlinear": "cubic",
}

TRAIN = ("e1", "e2", "weak", "strong")


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def init():
    base = run_path(str(
        HERE /
        "55_p3a7a_evidence_gated_escalation.py"
    ))

    src, _, sim, api = (
        base["init_modules"]()
    )

    pulse = sim["make_profiles"]()["pulse"]

    commands = {
        "e1": pulse * 0.25,
        "e2": pulse * 0.50,
        "weak": pulse * 0.75,
        "strong": pulse * 2.25,
        "vmid": pulse * 1.75,
        "vreverse": pulse * -1.75,
    }

    return src, sim, api, commands


def make_probe(
    src, sim, api, model, scene, mode,
    command, seed,
):
    obs, work = src["get_probe"](
        sim, api, model, scene,
        mode, command, seed
    )

    return obs, float(work)


def candidate_set(scores, cutoffs):
    """
    Receives only observable candidate scores
    and calibration cutoffs. No truth labels.
    """
    return sorted([
        law
        for law, score in scores.items()
        if score <= cutoffs[law]
    ])


def run_episode(
    src, sim, api, commands,
    scene, sid, mode, band,
    noise_seed, cutoffs=None,
    do_reverse=False,
):
    model = sim["make_model"](scene)

    mode_idx = src["MODES"].index(mode)
    band_idx = BANDS.index(band)

    base_seed = (
        noise_seed
        + sid * 1000
        + mode_idx * 20
        + band_idx * 100000
    )

    probes = {}
    works = {}

    for i, action in enumerate(TRAIN):
        probes[action], works[action] = (
            make_probe(
                src, sim, api,
                model, scene, mode,
                commands[action],
                base_seed + i,
            )
        )

    # Crucial: every law is refitted on
    # the same four observed interventions.
    _, fits = api["fit_library"](
        [probes[a] for a in TRAIN],
        src["LAWS"],
        NFEV,
    )

    # First challenge: not used for fitting.
    vmid, works["vmid"] = make_probe(
        src, sim, api,
        model, scene, mode,
        commands["vmid"],
        base_seed + 4,
    )

    vmid_scores = {
        law: float(
            api["prediction_rms"](
                vmid, fits[law]
            )
        )
        for law in src["LAWS"]
    }

    if not np.isfinite(
        list(vmid_scores.values())
    ).all():
        raise RuntimeError(
            "Nonfinite Vmid residual"
        )

    result = {
        "scene_id": sid,
        "band": band,
        "truth_PRIVATE": mode,
        "vmid_scores_m": vmid_scores,
        "vmid_best_law": min(
            vmid_scores,
            key=vmid_scores.get
        ),
        "online_work_J": sum(
            works[a]
            for a in (*TRAIN, "vmid")
        ),
    }

    # The set is frozen BEFORE the
    # independent Vreverse is observed.
    if cutoffs is not None:
        subset = candidate_set(
            vmid_scores, cutoffs
        )

        result["set"] = subset
        result["set_size"] = len(subset)

        if len(subset) == 0:
            result["decision"] = (
                "unresolved_none"
            )
        elif len(subset) == 1:
            result["decision"] = subset[0]
        else:
            result["decision"] = "ambiguous"

    if do_reverse:
        reverse, reverse_work = make_probe(
            src, sim, api,
            model, scene, mode,
            commands["vreverse"],
            base_seed + 5,
        )

        result["reverse_work_J_PRIVATE"] = (
            reverse_work
        )

        result["reverse_scores_m_PRIVATE"] = {
            law: float(
                api["prediction_rms"](
                    reverse, fits[law]
                )
            )
            for law in src["LAWS"]
        }

        if not np.isfinite(
            list(
                result[
                    "reverse_scores_m_PRIVATE"
                ].values()
            )
        ).all():
            raise RuntimeError(
                "Nonfinite reverse residual"
            )

    return result


def finite_sample_cutoff(scores):
    """
    Split-calibration rank:
        k = ceil((n + 1)(1 - alpha))

    The per-law inclusion interpretation
    assumes exchangeable calibration/test
    scores in that law and band.
    """
    ordered = sorted(float(x) for x in scores)
    n = len(ordered)

    rank = ceil((n + 1) * (1 - ALPHA))

    if rank > n:
        raise RuntimeError(
            "Calibration size too small"
        )

    return ordered[rank - 1]


def calibrate(
    src, sim, api, commands,
    scenes, noise_seed,
):
    collected = {
        band: {
            law: [] for law in src["LAWS"]
        }
        for band in BANDS
    }

    for sid, scene_set in enumerate(scenes):
        for band in BANDS:
            scene = scene_set[band]

            for mode in KNOWN:
                episode = run_episode(
                    src, sim, api, commands,
                    scene, sid, mode, band,
                    noise_seed,
                    cutoffs=None,
                    do_reverse=False,
                )

                true_law = TRUTH_TO_LAW[mode]

                collected[band][true_law].append(
                    episode["vmid_scores_m"][
                        true_law
                    ]
                )

    cutoffs = {}

    for band in BANDS:
        cutoffs[band] = {}

        print("\nCALIBRATION BAND:", band)

        for law in src["LAWS"]:
            values = collected[band][law]

            if len(values) != len(scenes):
                raise RuntimeError(
                    "Calibration count mismatch"
                )

            cutoff = finite_sample_cutoff(
                values
            )

            cutoffs[band][law] = cutoff

            print(
                " ", law,
                "q =",
                round(1000 * cutoff, 4),
                "mm"
            )

    return cutoffs


def evaluate(rows):
    output = {}

    for mode in MODES:
        subset = [
            r for r in rows
            if r["truth_PRIVATE"] == mode
        ]

        counts = Counter()
        best_correct = 0
        reverse_errors = []

        for row in subset:
            chosen = row["set"]
            size = len(chosen)

            if size == 0:
                counts["none"] += 1

            elif size == 1:
                counts["singleton"] += 1

            else:
                counts["ambiguous"] += 1

            if mode == "saturating":
                if size == 1:
                    counts["unknown_false_singleton"] += 1
            else:
                true_law = TRUTH_TO_LAW[mode]

                best_correct += int(
                    row["vmid_best_law"]
                    == true_law
                )

                if true_law in chosen:
                    counts["contains_true"] += 1

                if size == 1:
                    if chosen[0] == true_law:
                        counts["correct_singleton"] += 1
                    else:
                        counts["wrong_singleton"] += 1

            if size == 1:
                reverse_errors.append(
                    1000 * row[
                        "reverse_scores_m_PRIVATE"
                    ][chosen[0]]
                )

        item = {
            "n": len(subset),
            "counts": dict(counts),
            "median_singleton_reverse_RMS_mm": (
                float(np.median(reverse_errors))
                if reverse_errors else None
            ),
        }

        if mode != "saturating":
            item["forced_vmid_top1_correct"] = (
                best_correct
            )

        output[mode] = item

        print("\nMODE:", mode)
        print("  Cases:", len(subset))
        print("  Outcomes:", dict(counts))

        print(
            "  Unique-set Vreverse RMS median:",
            item[
                "median_singleton_reverse_RMS_mm"
            ],
            "mm"
        )

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    historic = load_json(HISTORIC)

    if historic["smoke"]:
        raise RuntimeError(
            "Formal source provenance required"
        )

    for name, expected in (
        historic["source_sha256"].items()
    ):
        if sha256(HERE / name) != expected:
            raise RuntimeError(
                "Source hash mismatch: "
                + name
            )

    src, sim, api, commands = init()

    offset = 100000 if args.smoke else 0

    # 9 calibration scenes allow a finite
    # order statistic even in smoke.
    n_cal = 9 if args.smoke else N_CAL
    n_test = 2 if args.smoke else N_TEST

    calibration_scenes = src["make_scenes"](
        sim, CAL_SEED + offset, n_cal
    )

    test_scenes = src["make_scenes"](
        sim, TEST_SEED + offset, n_test
    )

    print(
        "\nP3-A7D CALIBRATED "
        "HYPOTHESIS-SET AUDIT"
    )

    print(
        "Calibration scenes:", n_cal,
        "| Test scenes:", n_test
    )

    print(
        "Nominal alpha:", ALPHA
    )

    cutoffs = calibrate(
        src, sim, api, commands,
        calibration_scenes,
        CAL_NOISE + offset,
    )

    summaries = {}
    audits = {}

    for band in BANDS:
        records = []

        for sid, scene_set in enumerate(
            test_scenes
        ):
            for mode in MODES:
                records.append(
                    run_episode(
                        src, sim, api,
                        commands,
                        scene_set[band],
                        sid, mode, band,
                        TEST_NOISE + offset,
                        cutoffs=cutoffs[band],
                        do_reverse=True,
                    )
                )

        if len(records) != (
            n_test * len(MODES)
        ):
            raise RuntimeError(
                "Test matrix incomplete"
            )

        print("\n" + "=" * 68)
        print("BAND:", band)
        print("=" * 68)

        summaries[band] = evaluate(
            records
        )

        audits[band] = records

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "fresh"
    )

    report_path = OUT / (
        f"p3a7d_{suffix}_report.json"
    )

    private_path = OUT / (
        f"p3a7d_{suffix}_audit_PRIVATE.json"
    )

    report = {
        "protocol":
            "P3-A7D calibrated hypothesis sets",
        "smoke": args.smoke,
        "calibration_seed": CAL_SEED + offset,
        "test_seed": TEST_SEED + offset,
        "calibration_scenes": n_cal,
        "test_scenes": n_test,
        "alpha": ALPHA,
        "cutoffs_m": cutoffs,
        "summary": summaries,
        "E4_accessed": False,
        "test_tuning": False,
        "online_uses_truth": False,
        "online_uses_reverse": False,
        "parameter_refit_for_all_laws": True,
        "limitations": [
            "Fixed actions; not adaptive",
            "MuJoCo simulation only",
            "Only one unknown mechanism",
            "Model-set coverage does not "
            "guarantee unknown rejection",
            "Multiple accepted laws are "
            "not a correct unique identification",
            "Calibration conditional on "
            "known mechanism and gap band",
            "Finite-sample interpretation "
            "requires score exchangeability",
            "Vreverse diagnostic only",
            "Two bands share physical scenes",
            "No RGB perception or E4",
        ],
        "source_sha256": {
            name: sha256(HERE / name)
            for name in (
                "16_p2d1b_probe_diversity.py",
                "22_p3a1b_structural_revision.py",
                "27_p3a3b_evidence_vs_structure.py",
                "47_p3a6h_six_mechanism_openworld_audit.py",
                "55_p3a7a_evidence_gated_escalation.py",
                "57_p3a7c_parameter_structure_attribution.py",
            )
        },
    }

    report_path.write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )

    private_path.write_text(
        json.dumps(
            audits, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A7D HYPOTHESIS-SET "
        "AUDIT COMPLETE"
    )
    print(
        "No E4, no oracle online, "
        "no test threshold tuning."
    )
    print("Saved:", report_path)
    print("Saved:", private_path)


if __name__ == "__main__":
    main()
