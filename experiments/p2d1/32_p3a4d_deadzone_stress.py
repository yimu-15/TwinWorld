
"""
P3-A4D: Frozen-policy deadzone activation stress audit.

The same base scenes and observation noise are reused
across three gap bands.

Policy inputs:
    E1/E2 evidence and selected E3 observations only.

Evaluation-only information:
    true mechanism, gap, E4 activation and E4 errors.

No policy retraining or threshold tuning.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a4d"

REFERENCE = (
    HERE / "results" / "p3a4c"
    / "p3a4c_prospective_report.json"
)

SEED = 20261128

KNOWN = (
    "independent",
    "rigid",
    "elastic",
)

MODES = KNOWN + (
    "deadzone",
    "nonlinear",
    "saturating",
)

POLICIES = (
    "old_stop_or_strong",
    "hypothesis_aware",
    "always_strong",
)

BANDS = (
    "original",
    "high_in_range",
    "above_original_range",
)

TESTS = (
    "pulse_2p75",
    "oscillation_2p25",
)


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def make_scene_banks(original_sampler, n):
    """
    Create paired scene banks.

    All physical scene parameters are identical
    across bands except the deadzone gap.

    In non-deadzone laws the gap is unused.
    """
    rng = np.random.default_rng(SEED + 17)

    base = [
        original_sampler(rng)
        for _ in range(n)
    ]

    banks = {}

    for band in BANDS:
        scenes = []

        for scene in base:
            s = dict(scene)

            q = np.clip(
                (s["gap"] - 0.015) / 0.105,
                0.0,
                1.0,
            )

            if band == "high_in_range":
                s["gap"] = float(
                    0.09 + 0.03 * q
                )

            elif band == "above_original_range":
                s["gap"] = float(
                    0.12 + 0.06 * q
                )

            scenes.append(s)

        banks[band] = scenes

    return banks


def sample_from_bank(bank):
    cursor = 0

    def sample(_rng):
        nonlocal cursor

        if cursor >= len(bank):
            raise RuntimeError(
                "Scene bank exhausted"
            )

        value = dict(bank[cursor])
        cursor += 1

        return value

    return sample


def activation_diagnostic(sim, profiles, scenes):
    """
    PRIVATE evaluation-only physical activation audit.

    This information must never enter the action
    selector, model fitter, or stopping criterion.

    Source 16 observes:
        A = qpos[0]
        B = 1.2 + qpos[1]

    Therefore the simulated relative coordinate is:
        r = B_observed - 1.2 - A_observed
    """
    commands = {
        "e1": profiles["pulse"] * 0.25,
        "e2": profiles["pulse"] * 0.50,
        "e4_pulse": profiles["pulse"] * 2.75,
    }

    records = {}

    for sid, scene in enumerate(scenes):
        model = sim["make_model"](scene)
        maxima = {}

        for name, command in commands.items():
            _, active, _, _ = sim["simulate"](
                model,
                scene,
                "deadzone",
                command,
            )

            relative = (
                active[:, 1]
                - 1.2
                - active[:, 0]
            )

            maxima[name] = float(
                np.max(np.abs(relative))
            )

        gap = float(scene["gap"])

        records[sid] = {
            "gap_m_PRIVATE": gap,
            "max_relative_m_PRIVATE": maxima,

            "early_activated_PRIVATE": bool(
                max(
                    maxima["e1"],
                    maxima["e2"],
                ) > gap + 1e-8
            ),

            "E4_activated_PRIVATE": bool(
                maxima["e4_pulse"]
                > gap + 1e-8
            ),
        }

    return records


def average_E4_mm(row, policy):
    errors = row["policies"][policy]["E4_m"]

    return 1000.0 * float(np.mean([
        errors[name]
        for name in TESTS
    ]))


def evaluate_band(records, activation, evaluator, seed):
    """
    Evaluate frozen policies under one gap condition.
    """
    summary = evaluator["summarize"](
        records,
        seed,
    )

    paired = {}

    for i, policy in enumerate(POLICIES[:2]):
        paired[policy] = {
            "work_minus_strong_J": evaluator[
                "bootstrap"
            ](
                records,

                lambda r: (
                    r["policies"][policy]["work_J"]
                    -
                    r["policies"]["always_strong"]
                    ["work_J"]
                ),

                seed + 50 + i,
            ),

            "E4_minus_strong_mm": evaluator[
                "bootstrap"
            ](
                records,

                lambda r: (
                    average_E4_mm(r, policy)
                    -
                    average_E4_mm(
                        r,
                        "always_strong",
                    )
                ),

                seed + 60 + i,
            ),
        }

    deadzone_rows = [
        r for r in records
        if r["truth_PRIVATE_EVAL_ONLY"]
        == "deadzone"
    ]

    # Private physical audit.
    # Count cases where the force law is hidden
    # during E1/E2 but activates under E4.
    deadzone_audit = {
        "total_deadzone": len(deadzone_rows),

        "hidden_at_E1_E2": sum(
            not activation[
                r["scene_id"]
            ]["early_activated_PRIVATE"]
            for r in deadzone_rows
        ),

        "activated_at_E4": sum(
            activation[
                r["scene_id"]
            ]["E4_activated_PRIVATE"]
            for r in deadzone_rows
        ),

        "hidden_early_but_active_E4": sum(
            (
                not activation[
                    r["scene_id"]
                ]["early_activated_PRIVATE"]
            )
            and (
                activation[
                    r["scene_id"]
                ]["E4_activated_PRIVATE"]
            )
            for r in deadzone_rows
        ),
    }

    for policy in POLICIES[:2]:
        stopped = [
            r for r in deadzone_rows
            if r["policies"][policy]["action"]
            == "none"
        ]

        activated_stops = [
            r for r in stopped
            if activation[
                r["scene_id"]
            ]["E4_activated_PRIVATE"]
        ]

        deadzone_audit[policy] = {
            "stopped_deadzone": len(stopped),

            "stopped_E4_activated_deadzone":
                len(activated_stops),

            "stopped_scene_ids_PRIVATE": [
                r["scene_id"]
                for r in stopped
            ],

            "E4_error_delta_mm_on_stopped_deadzone": (
                float(np.mean([
                    average_E4_mm(r, policy)
                    -
                    average_E4_mm(
                        r,
                        "always_strong",
                    )
                    for r in stopped
                ]))
                if stopped else None
            ),
        }

    deadzone_audit["stopping_by_policy_PRIVATE"] = {}

    for policy in POLICIES:
        stopped = [
            r for r in records
            if r["policies"][policy]["action"]
            == "none"
        ]

        deadzone_audit[
            "stopping_by_policy_PRIVATE"
        ][policy] = {
            "total": len(stopped),

            "true_modes": dict(Counter(
                r["truth_PRIVATE_EVAL_ONLY"]
                for r in stopped
            )),
        }

    return {
        "standard_metrics": summary,
        "paired_comparisons": paired,
        "deadzone_activation_PRIVATE":
            deadzone_audit,
    }


def self_test():
    fake_sampler = lambda _rng: {
        "gap": 0.0675,
        "mass_a": 1.0,
    }

    banks = make_scene_banks(
        fake_sampler,
        3,
    )

    assert np.isclose(
        banks["original"][0]["gap"],
        0.0675,
    )

    assert np.isclose(
        banks["high_in_range"][0]["gap"],
        0.105,
    )

    assert np.isclose(
        banks["above_original_range"][0]["gap"],
        0.15,
    )

    sampler = sample_from_bank(
        banks["original"]
    )

    assert sampler(None)["gap"] == 0.0675

    print(
        "P3-A4D PURE STRESS SELF-TEST PASS "
        "(not MuJoCo validation)"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--self-test-only",
        action="store_true",
    )

    args = parser.parse_args()

    self_test()

    if args.self_test_only:
        return

    if not REFERENCE.exists():
        raise FileNotFoundError(
            "Run P3-A4C first: "
            + str(REFERENCE)
        )

    prior = json.loads(
        REFERENCE.read_text(
            encoding="utf-8"
        )
    )

    assert prior["seed"] == 20261207

    # Verify frozen source provenance.
    for name, expected_hash in (
        prior["source_sha256"].items()
    ):
        if digest(HERE / name) != expected_hash:
            raise RuntimeError(
                "Frozen source changed: " + name
            )

    # Freeze known-only P3-A4C thresholds.
    # Never recalibrate against gap-stress labels.
    cuts = prior["calibration_known_only"]

    stage = run_path(str(
        HERE /
        "26_p3a3a_weak_evidence_stress.py"
    ))

    old = run_path(str(
        HERE /
        "29_p3a4a_evidence_stopping.py"
    ))

    router = run_path(str(
        HERE /
        "30_p3a4b_mechanism_aware_replay.py"
    ))

    evaluator = run_path(str(
        HERE /
        "31_p3a4c_prospective_validation.py"
    ))

    helper = run_path(str(
        HERE /
        "27_p3a3b_evidence_vs_structure.py"
    ))

    sim = run_path(str(
        HERE /
        "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE /
        "22_p3a1b_structural_revision.py"
    ))

    assert old["STOP_BIC_MARGIN"] == -10.0
    assert router["RIGID_BIC_CUTOFF"] == -10.0

    # Held-out force law used only by the simulator.
    helper["install_saturating"](sim)

    profiles = sim["make_profiles"]()
    original_sampler = sim["sample_scene"]

    n_scenes = 2 if args.smoke else 24
    nfev = 12 if args.smoke else 40

    banks = make_scene_banks(
        original_sampler,
        n_scenes,
    )

    report = {
        "study":
            "P3-A4D frozen-policy targeted gap stress",

        "status":
            "Targeted stress, not external validation",

        "seed": SEED,
        "scenes_per_band": n_scenes,

        "noise_control":
            "Same random seed and observation order",

        "base_scene_control":
            "Same scenes, changing only gap",

        "calibration":
            "Frozen P3-A4C known-only thresholds",

        "reference_seed": prior["seed"],

        "bands": {},

        "limitations": [
            "Physical activation is evaluation-only.",
            "Policies use no private truth labels.",
            "Stress groups motivated by prior failures.",
            "An unactivated deadzone may be observationally indistinguishable from independence.",
            "Candidate law library remains handcrafted.",
            "E4 B prediction conditions on observed E4 A.",
            "Inconclusive results use fallback predictions only for diagnostics.",
        ],
    }

    private_audit = {}

    for i, band in enumerate(BANDS):

        # Inject a predetermined scene bank.
        # The original sampler source is unchanged.
        sim["sample_scene"] = sample_from_bank(
            banks[band]
        )

        # Same noise seed/order across all three bands.
        rows, _ = stage["make_dataset"](
            sim,
            api,
            profiles,
            n_scenes,
            MODES,
            SEED + 222,
            nfev,
            evaluation=True,
        )

        sim["sample_scene"] = original_sampler

        # Policies make decisions using only E1/E2
        # and their selected E3 action.
        records = evaluator["evaluate"](
            rows,
            cuts,
            api,
            stage,
            old,
            router,
        )

        # Evaluate physical activation separately.
        activation = activation_diagnostic(
            sim,
            profiles,
            banks[band],
        )

        stats = evaluate_band(
            records,
            activation,
            evaluator,
            SEED + 1000 + i,
        )

        report["bands"][band] = stats

        private_audit[band] = {
            "trajectories": records,

            "activation_by_scene_PRIVATE":
                activation,

            "scene_parameters_PRIVATE":
                banks[band],
        }

        print("\n" + "=" * 65)
        print("BAND:", band)
        print("SCENES:", n_scenes)
        print("=" * 65)

        physical = stats[
            "deadzone_activation_PRIVATE"
        ]

        print(
            "Deadzone activation audit:",
            {
                k: v
                for k, v in physical.items()
                if k != "stopping_by_policy_PRIVATE"
            },
        )

        for policy in POLICIES:
            item = stats[
                "standard_metrics"
            ][policy]

            print(
                policy,
                "actions=", item["actions"],
                "work_J=",
                round(item["mean_work_J"], 4),
                "wrong_stops=",
                item["stops"]["wrong_count"],
                "correct_new=",
                item["new_correct_edit"]["count"],
                "E4mm=",
                {
                    k: round(v, 3)
                    for k, v in item[
                        "E4_all_fallback_diagnostic_mm"
                    ].items()
                },
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "smoke" if args.smoke
        else "stress"
    )

    report_path = OUT / (
        f"p3a4d_{suffix}_report.json"
    )

    audit_path = OUT / (
        f"p3a4d_{suffix}_audit_PRIVATE.json"
    )

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
            private_audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A4D GAP STRESS AUDIT COMPLETE"
    )

    print(
        "Frozen policies and thresholds; "
        "no tuning on stress outcomes."
    )

    print("Saved:", report_path)


if __name__ == "__main__":
    main()
