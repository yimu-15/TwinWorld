
"""
P3-A6M: Noise Stability of Unknown-Law Acceptance

Question:
Does low-excitation Saturating false acceptance
persist when only observation noise changes?

Frozen:
- Physical scenes
- Weak/Strong actions
- Five candidate laws
- A6I decision thresholds

Formal: 48 Saturating scenes, 8 new noise seeds.
Smoke: 2 scenes, 1 new noise seed.

No E4. No policy tuning. No oracle fitting.
This is retrospective diagnosis, not new holdout.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from runpy import run_path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"

A6K = ROOT / "p3a6k"
A6I = ROOT / "p3a6i"
OUT = ROOT / "p3a6m"

SCENE_SEED = 20270115
NOISE_SEED = 2027011501

N_SCENES = 48
N_NEW_NOISE = 8
NOISE_OFFSET = 10000000


def read(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_modules():
    src = run_path(str(
        HERE /
        "47_p3a6h_six_mechanism_openworld_audit.py"
    ))

    policy = run_path(str(
        HERE /
        "48_p3a6i_calibrated_revision.py"
    ))

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    api = run_path(str(
        HERE / "22_p3a1b_structural_revision.py"
    ))

    extension = run_path(str(
        HERE / "27_p3a3b_evidence_vs_structure.py"
    ))

    extension["install_saturating"](sim)

    return src, policy, sim, api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true"
    )
    args = parser.parse_args()

    historic = read(
        A6K / "p3a6k_fresh_report.json"
    )

    audit = read(
        A6K / "p3a6k_fresh_audit_PRIVATE.json"
    )

    frozen = read(
        A6I / "p3a6i_fresh_report.json"
    )

    if historic["smoke"] or frozen["smoke"]:
        raise RuntimeError(
            "Formal frozen results required"
        )

    thresholds = historic["frozen_thresholds"]

    if thresholds != frozen["thresholds"]:
        raise RuntimeError(
            "Frozen policy mismatch"
        )

    # Verify physical and policy sources are
    # unchanged since P3-A6K.
    for filename, expected in (
        historic["source_sha256"].items()
    ):
        if sha256(HERE / filename) != expected:
            raise RuntimeError(
                "Historical source changed: "
                + filename
            )

    src, policy, sim, api = load_modules()

    scenes = src["make_scenes"](
        sim, SCENE_SEED, N_SCENES
    )

    commands = src["make_commands"](sim)
    profiles = sim["make_profiles"]()

    commands["weak"] = (
        profiles["pulse"] * 0.75
    )
    commands["strong"] = (
        profiles["pulse"] * 1.10
    )

    old_rows = [
        r for r in audit["low"]["original"]
        if r["truth_PRIVATE"] == "saturating"
    ]

    old_index = {
        int(r["scene_id"]): r
        for r in old_rows
    }

    if len(old_index) != N_SCENES:
        raise RuntimeError(
            "Expected 48 historical Saturating cases"
        )

    original_failures = {
        sid for sid, row in old_index.items()
        if row["candidate_outcome_PRIVATE"]
        == "unknown_false_accept"
    }

    if len(original_failures) != 17:
        raise RuntimeError(
            "Historical failure group mismatch"
        )

    ids = (
        range(2) if args.smoke
        else range(N_SCENES)
    )

    new_noise_count = (
        1 if args.smoke
        else N_NEW_NOISE
    )

    rows = []
    per_scene = {}

    print(
        "\nP3-A6M NOISE-STABILITY AUDIT"
    )
    print(
        "Frozen absolute cutoff:",
        round(
            1000 * thresholds["tau_absolute_m"],
            4,
        ),
        "mm",
    )
    print(
        "Original Low false accepts:",
        len(original_failures), "/48"
    )
    print(
        "Replayed scenes:", len(ids)
    )
    print(
        "New noise seeds:", new_noise_count
    )

    for sid in ids:
        scene = scenes[sid]["original"]
        accepted = 0
        all_decisions = []
        all_rms = []

        for repeat in range(
            new_noise_count + 1
        ):
            noise_seed = (
                NOISE_SEED
                + repeat * NOISE_OFFSET
            )

            episode = src["run_episode"](
                sim,
                api,
                scene,
                sid,
                "saturating",
                commands,
                noise_seed,
            )

            decision, reason = policy["decide"](
                episode,
                thresholds,
            )

            error_mm = (
                1000 * episode["minimum_RMS_m"]
            )

            if repeat == 0:
                # Original observation replay must
                # reproduce P3-A6K exactly.
                expected = old_index[sid]

                if not np.isclose(
                    error_mm,
                    expected["minimum_RMS_mm"],
                    atol=1e-6,
                    rtol=0,
                ):
                    raise RuntimeError(
                        "Original RMS replay failed: "
                        f"scene={sid}"
                    )

                if (
                    decision
                    != expected["candidate_decision"]
                ):
                    raise RuntimeError(
                        "Original decision replay failed: "
                        f"scene={sid}"
                    )

            else:
                accepted += int(
                    decision != "unresolved"
                )
                all_decisions.append(decision)
                all_rms.append(error_mm)

            rows.append({
                "scene_id": sid,
                "repeat": repeat,
                "noise_seed": noise_seed,
                "truth_PRIVATE": "saturating",
                "was_original_false_accept_PRIVATE":
                    sid in original_failures,
                "minimum_RMS_mm": error_mm,
                "decision": decision,
                "reason": reason,
                "false_accept":
                    decision != "unresolved",
            })

        per_scene[sid] = {
            "original_false_accept":
                sid in original_failures,
            "new_noise_accept_count": accepted,
            "new_noise_total": new_noise_count,
            "new_noise_decisions":
                dict(Counter(all_decisions)),
            "new_noise_RMS_median_mm":
                float(np.median(all_rms)),
            "new_noise_RMS_min_mm":
                float(min(all_rms)),
            "new_noise_RMS_max_mm":
                float(max(all_rms)),
        }

    print("\n" + "=" * 70)
    print("FROZEN-WORLD NEW-NOISE RESULTS")
    print("=" * 70)

    groups = {
        "original_false_accept": [
            sid for sid in ids
            if sid in original_failures
        ],
        "original_correct_reject": [
            sid for sid in ids
            if sid not in original_failures
        ],
    }

    summary = {}

    for group_name, group_ids in groups.items():
        if not group_ids:
            continue

        counts = [
            per_scene[sid][
                "new_noise_accept_count"
            ]
            for sid in group_ids
        ]

        total_accepts = sum(counts)
        total_tests = (
            len(group_ids) * new_noise_count
        )

        always_accepted = sum(
            c == new_noise_count
            for c in counts
        )

        always_rejected = sum(
            c == 0
            for c in counts
        )

        summary[group_name] = {
            "n_scenes": len(group_ids),
            "accept_events": total_accepts,
            "total_noise_trials": total_tests,
            "always_accepted_scenes":
                always_accepted,
            "always_rejected_scenes":
                always_rejected,
            "per_scene_accept_counts": counts,
        }

        print("\nGROUP:", group_name)
        print("  Scenes:", len(group_ids))
        print(
            "  False accepts across new noise:",
            total_accepts, "/", total_tests,
        )
        print(
            "  Always accepted:",
            always_accepted, "/", len(group_ids),
        )
        print(
            "  Always rejected:",
            always_rejected, "/", len(group_ids),
        )
        print(
            "  Acceptance count distribution:",
            dict(Counter(counts)),
        )

    result = {
        "protocol":
            "P3-A6M fixed-scene noise stability",
        "smoke": args.smoke,
        "original_noise_seed": NOISE_SEED,
        "n_new_noise": new_noise_count,
        "frozen_thresholds": thresholds,
        "original_replay_verified": True,
        "E4_accessed": False,
        "thresholds_tuned": False,
        "limitations": [
            "Repeated noise on seen failure scenes",
            "Not an independent scene-level holdout",
            "Noise repetitions are not independent worlds",
            "Truth label used only to select audit group",
            "No active action policy",
            "No force-level identifiability proof",
        ],
        "summary": summary,
        "per_scene_PRIVATE": per_scene,
        "records_PRIVATE": rows,
    }

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "formal"
    )

    path = OUT / (
        f"p3a6m_{suffix}_audit_PRIVATE.json"
    )

    path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nP3-A6M NOISE-STABILITY "
        "AUDIT PASS"
    )
    print(
        "No E4, no parameter or threshold tuning."
    )
    print("Saved:", path)


if __name__ == "__main__":
    main()
