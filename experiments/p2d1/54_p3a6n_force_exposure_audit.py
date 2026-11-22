
"""
P3-A6N: Force-Level Mechanism Distinguishability Audit

Uses the 48 existing Saturating scenes.

Compares Saturating and same-parameter Elastic
forces at EXACTLY the same high-resolution states.

Actions:
  Reference, Moderate, Low
  Weak and Strong within each regime

No fitting or threshold tuning.
No E4 access.
True parameters are PRIVATE audit information.
Not an online structural inference algorithm.
"""

from pathlib import Path
from runpy import run_path
from collections import Counter
import argparse
import csv
import hashlib
import json

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"

A6M = (
    ROOT / "p3a6m"
    / "p3a6m_formal_audit_PRIVATE.json"
)

A6K = (
    ROOT / "p3a6k"
    / "p3a6k_fresh_report.json"
)

OUT = ROOT / "p3a6n"

SCENE_SEED = 20270115
N_SCENES = 48

REGIMES = {
    "reference": (1.50, 2.25),
    "moderate": (1.25, 1.75),
    "low": (0.75, 1.10),
}


def load(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(
        np.sqrt(np.mean(values ** 2))
    )


def force_trace(sim, model, scene, command):
    """
    Replays the original high-frequency simulator.

    Returns force contrast evaluated on the
    SAME real Saturating state trajectory.

    Strictly checks the reconstructed trajectory
    against sim.simulate().
    """
    force_fn = sim["simulate"].__globals__.get(
        "internal_force"
    )

    if not callable(force_fn):
        raise RuntimeError(
            "Cannot access active physical force law"
        )

    dt = sim["DT"]
    steps = sim["STEPS"]
    sample_every = sim["SAMPLE_EVERY"]

    if len(command) != steps:
        raise RuntimeError(
            "Unexpected intervention length"
        )

    data = mujoco.MjData(model)
    data.eq_active[0] = 0
    data.qvel[:] = scene["velocity"]

    mujoco.mj_forward(model, data)

    for _ in range(steps):
        data.ctrl[0] = 0.0
        data.qfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)

    target_x = float(data.qpos[0])
    target_v = float(scene["velocity"])

    observed = [sim["observe"](data)]

    r_values = []
    vr_values = []

    fs_values = []
    fe_values = []

    for step, acceleration in enumerate(command):
        target_v += float(acceleration) * dt
        target_x += target_v * dt

        u = (
            sim["KP"] * (
                target_x - data.qpos[0]
            )
            + sim["KD"] * (
                target_v - data.qvel[0]
            )
        )

        r = float(
            data.qpos[1] - data.qpos[0]
        )

        vr = float(
            data.qvel[1] - data.qvel[0]
        )

        fs = float(
            force_fn(
                "saturating", r, vr, scene
            )
        )

        fe = float(
            force_fn(
                "elastic", r, vr, scene
            )
        )

        if not np.isfinite([fs, fe]).all():
            raise RuntimeError(
                "Nonfinite force evaluation"
            )

        r_values.append(r)
        vr_values.append(vr)

        fs_values.append(fs)
        fe_values.append(fe)

        data.qfrc_applied[:] = (-fs, fs)
        data.ctrl[0] = u

        mujoco.mj_step(model, data)

        if (step + 1) % sample_every == 0:
            observed.append(
                sim["observe"](data)
            )

    reconstructed = np.asarray(observed)

    _, actual, _, _ = sim["simulate"](
        model,
        scene,
        "saturating",
        command,
    )

    max_error = float(
        np.max(
            np.abs(
                reconstructed - actual
            )
        )
    )

    if max_error > 1e-9:
        raise RuntimeError(
            "Physical replay mismatch: "
            f"{max_error}"
        )

    r_values = np.asarray(r_values)
    vr_values = np.asarray(vr_values)

    fs_values = np.asarray(fs_values)
    fe_values = np.asarray(fe_values)

    force_delta = fs_values - fe_values

    # Counterfactual same-parameter Elastic
    # simulation: privileged diagnostic only.
    _, elastic_active, _, _ = sim["simulate"](
        model,
        scene,
        "elastic",
        command,
    )

    clean_B_difference = rms(
        actual[:, 1] - elastic_active[:, 1]
    )

    return {
        "force_difference_RMS_N":
            rms(force_delta),

        "force_difference_max_N":
            float(np.max(np.abs(force_delta))),

        "saturating_force_RMS_N":
            rms(fs_values),

        "elastic_force_RMS_N":
            rms(fe_values),

        "force_difference_relative":
            rms(force_delta)
            / max(rms(fe_values), 1e-12),

        "max_abs_relative_position_m":
            float(np.max(np.abs(r_values))),

        "p95_abs_relative_position_m":
            float(np.quantile(
                np.abs(r_values), 0.95
            )),

        "max_abs_relative_velocity_mps":
            float(np.max(np.abs(vr_values))),

        "same_parameter_clean_B_RMS_mm":
            1000.0 * clean_B_difference,

        "replay_max_error_m":
            max_error,
    }


def print_group(rows, group, regime, action):
    selected = [
        r for r in rows
        if (
            r["group"] == group
            and r["regime"] == regime
            and r["action"] == action
        )
    ]

    if not selected:
        return

    print(
        "\nGROUP:", group,
        "REGIME:", regime,
        "ACTION:", action,
        "N:", len(selected),
    )

    fields = (
        "force_difference_RMS_N",
        "force_difference_relative",
        "same_parameter_clean_B_RMS_mm",
        "p95_abs_relative_position_m",
    )

    for field in fields:
        values = [
            r[field] for r in selected
        ]

        print(
            " ", field,
            "median=", round(
                float(np.median(values)), 6
            ),
            "range=(",
            round(min(values), 6),
            ",",
            round(max(values), 6),
            ")",
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    historic = load(A6K)
    noise_audit = load(A6M)

    if (
        historic["smoke"]
        or noise_audit["smoke"]
        or not noise_audit["original_replay_verified"]
    ):
        raise RuntimeError(
            "Formal historical audits required"
        )

    # Verify unchanged physical model sources.
    for filename in (
        "16_p2d1b_probe_diversity.py",
        "27_p3a3b_evidence_vs_structure.py",
    ):
        expected = historic[
            "source_sha256"
        ][filename]

        if sha256(HERE / filename) != expected:
            raise RuntimeError(
                "Physical source was modified: "
                + filename
            )

    sim = run_path(str(
        HERE / "16_p2d1b_probe_diversity.py"
    ))

    ext = run_path(str(
        HERE / "27_p3a3b_evidence_vs_structure.py"
    ))

    ext["install_saturating"](sim)

    src = run_path(str(
        HERE /
        "47_p3a6h_six_mechanism_openworld_audit.py"
    ))

    scenes = src["make_scenes"](
        sim,
        SCENE_SEED,
        N_SCENES,
    )

    prior = noise_audit["per_scene_PRIVATE"]

    if len(prior) != N_SCENES:
        raise RuntimeError(
            "Noise audit scene count mismatch"
        )

    original_failures = [
        sid for sid in range(N_SCENES)
        if prior[str(sid)][
            "original_false_accept"
        ]
    ]

    if len(original_failures) != 17:
        raise RuntimeError(
            "Expected 17 original false accepts"
        )

    ids = (
        range(2)
        if args.smoke
        else range(N_SCENES)
    )

    pulse = sim["make_profiles"]()["pulse"]

    records = []

    print(
        "\nP3-A6N FORCE-LEVEL "
        "IDENTIFIABILITY AUDIT"
    )

    print(
        "Original false accepts:",
        len(original_failures),
        "/48"
    )

    print(
        "Scenes replayed:",
        len(ids)
    )

    for sid in ids:
        scene = scenes[sid]["original"]
        model = sim["make_model"](scene)

        original_failed = (
            sid in original_failures
        )

        group = (
            "original_false_accept"
            if original_failed
            else "original_correct_reject"
        )

        repeated_accepts = prior[str(sid)][
            "new_noise_accept_count"
        ]

        for regime, gains in REGIMES.items():
            weak_gain, strong_gain = gains

            commands = {
                "weak": pulse * weak_gain,
                "strong": pulse * strong_gain,
            }

            for action, command in (
                commands.items()
            ):
                metrics = force_trace(
                    sim, model, scene, command
                )

                record = {
                    "scene_id": sid,
                    "group": group,
                    "regime": regime,
                    "action": action,

                    "original_false_accept_PRIVATE":
                        original_failed,

                    "new_noise_accept_count_PRIVATE":
                        repeated_accepts,

                    "new_noise_total": 8,
                }

                record.update(metrics)

                records.append(record)

    expected = len(ids) * len(REGIMES) * 2

    if len(records) != expected:
        raise RuntimeError(
            "Unexpected force-audit count"
        )

    groups = (
        "original_false_accept",
        "original_correct_reject",
    )

    for regime in REGIMES:
        for action in ("weak", "strong"):
            for group in groups:
                print_group(
                    records, group,
                    regime, action
                )

    # Cross-scene association is descriptive.
    # No threshold or policy is fitted.
    if not args.smoke:
        from scipy.stats import spearmanr

        low_strong = sorted(
            [
                r for r in records
                if r["regime"] == "low"
                and r["action"] == "strong"
            ],
            key=lambda r: r["scene_id"]
        )

        force_delta = [
            r["force_difference_RMS_N"]
            for r in low_strong
        ]

        accept_frequency = [
            r["new_noise_accept_count_PRIVATE"] / 8
            for r in low_strong
        ]

        statistic = spearmanr(
            force_delta,
            accept_frequency,
        )

        print(
            "\nLOW STRONG FORCE GAP VS"
            " FALSE ACCEPTANCE FREQUENCY"
        )
        print(
            "Spearman rho:",
            float(statistic.statistic)
        )
        print(
            "Descriptive only: selected"
            " development scenes."
        )

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    suffix = (
        "smoke" if args.smoke
        else "formal"
    )

    csv_path = OUT / (
        f"p3a6n_{suffix}_force_PRIVATE.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0])
        )

        writer.writeheader()
        writer.writerows(records)

    report = {
        "protocol":
            "P3-A6N oracle force exposure audit",

        "smoke": args.smoke,
        "n_scenes": len(ids),

        "n_records": len(records),

        "force_comparison":
            "same physical r and vr",

        "counterfactual":
            "same-parameter Elastic rollout",

        "uses_oracle_parameters":
            "PRIVATE AUDIT ONLY",

        "E4_accessed": False,
        "threshold_tuned": False,
        "model_refitted": False,

        "limitations": [
            "Retrospective oracle diagnosis",
            "Does not establish global identifiability",
            "Force observations unavailable online",
            "Same development scenes as P3-A6K",
            "No new independent validation",
            "No adaptive action choice tested",
        ],
    }

    report_path = OUT / (
        f"p3a6n_{suffix}_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "\nP3-A6N FORCE-LEVEL "
        "AUDIT PASS"
    )

    print(
        "No E4, no thresholds changed."
    )
    print("Saved:", csv_path)
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
