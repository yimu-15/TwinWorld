
from pathlib import Path
from runpy import run_path
from itertools import combinations
from collections import Counter
import json

import mujoco
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p2d0"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse P2-A three-mechanism model.
BASE_XML = run_path(str(
    ROOT / "scripts" / "10_p2a_three_mechanisms.py"
))["xml"]

SEED = 20261113
N_SCENES = 12

MODES = ("independent", "rigid", "elastic")
ACTIONS = ("steady", "pulse", "reversal")

DT = 0.002
STEPS = 500
SAMPLE_EVERY = 10
KP, KD = 250.0, 40.0


def command_accel(action, t):
    if action == "steady":
        return 0.4
    if action == "pulse":
        return 1.0 if t < 0.4 else 0.0
    if action == "reversal":
        if t < 0.25:
            return 0.8
        if t < 0.50:
            return -0.8
        return 0.0
    raise ValueError(action)


def make_xml(ma, mb):
    assert BASE_XML.count('mass="1"') == 2
    return BASE_XML.replace(
        'mass="1"', f'mass="{ma:.8f}"', 1
    ).replace(
        'mass="1"', f'mass="{mb:.8f}"', 1
    )


def simulate(params, mode, action):
    ma, mb, k, d, v0 = params

    model = mujoco.MjModel.from_xml_string(
        make_xml(ma, mb)
    )
    data = mujoco.MjData(model)

    data.eq_active[0] = int(mode == "rigid")
    model.tendon_stiffness[0] = (
        k if mode == "elastic" else 0.0
    )
    model.tendon_damping[0] = (
        d if mode == "elastic" else 0.0
    )

    data.qvel[:] = v0
    mujoco.mj_forward(model, data)

    def observe():
        return [
            float(data.qpos[0]),
            float(1.2 + data.qpos[1])
        ]

    passive = [observe()]

    for step in range(STEPS):
        data.ctrl[0] = 0.0
        mujoco.mj_step(model, data)
        if (step + 1) % SAMPLE_EVERY == 0:
            passive.append(observe())

    base_x = float(data.qpos[0])
    target_x = base_x
    target_v = v0

    active = [observe()]
    command_budget = 0.0

    for step in range(STEPS):
        t = step * DT
        acceleration = command_accel(action, t)

        command_budget += abs(acceleration) * DT

        target_v += acceleration * DT
        target_x += target_v * DT

        data.ctrl[0] = (
            KP * (target_x - data.qpos[0])
            + KD * (target_v - data.qvel[0])
        )

        mujoco.mj_step(model, data)

        if (step + 1) % SAMPLE_EVERY == 0:
            active.append(observe())

    return (
        np.asarray(passive),
        np.asarray(active),
        command_budget
    )


def main():
    rng = np.random.default_rng(SEED)

    # Audit-only candidate-action catalog.
    # Do not use mechanism labels or all candidate
    # outcomes as inputs to a future agent.
    all_active = []
    all_passive = []
    scores = {a: [] for a in ACTIONS}
    best_actions = []
    max_passive_gap = 0.0
    max_a_gap = 0.0
    budgets = {}

    for scene_id in range(N_SCENES):
        params = (
            rng.uniform(0.8, 1.4),
            rng.uniform(0.8, 1.4),
            rng.uniform(7.0, 16.0),
            rng.uniform(0.6, 2.0),
            rng.uniform(0.05, 0.25)
        )

        scene_active = []
        scene_passive = []
        scene_scores = {}

        for action in ACTIONS:
            rollouts = [
                simulate(params, mode, action)
                for mode in MODES
            ]

            passive = np.stack(
                [r[0] for r in rollouts]
            )
            active = np.stack(
                [r[1] for r in rollouts]
            )

            assert passive.shape == (3, 51, 2)
            assert active.shape == (3, 51, 2)
            assert np.isfinite(active).all()

            passive_gap = float(np.max(np.abs(
                passive - passive[:1]
            )))
            max_passive_gap = max(
                max_passive_gap, passive_gap
            )

            a_gap = float(np.max(
                np.ptp(active[:, :, 0], axis=0)
            ))
            max_a_gap = max(max_a_gap, a_gap)

            # Pairwise B-trajectory differences.
            pair_scores = []
            for i, j in combinations(range(3), 2):
                difference = (
                    active[i, :, 1]
                    - active[j, :, 1]
                )
                rms = float(np.sqrt(
                    np.mean(difference ** 2)
                ))
                pair_scores.append(rms)

            # Worst separated mechanism pair.
            score = min(pair_scores)
            scene_scores[action] = score
            scores[action].append(score)

            budgets[action] = float(
                rollouts[0][2]
            )

            scene_active.append(active)
            scene_passive.append(passive)

        best = max(scene_scores, key=scene_scores.get)
        best_actions.append(best)

        all_active.append(np.stack(scene_active))
        all_passive.append(np.stack(scene_passive))

    all_active = np.stack(all_active)
    all_passive = np.stack(all_passive)

    report = {
        "protocol": "P2-D0 probe feasibility audit",
        "mujoco_version": mujoco.__version__,
        "seed": SEED,
        "scene_count": N_SCENES,
        "mechanisms": list(MODES),
        "actions": list(ACTIONS),
        "max_passive_gap_m": max_passive_gap,
        "max_A_cross_mode_gap_m": max_a_gap,
        "command_abs_accel_integral": budgets,
        "mean_worst_pair_B_RMS_m": {
            a: float(np.mean(scores[a]))
            for a in ACTIONS
        },
        "min_worst_pair_B_RMS_m": {
            a: float(np.min(scores[a]))
            for a in ACTIONS
        },
        "privileged_best_action_counts": dict(
            Counter(best_actions)
        ),
        "note": (
            "Privileged counterfactual probe audit. "
            "NOT an active learned policy. "
            "No visual inputs or concept revision."
        )
    }

    (OUT / "p2d0_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    np.savez_compressed(
        OUT / "p2d0_audit_only.npz",
        active=all_active,
        passive=all_passive
    )

    print(json.dumps(report, indent=2))

    # Show scene 0: B trajectories for all probes.
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 3.7), dpi=150
    )
    times = np.linspace(0, 1, 51)

    for action_id, action in enumerate(ACTIONS):
        ax = axes[action_id]
        for mode_id, mode in enumerate(MODES):
            ax.plot(
                times,
                all_active[
                    0, action_id, mode_id, :, 1
                ],
                label=mode
            )

        ax.set_title(action)
        ax.set_xlabel("Time since intervention (s)")
        ax.set_ylabel("B position (m)")
        ax.grid(alpha=0.2)
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUT / "p2d0_probe_comparison.png",
        bbox_inches="tight"
    )
    plt.close(fig)

    # Engineering checks, not paper performance.
    assert max_passive_gap < 1e-8
    assert np.isfinite(all_active).all()
    assert all(
        abs(budgets[a] - 0.4) < 1e-6
        for a in ACTIONS
    )

    print("\nP2-D0 probe catalog PASS")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
