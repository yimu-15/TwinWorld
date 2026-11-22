
from pathlib import Path
from runpy import run_path
from itertools import combinations
import json

import mujoco
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "p2d1" / "results"
OUT.mkdir(parents=True, exist_ok=True)

XML = run_path(str(
    ROOT / "scripts" / "06_twinworld_mujoco.py"
))["XML"]

MODES = [
    "independent",
    "rigid",
    "elastic",
    "deadzone",
    "nonlinear"
]
ACTIONS = ["steady", "pulse", "reversal"]

DT = 0.002
STEPS = 500
SAMPLE_EVERY = 10
V0 = 0.15
KP, KD = 250.0, 40.0

K = 8.0
C = 1.5
GAP = 0.05
K3 = 400.0


def acceleration(action, t):
    if action == "steady":
        return 0.4
    if action == "pulse":
        return 1.0 if t < 0.4 else 0.0
    if action == "reversal":
        if t < 0.25:
            return 0.8
        if t < 0.5:
            return -0.8
        return 0.0
    raise ValueError(action)


def internal_force(mode, r, vr):
    if mode in ("independent", "rigid"):
        return 0.0

    if mode == "elastic":
        return -K * r - C * vr

    if mode == "deadzone":
        if abs(r) <= GAP:
            return 0.0
        stretch = np.sign(r) * (abs(r) - GAP)
        return -K * stretch - C * vr

    if mode == "nonlinear":
        return -K * r - K3 * r**3 - C * vr

    raise ValueError(mode)


def simulate(mode, action):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    assert model.neq == 1
    assert model.nv == 2

    data.eq_active[0] = int(mode == "rigid")
    data.qvel[:] = V0
    mujoco.mj_forward(model, data)

    def observe():
        return [
            float(data.qpos[0]),
            float(1.2 + data.qpos[1])
        ]

    passive = [observe()]

    for step in range(STEPS):
        data.ctrl[0] = 0.0
        data.qfrc_applied[:] = 0.0

        mujoco.mj_step(model, data)

        if (step + 1) % SAMPLE_EVERY == 0:
            passive.append(observe())

    base_x = float(data.qpos[0])
    target_x = base_x
    target_v = V0

    active = [observe()]
    command_budget = 0.0
    actuator_impulse = 0.0
    absolute_work = 0.0
    max_force_balance_error = 0.0

    for step in range(STEPS):
        t = step * DT
        a = acceleration(action, t)

        target_v += a * DT
        target_x += target_v * DT

        u = (
            KP * (target_x - data.qpos[0])
            + KD * (target_v - data.qvel[0])
        )
        data.ctrl[0] = u

        r = float(data.qpos[1] - data.qpos[0])
        vr = float(data.qvel[1] - data.qvel[0])
        f_b = internal_force(mode, r, vr)

        # Equal-and-opposite generalized forces.
        data.qfrc_applied[:] = [-f_b, f_b]

        max_force_balance_error = max(
            max_force_balance_error,
            abs(float(data.qfrc_applied.sum()))
        )

        command_budget += abs(a) * DT
        actuator_impulse += abs(u) * DT
        absolute_work += abs(u * data.qvel[0]) * DT

        mujoco.mj_step(model, data)

        if (step + 1) % SAMPLE_EVERY == 0:
            active.append(observe())

    return {
        "passive": np.asarray(passive),
        "active": np.asarray(active),
        "command_budget": command_budget,
        "actuator_impulse": actuator_impulse,
        "absolute_work": absolute_work,
        "force_balance_error": max_force_balance_error
    }


def main():
    # Elementary force-law checks.
    assert internal_force("deadzone", GAP / 2, 0) == 0
    assert internal_force("elastic", 0, 0) == 0

    results = {
        action: {
            mode: simulate(mode, action)
            for mode in MODES
        }
        for action in ACTIONS
    }

    report = {
        "protocol": "P2-D1A physical mechanism smoke",
        "mujoco_version": mujoco.__version__,
        "modes": MODES,
        "actions": ACTIONS,
        "parameters": {
            "k": K,
            "damping": C,
            "deadzone_m": GAP,
            "cubic_k": K3
        },
        "actions_summary": {}
    }

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4), dpi=150
    )

    for ax, action in zip(axes, ACTIONS):
        block = results[action]
        passive = np.stack([
            block[m]["passive"] for m in MODES
        ])
        active = np.stack([
            block[m]["active"] for m in MODES
        ])

        passive_gap = float(np.max(np.abs(
            passive - passive[:1]
        )))

        pairwise = {}
        for m1, m2 in combinations(MODES, 2):
            d = (
                block[m1]["active"][:, 1]
                - block[m2]["active"][:, 1]
            )
            pairwise[f"{m1}_vs_{m2}"] = float(
                np.sqrt(np.mean(d**2))
            )

        summary = {
            "passive_max_gap_m": passive_gap,
            "min_pairwise_B_rms_m": min(
                pairwise.values()
            ),
            "pairwise_B_rms_m": pairwise,
            "command_budget": float(
                block["independent"]["command_budget"]
            ),
            "actuator_impulse_Ns": {
                m: float(
                    block[m]["actuator_impulse"]
                ) for m in MODES
            },
            "absolute_actuator_work_J": {
                m: float(
                    block[m]["absolute_work"]
                ) for m in MODES
            },
            "force_balance_error_max": max(
                block[m]["force_balance_error"]
                for m in MODES
            )
        }

        report["actions_summary"][action] = summary

        assert passive_gap < 1e-8
        assert abs(summary["command_budget"] - 0.4) < 1e-6
        assert summary["force_balance_error_max"] < 1e-10
        assert np.isfinite(active).all()

        for mode in MODES:
            ax.plot(
                np.linspace(0, 1, 51),
                block[mode]["active"][:, 1],
                label=mode
            )

        ax.set_title(action)
        ax.set_xlabel("Time after intervention (s)")
        ax.set_ylabel("B position (m)")
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        OUT / "p2d1a_mechanisms.png",
        bbox_inches="tight"
    )
    plt.close(fig)

    # Audit data only, never future agent inputs.
    np.savez_compressed(
        OUT / "p2d1a_audit_only.npz",
        **{
            action: np.stack([
                results[action][m]["active"]
                for m in MODES
            ])
            for action in ACTIONS
        }
    )

    path = OUT / "p2d1a_report.json"
    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    for action, r in report["actions_summary"].items():
        print(
            f"{action}: "
            f"passive_gap={r['passive_max_gap_m']:.3e}, "
            f"min_B_RMS={r['min_pairwise_B_rms_m']:.6f}"
        )

    print("\nP2-D1A PHYSICS SMOKE PASS")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
