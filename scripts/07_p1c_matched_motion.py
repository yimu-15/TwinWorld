
from pathlib import Path
from runpy import run_path
import json

import mujoco
import numpy as np
import matplotlib.pyplot as plt

DT = 0.002
STEPS = 500
V0 = 0.15
TARGET_ACCEL = 0.4
KP, KD = 250.0, 40.0

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p1c"
OUT.mkdir(parents=True, exist_ok=True)

XML = run_path(
    str(Path(__file__).with_name(
        "06_twinworld_mujoco.py"
    ))
)["XML"]


def observe(data):
    return np.array([
        data.time,
        data.qpos[0],
        1.2 + data.qpos[1]
    ], dtype=float)


def simulate(coupled, accel):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Only the hidden physical constraint differs.
    data.eq_active[0] = int(coupled)
    data.qvel[:] = V0
    mujoco.mj_forward(model, data)

    passive = [observe(data)]

    for _ in range(STEPS):
        data.ctrl[0] = 0.0
        mujoco.mj_step(model, data)
        passive.append(observe(data))

    passive = np.asarray(passive)

    base_x = float(data.qpos[0])
    base_t = float(data.time)

    active = [observe(data)]
    forces = [0.0]

    for _ in range(STEPS):
        tau = float(data.time - base_t)

        # Identical desired A trajectory in both worlds.
        target = (
            base_x + V0 * tau
            + 0.5 * accel * tau**2
        )
        target_v = V0 + accel * tau

        # Same PD controller, different required force.
        u = (
            KP * (target - data.qpos[0])
            + KD * (target_v - data.qvel[0])
        )

        data.ctrl[0] = u
        mujoco.mj_step(model, data)

        active.append(observe(data))
        forces.append(float(u))

    return (
        passive,
        np.asarray(active),
        np.asarray(forces)
    )


def main():
    pi, ai, fi = simulate(False, TARGET_ACCEL)
    pc, ac, fc = simulate(True, TARGET_ACCEL)

    # Negative control: no additional acceleration.
    _, ai0, _ = simulate(False, 0.0)
    _, ac0, _ = simulate(True, 0.0)

    passive_gap = float(
        np.max(np.abs(pi[:, 1:] - pc[:, 1:]))
    )

    a_gap_max = float(
        np.max(np.abs(ai[:, 1] - ac[:, 1]))
    )

    b_gap_final = float(
        abs(ai[-1, 2] - ac[-1, 2])
    )

    placebo_b_gap = float(
        abs(ai0[-1, 2] - ac0[-1, 2])
    )

    joint_gap = float(np.max(np.abs(
        (ac[:, 1] - ac[0, 1])
        - (ac[:, 2] - ac[0, 2])
    )))

    force_gap = float(
        np.max(np.abs(fi - fc))
    )

    report = {
        "mujoco_version": mujoco.__version__,
        "protocol": "matched A motion using same PD law",
        "target_acceleration_mps2": TARGET_ACCEL,
        "passive_gap_max_m": passive_gap,
        "active_A_trajectory_gap_max_m": a_gap_max,
        "active_B_final_gap_m": b_gap_final,
        "placebo_B_final_gap_m": placebo_b_gap,
        "coupled_constraint_gap_max_m": joint_gap,
        "control_force_gap_max_N": force_gap,
        "note": (
            "PD controller uses privileged A state "
            "only for experimental control."
        )
    }

    print(json.dumps(report, indent=2))

    checks = {
        "passive_equivalent": passive_gap < 1e-8,
        "A_motion_matched": a_gap_max < 0.01,
        "B_intervention_differs": b_gap_final > 0.10,
        "no_intervention_placebo": placebo_b_gap < 1e-8,
        "coupling_consistent": joint_gap < 0.005
    }

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")

    (OUT / "p1c_report.json").write_text(
        json.dumps({
            "results": report,
            "checks": checks
        }, indent=2),
        encoding="utf-8"
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 3.7), dpi=150
    )

    for j, name in [
        (1, "Object A"),
        (2, "Object B")
    ]:
        ax = axes[j - 1]
        ax.plot(
            ai[:, 0], ai[:, j],
            label="Independent"
        )
        ax.plot(
            ac[:, 0], ac[:, j],
            "--", label="Coupled"
        )
        ax.set_title(name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.grid(alpha=0.2)
        ax.legend()

    axes[2].plot(
        ai[:, 0], fi,
        label="Independent"
    )
    axes[2].plot(
        ac[:, 0], fc,
        "--", label="Coupled"
    )
    axes[2].set_title("Controller force (audit only)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Force (N)")
    axes[2].grid(alpha=0.2)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(
        OUT / "p1c_matched_motion.png",
        bbox_inches="tight"
    )
    plt.close(fig)

    assert all(checks.values()), (
        "P1-C1 FAIL: inspect report"
    )

    print("\nP1-C1 PASS")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
