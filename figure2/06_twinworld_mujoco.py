
from pathlib import Path
import json

import mujoco
import numpy as np
import matplotlib.pyplot as plt

# P1-B 固定实验协议
DT = 0.002
PASSIVE_STEPS = 500
ACTIVE_STEPS = 500
INITIAL_VELOCITY = 0.15
INTERVENTION_FORCE = 1.0

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "p1b"
OUTPUT.mkdir(parents=True, exist_ok=True)

XML = """
<mujoco model="TwinWorld_P1B">
  <option timestep="0.002"
          gravity="0 0 0"
          integrator="Euler"/>

  <worldbody>
    <body name="A" pos="0 0 0">
      <joint name="slide_A"
             type="slide"
             axis="1 0 0"
             damping="0"
             frictionloss="0"/>
      <geom type="box"
            size="0.06 0.06 0.06"
            mass="1"
            contype="0"
            conaffinity="0"
            rgba="0.2 0.5 0.8 1"/>
    </body>

    <body name="B" pos="1.2 0 0">
      <joint name="slide_B"
             type="slide"
             axis="1 0 0"
             damping="0"
             frictionloss="0"/>
      <geom type="box"
            size="0.06 0.06 0.06"
            mass="1"
            contype="0"
            conaffinity="0"
            rgba="0.9 0.5 0.2 1"/>
    </body>
  </worldbody>

  <equality>
    <joint name="hidden_coupler"
           joint1="slide_B"
           joint2="slide_A"
           polycoef="0 1 0 0 0"/>
  </equality>

  <actuator>
    <motor name="force_A"
           joint="slide_A"
           gear="1"/>
  </actuator>
</mujoco>
"""


def observe(data):
    """Public position observation, no mechanism labels."""
    x_a = float(data.qpos[0])
    x_b = 1.2 + float(data.qpos[1])
    return np.array([data.time, x_a, x_b])


def rollout(model, coupled):
    data = mujoco.MjData(model)

    # Same model; only hidden constraint activation differs.
    assert model.neq == 1
    data.eq_active[0] = int(coupled)

    # Both objects initially have identical X velocity.
    data.qvel[:] = INITIAL_VELOCITY
    mujoco.mj_forward(model, data)

    passive = [observe(data)]

    # Phase 1: no intervention.
    for _ in range(PASSIVE_STEPS):
        data.ctrl[0] = 0.0
        mujoco.mj_step(model, data)
        passive.append(observe(data))

    active = [observe(data)]

    # Phase 2: identical force on A in both worlds.
    for _ in range(ACTIVE_STEPS):
        data.ctrl[0] = INTERVENTION_FORCE
        mujoco.mj_step(model, data)
        active.append(observe(data))

    return np.asarray(passive), np.asarray(active)


def delta_b_velocity(passive, active):
    """Estimate B's velocity change from observed positions."""
    v_before = (
        passive[-1, 2] - passive[-2, 2]
    ) / DT
    v_after = (
        active[-1, 2] - active[-2, 2]
    ) / DT
    return float(v_after - v_before)


def main():
    model = mujoco.MjModel.from_xml_string(XML)

    pi, ai = rollout(model, coupled=False)
    pc, ac = rollout(model, coupled=True)

    passive_diff = float(np.max(np.abs(
        pi[:, 1:] - pc[:, 1:]
    )))

    active_b_diff = float(abs(
        ai[-1, 2] - ac[-1, 2]
    ))

    independent_dv = delta_b_velocity(pi, ai)
    coupled_dv = delta_b_velocity(pc, ac)

    report = {
        "mujoco_version": mujoco.__version__,
        "dt": DT,
        "passive_steps": PASSIVE_STEPS,
        "active_steps": ACTIVE_STEPS,
        "intervention_force_N": INTERVENTION_FORCE,
        "passive_max_abs_diff": passive_diff,
        "active_B_final_abs_diff": active_b_diff,
        "independent_B_delta_velocity": independent_dv,
        "coupled_B_delta_velocity": coupled_dv,
        "note": (
            "P1-B physical simulation audit, "
            "not a trained agent result."
        )
    }

    print("\n=== TwinWorld P1-B ===")
    for key, value in report.items():
        print(f"{key}: {value}")

    # Predefined acceptance conditions.
    assert passive_diff < 1e-8, (
        "FAIL: passive trajectories are not equivalent"
    )
    assert active_b_diff > 0.10, (
        "FAIL: intervention did not separate worlds"
    )
    assert abs(independent_dv) < 1e-5, (
        "FAIL: independent B unexpectedly accelerated"
    )
    assert coupled_dv > 0.20, (
        "FAIL: coupled B did not respond sufficiently"
    )

    with open(
        OUTPUT / "p1b_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    # Audit-only data: do not use the mechanism
    # labels as machine-learning model inputs.
    np.savez_compressed(
        OUTPUT / "p1b_audit_only.npz",
        passive_independent=pi,
        passive_coupled=pc,
        active_independent=ai,
        active_coupled=ac
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4), dpi=150
    )

    axes[0].plot(
        pi[:, 0], pi[:, 2],
        label="Independent"
    )
    axes[0].plot(
        pc[:, 0], pc[:, 2],
        "--", label="Coupled"
    )
    axes[0].set_title("Passive observation")

    axes[1].plot(
        ai[:, 0], ai[:, 2],
        label="Independent"
    )
    axes[1].plot(
        ac[:, 0], ac[:, 2],
        "--", label="Coupled"
    )
    axes[1].set_title("Intervention on A")

    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Object B position (m)")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "p1b_comparison.png",
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nP1-B PASS")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()
