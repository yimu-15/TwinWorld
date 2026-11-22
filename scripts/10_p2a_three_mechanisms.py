
from pathlib import Path
from runpy import run_path
import json

import mujoco
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p2a"
OUT.mkdir(parents=True, exist_ok=True)

base_xml = run_path(str(
    ROOT / "scripts" / "06_twinworld_mujoco.py"
))["XML"]

# Add a spring-damper between the two slide joints.
tendon_xml = """
<tendon>
  <fixed name="elastic_link"
         stiffness="8"
         damping="1.5"
         springlength="0">
    <joint joint="slide_A" coef="-1"/>
    <joint joint="slide_B" coef="1"/>
  </fixed>
</tendon>
"""

xml = base_xml.replace(
    "<actuator>",
    tendon_xml + "\n<actuator>", 1
)

MODES = ("independent", "rigid", "elastic")
DT = 0.002
STEPS = 500
V0 = 0.15
ACCEL = 0.4
KP = 250.0
KD = 40.0


def simulate(mode):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    assert model.neq == 1
    assert model.ntendon == 1

    data.eq_active[0] = int(mode == "rigid")

    # Only elastic mode activates spring forces.
    model.tendon_stiffness[0] = (
        8.0 if mode == "elastic" else 0.0
    )
    model.tendon_damping[0] = (
        1.5 if mode == "elastic" else 0.0
    )

    data.qvel[:] = V0
    mujoco.mj_forward(model, data)

    def state():
        return np.array([
            data.time,
            data.qpos[0],
            1.2 + data.qpos[1]
        ], dtype=float)

    passive = [state()]

    for _ in range(STEPS):
        data.ctrl[0] = 0.0
        mujoco.mj_step(model, data)
        passive.append(state())

    base_x = float(data.qpos[0])
    base_t = float(data.time)

    active = [state()]

    for _ in range(STEPS):
        tau = data.time - base_t

        target_x = (
            base_x + V0 * tau
            + 0.5 * ACCEL * tau**2
        )
        target_v = V0 + ACCEL * tau

        data.ctrl[0] = (
            KP * (target_x - data.qpos[0])
            + KD * (target_v - data.qvel[0])
        )

        mujoco.mj_step(model, data)
        active.append(state())

    return np.asarray(passive), np.asarray(active)


def main():
    results = {
        mode: simulate(mode) for mode in MODES
    }

    reference = results["independent"][0]

    passive_gaps = {}
    response_gaps = {}

    for mode in MODES:
        passive, active = results[mode]

        passive_gaps[mode] = float(np.max(
            np.abs(passive - reference)
        ))

        response_gaps[mode] = float(
            active[-1, 2] - (
                passive[-1, 2] + V0 * 1.0
            )
        )

    final_b = {
        mode: results[mode][1][-1, 2]
        for mode in MODES
    }

    report = {
        "mujoco_version": mujoco.__version__,
        "modes": list(MODES),
        "passive_gaps": passive_gaps,
        "B_excess_displacement_m": response_gaps,
        "B_final_positions_m": final_b,
        "note": (
            "Physical three-mechanism smoke test. "
            "Not a learned model result."
        )
    }

    print(json.dumps(report, indent=2))

    traces = {
        mode: results[mode][1]
        for mode in MODES
    }

    # Save audit data, not a training dataset.
    np.savez_compressed(
        OUT / "p2a_audit_only.npz",
        **traces
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4), dpi=150
    )

    for mode in MODES:
        passive, active = results[mode]

        axes[0].plot(
            passive[:, 0],
            passive[:, 2],
            label=mode
        )
        axes[1].plot(
            active[:, 0],
            active[:, 2],
            label=mode
        )

    axes[0].set_title("Passive observations")
    axes[1].set_title("Intervention on A")

    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Object B position (m)")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUT / "p2a_three_mechanisms.png",
        bbox_inches="tight"
    )
    plt.close(fig)

    (OUT / "p2a_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    assert all(
        v < 1e-8
        for v in passive_gaps.values()
    ), "Passive observations differ"

    assert abs(
        final_b["rigid"]
        - final_b["independent"]
    ) > 0.10, "Rigid response not separated"

    assert np.max(np.abs(
        traces["elastic"][:, 2]
        - traces["independent"][:, 2]
    )) > 1e-3, "Elastic mode not distinguishable"

    assert np.max(np.abs(
        traces["elastic"][:, 2]
        - traces["rigid"][:, 2]
    )) > 1e-3, "Elastic and rigid indistinguishable"

    print("\nP2-A physics smoke test PASS")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
