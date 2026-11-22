
from pathlib import Path
from runpy import run_path
import json

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p1c3"
OUT.mkdir(parents=True, exist_ok=True)

XML = run_path(
    str(ROOT / "scripts" / "06_twinworld_mujoco.py")
)["XML"]

STEPS = 500
SIGMA = 0.005
KP = 250.0
KD = 40.0


def simulate(m_a, m_b, v0, accel, coupled):
    # Same masses and initial conditions within a pair.
    assert XML.count('mass="1"') == 2

    xml = XML.replace(
        'mass="1"', f'mass="{m_a:.8f}"', 1
    ).replace(
        'mass="1"', f'mass="{m_b:.8f}"', 1
    )

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    data.eq_active[0] = int(coupled)
    data.qvel[:] = v0
    mujoco.mj_forward(model, data)

    def observe():
        return [
            data.time,
            data.qpos[0],
            1.2 + data.qpos[1]
        ]

    passive = [observe()]
    for _ in range(STEPS):
        data.ctrl[0] = 0
        mujoco.mj_step(model, data)
        passive.append(observe())

    base_x = float(data.qpos[0])
    base_t = float(data.time)

    active = [observe()]

    for _ in range(STEPS):
        tau = data.time - base_t
        target = (
            base_x + v0 * tau
            + 0.5 * accel * tau**2
        )
        target_v = v0 + accel * tau

        data.ctrl[0] = (
            KP * (target - data.qpos[0])
            + KD * (target_v - data.qvel[0])
        )

        mujoco.mj_step(model, data)
        active.append(observe())

    return np.asarray(passive), np.asarray(active)


def observed_features(passive, active, accel, rng):
    # New independent sensor noise.
    p = passive.copy()
    p[:, 1:] += rng.normal(
        0, SIGMA, p[:, 1:].shape
    )

    final = active[-1].copy()
    final[1:] += rng.normal(0, SIGMA, 2)

    scores = []

    for j in (1, 2):
        # Estimate passive velocity from 100 samples,
        # not from hidden simulator states.
        slope, intercept = np.polyfit(
            p[-100:, 0], p[-100:, j], 1
        )

        expected = (
            slope * active[-1, 0] + intercept
        )

        if j == 1:
            expected += 0.5 * accel

        scores.append(
            (final[j] - expected) / accel
        )

    return scores


def make_split(seed, pairs, mass_range):
    rng = np.random.default_rng(seed)
    rows = []
    max_a_gap = 0.0
    min_b_gap = float("inf")

    for _ in range(pairs):
        ma = rng.uniform(*mass_range)
        mb = rng.uniform(*mass_range)
        v0 = rng.uniform(0.05, 0.25)
        accel = rng.uniform(0.15, 0.60)

        pi, ai = simulate(ma, mb, v0, accel, False)
        pc, ac = simulate(ma, mb, v0, accel, True)

        passive_gap = np.max(
            np.abs(pi[:, 1:] - pc[:, 1:])
        )
        assert passive_gap < 1e-8

        a_gap = np.max(
            np.abs(ai[:, 1] - ac[:, 1])
        )
        b_gap = abs(ai[-1, 2] - ac[-1, 2])

        max_a_gap = max(max_a_gap, float(a_gap))
        min_b_gap = min(min_b_gap, float(b_gap))

        for label, p, a in (
            (0, pi, ai),
            (1, pc, ac)
        ):
            fa, fb = observed_features(
                p, a, accel, rng
            )
            rows.append((label, fa, fb))

    return np.asarray(rows), {
        "max_A_gap_m": max_a_gap,
        "min_B_gap_m": min_b_gap
    }


def calibrate(rows, feature):
    labels = rows[:, 0].astype(int)
    values = rows[:, feature]

    m0 = float(np.median(values[labels == 0]))
    m1 = float(np.median(values[labels == 1]))

    return {
        "threshold": (m0 + m1) / 2,
        "coupled_is_larger": bool(m1 > m0)
    }


def evaluate(rows, feature, rule):
    values = rows[:, feature]
    labels = rows[:, 0].astype(int)
    threshold = rule["threshold"]

    if rule["coupled_is_larger"]:
        pred = values > threshold
    else:
        pred = values < threshold

    return float(np.mean(pred == labels))


def main():
    validation, val_meta = make_split(
        101, 80, (0.8, 1.2)
    )
    test, test_meta = make_split(
        202, 120, (1.2, 1.6)
    )

    # Calibrate using validation data only.
    rule_a = calibrate(validation, 1)
    rule_b = calibrate(validation, 2)

    results = {
        "protocol": "P1-C3 nonprivileged diagnostic",
        "noise_std_m": SIGMA,
        "validation_pairs": 80,
        "test_pairs": 120,
        "validation": val_meta,
        "test": test_meta,
        "A_only_accuracy": evaluate(
            test, 1, rule_a
        ),
        "B_only_accuracy": evaluate(
            test, 2, rule_b
        ),
        "A_rule": rule_a,
        "B_rule": rule_b,
        "note": (
            "Fixed physics-informed residual diagnostic. "
            "Not a learned world model. "
            "Test labels used only for evaluation."
        )
    }

    print(json.dumps(results, indent=2))

    (OUT / "p1c3_report.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    print("\nP1-C3 execution completed")
    print("Saved:", OUT / "p1c3_report.json")


if __name__ == "__main__":
    main()
