"""TwinWorld-v2 P2-B: build paired, randomized MuJoCo dynamics dataset.

Run in the user's existing TwinWorld_P0 project:
  .\\Giant\\Scripts\\python.exe scripts\\11_p2b_build_dataset.py

Requires scripts/10_p2a_three_mechanisms.py from P2-A. This script
produces *state-position* observations, not visual images, and does
not train any model.
"""
from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p2b"
OUT.mkdir(parents=True, exist_ok=True)

XML = run_path(str(ROOT / "scripts" / "10_p2a_three_mechanisms.py"))["xml"]
MODES = ("independent", "rigid", "elastic")
DT = 0.002
PASSIVE_STEPS = 500
ACTIVE_STEPS = 500
STRIDE = 10  # sampled positions at 50 Hz
NOISE_STD = 0.005  # m, applied to public observations only
KP, KD = 250.0, 40.0

SPLITS = {
    "train": {"seed": 1101, "pairs": 60, "mass": (0.8, 1.4),
              "spring": (7.0, 16.0), "damper": (0.6, 2.0)},
    "val": {"seed": 2202, "pairs": 20, "mass": (0.8, 1.4),
            "spring": (7.0, 16.0), "damper": (0.6, 2.0)},
    "test": {"seed": 3303, "pairs": 30, "mass": (1.45, 2.0),
             "spring": (3.0, 9.0), "damper": (0.3, 1.6)},
}


def acceleration_at(step: int, magnitude: float, profile: int) -> float:
    """Three deterministic target-motion programs, identical within a triplet."""
    if profile == 0:  # continuous acceleration
        return magnitude
    if profile == 1:  # initial pulse, then coast
        return magnitude if step < ACTIVE_STEPS // 2 else 0.0
    if profile == 2:  # acceleration followed by deceleration
        return magnitude if step < ACTIVE_STEPS // 2 else -magnitude
    raise ValueError(f"Invalid profile {profile}")


def rollout(mode: str, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return passive, active observations and commanded acceleration profile.

    No mechanism metadata, control force, mass, tendon, or qfrc is returned.
    These are noiseless world-position arrays for internal dataset audit.
    """
    ma, mb = params["mass_a"], params["mass_b"]
    assert XML.count('mass="1"') == 2, "Upstream XML schema changed"
    xml = (XML.replace('mass="1"', f'mass="{ma:.8f}"', 1)
              .replace('mass="1"', f'mass="{mb:.8f}"', 1))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    assert model.neq == 1 and model.ntendon == 1

    data.eq_active[0] = int(mode == "rigid")
    model.tendon_stiffness[0] = params["spring"] if mode == "elastic" else 0.0
    model.tendon_damping[0] = params["damper"] if mode == "elastic" else 0.0
    data.qpos[:] = params["offset"]
    data.qvel[:] = params["velocity"]
    mujoco.mj_forward(model, data)

    def positions() -> np.ndarray:
        return np.array([data.qpos[0], 1.2 + data.qpos[1]], dtype=np.float64)

    passive = [positions()]
    for _ in range(PASSIVE_STEPS):
        data.ctrl[0] = 0.0
        mujoco.mj_step(model, data)
        passive.append(positions())

    x_desired = float(data.qpos[0])
    v_desired = params["velocity"]
    active = [positions()]
    commands = [0.0]
    for i in range(ACTIVE_STEPS):
        a_command = acceleration_at(i, params["accel"], params["profile"])
        v_desired += a_command * DT
        x_desired += v_desired * DT
        data.ctrl[0] = (KP * (x_desired - data.qpos[0])
                        + KD * (v_desired - data.qvel[0]))
        mujoco.mj_step(model, data)
        active.append(positions())
        commands.append(a_command)

    p = np.asarray(passive)
    a = np.asarray(active)
    c = np.asarray(commands)
    assert p.shape == (501, 2) and a.shape == (501, 2)
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(a))
    return p, a, c


def main() -> None:
    audit: dict = {"mujoco_version": mujoco.__version__, "splits": {}}
    for split, cfg in SPLITS.items():
        rng = np.random.default_rng(cfg["seed"])
        public_rows = []
        private_rows = []
        passive_diffs = []
        a_gaps = []
        pairwise_b_final = []

        for pair_id in range(cfg["pairs"]):
            params = {
                "mass_a": float(rng.uniform(*cfg["mass"])),
                "mass_b": float(rng.uniform(*cfg["mass"])),
                "spring": float(rng.uniform(*cfg["spring"])),
                "damper": float(rng.uniform(*cfg["damper"])),
                "velocity": float(rng.uniform(0.05, 0.25)),
                "offset": float(rng.uniform(-0.2, 0.2)),
                "accel": float(rng.uniform(0.15, 0.60)),
                "profile": int(rng.integers(0, 3)),
            }
            worlds = {mode: rollout(mode, params) for mode in MODES}
            baseline_passive = worlds["independent"][0]
            passive_diff = max(float(np.max(np.abs(
                worlds[mode][0] - baseline_passive))) for mode in MODES)
            passive_diffs.append(passive_diff)
            assert passive_diff < 1e-8, (split, pair_id, passive_diff)

            commanded = worlds["independent"][2]
            assert all(np.array_equal(commanded, worlds[mode][2])
                       for mode in MODES)
            active_a = np.stack([worlds[mode][1][:, 0] for mode in MODES])
            a_gaps.append(float(np.max(np.ptp(active_a, axis=0))))
            b_final = [worlds[mode][1][-1, 1] for mode in MODES]
            pairwise_b_final.append([
                abs(b_final[0] - b_final[1]),
                abs(b_final[0] - b_final[2]),
                abs(b_final[1] - b_final[2]),
            ])

            # Identical passive sensor noise *within the matched triplet*.
            # It is independent of mechanism, thus cannot encode a class label.
            passive_sensor_noise = rng.normal(
                0, NOISE_STD, size=(51, 2))

            for label, mode in enumerate(MODES):
                p, a, c = worlds[mode]
                p_obs = p[::STRIDE] + passive_sensor_noise
                a_obs = a[::STRIDE] + rng.normal(0, NOISE_STD, (51, 2))
                public_rows.append((p_obs, a_obs, c[::STRIDE]))
                private_rows.append({"label": label, "pair_id": pair_id,
                                     **params})

        # Shuffle episodes: never leave the class order 0,1,2 in public arrays.
        permutation = rng.permutation(len(public_rows))
        obs_p = np.stack([public_rows[i][0] for i in permutation]).astype('f4')
        obs_a = np.stack([public_rows[i][1] for i in permutation]).astype('f4')
        cmd = np.stack([public_rows[i][2] for i in permutation]).astype('f4')
        hidden = [private_rows[i] for i in permutation]

        # Model-facing file: labels and physical parameters NEVER included.
        np.savez_compressed(
            OUT / f"{split}_observations.npz",
            passive_xy=obs_p, active_xy=obs_a,
            commanded_accel=cmd,
        )
        # Evaluator/training-label file: use train labels for supervised
        # baselines; test labels must be opened only by evaluation code.
        np.savez_compressed(
            OUT / f"{split}_private.npz",
            mechanism=np.array([r["label"] for r in hidden], dtype='i1'),
            pair_id=np.array([r["pair_id"] for r in hidden], dtype='i4'),
            mass_a=np.array([r["mass_a"] for r in hidden], dtype='f8'),
            mass_b=np.array([r["mass_b"] for r in hidden], dtype='f8'),
            spring=np.array([r["spring"] for r in hidden], dtype='f8'),
            damper=np.array([r["damper"] for r in hidden], dtype='f8'),
            velocity=np.array([r["velocity"] for r in hidden], dtype='f8'),
            offset=np.array([r["offset"] for r in hidden], dtype='f8'),
            accel=np.array([r["accel"] for r in hidden], dtype='f8'),
            profile=np.array([r["profile"] for r in hidden], dtype='i1'),
        )

        assert obs_p.shape == (3 * cfg["pairs"], 51, 2)
        assert obs_a.shape == obs_p.shape
        assert cmd.shape == (3 * cfg["pairs"], 51)
        assert np.all(np.isfinite(obs_p)) and np.all(np.isfinite(obs_a))
        assert np.array_equal(np.bincount([r["label"] for r in hidden]),
                              np.full(3, cfg["pairs"]))

        b_gap = np.asarray(pairwise_b_final)
        split_audit = {
            "pairs": cfg["pairs"], "episodes": 3 * cfg["pairs"],
            "passive_max_abs_diff_noise_free_m": max(passive_diffs),
            "active_A_max_cross_mode_gap_m": max(a_gaps),
            "B_final_pairwise_gap_min_m": b_gap.min(axis=0).tolist(),
            "B_final_pairwise_gap_median_m": np.median(b_gap, axis=0).tolist(),
            "B_gap_order": ["independent_vs_rigid", "independent_vs_elastic", "rigid_vs_elastic"],
            "mechanism_counts": [cfg["pairs"]] * 3,
        }
        audit["splits"][split] = split_audit
        print(f"{split}: {split_audit}")

    manifest = {
        "protocol": "TwinWorld-v2 P2-B randomized state-trajectory dataset",
        "note": "Not RGB video, not visual concept revision, not trained performance.",
        "modes": list(MODES),
        "public_fields": ["passive_xy", "active_xy", "commanded_accel"],
        "private_fields": ["mechanism", "pair_id", "mass_a", "mass_b",
                           "spring", "damper", "velocity", "offset", "accel", "profile"],
        "sampling_hz": 50, "position_noise_std_m": NOISE_STD,
        "observation_shape_per_episode": [51, 2],
        "dt_simulation_seconds": DT, "target_motion_controller": {"kp": KP, "kd": KD},
        "splits": SPLITS,
        "data_policy": "Never pass test_private.npz, exact actual forces, or constraint states to a model.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\nP2-B DATASET BUILD PASS (engineering checks only)")
    print("Saved to:", OUT)


if __name__ == "__main__":
    main()
