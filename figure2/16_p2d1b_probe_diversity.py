"""P2-D1B: randomized 5-mechanism / 5-probe feasibility audit.

This is privileged offline counterfactual auditing, not an active agent.
Run inside the existing Giant Python environment in experiments/p2d1/.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
from runpy import run_path
import json

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "p2d1" / "results" / "p2d1b"
OUT.mkdir(parents=True, exist_ok=True)
BASE_XML = run_path(str(ROOT / "scripts" / "06_twinworld_mujoco.py"))["XML"]

SEED = 20261114
N_SCENES = 24
DT = 0.002
STEPS = 500
SAMPLE_EVERY = 10
KP, KD = 250.0, 40.0
MODES = ("independent", "rigid", "elastic", "deadzone", "nonlinear")
ACTIONS = ("steady", "pulse", "reversal", "oscillation", "sweep")
PAIRS = list(combinations(range(len(MODES)), 2))


def make_profiles():
    t = np.arange(STEPS) * DT
    raw = {
        "steady": np.ones(STEPS),
        "pulse": (t < 0.4).astype(float),
        "reversal": np.where(t < 0.25, 1.0, np.where(t < 0.5, -1.0, 0.0)),
        "oscillation": np.sin(2 * np.pi * 2.0 * (t + DT / 2)),
        "sweep": np.where(t < 0.5, 0.15, 0.65),
    }
    profiles = {}
    for name, values in raw.items():
        budget = np.sum(np.abs(values)) * DT
        assert budget > 0
        profiles[name] = values * (0.4 / budget)
        assert abs(np.sum(np.abs(profiles[name])) * DT - 0.4) < 1e-10
    return profiles


def sample_scene(rng):
    # The five mechanisms share all scene parameters within a scene.
    return {
        "mass_a": float(rng.uniform(0.8, 1.6)),
        "mass_b": float(rng.uniform(0.8, 1.6)),
        "velocity": float(rng.uniform(0.05, 0.25)),
        "k": float(rng.uniform(5.0, 16.0)),
        "c": float(rng.uniform(0.5, 2.2)),
        "gap": float(rng.uniform(0.015, 0.12)),
        "k3": float(rng.uniform(100.0, 850.0)),
    }


def make_model(scene):
    assert BASE_XML.count('mass="1"') == 2
    xml = BASE_XML.replace('mass="1"', f'mass="{scene["mass_a"]:.10f}"', 1)
    xml = xml.replace('mass="1"', f'mass="{scene["mass_b"]:.10f}"', 1)
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nv == 2 and model.neq == 1 and model.nu == 1
    return model


def internal_force(mode, r, vr, scene):
    k, c, gap, k3 = scene["k"], scene["c"], scene["gap"], scene["k3"]
    if mode in ("independent", "rigid"):
        return 0.0
    if mode == "elastic":
        return -k * r - c * vr
    if mode == "deadzone":
        if abs(r) <= gap:
            return 0.0
        return -k * np.sign(r) * (abs(r) - gap) - c * vr
    if mode == "nonlinear":
        return -k * r - k3 * r**3 - c * vr
    raise ValueError(mode)


def observe(data):
    return (float(data.qpos[0]), float(1.2 + data.qpos[1]))


def simulate(model, scene, mode, command):
    data = mujoco.MjData(model)
    data.eq_active[0] = int(mode == "rigid")
    data.qvel[:] = scene["velocity"]
    mujoco.mj_forward(model, data)

    passive = [observe(data)]
    for step in range(STEPS):
        data.ctrl[0] = 0.0
        data.qfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)
        if (step + 1) % SAMPLE_EVERY == 0:
            passive.append(observe(data))

    target_x = float(data.qpos[0])
    target_v = scene["velocity"]
    active = [observe(data)]
    abs_work = 0.0
    abs_impulse = 0.0
    for step, a in enumerate(command):
        target_v += a * DT
        target_x += target_v * DT
        u = float(KP * (target_x - data.qpos[0])
                  + KD * (target_v - data.qvel[0]))
        r = float(data.qpos[1] - data.qpos[0])
        vr = float(data.qvel[1] - data.qvel[0])
        f_b = internal_force(mode, r, vr, scene)
        data.qfrc_applied[:] = (-f_b, f_b)
        data.ctrl[0] = u
        abs_impulse += abs(u) * DT
        abs_work += abs(u * float(data.qvel[0])) * DT
        mujoco.mj_step(model, data)
        if (step + 1) % SAMPLE_EVERY == 0:
            active.append(observe(data))

    passive = np.asarray(passive, dtype=np.float64)
    active = np.asarray(active, dtype=np.float64)
    assert passive.shape == (51, 2) and active.shape == (51, 2)
    assert np.isfinite(passive).all() and np.isfinite(active).all()
    return passive, active, abs_work, abs_impulse


def main():
    rng = np.random.default_rng(SEED)
    profiles = make_profiles()
    traces = np.empty((N_SCENES, len(ACTIONS), len(MODES), 51, 2))
    passive_gap_max = 0.0
    a_gap_max = 0.0
    scores = np.empty((N_SCENES, len(ACTIONS)))
    pair_scores = np.empty((N_SCENES, len(ACTIONS), len(PAIRS)))
    works = np.empty((N_SCENES, len(ACTIONS), len(MODES)))
    impulses = np.empty_like(works)
    excursions = np.empty_like(works)
    private_scenes = []

    for si in range(N_SCENES):
        scene = sample_scene(rng)
        private_scenes.append(scene)
        model = make_model(scene)
        for ai, action in enumerate(ACTIONS):
            rollouts = [simulate(model, scene, mode, profiles[action])
                        for mode in MODES]
            passive = np.stack([r[0] for r in rollouts])
            active = np.stack([r[1] for r in rollouts])
            passive_gap_max = max(passive_gap_max,
                                  float(np.max(np.abs(passive - passive[:1]))))
            a_gap_max = max(a_gap_max,
                            float(np.max(np.ptp(active[:, :, 0], axis=0))))
            traces[si, ai] = active
            works[si, ai] = [r[2] for r in rollouts]
            impulses[si, ai] = [r[3] for r in rollouts]
            excursions[si, ai] = [float(np.max(np.abs(
                r[1][:, 0] - r[1][0, 0]))) for r in rollouts]
            for pi, (mi, mj) in enumerate(PAIRS):
                difference = active[mi, :, 1] - active[mj, :, 1]
                pair_scores[si, ai, pi] = float(np.sqrt(np.mean(difference**2)))
            scores[si, ai] = float(np.min(pair_scores[si, ai]))

    assert passive_gap_max < 1e-8, "Passive mismatch; inspect physical model"
    assert np.isfinite(traces).all() and np.isfinite(scores).all()
    assert np.all(works >= 0) and np.all(impulses >= 0)

    scene_winners = np.argmax(scores, axis=1)
    static_best_index = int(np.argmax(np.mean(scores, axis=0)))
    static_best = float(np.mean(scores[:, static_best_index]))
    privileged_best = float(np.mean(np.max(scores, axis=1)))
    mean_work = works.mean(axis=2)
    efficiency = scores / np.maximum(mean_work, 1e-8)
    pair_winners = np.argmax(pair_scores, axis=1)
    report = {
        "protocol": "P2-D1B randomized offline counterfactual probe audit",
        "engine": mujoco.__version__, "seed": SEED,
        "n_scenes": N_SCENES, "modes": list(MODES),
        "actions": list(ACTIONS), "n_rollouts": int(N_SCENES * len(ACTIONS) * len(MODES)),
        "nominal_command_budget_each": 0.4,
        "passive_max_gap_m": passive_gap_max,
        "A_cross_mechanism_max_gap_m": a_gap_max,
        "actions_summary": {
            action: {
                "mean_min_pair_B_RMS_m": float(np.mean(scores[:, ai])),
                "minimum_min_pair_B_RMS_m": float(np.min(scores[:, ai])),
                "mean_abs_work_proxy_J": float(np.mean(works[:, ai, :])),
                "mean_abs_impulse_Ns": float(np.mean(impulses[:, ai, :])),
                "mean_A_excursion_m": float(np.mean(excursions[:, ai, :])),
                "mean_score_over_work": float(np.mean(efficiency[:, ai])),
                "privileged_scene_wins": int(np.sum(scene_winners == ai)),
            }
            for ai, action in enumerate(ACTIONS)
        },
        "fixed_best_action": ACTIONS[static_best_index],
        "fixed_best_mean_score_m": static_best,
        "privileged_scene_oracle_mean_score_m": privileged_best,
        "privileged_adaptivity_headroom_fraction": float(
            (privileged_best - static_best) / max(static_best, 1e-12)
        ),
        "pairwise_best_action_counts": {
            f"{MODES[i]}_vs_{MODES[j]}": {
                ACTIONS[ai]: int(np.sum(pair_winners[:, pi] == ai))
                for ai in range(len(ACTIONS))
            }
            for pi, (i, j) in enumerate(PAIRS)
        },
        "notes": [
            "Privileged offline counterfactual audit, NOT a learned policy.",
            "Mechanism labels / all candidate outcomes are never agent inputs.",
            "Budget equality refers to commanded acceleration absolute integral, NOT real work.",
            "Score is B-trajectory RMS separation, NOT mutual information or classification accuracy.",
            "This is a development/feasibility set, NOT the final confirmatory test."
        ],
    }
    (OUT / "p2d1b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "p2d1b_audit_only.npz", active=traces,
                        scores=scores, pair_scores=pair_scores,
                        works=works, impulses=impulses, excursions=excursions)
    (OUT / "p2d1b_scene_parameters_PRIVATE.json").write_text(
        json.dumps(private_scenes, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=150)
    for ai, action in enumerate(ACTIONS):
        ax = axes.flat[ai]
        for mi, mode in enumerate(MODES):
            ax.plot(np.arange(51) * 0.02, traces[0, ai, mi, :, 1], label=mode)
        ax.set_title(action)
        ax.set_xlabel("Time since intervention (s)")
        ax.set_ylabel("B position (m)")
        ax.grid(alpha=0.25)
    axes.flat[5].axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "p2d1b_scene0_probes.png", bbox_inches="tight")
    plt.close(fig)

    print("P2-D1B randomized audit")
    print("scenes:", N_SCENES, "rollouts:", report["n_rollouts"])
    for action, info in report["actions_summary"].items():
        print(f"{action:12s} minpair_RMS={info['mean_min_pair_B_RMS_m']:.6f}m "
              f"abs_work={info['mean_abs_work_proxy_J']:.4f}J "
              f"wins={info['privileged_scene_wins']}")
    print("best fixed action:", report["fixed_best_action"])
    print("privileged headroom fraction:",
          f"{report['privileged_adaptivity_headroom_fraction']:.4%}")
    print("max passive gap:", passive_gap_max)
    print("max A leakage gap:", a_gap_max)
    print("P2-D1B ENGINEERING CHECKS PASS (NOT an active-agent result)")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
