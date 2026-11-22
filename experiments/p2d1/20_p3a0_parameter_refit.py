"""TwinWorld P3-A0: known-law parameter refitting and cross-excitation falsification.

DEVELOPMENT-ONLY physics experiment. This is NOT an AI concept-revision result.

Inputs to the inference routine: noisy observable A/B trajectories from a low-amplitude
identification intervention, and observable A of a separately executed high-amplitude
confirmation intervention. Ground-truth physical parameters and mechanism labels are
NEVER used by the inference routine. Ground truth is used only to generate/evaluate worlds.

Uses 16_p2d1b_probe_diversity.py unchanged for MuJoCo physics. The inference model
conditions on observed A (rather than claiming an exact full-actuator simulator model).
For linear elasticity, k and c are identifiable only through k/m_B and c/m_B here.

Run from repository root:
  .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\20_p3a0_parameter_refit.py --smoke
  .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\20_p3a0_parameter_refit.py

Needs: numpy, scipy, mujoco. Results are saved in results/p3a0/.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "p3a0"
DT_OBS = 0.02
SIGMA = 0.005
MODES = ("independent", "rigid", "elastic", "deadzone", "nonlinear")
KNOWN = MODES[:3]
IDENT_AMP = 1.0
CONFIRM_AMP = 2.0
SEED = 20261117


def noisy_observation(passive: np.ndarray, active: np.ndarray, rng: np.random.Generator):
    p = passive + rng.normal(0, SIGMA, passive.shape)
    a = active + rng.normal(0, SIGMA, active.shape)
    return p, a


class Probe:
    """A-conditioned B model using PUBLIC 50-Hz position observations only."""
    def __init__(self, passive: np.ndarray, active: np.ndarray):
        assert passive.shape == active.shape == (51, 2)
        self.t = np.arange(51) * DT_OBS
        tp = self.t - self.t[-1]  # passive samples cover t=-1,...,0
        v0 = np.empty(2)
        x0 = np.empty(2)
        for j in range(2):
            v0[j], x0[j] = np.polyfit(tp, passive[:, j], deg=1)
        self.b0, self.vb0 = float(x0[1]), float(v0[1])
        # Smooth A using public noisy positions; anchor to passive endpoint.
        smooth_a = savgol_filter(active[:, 0], window_length=11, polyorder=3)
        smooth_a = smooth_a - smooth_a[0] + x0[0]
        spline = CubicSpline(self.t, smooth_a, bc_type="natural")
        tm = np.arange(101) * (DT_OBS / 2)
        self.a_half = spline(tm)
        self.va_half = spline.derivative()(tm)
        self.a0 = float(self.a_half[0])
        self.y_b = active[:, 1].astype(float)

    def predict(self, structure: str, pars=None) -> np.ndarray:
        if structure == "independent":
            return self.b0 + self.vb0 * self.t
        if structure == "rigid":
            return self.b0 + (self.a_half[::2] - self.a0)
        if structure != "elastic":
            raise ValueError(structure)
        alpha, beta = float(pars[0]), float(pars[1])
        y = np.array([self.b0, self.vb0], dtype=float)
        result = [float(y[0])]
        h = DT_OBS

        def fun(state, half_idx):
            a, va = self.a_half[half_idx], self.va_half[half_idx]
            r = (state[0] - self.b0) - (a - self.a0)
            vr = state[1] - va
            return np.array([state[1], -alpha * r - beta * vr])

        for i in range(50):
            k1 = fun(y, 2 * i)
            k2 = fun(y + h * k1 / 2, 2 * i + 1)
            k3 = fun(y + h * k2 / 2, 2 * i + 1)
            k4 = fun(y + h * k3, 2 * i + 2)
            y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            result.append(float(y[0]))
        return np.asarray(result)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_candidates(identify: Probe):
    fits = {}
    for struct in ("independent", "rigid"):
        prediction = identify.predict(struct)
        residual = identify.y_b - prediction
        fits[struct] = {"pars": None, "fit_rms_m": rms(identify.y_b, prediction),
                        "bic": float(np.sum((residual / SIGMA) ** 2))}

    best_elastic = None
    for start in ([3.0, 1.0], [12.0, 2.0]):
        opt = least_squares(
            lambda th: (identify.predict("elastic", th) - identify.y_b) / SIGMA,
            np.asarray(start), bounds=([0.01, 0.001], [70.0, 15.0]),
            max_nfev=70, ftol=1e-6, xtol=1e-6, gtol=1e-6,
        )
        if best_elastic is None or np.sum(opt.fun ** 2) < np.sum(best_elastic.fun ** 2):
            best_elastic = opt
    fit_rms = rms(identify.y_b, identify.predict("elastic", best_elastic.x))
    fits["elastic"] = {"pars": [float(x) for x in best_elastic.x],
                       "fit_rms_m": fit_rms,
                       "bic": float(np.sum(best_elastic.fun ** 2) + 2 * np.log(51))}
    selected = min(fits, key=lambda k: fits[k]["bic"])
    return selected, fits


def analyze_episode(sim, scene, mechanism, profiles, rng):
    model = sim["make_model"](scene)
    command_id = profiles["pulse"] * IDENT_AMP
    command_confirm = profiles["pulse"] * CONFIRM_AMP
    passive_id, active_id, work_id, _ = sim["simulate"](model, scene, mechanism, command_id)
    passive_conf, active_conf, work_conf, _ = sim["simulate"](model, scene, mechanism, command_confirm)
    passive_gap = float(np.max(np.abs(passive_id - passive_conf)))
    assert passive_gap < 1e-8, "Mismatched passive starts across interventions"
    p_id, a_id = noisy_observation(passive_id, active_id, rng)
    p_cf, a_cf = noisy_observation(passive_conf, active_conf, rng)
    probe_id, probe_cf = Probe(p_id, a_id), Probe(p_cf, a_cf)
    selected, fits = fit_candidates(probe_id)
    confirm_scores = {name: rms(probe_cf.y_b, probe_cf.predict(name, fit["pars"]))
                      for name, fit in fits.items()}
    return {
        "structure_truth_PRIVATE_EVAL_ONLY": mechanism,
        "selected_known": selected,
        "fit_rms_m": fits[selected]["fit_rms_m"],
        "confirmed_selected_rms_m": confirm_scores[selected],
        "confirmed_min_known_rms_m": min(confirm_scores.values()),
        "fitted_alpha_beta_if_selected_elastic": fits["elastic"]["pars"],
        "confirmation_candidate_rms_m": confirm_scores,
        "work_identification_J": float(work_id),
        "work_confirmation_J": float(work_conf),
    }


def get_split(sim, profiles, seed, count, modes, mass_shift):
    rng_scene = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed + 50_000)
    rows = []
    for s in range(count):
        scene = sim["sample_scene"](rng_scene)
        if mass_shift:
            scene["mass_a"] = float(rng_scene.uniform(1.8, 2.3))
            scene["mass_b"] = float(rng_scene.uniform(1.8, 2.3))
        for mode in modes:
            result = analyze_episode(sim, scene, mode, profiles, rng_noise)
            result["scene_id"] = s
            rows.append(result)
    return rows


def grouped_rate(rows, condition, seed):
    # One independent scene group, not one episode, is the resampling unit.
    by_group = {}
    for r in rows:
        by_group.setdefault(r["scene_id"], []).append(float(condition(r)))
    per_scene = np.asarray([np.mean(by_group[k]) for k in sorted(by_group)])
    rng = np.random.default_rng(seed)
    samples = per_scene[rng.integers(len(per_scene), size=(2000, len(per_scene)))].mean(axis=1)
    return {"rate": float(per_scene.mean()), "scene_bootstrap_95_ci":
            [float(v) for v in np.quantile(samples, [0.025, 0.975])],
            "scenes": len(per_scene), "episodes": len(rows)}


def self_test():
    t = np.arange(51) * DT_OBS
    tp = t - 1.0
    v0 = 0.15
    pre = np.stack((v0 * (tp + 1), 1.2 + v0 * (tp + 1)), axis=1)
    act_a = v0 + v0 * t + 0.20 * t * t
    rigid_b = 1.35 + (act_a - act_a[0])
    probe = Probe(pre, np.stack((act_a, rigid_b), axis=1))
    assert rms(probe.predict("rigid"), rigid_b) < 2e-3
    assert rms(probe.predict("independent"), 1.35 + v0 * t) < 1e-8
    elastic_b = probe.predict("elastic", [8.0, 1.2])
    assert np.isfinite(elastic_b).all()
    select, fits = fit_candidates(Probe(pre, np.stack((act_a, elastic_b), axis=1)))
    assert fits["elastic"]["fit_rms_m"] < 0.003
    print("P3-A0 numerical self-test PASS (does not verify MuJoCo model fidelity)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 scenes per split; no statistical conclusion")
    args = ap.parse_args()
    self_test()
    sim = run_path(str(HERE / "16_p2d1b_probe_diversity.py"))
    assert tuple(sim["MODES"]) == MODES
    profiles = sim["make_profiles"]()
    n_cal, n_dev, n_shift = (2, 2, 2) if args.smoke else (12, 18, 18)
    calibration = get_split(sim, profiles, SEED + 11, n_cal, KNOWN, mass_shift=False)
    development = get_split(sim, profiles, SEED + 22, n_dev, MODES, mass_shift=False)
    mass_shift = get_split(sim, profiles, SEED + 33, n_shift, KNOWN, mass_shift=True)

    # Calibrate using ONLY known-law, independently generated calibration scenes.
    score_key = "confirmed_selected_rms_m"
    threshold = float(np.quantile([r[score_key] for r in calibration], 0.95))
    dev_known = [r for r in development if r["structure_truth_PRIVATE_EVAL_ONLY"] in KNOWN]
    dev_unknown = [r for r in development if r["structure_truth_PRIVATE_EVAL_ONLY"] not in KNOWN]
    eval_groups = {
        "id_known_false_rejection": (dev_known, lambda r: r[score_key] > threshold),
        "mass_shift_known_false_rejection": (mass_shift, lambda r: r[score_key] > threshold),
        "structure_unknown_rejection": (dev_unknown, lambda r: r[score_key] > threshold),
    }
    rates = {key: grouped_rate(rows, fn, SEED + i + 1)
             for i, (key, (rows, fn)) in enumerate(eval_groups.items())}
    breakdown = {mode: grouped_rate([r for r in dev_unknown if r["structure_truth_PRIVATE_EVAL_ONLY"] == mode],
                                    lambda r: r[score_key] > threshold, SEED + 7 + i)
                 for i, mode in enumerate(MODES[3:])}
    report = {
        "protocol": "P3-A0 A-conditioned known-law refit and held-out excitation",
        "development_only": True,
        "simulator": "MuJoCo simulator from 16_p2d1b_probe_diversity.py",
        "scene_groups": {"cal_known": n_cal, "dev_all": n_dev, "mass_shift_known": n_shift},
        "seed": SEED, "noise_std_m": SIGMA, "identification_pulse": IDENT_AMP,
        "confirmation_pulse": CONFIRM_AMP,
        "candidate_laws": list(KNOWN),
        "model_selection": "identification-only BIC, elastic alpha=k/mB beta=c/mB",
        "score": score_key, "known_only_calibration_q95_threshold_m": threshold,
        "rates": rates, "unknown_breakdown": breakdown,
        "known_model_selection_counts_on_dev": dict(Counter(r["selected_known"] for r in development)),
        "note": [
            "No RGB input, no causal concept edit, no active action-selection policy.",
            "No test-set tuning: all new scenes are exploratory development scenes.",
            "A-conditioned continuous B dynamics is an approximation to the discrete MuJoCo simulator.",
            "Sensor-noise assumptions, controller differences, and model approximation may produce false rejections.",
            "A-only leakage must be independently measured.",
            "The optimistic min-known confirmation residual is logged for audit, NOT used to select models.",
            "Exact mass and k/c cannot be separated from alpha/beta using conditional B dynamics alone.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "development"
    (OUT / f"p3a0_{suffix}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Keep labels only in audit file; do not expose these to future inference algorithms.
    (OUT / f"p3a0_{suffix}_audit_PRIVATE.json").write_text(json.dumps({
        "calibration": calibration, "development": development, "mass_shift": mass_shift
    }, indent=2), encoding="utf-8")
    print("\nP3-A0:", suffix)
    print("scenes:", report["scene_groups"], "calibrated threshold(m):", round(threshold, 6))
    for k, v in rates.items():
        print(f"{k}: {v['rate']:.3f}, scene CI95={v['scene_bootstrap_95_ci']}")
    for k, v in breakdown.items():
        print(f"  {k}: {v['rate']:.3f}")
    print("P3-A0 EXPERIMENT FINISHED (NOT concept revision)")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
