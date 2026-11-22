"""P3-A1B: evidence-gated *restricted* structural revision (development only).

Uses unchanged physical simulator in 16_p2d1b_probe_diversity.py and the
A-conditioned, noisy-position Probe from 20_p3a0_parameter_refit.py.

E1=1.0x and E2=2.0x: joint parameter fits and edit proposal (BIC).
E3=1.5x: calibration-compatible predeclared acceptance gate, NOT refitting
           or selecting among edits. The known-only data-matched comparator
           MAY use E3 for refitting, and must be reported.
E4=2.5x: completely held out until predictions & edit decisions are frozen.

All predictions are B positions CONDITIONAL on the observed A trajectory in
that probe, not open-loop 2-object motion forecasting. There is no RGB, no
unrestricted new concept discovery and no learned active action selection.

Run from repository root using Giant Python 3.11:
 .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\22_p3a1b_structural_revision.py --self-test-only
 .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\22_p3a1b_structural_revision.py --smoke
 .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\22_p3a1b_structural_revision.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from runpy import run_path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results" / "p3a1b"
SEED = 20261119
SIGMA = 0.005
DT_OBS = 0.02
AMPS = (1.0, 2.0, 1.5, 2.5)
KNOWN = ("independent", "rigid", "elastic")
EDITS = ("deadzone", "cubic")
ALL = KNOWN + ("deadzone", "nonlinear")
EDIT_GROUND_TRUTH = {"deadzone": "deadzone", "nonlinear": "cubic"}
BIC_MARGIN = 2.0
BOUNDS = {
    "elastic": ([0.01, 0.001], [70., 15.]),
    "deadzone": ([0.01, 0.001, 0.001], [70., 15., 0.20]),
    "cubic": ([0.01, 0.001, 0.01], [70., 15., 2500.]),
}
STARTS = {
    "elastic": ([6., 1.], [15., 2.]),
    "deadzone": ([6., 1., 0.03], [12., 1.5, 0.09]),
    "cubic": ([6., 1., 100.], [12., 1.5, 800.]),
}

# Pure numerical code from the same A-conditioned model family as P3-A0.
base = run_path(str(HERE / "20_p3a0_parameter_refit.py"))
BaseProbe = base["Probe"]
noisy_observation = base["noisy_observation"]


class RevisionProbe(BaseProbe):
    """Add two candidate force laws without changing earlier files.

    r = (x_B-b0)-(x_A-a0), vr=v_B-v_A.
    deadzone implements the exact deadzone logic in simulator script 16:
    both stiffness and damping vanish inside |r|<=gap.
    Cubic uses effective alpha=k/m_B, beta=c/m_B, gamma=k3/m_B.
    """
    def predict(self, structure, pars=None):
        if structure in KNOWN:
            return super().predict(structure, pars)
        if structure not in EDITS:
            raise ValueError(f"Unknown law {structure}")
        alpha, beta, extra = map(float, pars)
        y = np.array([self.b0, self.vb0], dtype=float)
        trajectory = [float(y[0])]
        h = DT_OBS

        def derivative(state, idx):
            a, va = self.a_half[idx], self.va_half[idx]
            r = (state[0] - self.b0) - (a - self.a0)
            vr = state[1] - va
            if structure == "deadzone":
                if abs(r) <= extra:
                    force = 0.0
                else:
                    force = (-alpha * np.sign(r) * (abs(r) - extra)
                             - beta * vr)
            else:
                force = -alpha * r - beta * vr - extra * r ** 3
            return np.array((state[1], force), dtype=float)

        # Identical 50-step RK4 convention as the old Probe.
        for i in range(50):
            k1 = derivative(y, 2 * i)
            k2 = derivative(y + h * k1 / 2, 2 * i + 1)
            k3 = derivative(y + h * k2 / 2, 2 * i + 1)
            k4 = derivative(y + h * k3, 2 * i + 2)
            y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
            trajectory.append(float(y[0]))
        prediction = np.asarray(trajectory)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite trajectory in candidate model")
        return prediction


def prediction_rms(p: RevisionProbe, fit: dict) -> float:
    y = p.predict(fit["kind"], fit["params"])
    return float(np.sqrt(np.mean((p.y_b - y) ** 2)))


def fit_one(probes: list[RevisionProbe], structure: str, nfev: int) -> dict:
    n = sum(len(p.y_b) for p in probes)
    if structure in ("independent", "rigid"):
        pars, complexity = None, 0
        residual = np.concatenate([
            (p.predict(structure) - p.y_b) / SIGMA for p in probes
        ])
        success = True
    else:
        lower, upper = BOUNDS[structure]
        best = None
        for start in STARTS[structure]:
            opt = least_squares(
                lambda th: np.concatenate([
                    (p.predict(structure, th) - p.y_b) / SIGMA
                    for p in probes
                ]),
                x0=np.asarray(start, float),
                bounds=(lower, upper),
                max_nfev=nfev, ftol=1e-5, xtol=1e-5, gtol=1e-5,
            )
            cost = float(np.dot(opt.fun, opt.fun))
            if best is None or cost < best[0]:
                best = (cost, opt)
        assert best is not None
        opt = best[1]
        pars = list(map(float, opt.x))
        complexity = len(pars)
        residual = opt.fun
        success = bool(opt.success)
    rss_norm = float(np.dot(residual, residual))
    return {
        "kind": structure, "params": pars, "k": complexity,
        "bic": rss_norm + complexity * np.log(n),
        "fit_rms_m": float(np.sqrt(rss_norm / n) * SIGMA),
        "optimizer_success": success,
    }


def fit_library(probes, laws, nfev=40):
    fits = {law: fit_one(probes, law, nfev) for law in laws}
    chosen = min(fits, key=lambda k: fits[k]["bic"])
    return fits[chosen], fits


def evaluate_early(probes, nfev):
    # E4 is never passed to this function.
    assert len(probes) == 3
    known, known_fits = fit_library(probes[:2], KNOWN, nfev)
    edit, edit_fits = fit_library(probes[:2], EDITS, nfev)
    known_e3 = prediction_rms(probes[2], known)
    edit_e3 = prediction_rms(probes[2], edit)
    return {
        "known": known, "edit": edit,
        "known_e3": known_e3, "edit_e3": edit_e3,
        "gain_e3": known_e3 - edit_e3,
        "bic_advantage": known["bic"] - edit["bic"],
        "known_three": fit_library(probes, KNOWN, nfev)[0],
        "unrestricted_12": min((known, edit), key=lambda x: x["bic"]),
        "optimizer_success_by_family": {
            **{k: v["optimizer_success"] for k, v in known_fits.items()},
            **{k: v["optimizer_success"] for k, v in edit_fits.items()},
        },
    }


def decision(e, cutoff):
    # E1/E2 select ONE candidate edit; E3 only gates it. No optimizing on E3.
    if e["known_e3"] <= cutoff["known_e3_q95_m"]:
        return "keep", e["known"]
    if (e["bic_advantage"] > BIC_MARGIN
        and e["gain_e3"] > cutoff["positive_gain_q95_m"]
        and e["edit_e3"] <= cutoff["edit_e3_q95_m"]):
        return "edit", e["edit"]
    return "inconclusive", e["known"]


def numeric_self_test():
    # Pure test independent of MuJoCo or private simulator truth.
    t = np.arange(51) * DT_OBS
    tp = t - 1
    pre = np.stack((0.15 + .15 * (tp+1), 1.20 + .15 * (tp+1)), axis=1)
    x_a = .30 + .15*t + .25*t*t
    dummy = RevisionProbe(pre, np.stack((x_a, 1.35+.15*t), axis=1))
    for law, pars in (("deadzone", [8., 1.2, .035]),
                      ("cubic", [8., 1.2, 250.])):
        y = dummy.predict(law, pars)
        assert y.shape == (51,) and np.isfinite(y).all()
        probe = RevisionProbe(pre, np.stack((x_a, y), axis=1))
        assert prediction_rms(probe, {"kind": law, "params": pars}) < 1e-8
    assert np.allclose(dummy.predict("independent"), 1.35 + .15*t, atol=1e-8)
    # Rule decision unit checks (independent of ground truth and E4).
    fake = {"known_e3": .005, "edit_e3": .003,
            "bic_advantage": 50, "gain_e3": .002,
            "known": {"kind": "elastic"}, "edit": {"kind": "cubic"}}
    cutoff = {"known_e3_q95_m": .010, "edit_e3_q95_m": .010,
              "positive_gain_q95_m": .001}
    assert decision(fake, cutoff)[0] == "keep"
    fake["known_e3"] = .020
    fake["gain_e3"] = .017
    assert decision(fake, cutoff)[0] == "edit"
    fake["edit_e3"] = .012
    assert decision(fake, cutoff)[0] == "inconclusive"
    assert abs(BIC_MARGIN-2.0)<1e-12
    print("P3-A1B PURE NUMERICAL SELF-TEST PASS (not MuJoCo validation)")


def make_episodes(sim, profiles, n_scenes, modes, seed, shifted, nfev):
    scene_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed+50000)
    episodes = []
    gaps = []
    for scene_id in range(n_scenes):
        scene = sim["sample_scene"](scene_rng)
        if shifted:
            scene["mass_a"] = float(scene_rng.uniform(1.8, 2.3))
            scene["mass_b"] = float(scene_rng.uniform(1.8, 2.3))
        model = sim["make_model"](scene)
        # Physically paired counterfactuals are generated ONLY for audit,
        # never passed to an inference decision on one episode.
        a_per_mode = []
        p_per_mode = []
        for mode in modes:
            probes = []
            works = []
            clean_a = []
            clean_p = []
            for amp in AMPS:
                passive, active, work, _ = sim["simulate"](
                    model, scene, mode, profiles["pulse"] * amp
                )
                clean_a.append(active)
                clean_p.append(passive)
                pp, aa = noisy_observation(passive, active, noise_rng)
                probes.append(RevisionProbe(pp, aa))
                works.append(float(work))
            p_per_mode.append(clean_p)
            a_per_mode.append(clean_a)
            assert len(probes)==4
            # Design guarantees at most E1-E3 is visible to inference.
            inferred = evaluate_early(probes[:3], nfev)
            episodes.append({
                "scene_id": scene_id,
                "truth_PRIVATE": mode, "inferred": inferred,
                "E4": probes[3], "works_J": works,
            })
        p_per_mode = np.asarray(p_per_mode)
        a_per_mode = np.asarray(a_per_mode)
        gaps.append({
            "max_passive_gap_m": float(np.max(np.abs(
                p_per_mode - p_per_mode[:1]))),
            "max_cross_mode_A_gap_m": float(np.max(np.ptp(
                a_per_mode[:, :, :, 0], axis=0))),
        })
    return episodes, gaps


def calibrate(episodes):
    # ONLY known-mechanism calibration scenes. No unknown/test label tuning.
    e = [row["inferred"] for row in episodes]
    known_scores = [q["known_e3"] for q in e]
    edit_scores = [q["edit_e3"] for q in e]
    positive_gains = [max(q["gain_e3"], 0.) for q in e]
    cutoff = {
        "known_e3_q95_m": float(np.quantile(known_scores, .95)),
        "edit_e3_q95_m": float(np.quantile(edit_scores, .95)),
        "positive_gain_q95_m": float(np.quantile(positive_gains, .95)),
    }
    return cutoff


def grouped_bootstrap(rows, predicate, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_id"]].append(float(predicate(row)))
    if not groups:
        return None
    group_rates = np.asarray([np.mean(groups[k]) for k in sorted(groups)])
    rng = np.random.default_rng(seed)
    draw = group_rates[rng.integers(
        len(group_rates), size=(2000, len(group_rates))
    )].mean(axis=1)
    return {
        "rate": float(group_rates.mean()),
        "CI95_scene_bootstrap": list(map(float, np.quantile(draw, [.025, .975]))),
        "scenes": len(groups), "episodes": len(rows)
    }


def compile_results(episodes, cutoff, seed):
    rows = []
    for ep in episodes:
        e, e4 = ep["inferred"], ep["E4"]
        status, selected = decision(e, cutoff)
        # E4 is FIRST touched here, after decisions have been finalized.
        score = {
            "known_12": prediction_rms(e4, e["known"]),
            "known_123": prediction_rms(e4, e["known_three"]),
            "unrestricted_12": prediction_rms(e4, e["unrestricted_12"]),
            "gated_revision": prediction_rms(e4, selected),
        }
        rows.append({
            "scene_id": ep["scene_id"],
            "truth_PRIVATE_EVAL_ONLY": ep["truth_PRIVATE"],
            "known_kind": e["known"]["kind"],
            "edit_proposal_kind": e["edit"]["kind"],
            "decision": status,
            "final_kind": selected["kind"],
            "known_e3_m": e["known_e3"],
            "edit_e3_m": e["edit_e3"],
            "gain_e3_m": e["gain_e3"],
            "bic_advantage": e["bic_advantage"],
            "E4_RMS_m": score,
            "total_work_proxy_J": sum(ep["works_J"]),
            "optimizer_success": e["optimizer_success_by_family"],
        })
    known = [r for r in rows if r["truth_PRIVATE_EVAL_ONLY"] in KNOWN]
    unknown = [r for r in rows if r["truth_PRIVATE_EVAL_ONLY"] in EDIT_GROUND_TRUTH]

    stats = {
        "n_scenes": len(set(r["scene_id"] for r in rows)),
        "n_episodes": len(rows),
        "decision_counts": dict(Counter(r["decision"] for r in rows)),
        "E4_RMS_m_mean_by_method": {
            method: float(np.mean([r["E4_RMS_m"][method] for r in rows]))
            for method in ("known_12", "known_123", "unrestricted_12", "gated_revision")
        },
        "E4_RMS_m_median_by_method": {
            method: float(np.median([r["E4_RMS_m"][method] for r in rows]))
            for method in ("known_12", "known_123", "unrestricted_12", "gated_revision")
        },
        "E4_mean_by_truth_m": {
            truth: {
                method: float(np.mean([r["E4_RMS_m"][method] for r in rows
                                     if r["truth_PRIVATE_EVAL_ONLY"] == truth]))
                for method in ("known_12", "known_123", "unrestricted_12", "gated_revision")
            }
            for truth in sorted(set(r["truth_PRIVATE_EVAL_ONLY"] for r in rows))
        },
        "optimizer_failure_episodes": int(sum(
            not all(r["optimizer_success"].values()) for r in rows
        )),
        "false_edit_known": grouped_bootstrap(
            known, lambda r: r["decision"] == "edit", seed + 1),
        "inconclusive_known": grouped_bootstrap(
            known, lambda r: r["decision"] == "inconclusive", seed + 2),
        "correct_edit_unknown": grouped_bootstrap(
            unknown, lambda r: (r["decision"] == "edit" and
               r["final_kind"] == EDIT_GROUND_TRUTH[r["truth_PRIVATE_EVAL_ONLY"]]),
            seed + 3),
        "any_edit_unknown": grouped_bootstrap(
            unknown, lambda r: r["decision"] == "edit", seed + 4),
        "inconclusive_unknown": grouped_bootstrap(
            unknown, lambda r: r["decision"] == "inconclusive", seed + 5),
        "wrong_edit_unknown": grouped_bootstrap(
            unknown, lambda r: r["decision"] == "edit" and
                r["final_kind"] != EDIT_GROUND_TRUTH[r["truth_PRIVATE_EVAL_ONLY"]],
            seed + 6),
        "correct_edit_by_unknown_kind": {
            mode: grouped_bootstrap([r for r in unknown if r["truth_PRIVATE_EVAL_ONLY"] == mode],
                lambda r: r["decision"] == "edit" and
                    r["final_kind"] == EDIT_GROUND_TRUTH[r["truth_PRIVATE_EVAL_ONLY"]],
                seed + 7 + i)
            for i, mode in enumerate(EDIT_GROUND_TRUTH)
        },
    }
    return rows, stats


def make_report(cal, dev, shift, cal_gaps, dev_gaps, shift_gaps, cutoff, smoke):
    cal_rows, cal_stats = compile_results(cal, cutoff, SEED + 100)
    dev_rows, dev_stats = compile_results(dev, cutoff, SEED + 200)
    shift_rows, shift_stats = compile_results(shift, cutoff, SEED + 300)
    report = {
        "protocol": "P3-A1B restricted structural edits, 4 excitations",
        "development_only": True,
        "smoke": smoke,
        "seed": SEED,
        "noise_std_m": SIGMA,
        "amplitudes": list(AMPS),
        "calibration_only_known": True,
        "calibration_thresholds": cutoff,
        "bic_advantage_threshold": BIC_MARGIN,
        "split_scene_counts": {"calibration": len(set(x["scene_id"] for x in cal)),
            "development": len(set(x["scene_id"] for x in dev)),
            "mass_shift": len(set(x["scene_id"] for x in shift))},
        "split_stats": {
            "calibration_in_sample": cal_stats,
            "development": dev_stats,
            "mass_shift": shift_stats,
        },
        "audit_physics": {
            "max_passive_gap_m": max(x["max_passive_gap_m"] for x in (cal_gaps+dev_gaps+shift_gaps)),
            "max_A_cross_mode_gap_m": max(x["max_cross_mode_A_gap_m"] for x in (cal_gaps+dev_gaps+shift_gaps)),
        },
        "constraints": [
            "Candidate edits are hand-specified (deadzone, cubic) and not discovered from scratch.",
            "E4 was not used in model fitting, editing choice, gate selection or calibration.",
            "B predictions condition on observed E4 A, not full counterfactual prediction.",
            "Known_123 is matched-E3-information known-only baseline.",
            "BIC treats temporally correlated sampled trajectories as independent residuals; descriptive proxy.",
            "No RGB; no adaptive action selection. Model fitting uses observed B and A only.",
            "Optimistic simulation and handcrafted model families; development, not confirmatory evidence.",
        ],
    }
    return report, {"calibration": cal_rows, "development": dev_rows, "mass_shift": shift_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    numeric_self_test()
    if args.self_test_only:
        return
    sim = run_path(str(HERE / "16_p2d1b_probe_diversity.py"))
    assert tuple(sim["MODES"]) == ALL
    prof = sim["make_profiles"]()
    nfev = 25 if args.smoke else 50
    ncal, ndev, nshift = ((2, 2, 2) if args.smoke else (12, 18, 18))
    cal, cg = make_episodes(sim, prof, ncal, KNOWN, SEED + 11, False, nfev)
    dev, dg = make_episodes(sim, prof, ndev, ALL, SEED + 22, False, nfev)
    shift, sg = make_episodes(sim, prof, nshift, KNOWN, SEED + 33, True, nfev)
    cutoff = calibrate(cal)
    report, private = make_report(cal, dev, shift, cg, dg, sg, cutoff, args.smoke)
    report["mujoco_version"] = getattr(sim.get("mujoco", None), "__version__", "synthetic-or-unknown")
    report["source_sha256"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("16_p2d1b_probe_diversity.py", "20_p3a0_parameter_refit.py",
                     "21_p3a1a_reconsider_known.py") if (HERE / name).exists()
    }
    assert report["audit_physics"]["max_passive_gap_m"] < 1e-8
    suffix = "smoke" if args.smoke else "development"
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / f"p3a1b_{suffix}_report.json"
    private_path = OUT / f"p3a1b_{suffix}_audit_PRIVATE.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    private_path.write_text(json.dumps(private, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"P3-A1B {suffix}: calibration / development / mass shift scenes = {ncal}/{ndev}/{nshift}")
    print("calibration E3 thresholds (mm):", {k: round(v*1000, 3) for k,v in cutoff.items()})
    for name in ("development", "mass_shift"):
        st = report["split_stats"][name]
        print(f"{name} | decisions={st['decision_counts']}")
        for key in ("false_edit_known", "inconclusive_known", "correct_edit_unknown", "inconclusive_unknown"):
            if st[key] is not None:
                print(f"  {key}: {st[key]['rate']:.3f}, sceneCI={st[key]['CI95_scene_bootstrap']}")
        print("  E4 mean RMS(mm):", {k: round(v*1000, 3) for k,v in st["E4_RMS_m_mean_by_method"].items()})
    print("P3-A1B EXECUTION COMPLETE (development only, NOT unrestricted concept discovery)")
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
