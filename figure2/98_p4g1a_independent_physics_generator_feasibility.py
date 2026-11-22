"""P4-G1A / Experiment #98: independent dynamics generator feasibility.

This is a PREPARATORY, TRUTH-AWARE generator audit, not a mechanism-identification
benchmark, independent-hardware claim, or a MuJoCo re-run. No fitting, test labels
for decisions, E4, action choice, thresholds, or historical files are used.

Two predeclared arms:
  IDEAL_ODE: standalone Newtonian two-mass solver, same three force-law forms.
  SHIFTED_ODE: same standalone solver with common per-mass ground drag and
               first-order actuation lag (new dynamics absent from candidate fit).
Both are integrated using a self-contained RK4 stepper, not MuJoCo; G1B will
only be launched if shapes, invariants, finite states and nontrivial physical
shift pass. Do NOT describe these as two independent simulator engines; they are
one independent implementation with two fixed dynamical assumptions.

Grounding: original P2D1B physical force laws and controller coefficients;
M2 mismatch ratios and four-train plus Mid/Half/Full commands are copied from
frozen protocols. Because base scene sampling is freshly implemented, its
exact distributional equivalence to earlier MuJoCo scene samplers is NOT
asserted. All G1A results are feasibility diagnostics, never scientific effect
estimates. The main SCI paper's E4 is never touched.

Run from anywhere:
  python 98_p4g1a_independent_physics_generator_feasibility.py --smoke
  python 98_p4g1a_independent_physics_generator_feasibility.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

SELF = Path(__file__).resolve()
HERE = SELF.parent
OUT = HERE / "results" / "p4g1a"
DT, STEPS, SAMPLE_EVERY = 0.002, 500, 10
KP, KD = 250.0, 40.0
N_SMOKE, N_FEASIBILITY = 1, 4
SEED, SMOKE_SEED = 20291201, 20391201
BANDS = ("original", "above_original_range")
CONDITIONS = {
    "elastic": (("matched", "M0"), ("alpha_high", "M2")),
    "deadzone": (("matched", "M0"), ("alpha_high", "M2"), ("gap_high", "M2")),
    "nonlinear": (("matched", "M0"), ("alpha_high", "M2"), ("gamma_high", "M2")),
}
# Fixed physical shifts; no post-result selection or tuning allowed.
ARMS = {
    "IDEAL_ODE": {"viscous_rate_s_inv": 0.0, "dry_acc_ms2": 0.0,
                  "velocity_smoothing_ms": 0.05, "actuator_tau_s": 0.0},
    "SHIFTED_ODE": {"viscous_rate_s_inv": 0.08, "dry_acc_ms2": 0.02,
                    "velocity_smoothing_ms": 0.05, "actuator_tau_s": 0.025},
}
ACTION_AMPS = {"e1": 0.25, "e2": 0.50, "weak": 0.75,
               "strong": 2.25, "mid": 1.75, "HALF": 1.375, "FULL": 2.75}
ACTIONS = tuple(ACTION_AMPS)


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest_object(obj):
    txt = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".writing")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def sample_scene(rng, band):
    # Same nominal supports as the historical mechanism study. Not an attempt
    # to reproduce old scene-seed pseudorandom streams.
    gap_range = (0.015, 0.12) if band == "original" else (0.12, 0.18)
    return {
        "mass_a": float(rng.uniform(0.8, 1.6)),
        "mass_b": float(rng.uniform(0.8, 1.6)),
        "velocity": float(rng.uniform(0.05, 0.25)),
        "k": float(rng.uniform(5.0, 16.0)),
        "c": float(rng.uniform(0.5, 2.2)),
        "gap": float(rng.uniform(*gap_range)),
        "k3": float(rng.uniform(100.0, 850.0)),
    }


def apply_condition(scene, mode, axis, level):
    changed = dict(scene)
    if level == "M0":
        need(axis == "matched", "Invalid M0 matching axis")
    elif level == "M2":
        if axis == "alpha_high":
            changed["k"] = 20.0 * changed["mass_b"] * 1.10
        elif axis == "gamma_high" and mode == "nonlinear":
            changed["k3"] = 1062.5 * changed["mass_b"] * 1.10
        elif axis == "gap_high" and mode == "deadzone":
            changed["gap"] = 0.18 * 1.05
        else:
            raise RuntimeError(f"Invalid M2 condition {mode}/{axis}")
    else:
        raise RuntimeError(f"Unknown level {level}")
    return changed


def force_b(mode, relative_x, relative_v, scene):
    k, c = scene["k"], scene["c"]
    if mode == "elastic":
        return -k * relative_x - c * relative_v
    if mode == "deadzone":
        g = scene["gap"]
        if abs(relative_x) <= g:
            return 0.0
        return -k * math.copysign(abs(relative_x)-g, relative_x) - c * relative_v
    if mode == "nonlinear":
        return -k * relative_x - scene["k3"] * relative_x**3 - c * relative_v
    raise RuntimeError(f"Unsupported mode: {mode}")


def derivatives(y, u_desired, scene, mode, dynamics):
    xa, xb, va, vb, ua = y
    f_b = force_b(mode, xb-xa, vb-va, scene)
    v_rate = dynamics["viscous_rate_s_inv"]
    f_rate = dynamics["dry_acc_ms2"]
    v_smooth = dynamics["velocity_smoothing_ms"]
    def drag_acc(v):
        return v_rate*v + f_rate*math.tanh(v/v_smooth)
    tau = dynamics["actuator_tau_s"]
    actual_u = ua if tau else u_desired
    return (va, vb,
            (actual_u-f_b)/scene["mass_a"] - drag_acc(va),
            f_b/scene["mass_b"] - drag_acc(vb),
            (u_desired-ua)/tau if tau else 0.0)


def rk4_step(y, dt, u, scene, mode, dynamics):
    def add(x, k, factor):
        return tuple(a+factor*b for a,b in zip(x,k))
    k1 = derivatives(y, u, scene, mode, dynamics)
    k2 = derivatives(add(y,k1,dt/2), u, scene, mode, dynamics)
    k3 = derivatives(add(y,k2,dt/2), u, scene, mode, dynamics)
    k4 = derivatives(add(y,k3,dt), u, scene, mode, dynamics)
    return tuple(float(x + dt*(a+2*b+2*c+d)/6)
                 for x,a,b,c,d in zip(y,k1,k2,k3,k4))


def command_profiles():
    # Frozen P2D1B pulse is 1 for t<0.4, 0 elsewhere, with integral 0.4.
    t = np.arange(STEPS) * DT
    pulse = (t < 0.4).astype(float)
    need(abs(float(pulse.sum()*DT)-0.4)<1e-12, "Pulse budget changed")
    return {name: pulse*amp for name,amp in ACTION_AMPS.items()}


def simulate(scene, mode, command, dynamics):
    cmd = np.asarray(command, dtype=float)
    need(cmd.shape == (STEPS,) and np.isfinite(cmd).all(), "Invalid command")
    # Relative coordinates start at zero; world-visible B is offset by 1.2m.
    v0 = scene["velocity"]
    state = (0.0, 0.0, v0, v0, 0.0)
    def observation(s):
        return (s[0], 1.2+s[1])
    passive = [observation(state)]
    for step in range(STEPS):
        state = rk4_step(state, DT, 0.0, scene, mode, dynamics)
        if (step+1) % SAMPLE_EVERY == 0:
            passive.append(observation(state))
    target_x = state[0]
    target_v = v0
    active = [observation(state)]
    for step, a in enumerate(cmd):
        target_v += float(a)*DT
        target_x += target_v*DT
        u = KP*(target_x-state[0])+KD*(target_v-state[2])
        state = rk4_step(state, DT, u, scene, mode, dynamics)
        if (step+1) % SAMPLE_EVERY == 0:
            active.append(observation(state))
    p = np.asarray(passive, dtype=np.float64)
    a = np.asarray(active, dtype=np.float64)
    need(p.shape == a.shape == (51,2) and
         bool(np.isfinite(p).all()) and bool(np.isfinite(a).all()),
         f"Invalid generator output {mode}")
    return p,a


def metric_rms_mm(x,y):
    return float(1e3*np.sqrt(np.mean((np.asarray(x)-np.asarray(y))**2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    n = N_SMOKE if args.smoke else N_FEASIBILITY
    seed = SMOKE_SEED if args.smoke else SEED
    print("="*72)
    print("P4-G1A — INDEPENDENT DYNAMICS GENERATOR / PHYSICS-SHIFT FEASIBILITY")
    print("="*72)
    print(f"Smoke: {args.smoke} | n_scenes={n} | no MuJoCo, no fitting, E4 sealed")
    rng = np.random.default_rng(seed)
    commands = command_profiles()
    rows = []
    fingerprints = []
    passive_gaps = []
    passive_cross_law_gaps = []
    shift_gaps = []
    dose_gaps = []
    for sid in range(n):
        for band in BANDS:
            base = sample_scene(rng, band)
            fingerprints.append(digest_object(base))
            reference_passives = {}
            for mode, conditions in CONDITIONS.items():
                for axis, level in conditions:
                    truth = apply_condition(base, mode, axis, level)
                    generated = {}
                    for arm, dyn in ARMS.items():
                        probes = {name: simulate(truth,mode,command,dyn)
                                  for name,command in commands.items()}
                        # Every action must restart from the same passive world.
                        ref_passive=probes["e1"][0]
                        passive_gaps.extend(float(np.max(np.abs(p-ref_passive)))
                                            for p,_ in probes.values())
                        if arm not in reference_passives:
                            reference_passives[arm] = ref_passive
                        passive_cross_law_gaps.append(float(np.max(np.abs(
                            reference_passives[arm]-ref_passive))))
                        generated[arm] = probes
                    # Nothing uses truth as a candidate fit or decision input.
                    change = metric_rms_mm(generated["IDEAL_ODE"]["FULL"][1][:,1],
                                           generated["SHIFTED_ODE"]["FULL"][1][:,1])
                    dgap = metric_rms_mm(generated["SHIFTED_ODE"]["HALF"][1][:,1],
                                         generated["SHIFTED_ODE"]["FULL"][1][:,1])
                    shift_gaps.append(change); dose_gaps.append(dgap)
                    row = {"scene_id":sid,"band":band,"mode_PRIVATE":mode,
                           "condition_PRIVATE":[axis,level],
                           "scene_sha256_PRIVATE":digest_object(truth),
                           "ideal_vs_shifted_B_rms_mm":change,
                           "shifted_half_vs_full_B_rms_mm":dgap,
                           "digest_PRIVATE":{
                               arm:{action:digest_object({
                                   "passive":pair[0].tolist(),"active":pair[1].tolist()})
                                   for action,pair in probes.items()}
                               for arm,probes in generated.items()}}
                    rows.append(row)
        print(f"[DONE] independent scene {sid+1}/{n}: 16 physical conditions",flush=True)
    need(len(rows)==16*n and len(set(fingerprints))==2*n,"Incomplete/duplicate scenes")
    need(max(passive_gaps) < 1e-10,"Passive trajectory differs across actions")
    need(max(passive_cross_law_gaps) < 1e-10,
         "Passive trajectory leaks force-law identity across conditions")
    need(all(x>1e-10 for x in shift_gaps),"Physical shift unexpectedly absent")
    need(all(x>1e-10 for x in dose_gaps),"Full/Half actions unexpectedly identical")
    need(abs(force_b("deadzone",0.01,0.3,
                     {"k":12.0,"c":1.0,"gap":0.04,"k3":100.0})) < 1e-15,
         "Deadzone force support changed")
    report={"protocol":"P4-G1A stand-alone ODE feasibility, not a predictive benchmark",
            "smoke":args.smoke,"n_scenes":n,"n_cases":len(rows),
            "seed":seed,"generator":"self-contained two-mass RK4, no MuJoCo",
            "arms_fixed_before_outcomes":ARMS,"action_amplitudes":ACTION_AMPS,
            "step_s":DT,"steps":STEPS,"sample_every":SAMPLE_EVERY,
            "source_sha256":hashlib.sha256(SELF.read_bytes()).hexdigest(),
            "scene_fingerprints_PRIVATE":fingerprints,
            "max_passive_within_case_difference_m":max(passive_gaps),
            "max_passive_cross_law_difference_m":max(passive_cross_law_gaps),
            "ideal_vs_shifted_B_rms_mm_median":float(np.median(shift_gaps)),
            "ideal_vs_shifted_B_rms_mm_min":min(shift_gaps),
            "shifted_half_vs_full_B_rms_mm_median":float(np.median(dose_gaps)),
            "new_fits":0,"classifier_decisions":0,"E4_accessed":False,
            "truth_used_for_fit_or_selection":False,
            "cross_simulator_claim":False,
            "limitations":[
                "One independent ODE generator with two arms, not two independent physics engines",
                "Synthetic distribution only; no real hardware data",
                "Distinct dynamical assumptions + integration implementation; shift settings are fixed, not optimized",
                "No mechanism classification, heldout accuracy or statistical hypothesis testing in G1A",
                "A/B controller is an explicit idealized model, not necessarily MuJoCo-XML identical",
                "Formal G1B must freeze generator assumptions, fit budget, noise, endpoints and coverage beforehand"
            ]}
    label="smoke" if args.smoke else "feasibility"
    rpath=OUT/f"p4g1a_{label}_report.json"
    apath=OUT/f"p4g1a_{label}_audit_PRIVATE.json"
    atomic_json(rpath,report)
    atomic_json(apath,{"rows_PRIVATE":rows})
    print("[PASS] Stand-alone ODE outputs (51,2); passive parity; nontrivial SHIFT and dose")
    print("[SUMMARY] shift median mm",report["ideal_vs_shifted_B_rms_mm_median"])
    print("[SUMMARY] half/full B median mm",report["shifted_half_vs_full_B_rms_mm_median"])
    print("[NOTE] No classification outcome, no cross-simulator evidence, no E4")
    print("Saved:",rpath)
    print("Saved:",apath)
    print("P4-G1A COMPLETE")


if __name__=="__main__":
    main()
