"""P4-G1B / experiment #99 — independent-ODE identification confirmation.

Prospective, immutable after the first fresh outcome. The physical GENERATOR is
P98's standalone RK4 two-mass code, with IDEAL_ODE and SHIFTED_ODE (one engine,
two fixed assumptions). The CANDIDATE INFERENCE is the unchanged historical
A-conditioned B-position RevisionProbe family, refitted on four ODE training
observations with P77/P92's 2+3 SLSQP starts and WWW/OOO/WOO domains.
This is structural-transfer-with-refitting, NOT zero-shot parameter transfer,
independent-hardware validation, a learned agent, or cross-simulator fidelity
proof. The SHIFT nuisance is deliberately omitted from deployed candidates.

A separate POST-SELECTION oracle-prediction diagnostic uses the ground-truth
parameters and the generator's drag, solely to assess interface/model mismatch;
its results NEVER participate in fitting, optimizer starts, or model choice.

Primary prospective endpoint: MATCHED5/WOO/SHIFTED_ODE true Nonlinear-M2
MID_PLUS_FULL minus MID_PLUS_HALF identification accuracy (96 paired cases).
Secondary: same IDEAL_ODE endpoint and their paired difference of effects.
Every unavailable fit counts as incorrect in the full denominator. Bootstraps
resample the 24 base scenes. E4 is never read. Reports/checkpoints are new files.

Usage (from the repository root):
  .\\Giant\\Scripts\\python.exe experiments\\p2d1\\99_p4g1b_independent_generator_identification.py --smoke
  .\\Giant\\Scripts\\python.exe experiments\\p2d1\\99_p4g1b_independent_generator_identification.py
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from runpy import run_path
import sys

import numpy as np
import scipy

SELF = Path(__file__).resolve()
HERE = SELF.parent
BASE = HERE / "results"
OUT = BASE / "p4g1b"
P16 = HERE / "16_p2d1b_probe_diversity.py"
P22 = HERE / "22_p3a1b_structural_revision.py"
P77 = HERE / "77_p4m2b2a_matched_domain_fit.py"
P92 = HERE / "92_p4s1b_full_solver_budget_audit.py"
P97 = HERE / "97_p4b1b_fresh_classical_baseline_confirmation_v2.py"
P98 = HERE / "98_p4g1a_independent_physics_generator_feasibility.py"
R97 = BASE / "p4b1b_v2" / "p4b1b_v2_fresh_report.json"
R98 = BASE / "p4g1a" / "p4g1a_feasibility_report.json"
R98_SMOKE = BASE / "p4g1a" / "p4g1a_smoke_report.json"

GEN_ARMS = ("IDEAL_ODE", "SHIFTED_ODE")
FIT_ARMS = ("FROZEN2", "MATCHED5")
LAWS = ("elastic", "deadzone", "cubic")
DOMAINS = ("wide", "outer_box")
CONFIGS = ("WWW", "OOO", "WOO")
METHODS = ("MID", "HALF_ONLY", "FULL_ONLY", "MID_PLUS_HALF", "MID_PLUS_FULL")
MODES = ("elastic", "deadzone", "nonlinear")
TRUTH = {"elastic": "elastic", "deadzone": "deadzone", "nonlinear": "cubic"}
CONDITIONS = {
    "elastic": (("matched", "M0"), ("alpha_high", "M2")),
    "deadzone": (("matched", "M0"), ("alpha_high", "M2"), ("gap_high", "M2")),
    "nonlinear": (("matched", "M0"), ("alpha_high", "M2"), ("gamma_high", "M2")),
}
BANDS = ("original", "above_original_range")
TRAIN = ("e1", "e2", "weak", "strong")
ACTIONS = TRAIN + ("mid", "HALF", "FULL")

# All seeds/protocol constants fixed by this released file BEFORE outcomes.
SCENE_SEED = 20291227
SMOKE_SCENE_SEED = 20391227
NOISE_SEED = 2029122701
SMOKE_NOISE_SEED = 2039122701
BOOT_SEED = 2029122702
N_SCENES, N_BOOT = 24, 2000
SIGMA = 0.005
N_TRAIN_RESIDUALS = 204
GAP_TOL = 1e-6
START_FRACTIONS = (0.10, 0.50, 0.90)
MIN_INTERPRETABLE_COVERAGE = 0.90  # reporting guard only, never discards cases


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    need(path.is_file(), f"Missing frozen prerequisite: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    need(path.is_file(), f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".writing")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def obj_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def stable_rng(base, *keys):
    message = json.dumps([base, *keys], sort_keys=True, separators=(",", ":"))
    seed = int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def original_force_from_ast():
    """Extract ONLY original internal_force definition; never execute MuJoCo imports."""
    source = ast.parse(P16.read_text(encoding="utf-8"), filename=str(P16))
    defs = [x for x in source.body if isinstance(x, ast.FunctionDef)
            and x.name == "internal_force"]
    need(len(defs) == 1, "Original MuJoCo force function absent/ambiguous")
    module = ast.Module(body=defs, type_ignores=[])
    scope = {"np": np}
    exec(compile(ast.fix_missing_locations(module), str(P16), "exec"), scope)
    return scope["internal_force"]


def check_physical_interface(gen, p77, p92, p22):
    need(gen["DT"] == .002 and gen["STEPS"] == 500 and gen["SAMPLE_EVERY"] == 10,
         "Independent generator integration/sampling changed")
    need((gen["KP"], gen["KD"]) == (250.0, 40.0), "Generator PD changed")
    need(gen["SEED"] == 20291201 and gen["SMOKE_SEED"] == 20391201,
         "P98 scene-seed lineage changed")
    need(gen["BANDS"] == BANDS and gen["CONDITIONS"] == CONDITIONS,
         "P98 condition grid changed")
    need(gen["ARMS"] == {
        "IDEAL_ODE": {"viscous_rate_s_inv": 0.0, "dry_acc_ms2": 0.0,
                      "velocity_smoothing_ms": 0.05, "actuator_tau_s": 0.0},
        "SHIFTED_ODE": {"viscous_rate_s_inv": 0.08, "dry_acc_ms2": 0.02,
                        "velocity_smoothing_ms": 0.05, "actuator_tau_s": 0.025}
    }, "P98 fixed physics shifts changed")
    need(gen["ACTION_AMPS"] == {"e1": .25, "e2": .5, "weak": .75,
                                 "strong": 2.25, "mid": 1.75,
                                 "HALF": 1.375, "FULL": 2.75},
         "P98 action amplitudes changed")
    commands = gen["command_profiles"]()
    need(tuple(commands) == ACTIONS and all(v.shape == (500,) for v in commands.values()),
         "Pulse interface changed")
    pulse = np.r_[np.ones(200), np.zeros(300)]
    need(all(np.array_equal(v, pulse * gen["ACTION_AMPS"][a])
             for a,v in commands.items()), "P98 pulse differs from frozen P2D1B/P60")
    need(tuple(p77["TRAIN"]) == TRAIN and p77["VALID"] == "mid"
         and math.isclose(p77["SIGMA"], SIGMA) and p77["MAXITER"] == 150,
         "P77 train/noise/optimization protocol changed")
    need(tuple(p77["LAWS"]) == LAWS and tuple(p92["LAWS"]) == LAWS
         and tuple(p92["DOMAINS"]) == DOMAINS, "Candidate law/domain protocol changed")
    need(tuple(p92["START_FRACTIONS"]) == START_FRACTIONS and
         p92["MAXITER"] == 150, "P92 five-start budget changed")
    need(p22["RevisionProbe"].__name__ == "RevisionProbe", "Inference API missing")
    # Mechanism law agreement is checked WITHOUT executing any MuJoCo source imports.
    force_ref = original_force_from_ast()
    sample = {"k": 14.0, "c": 1.7, "gap": .06, "k3": 500.0}
    for mode in MODES:
        for r in (-.4, -.07, -.025, .0, .025, .07, .4):
            for vr in (-.2, 0.0, .3):
                a = float(gen["force_b"](mode, r, vr, sample))
                b = float(force_ref(mode, r, vr, sample))
                need(np.isclose(a,b,rtol=0,atol=2e-13),
                     f"MuJoCo force formula differs from ODE at {mode}/{r}/{vr}: {a} vs {b}")
    for law in LAWS:
        wlo,whi=p77["specifications"](law,"wide")
        olo,ohi=p77["specifications"](law,"outer_box")
        need(np.all(wlo<=olo) and np.all(ohi<=whi),"Outer not inside Wide")
        need(len(p77["initial_points"](law))==2,"Original 2 starts changed")
        need(len(p92["common_starts"](p77,law))==3,"P92 common starts changed")
    return commands


def preflight():
    paths={"P16":P16,"P22":P22,"P77":P77,"P92":P92,
           "P97":P97,"P98":P98,"R97":R97,"R98":R98,
           "R98_SMOKE":R98_SMOKE,"SELF":SELF}
    manifest={name:sha(path) for name,path in paths.items()}
    r98=read(R98); s98=read(R98_SMOKE); r97=read(R97)
    need(not r98["smoke"] and r98["n_scenes"]==4 and r98["n_cases"]==64,
         "P98 formal feasibility incomplete")
    need(s98["smoke"] and s98["n_scenes"]==1, "P98 smoke evidence missing")
    need(manifest["P98"]==r98["source_sha256"]==s98["source_sha256"],
         "P98 source differs from reports")
    need(not r97["smoke"] and r97["n_scenes"]==24 and r97["n_cases"]==384,
         "P97 v2 fresh report incomplete")
    need(manifest["P97"]==r97["source_sha256"]["SELF"] and
         manifest["P92"]==r97["source_sha256"]["P92"],
         "P97/P92 source provenance mismatch")
    need(r97["n_training_residuals"]==N_TRAIN_RESIDUALS,
         "Corrected 204-residual baseline absent")
    need(not r98["E4_accessed"] and not s98["E4_accessed"]
         and not r97["E4_accessed"], "E4 flag in historical evidence")
    gen=run_path(str(P98))
    p77=run_path(str(P77))
    p92=run_path(str(P92))
    p22=run_path(str(P22))
    commands=check_physical_interface(gen,p77,p92,p22)
    return gen,p77,p92,p22,commands,manifest,(r98,s98,r97)


def gen_scenes(gen, smoke, reports):
    n=1 if smoke else N_SCENES
    rng=np.random.default_rng(SMOKE_SCENE_SEED if smoke else SCENE_SEED)
    scenes=[]; hashes=[]
    for sid in range(n):
        pair={band:gen["sample_scene"](rng,band) for band in BANDS}
        scenes.append(pair)
        hashes.extend(gen["digest_object"](pair[b]) for b in BANDS)
    need(len(hashes)==len(set(hashes))==2*n, "New scenes duplicate")
    r98,s98,r97=reports
    old=set(r98["scene_fingerprints_PRIVATE"])|set(s98["scene_fingerprints_PRIVATE"])
    need(not old.intersection(hashes), "Exact overlap with G1A feasibility/smoke scenes")
    if not smoke:
        checksmoke,_=gen_scenes(gen,True,reports)
        # compare after constructing the out-of-sample smoke scene
        smoke_hashes={gen["digest_object"](p[b]) for p in checksmoke for b in BANDS}
        need(not smoke_hashes.intersection(hashes),"Smoke/formal scene overlap")
    return scenes,hashes


def sample_noisy_probes(gen,Probe,scene,mode,commands,sid,band,axis,level,noise_seed):
    """Identical additive noise across arms; HALF/FULL share noise arrays.

    The true scene/mode are used ONLY by the generator. All fitting receives
    Probe objects constructed from noisy public A and B position trajectories.
    """
    probes={}; clean={}; clean_orig={}
    for dyn_name in GEN_ARMS:
        dyn=gen["ARMS"][dyn_name]
        c={name:gen["simulate"](scene,mode,commands[name],dyn) for name in ACTIONS}
        need(all(p.shape==a.shape==(51,2) for p,a in c.values()),
             "ODE output not (51,2)")
        # Passive data for each action must be the same within each arm.
        passive0=c["e1"][0]
        need(all(np.array_equal(passive0,p) for p,_ in c.values()),
             "Passive trajectory differs across actions")
        clean[dyn_name]=c
    arm_names=list(clean)
    need(not np.array_equal(clean[arm_names[0]]["FULL"][1],
                                clean[arm_names[1]]["FULL"][1]),
         "Frozen IDEAL/SHIFTED FULL signals unexpectedly identical")
    for dyn_name in GEN_ARMS:
        c=clean[dyn_name]; probes[dyn_name]={}
        for name in ACTIONS:
            noise_tag="HALF_FULL_PAIR" if name in ("HALF","FULL") else name
            rng=stable_rng(noise_seed,sid,band,mode,axis,level,noise_tag)
            pn=rng.normal(0,SIGMA,(51,2))
            an=rng.normal(0,SIGMA,(51,2))
            passive,active=c[name]
            z=Probe(passive+pn,active+an)
            need(np.asarray(z.y_b).shape==(51,) and
                 np.asarray(z.a_half).shape==(101,),
                 "RevisionProbe observation shape mismatch")
            probes[dyn_name][name]=z
        clean_orig[dyn_name]={name:gen["digest_object"]({
               "p":c[name][0].tolist(),"a":c[name][1].tolist()})
               for name in ACTIONS}
    return probes,clean,clean_orig


def domain(cfg,law):
    return "wide" if cfg[LAWS.index(law)]=="W" else "outer_box"


def validity(fit):
    return bool(fit is not None and fit.get("params_PRIVATE") is not None and
                fit.get("converged") is True and math.isfinite(float(fit["mse"])))


def fit_all(probes,p77,p92,starts):
    training=[probes[k] for k in TRAIN]
    results={arm:{d:{} for d in DOMAINS} for arm in FIT_ARMS}
    fitlogs={d:{} for d in DOMAINS}
    for law in LAWS:
        loss=p92["objective"](training,law,SIGMA)
        for d in DOMAINS:
            old=p77["fit_domain"](training,law,d)
            extra=p92["extra_fit"](loss,p77,law,d,starts[law])
            new=p92["selected_fit"](old,extra,loss,p77,law,d)
            old_fit=(None if not (old.get("valid") is True
                   and old.get("converged") is True
                   and old.get("params_PRIVATE") is not None
                   and old.get("train_mse") is not None
                   and math.isfinite(float(old["train_mse"]))) else
                   {"mse":float(old["train_mse"]),"params_PRIVATE":old["params_PRIVATE"],
                    "converged":True,"origin":"original_two_starts"})
            for arm,fit in (("FROZEN2",old_fit),("MATCHED5",new)):
                if not validity(fit):
                    results[arm][d][law]=None
                else:
                    pars=fit["params_PRIVATE"]
                    vals={}
                    for name in ("mid","HALF","FULL"):
                        actual=np.asarray(probes[name].y_b,dtype=float)
                        predicted=np.asarray(probes[name].predict(law,pars),dtype=float)
                        need(actual.shape==predicted.shape==(51,) and np.isfinite(predicted).all(),
                             "Invalid frozen candidate prediction")
                        vals[name]=float(1000*np.sqrt(np.mean((predicted-actual)**2)))
                    results[arm][d][law]={"mse":float(fit["mse"]),
                         "mid":vals["mid"],"half":vals["HALF"],"full":vals["FULL"],
                         "origin":fit["origin"],"params_PRIVATE":[float(v) for v in pars]}
            fitlogs[d][law]={"original":{"valid":bool(old["valid"]),
                                   "converged":bool(old["converged"]),
                                   "mse":old["train_mse"],"nfev":int(old["nfev"])},
                 "extra":extra,"best5":None if new is None else
                 {"origin":new["origin"],"mse":new["mse"],"converged":new["converged"]},
                 "nfev_total":int(old["nfev"])+sum(int(x["nfev"]) for x in extra)}
    return results,fitlogs


def select_results(fits,score_criteria):
    decisions={}
    for arm in FIT_ARMS:
        decisions[arm]={}
        for cfg in CONFIGS:
            d={}
            chosen={law:fits[arm][domain(cfg,law)][law] for law in LAWS}
            if any(x is None for x in chosen.values()):
                for meth in METHODS+("TRAIN_PEM","PROFILE_SIGMA_BIC"):
                    d[meth]={"available":False,"winner":None}
            else:
                for meth in METHODS:
                    scores={}
                    for law in LAWS:
                        x=chosen[law]
                        if meth=="MID":v=x["mid"]
                        elif meth=="HALF_ONLY":v=x["half"]
                        elif meth=="FULL_ONLY":v=x["full"]
                        elif meth=="MID_PLUS_HALF":v=math.sqrt((x["mid"]**2+x["half"]**2)/2)
                        elif meth=="MID_PLUS_FULL":v=math.sqrt((x["mid"]**2+x["full"]**2)/2)
                        scores[law]=v
                    winner=min(LAWS,key=lambda l:(scores[l],LAWS.index(l)))
                    d[meth]={"available":True,"winner":winner,
                             "scores_mm_PRIVATE":scores}
                for meth in ("TRAIN_PEM","PROFILE_SIGMA_BIC"):
                    scores={law:score_criteria(chosen[law]["mse"],law,meth) for law in LAWS}
                    winner=min(LAWS,key=lambda l:(scores[l],LAWS.index(l)))
                    d[meth]={"available":True,"winner":winner,
                             "scores_PRIVATE":scores}
            decisions[arm][cfg]=d
    return decisions


def native_oracle_diagnostic(gen,probes,scene,mode,clean):
    """Ground-truth parameters for AFTER-DECISION, retrospective checks only.

    This is not an independent model-selection arm or an operational oracle.
    Source: generator's physical parameters and known drag of each dynamics arm.
    """
    mb=scene["mass_b"]
    law=TRUTH[mode]
    pars=[scene["k"]/mb,scene["c"]/mb]
    if law=="deadzone":pars.append(scene["gap"])
    elif law=="cubic":pars.append(scene["k3"]/mb)
    outputs={}
    for ga in GEN_ARMS:
        v=probes[ga]["mid"]
        hist=np.asarray(v.predict(law,pars),dtype=float)
        # True-law evaluation against noiseless generator trajectory; A-conditioning
        # still comes from noisy probe. This is NOT an accuracy or deployable method.
        y=np.asarray(clean[ga]["mid"][1][:,1],dtype=float)
        outputs[ga] = {"true_effective_params_PRIVATE":pars,
                       "oracle_law_rms_to_clean_B_mm":float(1000*np.sqrt(np.mean((hist-y)**2)))}
    return outputs


def case_key(sid,band,mode,axis,level):
    return [sid,band,mode,axis,level]


def run_case(gen,p77,p92,Probe,scorefn,commands,starts,base,sid,band,mode,axis,level,noise_seed):
    scene=gen["apply_condition"](base,mode,axis,level)
    probs,clean,digests=sample_noisy_probes(gen,Probe,scene,mode,commands,
                              sid,band,axis,level,noise_seed)
    # Fit and decide FIRST. The truth labels, physical oracle parameters and
    # true mechanism are not passed to any scoring or optimizer call.
    all_decisions={}; logs={}; nested={}; fitted_all={}
    for ga in GEN_ARMS:
        fitted,fitlog=fit_all(probs[ga],p77,p92,starts)
        all_decisions[ga]=select_results(fitted,scorefn)
        fitted_all[ga]=fitted
        logs[ga]=fitlog
        nested[ga]={}
        for law in LAWS:
            w=fitted["MATCHED5"]["wide"][law]
            o=fitted["MATCHED5"]["outer_box"][law]
            gap=None if w is None or o is None else float(w["mse"]-o["mse"])
            nested[ga][law]={"gap":gap,"violation":gap is not None and gap>GAP_TOL}
    oracle=native_oracle_diagnostic(gen,probs,scene,mode,clean)
    return {"key_PRIVATE":case_key(sid,band,mode,axis,level),
            "scene_id":sid,"mode_PRIVATE":mode,"true_law_PRIVATE":TRUTH[mode],
            "condition_PRIVATE":level,"physical_scene_sha256_PRIVATE":gen["digest_object"](scene),
            "physical_scene_parameters_PRIVATE":scene,
            "clean_observation_fingerprints_PRIVATE":digests,
            "decisions":all_decisions,"fit_results_PRIVATE":fitted_all,
            "fit_audit_PRIVATE":logs,
            "nested_audit_PRIVATE":nested,
            "oracle_after_selection_PRIVATE":oracle}


def outcome(d,truth):
    return int(d["available"] and d["winner"]==truth)


def summarize(rows):
    n=len(rows)
    stats={}
    for ga in GEN_ARMS:
        stats[ga]={}
        for arm in FIT_ARMS:
            stats[ga][arm]={}
            for cfg in CONFIGS:
                ds={}
                for method in METHODS+("TRAIN_PEM","PROFILE_SIGMA_BIC"):
                    decisions=[r["decisions"][ga][arm][cfg][method] for r in rows]
                    counts=sum(outcome(d,r["true_law_PRIVATE"]) for d,r in zip(decisions,rows))
                    cover=sum(int(d["available"]) for d in decisions)
                    ds[method]={"correct":counts,"total":n,"available":cover,
                        "coverage":cover/n,"accuracy_all":counts/n,
                        "accuracy_available":counts/cover if cover else None,
                        "by_mode_PRIVATE":{mode:{"n":sum(r["mode_PRIVATE"]==mode for r in rows),
                            "correct":sum(outcome(d,r["true_law_PRIVATE"])
                                for d,r in zip(decisions,rows) if r["mode_PRIVATE"]==mode)}
                             for mode in MODES}}
                stats[ga][arm][cfg]=ds
    return stats


def primary_pairs(rows,smoke):
    nm2=[r for r in rows if r["mode_PRIVATE"]=="nonlinear"
                             and r["condition_PRIVATE"]=="M2"]
    result={}; deltas={}; pair_counts={}
    for ga in GEN_ARMS:
        pairs=[]; status=Counter()
        for r in nm2:
            d=r["decisions"][ga]["MATCHED5"]["WOO"]
            half=outcome(d["MID_PLUS_HALF"],"cubic")
            full=outcome(d["MID_PLUS_FULL"],"cubic")
            pairs.append((r["scene_id"],full-half))
            status[(half,full)]+=1
        diff=sum(x for _,x in pairs)
        result[ga]={"n":len(pairs),"half_correct":sum(status[(1,j)] for j in (0,1)),
                    "full_correct":sum(status[(i,1)] for i in (0,1)),
                    "net":diff,"effect":diff/len(pairs)}
        pair_counts[ga]={"both_correct":status[(1,1)],
                         "corrected_by_full":status[(0,1)],
                         "harmed_by_full":status[(1,0)],
                         "both_wrong":status[(0,0)]}
        deltas[ga]=pairs
    dd=[(a[0],b[1]-a[1]) for a,b in zip(deltas["IDEAL_ODE"],deltas["SHIFTED_ODE"])]
    need(all(a[0]==b[0] for a,b in zip(deltas["IDEAL_ODE"],deltas["SHIFTED_ODE"])),
         "ODE branch pairing mismatch")
    result["SHIFT_MINUS_IDEAL_DD"]={"n":len(dd),"net":sum(x for _,x in dd),
                            "effect":sum(x for _,x in dd)/len(dd)}
    return result,deltas,dd,pair_counts


def cluster_bootstrap(deltas,dd):
    per={}
    for ga,pairs in {**deltas,"SHIFT_MINUS_IDEAL_DD":dd}.items():
        grp=defaultdict(list)
        for sid,delta in pairs: grp[sid].append(delta)
        need(set(grp)==set(range(N_SCENES)) and all(len(v)==4 for v in grp.values()),
             f"Cluster shape mismatch: {ga}")
        per[ga]=np.asarray([np.mean(grp[s]) for s in range(N_SCENES)],dtype=float)
    ix=np.random.default_rng(BOOT_SEED).integers(0,N_SCENES,size=(N_BOOT,N_SCENES))
    return {"unit":"base_scene","n_clusters":N_SCENES,"resamples":N_BOOT,
            "seed":BOOT_SEED,
            "CI95":{ga:[float(q) for q in np.quantile(values[ix].mean(1),[.025,.975])]
                    for ga,values in per.items()}}



def smoke_interface(gen,p77,p92,Probe,commands,starts,scenes,manifest,reports):
    """One small non-inferential check; deliberate reduced optimizer budget.

    It is *not* the matched-five-start formal protocol. It exercises both ODE
    generators and the exact transfer-probe interface, plus the historical
    original-two and one extra optimizer call on a single candidate/domain.
    """
    sid=0;band="original";mode="elastic";axis="matched";level="M0"
    scene=gen["apply_condition"](scenes[0][band],mode,axis,level)
    probes,clean,digests=sample_noisy_probes(gen,Probe,scene,mode,commands,
                         sid,band,axis,level,SMOKE_NOISE_SEED)
    for ga in GEN_ARMS:
        sample=probes[ga]["mid"]
        need(sample.y_b.shape==(51,) and sample.a_half.shape==(101,),
             "Smoke probe incompatible with candidate model")
        prediction=sample.predict("elastic",[10.0,1.2])
        need(prediction.shape==(51,) and np.isfinite(prediction).all(),
             "Smoke transfer prediction nonfinite")
    training=[probes["IDEAL_ODE"][a] for a in TRAIN]
    law="elastic";dom="wide"
    old=p77["fit_domain"](training,law,dom)
    objective=p92["objective"](training,law,SIGMA)
    extra=p92["extra_fit"](objective,p77,law,dom,starts[law][:1])
    chosen=p92["selected_fit"](old,extra,objective,p77,law,dom)
    need(len(extra)==1 and int(old["nfev"])>=0 and int(extra[0]["nfev"])>=0,
         "Smoke optimization interface failed")
    report={"protocol":"P4-G1B smoke-only compatibility check, not full solver budget",
            "smoke":True,"n_scenes":1,"n_cases":1,"arms":list(GEN_ARMS),
            "source_sha256":manifest,"source_ODE_digests_PRIVATE":digests,
            "observations_are_B_51_A_half_101":True,
            "small_fit_test":{"candidate":law,"domain":dom,
                 "original_2_converged":bool(old.get("converged")),
                 "one_extra_converged":bool(extra[0]["converged"]),
                 "best_available":chosen is not None,
                 "nfev_original":int(old["nfev"]),
                 "nfev_one_extra":int(extra[0]["nfev"])},
            "E4_accessed":False,"scientific_effect_estimate":False,
            "formal_case_results_observed":False}
    audit={"generator_clean_probe_digests_PRIVATE":digests,
            "diagnostic_fits_PRIVATE":{"original":old,"one_extra":extra,
                     "selected":chosen}}
    rp=OUT/"p4g1b_smoke_report.json"
    ap=OUT/"p4g1b_smoke_audit_PRIVATE.json"
    atomic_json(rp,report);atomic_json(ap,audit)
    print("[PASS] Two generator branches, probe interface and small optimizer-call smoke")
    print("[CAUTION] Smoke uses 2+1 starts for one Elastic/Wide case only, no conclusions")
    print("Saved:",rp,"\nSaved:",ap)
    print("P4-G1B SMOKE COMPLETE")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke",action="store_true",help="1 out-of-cohort scene; no scientific inference")
    args=ap.parse_args()
    print("="*75)
    print("P4-G1B — FROZEN ODE GENERATOR IDENTIFICATION / PHYSICS SHIFT")
    print("="*75)
    print("Smoke:",args.smoke,"| E4 sealed | matched structural inference, new fits")
    gen,p77,p92,p22,commands,manifest,reports=preflight()
    scenes,fps=gen_scenes(gen,args.smoke,reports)
    starts={law:p92["common_starts"](p77,law) for law in LAWS}
    scorefn=run_path(str(P97))["score_corrected"]
    noise_seed=SMOKE_NOISE_SEED if args.smoke else NOISE_SEED
    print("[PASS] P98/P97 source hashes; original-law force and pulse; 204 B residuals")
    print("[PASS] Outer inside Wide, two original + three predeclared common starts")
    print("[PASS] Independent ODE scenes vs P98 feasibility and smoke (exact hash)")
    print("[PROTOCOL]",len(scenes),"scenes x 2 bands x 8 conditions x 2 ODE physics arms")
    print("[CAUTION] MuJoCo A/B controls are NOT numerically identical to ODE controls")
    if args.smoke:
        smoke_interface(gen,p77,p92,p22["RevisionProbe"],commands,starts,scenes,manifest,reports)
        return
    manifest_path=OUT/"p4g1b_source_manifest.json"
    common_meta={"source_sha256":manifest,"scene_seed":SCENE_SEED,
                 "smoke_scene_seed":SMOKE_SCENE_SEED,"noise_seed":NOISE_SEED,
                 "smoke_noise_seed":SMOKE_NOISE_SEED,"boot_seed":BOOT_SEED,
                 "scene_fingerprints":fps if not args.smoke else None,
                 "protocol":"Frozen G1B ODE two-arm structural-transfer-with-refitting"}
    if manifest_path.exists():
        prev=read(manifest_path)
        need(prev==common_meta,"Frozen protocol or source changed; do NOT combine checkpoints")
    elif not args.smoke:
        atomic_json(manifest_path,common_meta)
    rows=[]
    for sid,scene_pair in enumerate(scenes):
        ck=OUT/"checkpoints"/f"scene_{sid:02d}.json"
        if not args.smoke and ck.exists():
            payload=read(ck)
            need(payload["source_sha256"]==manifest and payload["scene_id"]==sid
                 and payload["base_fingerprints"]==fps[2*sid:2*sid+2]
                 and len(payload["rows"])==16
                 and payload.get("rows_sha256")==obj_hash(payload["rows"]),
                 f"Checkpoint provenance mismatch: {ck}")
            block=payload["rows"]
            print(f"[RESUME] scene {sid+1}/{len(scenes)}: 16 paired ODE cases",flush=True)
        else:
            block=[]
            for band in BANDS:
                for mode in MODES:
                    conditions=CONDITIONS[mode]
                    for axis,level in conditions:
                        block.append(run_case(gen,p77,p92,p22["RevisionProbe"],scorefn,
                                              commands,starts,scene_pair[band],sid,
                                              band,mode,axis,level,noise_seed))
            need(len(block)==16,"Incomplete conditions per scene")
            if not args.smoke:
                atomic_json(ck,{"source_sha256":manifest,"scene_id":sid,
                       "base_fingerprints":fps[2*sid:2*sid+2],
                       "rows_sha256":obj_hash(block),"rows":block})
            print(f"[DONE] scene {sid+1}/{len(scenes)}: 16 paired ODE cases",flush=True)
        rows.extend(block)
    n=len(rows)
    need(n==16*len(scenes) and len({tuple(r["key_PRIVATE"]) for r in rows})==n,
         "Duplicate/missing formal case")
    stats=summarize(rows)
    primary,pairs,dd,pair_status=primary_pairs(rows,args.smoke)
    if args.smoke:
        ci=None
    else:
        need(len(pairs["IDEAL_ODE"])==len(pairs["SHIFTED_ODE"])==96,
             "Primary 96-case denominator changed")
        ci=cluster_bootstrap(pairs,dd)
    primary_coverage={ga:{"min_full_half_coverage":min(
                    stats[ga]["MATCHED5"]["WOO"][m]["coverage"]
                    for m in ("MID_PLUS_HALF","MID_PLUS_FULL")),
                "passes_predeclared_90pct_guard":all(
                    stats[ga]["MATCHED5"]["WOO"][m]["coverage"]>=MIN_INTERPRETABLE_COVERAGE
                    for m in ("MID_PLUS_HALF","MID_PLUS_FULL"))}
                for ga in GEN_ARMS}
    nested={ga:dict(Counter(law for row in rows for law in LAWS
                       if row["nested_audit_PRIVATE"][ga][law]["violation"]))
            for ga in GEN_ARMS}
    nfev={ga:{d:{"original":sum(row["fit_audit_PRIVATE"][ga][d][law]["original"]["nfev"]
                                   for row in rows for law in LAWS),
                 "extra":sum(sum(int(x["nfev"]) for x in row["fit_audit_PRIVATE"][ga][d][law]["extra"])
                                   for row in rows for law in LAWS)} for d in DOMAINS}
          for ga in GEN_ARMS}
    oracle={ga:{"true_model_clean_B_rms_mm_median":float(np.median([
        row["oracle_after_selection_PRIVATE"][ga]["oracle_law_rms_to_clean_B_mm"]
        for row in rows]))} for ga in GEN_ARMS}
    report={"protocol":"P4-G1B pre-frozen independent ODE two-arm identification",
        "smoke":args.smoke,"n_scenes":len(scenes),"n_cases_per_generator":n,
        "n_total_case_arm_pairs":n*len(GEN_ARMS),
        "n_fit_units":n*len(GEN_ARMS)*len(LAWS)*len(DOMAINS),
        "ODE_arms":gen["ARMS"],"action_amplitudes":gen["ACTION_AMPS"],
        "generator_source":"P98 standalone two-mass RK4; no MuJoCo runtime needed",
        "inference_source":"P22 RevisionProbe A-conditioned B; P77/P92 domain solver",
        "fit_budget":{"original":2,"extra":3,"maxiter":150,"ftol":1e-9},
        "n_training_scalar_residuals":N_TRAIN_RESIDUALS,
        "scene_seed":SMOKE_SCENE_SEED if args.smoke else SCENE_SEED,
        "noise_seed":noise_seed,"source_sha256":manifest,
        "scene_fingerprints_PRIVATE":fps,
        "stats":stats,"primary_contrasts":primary,"primary_pair_status":pair_status,
        "bootstrap":ci,"primary_coverage_interpretation_gate":primary_coverage,
        "nested_solver_violations":nested,"solver_nfev":nfev,
        "privileged_postdecision_diagnostic":oracle,
        "E4_accessed":False,"truth_used_for_fit_or_selection":False,
        "heldout_actions_used_for_fit":False,"threshold_tuning":False,
        "new_fits":n*len(GEN_ARMS)*len(LAWS)*len(DOMAINS),
        "is_zero_shot_parameter_transfer":False,
        "is_independent_hardware":False,"has_two_independent_ODE_engines":False,
        "limitations":[
            "IDEAL and SHIFTED share the same RK4 generator; not separate engines or hardware",
            "Inference parameters are re-fitted from four ODE training actions: NOT zero-shot",
            "MuJoCo has its own integrator and PD/actuator implementation; input-level force/pulse parity is not numerical trajectory parity",
            "SHIFTED generator adds ground drag and actuator lag, not in deployed candidates; this is model mismatch",
            "Candidate predicts B conditional on observed noisy A; does not reproduce unobserved A actuator dynamics",
            "Oracle true-parameter diagnostics are post-decision only, not a competing operational estimator",
            "FROZEN2 and MATCHED5 count starts; actual function evaluations differ by arm/domain",
            "Original SLSQP local minima and nested-domain gaps remain possible",
            "Same-generator fresh seeds and separate ODE physics do not prove hardware generalization",
            "Positive dose effect need not imply superlinear or information-theoretic gain",
            "Smoke has no inferential value; all primary claims use 24 scene-cluster units",
            "All case denominators retain abstentions as errors; no E4 accessed",
        ],"environment":{"python":sys.version,"numpy":np.__version__,
           "scipy":scipy.__version__,"platform":platform.platform()}}
    tag="smoke" if args.smoke else "fresh"
    rp=OUT/f"p4g1b_{tag}_report.json"
    apath=OUT/f"p4g1b_{tag}_audit_PRIVATE.json"
    atomic_json(rp,report);atomic_json(apath,rows)
    if not args.smoke:
        for ga in GEN_ARMS:
            for cfg in CONFIGS:
                vals=stats[ga]["MATCHED5"][cfg]
                print(f"{ga}/{cfg}/MATCHED5: " + " | ".join(
                    f"{m}:{vals[m]['correct']}/{n}"
                    for m in ("TRAIN_PEM","PROFILE_SIGMA_BIC","MID_PLUS_HALF","MID_PLUS_FULL")))
        print("[PRIMARY] SHIFTED Nonlinear-M2 Full−Half:",primary["SHIFTED_ODE"],
              "CI:",ci["CI95"]["SHIFTED_ODE"])
        print("[SECONDARY] IDEAL:",primary["IDEAL_ODE"],
              "difference-of-effects:",primary["SHIFT_MINUS_IDEAL_DD"])
    else:
        print("[PASS] One-scene Smoke; NO scientific effect estimate")
    if not all(x["passes_predeclared_90pct_guard"] for x in primary_coverage.values()):
        print("[WARNING] Primary coverage below 90%; effect must be treated as limited")
    print("[COVERAGE]",primary_coverage)
    print("[NESTED]",nested)
    print("[ORACLE-ONLY DIAGNOSTIC]",oracle)
    print("Saved:",rp,"\nSaved:",apath)
    print("P4-G1B COMPLETE")


if __name__=="__main__":
    main()
