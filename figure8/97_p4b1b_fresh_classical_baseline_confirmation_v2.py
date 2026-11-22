"""P4-B1B / Experiment 97 v2: independent fresh-scene classical selector confirmation.

PRE-OUTCOME PROTOCOL CORRECTION (2026-11-15):
The original 97 failed on the first smoke observation, before optimizer or
classification output.  RevisionProbe.y_b is (51,), NOT (51,2): the two-column
raw trajectory is the constructor input, but the fitting target is B position.
The preceding P96 pilot inadvertently used n=408, instead of the correct
number n=4*51=204 of scalar training residuals.  This v2 uses n=204 for
pseudo AIC/BIC and separately recomputes a corrected P96 pilot for comparison.
The immutable P96 pilot, original P97 script and all frozen earlier runs remain
untouched.  No P97 original fresh-case outcomes were observed before correction.
Both numerical amendments are determined by preexisting source-level semantics,
not by prospective new-scene classification results.

Freeze BEFORE any formal P4-B1B outcomes are inspected. New P90-style 24 scenes
x 2 physical bands x 8 conditions = 384 cases. Every candidate receives
original 2 SLSQP starts + the SAME fixed 3 extra starts from P92; no truth,
MID, HALF or FULL observations enter parameter fitting or solver selection.

Distinct observation-budget families (NEVER treat them as matched budgets):
- TRAIN-ONLY: PEM, known-sigma AIC/BIC, profiled-sigma pseudo-AIC/BIC
  evaluated on exactly the same four training actions.
- HELD-OUT ACTIONS: frozen fitted candidates evaluated by prediction RMS on
  MID, MID+HALF and MID+FULL, as in P90; these consume additional observations.

Primary prospective contrast: MATCHED5/WOO PROFILE_SIGMA_BIC - TRAIN_PEM,
on equal four-training-action observations. Historical P96 informed choice;
this is validation on a new cohort, NOT a blind design independent of P96.
Secondary replication: MATCHED5/WOO Nonlinear-M2 recall MID_PLUS_FULL
minus MID_PLUS_HALF with paired action noise, four cases per base scene.

One-sided optimizer improvements, adaptive thresholds, hidden test-label use,
access to E4, optimizer claims of global optimality and retroactive seed
selection are prohibited. Errors fail closed; completed checkpoints are SHA-
guarded. The P96/P90/P92 inputs are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from runpy import run_path
import sys

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "p4b1b_v2"
SOURCES = {
    "P80": HERE / "80_p4m2c0_prior_mismatch_stress.py",
    "P88": HERE / "88_p4m2d1_fresh_action_prior_interaction.py",
    "P90": HERE / "90_p4m2d2b_amplitude_dose_response_v2.py",
    "P92": HERE / "92_p4s1b_full_solver_budget_audit.py",
    "P96": HERE / "96_p4b1a_frozen_classical_model_selection_baselines.py",
    "P97_ORIGINAL": HERE / "97_p4b1b_fresh_classical_baseline_confirmation.py",
    "R88": RESULTS / "p4m2d1" / "p4m2d1_fresh_report.json",
    "R90": RESULTS / "p4m2d2b" / "p4m2d2b_v2_fresh_report.json",
    "R92": RESULTS / "p4s1b" / "p4s1b_fresh_report.json",
    "R96": RESULTS / "p4b1a" / "p4b1a_pilot_report.json",
    "A96": RESULTS / "p4b1a" / "p4b1a_pilot_audit_PRIVATE.json",
}
SELF = Path(__file__).resolve()
LAWS = ("elastic", "deadzone", "cubic")
CONFIGS = ("WWW", "OOO", "WOO")
DOMAINS = ("wide", "outer_box")
ARMS = ("FROZEN2", "MATCHED5")
SELECTORS = ("TRAIN_PEM", "KNOWN_SIGMA_AIC", "KNOWN_SIGMA_BIC",
             "PROFILE_SIGMA_AIC", "PROFILE_SIGMA_BIC")
EVAL_METHODS = ("MID", "MID_PLUS_HALF", "MID_PLUS_FULL")
MODE_TO_LAW = {"elastic":"elastic", "deadzone":"deadzone", "nonlinear":"cubic"}
CONDITIONS = {
    "elastic": (("matched","M0"), ("alpha_high","M2")),
    "deadzone": (("matched","M0"), ("alpha_high","M2"), ("gap_high","M2")),
    "nonlinear": (("matched","M0"), ("alpha_high","M2"), ("gamma_high","M2")),
}
# Fixed now; NEVER change after first formal result.
SCENE_SEED = 20291123
FIT_NOISE_SEED = 2029112301
ACTION_NOISE_SEED = 2029112302
BOOT_SEED = 2029112303
N_SCENES = 24
SMOKE_OFFSET = 100000
N_BOOT = 2000
MIN_INTERPRETABLE_COVERAGE = 0.90
N_RESIDUALS = 4 * 51   # B-position target only: RevisionProbe.y_b is (51,)
P96_REPORTED_N = 4 * 51 * 2  # Historical pilot bug; never modify P96 files
K = {"elastic":2, "deadzone":3, "cubic":3}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    need(path.is_file(), f"Missing prerequisite: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    need(path.is_file(), f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def preflight():
    for path in SOURCES.values():
        need(path.is_file(), f"Missing required frozen source: {path}")
    r96, r90, r92, r88 = (read(SOURCES[name]) for name in ("R96", "R90", "R92", "R88"))
    need(r96["smoke"] is False and r90["smoke"] is False and
         r92["smoke"] is False and r88["smoke"] is False,
         "Formal historical sources required")
    need(all(r["E4_accessed"] is False for r in (r96,r90,r92,r88)),
         "Historical E4 flag")
    need(r96["n_cases"] == r90["n_cases"] == r92["n_cases"] == r88["n_cases"] == 384,
         "Historical dimensions changed")
    need(r96["new_fits"] == r96["new_simulations"] == 0 and
         r96["heldout_mid_half_full_used_for_selection"] is False and
         r96["truth_used_for_fit_or_selection"] is False,
         "P96 baseline protocol modified")
    for f_key,ref_key in (("P90","S90"),("P92","S92"),("R90","R90"),("R92","R92")):
        need(sha(SOURCES[f_key]) == r96["source_sha256"][ref_key],
             f"P96 upstream hash changed: {f_key}")
    need(tuple(r96["training_actions"]) == ("e1","e2","weak","strong") and
         r96["n_training_scalar_residuals"] == P96_REPORTED_N and
         r96["candidate_parameters"] == K,
         "P96 historical (incorrect) sample/parameter count changed")
    need(r96["score_definitions"] == {
        "TRAIN_PEM":"training normalized MSE (already the original optimizer objective)",
        "KNOWN_SIGMA_AIC":"n * normalized_MSE + 2*k",
        "KNOWN_SIGMA_BIC":"n * normalized_MSE + ln(n)*k",
        "PROFILE_SIGMA_AIC":"n * ln(normalized_MSE) + 2*k",
        "PROFILE_SIGMA_BIC":"n * ln(normalized_MSE) + ln(n)*k",
    },"P96 scoring definition changed")
    need(r92["starts"]["historical"] == 2 and r92["starts"]["common_extra"] == 3
         and tuple(r92["starts"]["fractions"]) == (0.1, 0.5, 0.9) and
         r92["starts"]["maxiter_per_start"] == 150,
         "P92 five-start budget changed")
    need(r90["scene_seed"] != SCENE_SEED and r88["scene_seed"] != SCENE_SEED,
         "Fresh cohort seed overlaps previous")
    manifest = {name:sha(path) for name,path in SOURCES.items()}
    manifest["SELF"] = sha(SELF)
    return (r96,r90,r92,r88), manifest


def validate_history_fps(p79,src,sim,scenes,reports,smoke):
    r96,r90,r92,r88 = reports
    fps=p79["fingerprint_scenes"](scenes)
    need(len(fps)==2*len(scenes) and len(set(fps))==len(fps),
         "Duplicate or incomplete fresh scene fingerprints")
    for label, report in (("P90",r90),("P88",r88)):
        old=set(report["scene_independence"].get("fingerprints",
                    report["scene_independence"].get("new_scene_fingerprints",[])))
        need(bool(old),f"Missing {label} old fingerprints")
        need(not (set(fps)&old), f"Reused scene parameters from {label}")
    p79["verify_scene_independence"](src,sim,scenes)
    if not smoke:
        prev=src["make_scenes"](sim,SCENE_SEED+SMOKE_OFFSET,1)
        need(not set(fps)&set(p79["fingerprint_scenes"](prev)),
             "Formal/smoke overlap")
    return fps


def mode_scenario(p80,p74,p77,base,mode,axis,level):
    if level=="M0":
        need(axis=="matched", "M0 conditions changed")
        return base
    scene,_=p80["make_stress_scene"](p74,p77,base,mode,axis,level)
    return scene


def fitting_result(old, extra, fit5):
    old_ok=bool(old.get("valid") is True and old.get("converged") is True
                and old.get("train_mse") is not None and
                np.isfinite(old["train_mse"]) and
                old.get("params_PRIVATE") is not None)
    if fit5 is not None:
        need(fit5["converged"] and math.isfinite(float(fit5["mse"])),
             "Selected MATCHED5 is not converged")
    return {
        "FROZEN2":None if not old_ok else {
            "origin":"original_two_starts", "mse":float(old["train_mse"]),
            "params_PRIVATE":old["params_PRIVATE"]},
        "MATCHED5":fit5,
    }


def fit_report(p77,p92,training,law,dom,starts):
    old=p77["fit_domain"](training,law,dom)
    objective=p92["objective"](training,law,float(p77["SIGMA"]))
    extra=p92["extra_fit"](objective,p77,law,dom,starts)
    chosen=p92["selected_fit"](old,extra,objective,p77,law,dom)
    sel=fitting_result(old,extra,chosen)
    return sel,{
        "original":{"valid":bool(old["valid"]),
                    "converged":bool(old["converged"]),
                    "mse":old["train_mse"],"nfev":int(old["nfev"])},
        "extra":extra,
        "best5":None if chosen is None else {"origin":chosen["origin"],
                   "mse":chosen["mse"],"converged":chosen["converged"]},
        "nfev_extra":sum(int(x["nfev"]) for x in extra),
        "nfev_total":int(old["nfev"])+sum(int(x["nfev"]) for x in extra),
    }


def domain_for(cfg,i):
    return "wide" if cfg[i]=="W" else "outer_box"


def score_corrected(mse,law,selector):
    """Same P96 criterion family, with corrected scalar residual count 204."""
    m=float(mse); k=K[law]; n=N_RESIDUALS
    need(math.isfinite(m) and m>=0, "Nonfinite/negative train MSE")
    if selector=="TRAIN_PEM": return m
    if selector=="KNOWN_SIGMA_AIC": return n*m+2*k
    if selector=="KNOWN_SIGMA_BIC": return n*m+math.log(n)*k
    need(m>0, "Profiled variance undefined for zero training MSE")
    if selector=="PROFILE_SIGMA_AIC": return n*math.log(m)+2*k
    if selector=="PROFILE_SIGMA_BIC": return n*math.log(m)+math.log(n)*k
    raise ValueError(selector)


def decide_corrected(candidates,selector):
    if not all(candidates[law]["available"] for law in LAWS):
        return {"available":False,"winner":None,"scores_PRIVATE":None,
                "selection_margin":None}
    scores={law:score_corrected(candidates[law]["mse"],law,selector)
            for law in LAWS}
    need(all(math.isfinite(s) for s in scores.values()),
         "Nonfinite corrected selector score")
    order=sorted(LAWS,key=lambda law:(scores[law],LAWS.index(law)))
    return {"available":True,"winner":order[0],"scores_PRIVATE":scores,
            "selection_margin":float(scores[order[1]]-scores[order[0]])}


def corrected_p96_pilot():
    """Read-only retrospective rescore to avoid comparing n=204 vs n=408.

    Historical pilot is exploratory, not used to choose fresh-scene settings.
    Only yields descriptive context, with the original P96 left immutable.
    """
    pilot_audit=read(SOURCES["A96"])
    rows=pilot_audit.get("rows_PRIVATE")
    need(isinstance(rows,list) and len(rows)==384,
         "Incomplete P96 pilot audit")
    out={}
    seen=set()
    for row in rows:
        key=tuple(row["key_PRIVATE"])
        need(key not in seen,"Duplicate P96 pilot row")
        seen.add(key)
    for arm in ("FROZEN","MATCHED5"):
        out[arm]={}
        for cfg in CONFIGS:
            out[arm][cfg]={}
            for sel in SELECTORS:
                winners=[decide_corrected(r["arms"][arm][cfg]["fits_PRIVATE"],sel)
                         for r in rows]
                correct=sum(int(d["available"] and
                        d["winner"]==r["true_law_PRIVATE"])
                        for r,d in zip(rows,winners))
                available=sum(int(d["available"]) for d in winners)
                out[arm][cfg][sel]={"correct":correct,"available":available,
                                  "n":384,"accuracy_all":correct/384}
                if sel=="TRAIN_PEM":
                    historic=read(SOURCES["R96"])["metrics"][arm][cfg][sel]
                    need(correct==historic["correct"] and
                         available==historic["available"],
                         f"Corrected pilot PEM must be unchanged: {arm}/{cfg}")
    return out


def obs_digest(probes):
    """Fingerprint the actual A-conditioned probe representation and B target.

    RevisionProbe's raw constructor inputs have shape (51,2), but y_b is
    active[:,1], a (51,) B position series. The A conditioning trajectory
    and derivative are 101-element RK4 half-step arrays. Checking/hash both
    is stronger than hashing B alone; no simulation/noise is regenerated.
    """
    h=hashlib.sha256()
    for name in ("e1","e2","weak","strong","mid","HALF","FULL"):
        p=probes[name]
        h.update(name.encode("utf-8")+b"\0")
        for attr,expected in (("y_b",(51,)),("a_half",(101,)),
                              ("va_half",(101,))):
            obs=np.asarray(getattr(p,attr),dtype="<f8")
            need(obs.shape==expected and bool(np.isfinite(obs).all()),
                 f"Probe field invalid: {name}.{attr} has shape {obs.shape}")
            h.update(attr.encode("utf-8")+b"\0"+obs.tobytes())
        for attr in ("b0","vb0","a0"):
            val=np.asarray(float(getattr(p,attr)),dtype="<f8")
            need(bool(np.isfinite(val)),f"Nonfinite probe scalar {name}.{attr}")
            h.update(attr.encode("utf-8")+b"\0"+val.tobytes())
    return h.hexdigest()


def run_case(sid,band,mode,axis,level,base,ctx):
    p80,p90,p92,p96,p74,p75,p77,sim,api,commands,high_cmd,starts,fit_seed=ctx
    scene=mode_scenario(p80,p74,p77,base,mode,axis,level)
    probe_set=p75["create_noisy_probes"](sim,api,commands,scene,
                                           sid,band,mode,fit_seed)
    training=[probe_set[name] for name in ("e1","e2","weak","strong")]
    paired=p90["pair_probes"](scene,sid,band,mode,sim,api,
                               high_cmd,float(p77["SIGMA"]))
    all_probes={**probe_set,**paired}
    fingerprints=obs_digest(all_probes)
    comps={arm:{dom:{} for dom in DOMAINS} for arm in ARMS}
    audits={dom:{} for dom in DOMAINS}
    for law in LAWS:
        for dom in DOMAINS:
            selected,audit=fit_report(p77,p92,training,law,dom,starts[law])
            audits[dom][law]=audit
            for arm in ARMS:
                x=selected[arm]
                if x is None:
                    comps[arm][dom][law]=None
                else:
                    params=x["params_PRIVATE"]
                    mid=p92["rms"](p75,probe_set["mid"],law,params)
                    half=p92["rms"](p75,paired["HALF"],law,params)
                    full=p92["rms"](p75,paired["FULL"],law,params)
                    comps[arm][dom][law]={"mse":float(x["mse"]),
                                          "mid":mid,"half":half,"full":full,
                                          "origin":x["origin"]}
    decisions={}
    for arm in ARMS:
        decisions[arm]={}
        for cfg in CONFIGS:
            candidates={}
            for i,law in enumerate(LAWS):
                item=comps[arm][domain_for(cfg,i)][law]
                candidates[law]={"available":item is not None,
                                 "mse":None if item is None else item["mse"]}
            train_sel={sel:decide_corrected(candidates,sel) for sel in SELECTORS}
            held_sel=p92["decisions_from"](comps[arm],cfg)
            decisions[arm][cfg]={"train":train_sel,
                                 "heldout":{m:held_sel[m] for m in EVAL_METHODS}}
    nested={}
    for law in LAWS:
        w=comps["MATCHED5"]["wide"][law]
        o=comps["MATCHED5"]["outer_box"][law]
        delta=None if w is None or o is None else float(w["mse"]-o["mse"])
        nested[law]={"gap":delta,"violation":delta is not None and delta>1e-6}
    # Truth is NOT read or used before any optimizer or model-selection decision.
    return {"key_PRIVATE":[sid,band,mode,axis,level],"scene_id":sid,
            "mode_PRIVATE":mode,"true_law_PRIVATE":MODE_TO_LAW[mode],
            "observation_sha256_PRIVATE":fingerprints,
            "decisions":decisions,"fit_audit_PRIVATE":audits,
            "nested_audit_PRIVATE":nested}


def outcome(decision,law):
    return int(decision["available"] and decision["winner"]==law)


def evaluate(rows):
    n=len(rows)
    stats={}
    for arm in ARMS:
        stats[arm]={}
        for cfg in CONFIGS:
            stats[arm][cfg]={"train":{},"heldout":{}}
            for kind,methods in (("train",SELECTORS),("heldout",EVAL_METHODS)):
                for method in methods:
                    ds=[r["decisions"][arm][cfg][kind][method] for r in rows]
                    acc=sum(outcome(d,r["true_law_PRIVATE"]) for d,r in zip(ds,rows))
                    available=sum(int(d["available"]) for d in ds)
                    stats[arm][cfg][kind][method]={
                        "correct":acc,"n":n,"available":available,
                        "coverage":available/n,"accuracy_all":acc/n,
                        "accuracy_available":acc/available if available else None,
                        "by_mode_PRIVATE":{mode:{"n":sum(r["mode_PRIVATE"]==mode for r in rows),
                           "correct":sum(outcome(d,r["true_law_PRIVATE"])
                                     for d,r in zip(ds,rows) if r["mode_PRIVATE"]==mode)}
                           for mode in MODE_TO_LAW},
                    }
    return stats


def paired_primary(rows):
    # All n=384 cases contribute (abstention scores incorrect).
    all_d=[(r["scene_id"],
            outcome(r["decisions"]["MATCHED5"]["WOO"]["train"]["PROFILE_SIGMA_BIC"],
                    r["true_law_PRIVATE"]) -
            outcome(r["decisions"]["MATCHED5"]["WOO"]["train"]["TRAIN_PEM"],
                    r["true_law_PRIVATE"])) for r in rows]
    nm2=[r for r in rows if r["mode_PRIVATE"]=="nonlinear" and
         r["key_PRIVATE"][4]=="M2"]
    all_h=[(r["scene_id"],
            outcome(r["decisions"]["MATCHED5"]["WOO"]["heldout"]["MID_PLUS_FULL"],"cubic") -
            outcome(r["decisions"]["MATCHED5"]["WOO"]["heldout"]["MID_PLUS_HALF"],"cubic"))
           for r in nm2]
    return {"BIC_vs_PEM":{"sum":sum(x[1] for x in all_d),"n":len(all_d),
                            "effect":sum(x[1] for x in all_d)/len(all_d)},
            "full_vs_half_nonlinear_M2":{"sum":sum(x[1] for x in all_h),
                              "n":len(all_h),
                              "effect":sum(x[1] for x in all_h)/len(all_h)}}, all_d, all_h


def paired_bootstrap(a,b):
    need(len(a)==384 and len(b)==96,"Bootstrap requires full fresh matrix")
    per_a=defaultdict(list); per_b=defaultdict(list)
    for s,x in a:per_a[s].append(x)
    for s,x in b:per_b[s].append(x)
    need(set(per_a)==set(per_b)==set(range(N_SCENES)) and
         all(len(v)==16 for v in per_a.values()) and
         all(len(v)==4 for v in per_b.values()),
         "Unbalanced base scene bootstrap")
    aa=np.array([np.mean(per_a[s]) for s in range(N_SCENES)])
    bb=np.array([np.mean(per_b[s]) for s in range(N_SCENES)])
    rng=np.random.default_rng(BOOT_SEED)
    indices=rng.integers(0,N_SCENES,size=(N_BOOT,N_SCENES))
    return {"unit":"base_scene_id","clusters":N_SCENES,
            "n_resamples":N_BOOT,"seed":BOOT_SEED,
            "BIC_vs_PEM_CI95":[float(x) for x in np.quantile(aa[indices].mean(1),[.025,.975])],
            "full_vs_half_nonlinear_M2_CI95":
            [float(x) for x in np.quantile(bb[indices].mean(1),[.025,.975])]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke",action="store_true",help="1 out-of-cohort smoke scene; no inference")
    args=ap.parse_args()
    print("="*77)
    print("P4-B1B — INDEPENDENT FRESH-SCENE CLASSICAL SELECTOR CONFIRMATION")
    print("="*77)
    print("Smoke:",args.smoke,"| E4 sealed | No adaptive baseline selection")
    print("[PRE-OUTCOME CORRECTION] B-position y_b=(51,), N=204; P96 n=408 retained as historical pilot")
    reports, manifest=preflight()
    p80=run_path(str(SOURCES["P80"]))
    p80["check_provenance"]()
    p90=run_path(str(SOURCES["P90"]))
    p92=run_path(str(SOURCES["P92"]))
    p96=run_path(str(SOURCES["P96"]))
    p74,p75,p77,p79,src,sim,api,commands=p80["initialize"]()
    need(tuple(p77["TRAIN"]) == ("e1","e2","weak","strong") and
         p77["VALID"]=="mid" and np.isclose(p77["SIGMA"],.005) and
         int(p77["MAXITER"])==150,"Frozen train/solver protocol changed")
    need(all(len(p77["initial_points"](law))==2 for law in LAWS),
         "Original start count changed")
    starts={law:p92["common_starts"](p77,law) for law in LAWS}
    high_cmd=np.asarray(commands["high"],dtype=float)
    need(high_cmd.ndim==1 and np.isclose(float(np.max(np.abs(high_cmd))),2.75),
         "Held-out excitation waveform changed")
    n=1 if args.smoke else N_SCENES
    seed=SCENE_SEED+(SMOKE_OFFSET if args.smoke else 0)
    fit_seed=FIT_NOISE_SEED+(SMOKE_OFFSET if args.smoke else 0)
    scenes=src["make_scenes"](sim,seed,n)
    fps=validate_history_fps(p79,src,sim,scenes,reports,args.smoke)
    print("[PASS] Historical lineage / nested bounds / independent scene fingerprints")
    print("[PROTOCOL] 24x2x8 new conditions; 2+3 fixed starts/law/domain;")
    print("[PROTOCOL] TRAIN_PEM/AIC/BIC four-action only; MID+HALF/FULL separately")
    manifest_path=OUT/"p4b1b_v2_source_manifest.json"
    if manifest_path.is_file():
        need(read(manifest_path)["source_sha256"]==manifest,
             "Previous checkpoint source hash differs; STOP, do not merge runs")
    elif not args.smoke:
        write_atomic(manifest_path,{"source_sha256":manifest,
                    "scene_seed":SCENE_SEED,
                    "fit_noise_seed":FIT_NOISE_SEED,
                    "action_noise_seed":ACTION_NOISE_SEED,
                    "protocol":"Frozen P4-B1B v2: 24 fresh scenes, corrected n=204, y_b shape=51"})
    # P90 stable_seed uses this value from its global context, so supply a
    # wrapper with a different, pre-frozen action-noise seed instead of
    # mutating P90 source or shared globals. The tool's pair_probes uses
    # its own stable_seed, replace the function in the loaded dictionary's
    # private globals only; DO NOT modify the original P90 source on disk.
    stable_old=p90["stable_seed"]
    stable_globals=p90["pair_probes"].__globals__
    def new_stable_seed(sid,band,mode):
        message=f"{ACTION_NOISE_SEED}|{sid}|{band}|{mode}|amplitude_pair"
        raw=hashlib.sha256(message.encode("utf-8")).digest()
        return int.from_bytes(raw[:8],"little")
    stable_globals["stable_seed"]=new_stable_seed
    need(stable_globals["stable_seed"] is new_stable_seed,
         "Local action seed wrapper failed")
    ctx=(p80,p90,p92,p96,p74,p75,p77,sim,api,commands,high_cmd,starts,fit_seed)
    rows=[]
    for sid,scene_pair in enumerate(scenes):
        filename=OUT/"checkpoints"/f"scene_{sid:02d}.json"
        if not args.smoke and filename.is_file():
            item=read(filename)
            need(item["source_sha256"]==manifest and
                 item["scene_id"]==sid and
                 item["scene_fingerprints"]==fps[2*sid:2*sid+2] and
                 len(item["rows"])==16,
                 f"Checkpoint mismatch: {filename}")
            chunk=item["rows"]
            print(f"[RESUME] scene {sid+1}/{n}: 16 cases",flush=True)
        else:
            chunk=[]
            for band in ("original","above_original_range"):
                for mode,conditions in CONDITIONS.items():
                    for axis,level in conditions:
                        chunk.append(run_case(sid,band,mode,axis,level,
                                              scene_pair[band],ctx))
            need(len(chunk)==16,"Case grid not 16/scene")
            if not args.smoke:
                write_atomic(filename,{"scene_id":sid,
                            "scene_fingerprints":fps[2*sid:2*sid+2],
                            "source_sha256":manifest,"rows":chunk})
            print(f"[DONE] scene {sid+1}/{n}: {len(chunk)} cases",flush=True)
        rows.extend(chunk)
    need(len(rows)==16*n and len({tuple(r["key_PRIVATE"]) for r in rows})==len(rows),
         "Incomplete/duplicate test grid")
    stats=evaluate(rows)
    primary,pairs1,pairs2=paired_primary(rows)
    pilot_corrected=None if args.smoke else corrected_p96_pilot()
    if not args.smoke:
        need(primary["BIC_vs_PEM"]["n"]==384 and
             primary["full_vs_half_nonlinear_M2"]["n"]==96,
             "Primary denominators changed")
        ci=paired_bootstrap(pairs1,pairs2)
    else:
        ci=None
    nfev={dom:{"original":sum(r["fit_audit_PRIVATE"][dom][law]["original"]["nfev"]
                           for r in rows for law in LAWS),
               "extra":sum(r["fit_audit_PRIVATE"][dom][law]["nfev_extra"]
                           for r in rows for law in LAWS)} for dom in DOMAINS}
    nested=Counter(law for r in rows for law in LAWS
                   if r["nested_audit_PRIVATE"][law]["violation"])
    report={
        "protocol":"P4-B1B P96-frozen classical selector on fresh P90-style cohort",
        "smoke":args.smoke,"n_scenes":n,"n_cases":len(rows),
        "n_fit_units":n*16*len(LAWS)*len(DOMAINS),
        "scene_seed":seed,"fit_noise_seed":fit_seed,
        "action_noise_seed":ACTION_NOISE_SEED,
        "source_sha256":manifest,"scene_fingerprints":fps,
        "starts":{"historical":2,"common_extra":3,"maxiter":150,
                  "fractions":[.10,.50,.90]},
        "n_training_residuals":N_RESIDUALS,
        "p96_historical_reported_n":P96_REPORTED_N,
        "protocol_amendment_v2":{
            "timing":"Before any fresh-scene fit or classification result; original P97 smoke and formal failed on observation shape",
            "bug":"RevisionProbe.y_b shape (51,), original P97 demanded (51,2)",
            "sample_count":"P96 historical n=408 counted two channels, actual 4*51=204 B-position residuals",
            "historical_p96_immutable":True,
            "old97_fresh_outcomes_observed":False,
            "primary_criterion":"PROFILE_SIGMA_BIC with corrected n=204, frozen at v2 code version",
        },
        "p96_pilot_corrected_retrospective":pilot_corrected,
        "training_actions":["e1","e2","weak","strong"],
        "train_only_selectors":list(SELECTORS),
        "heldout_methods":list(EVAL_METHODS),
        "primary_frozen_before_new_outcome":{
            "WOO_MATCHED5_PROFILE_SIGMA_BIC_minus_TRAIN_PEM":True,
            "WOO_MATCHED5_Nonlinear_M2_Full_minus_Half":True},
        "stats":stats,"contrasts":primary,"bootstrap":ci,
        "nested_solver_violations":dict(nested),"solver_nfev":nfev,
        "E4_accessed":False,"truth_used_for_fitting_or_selection":False,
        "heldout_actions_used_for_fitting":False,"baseline_threshold_tuning":False,
        "no_historical_files_modified":True,
        "environment":{"python":sys.version,"platform":platform.platform(),
                       "numpy":np.__version__,"scipy":scipy.__version__},
        "limitations":[
            "Fresh scenes within same MuJoCo generator, NOT cross-generator validation",
            "P96 was inspected before B1B's selector was frozen; not independent of pilot design",
            "PEM/AIC/BIC use four train actions, MID+FULL uses two additional actions",
            "Candidate fit parameters are shared across selectors; AIC/BIC are scoring baselines, not independent fitters",
            "Original P96 pilot accidentally counted 408 scalar residuals; actually 204 B-position target residuals; P96 unchanged and rescored separately",
            "Gaussian pseudo information criteria: correlated residuals/uncertain effective N",
            "Known noise sigma 0.005 and k=2/3 are nominal assumptions",
            "Same number of optimizer starts does not imply equal actual nfev",
            "Finite-start SLSQP is not guaranteed globally optimal",
            "Evaluation labels enter only after all fit/selection computations",
            "Nonlinear-M2 replication is a secondary endpoint; no adaptive choice by outcomes",
            "No E4 access, no physical hardware validation, no structural identifiability proof",
        ]}
    tag="smoke" if args.smoke else "fresh"
    rp=OUT/f"p4b1b_v2_{tag}_report.json"
    ap=OUT/f"p4b1b_v2_{tag}_audit_PRIVATE.json"
    write_atomic(rp,report)
    write_atomic(ap,rows)
    if not args.smoke:
        for arm in ARMS:
            for cfg in CONFIGS:
                tr=stats[arm][cfg]["train"]
                vals=" | ".join(f"{method}:{tr[method]['correct']}/{len(rows)}"
                               for method in SELECTORS)
                print(f"{arm}/{cfg} TRAIN_ONLY: {vals}")
        print("[P96 PILOT CORRECTION] MATCHED5/WOO PROFILE_BIC:",
              pilot_corrected["MATCHED5"]["WOO"]["PROFILE_SIGMA_BIC"],
              "TRAIN_PEM:",pilot_corrected["MATCHED5"]["WOO"]["TRAIN_PEM"])
        print("[PRIMARY] BIC minus PEM:",primary["BIC_vs_PEM"],
              "CI:",ci["BIC_vs_PEM_CI95"])
        print("[SECONDARY] Nonlinear-M2 FULL minus HALF:",
              primary["full_vs_half_nonlinear_M2"],
              "CI:",ci["full_vs_half_nonlinear_M2_CI95"])
    else:
        print("[PASS] Smoke is not a scientific effect estimate")
    print("[NESTED]",dict(nested),"[FUNCTION EVALUATIONS]",nfev)
    print("Saved:",rp)
    print("Saved:",ap)
    print("P4-B1B v2 COMPLETE")

if __name__=="__main__":
    main()
