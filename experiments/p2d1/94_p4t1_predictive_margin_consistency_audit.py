"""P4-T1A: frozen predictive-competition / theory-consistency audit.

Strictly read-only for all historical input data. No MuJoCo, no fitting,
no optimization, no E4, no parameter or threshold selection. Ground truth
is used ONLY to audit mechanisms and predictive margins after all decisions
were frozen. P88 and P90 are different, non-pooled scene cohorts.

Outputs: p4t1a_margin_report.json + p4t1a_margin_audit_PRIVATE.json.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
P88 = RES / "p4m2d1"
P89 = RES / "p4m2d2a"
P90 = RES / "p4m2d2b"
P93 = RES / "p4s1c"
OUT = RES / "p4t1a"
PATHS = {
    "R88": P88 / "p4m2d1_fresh_report.json",
    "A88": P88 / "p4m2d1_fresh_audit_PRIVATE.json",
    "R89": P89 / "p4m2d2a_fresh_report.json",
    "R90": P90 / "p4m2d2b_v2_fresh_report.json",
    "A90": P90 / "p4m2d2b_v2_fresh_audit_PRIVATE.json",
    "R93": P93 / "p4s1c_solver_closure_report.json",
}
LAWS = ("elastic", "deadzone", "cubic")
CONFIGS = ("WWW", "OOO", "WOO")
TRUE_LAW = {"elastic": "elastic", "deadzone": "deadzone", "nonlinear": "cubic"}
OBS88 = ("MID", "HIGH_ONLY", "MID_PLUS_HIGH")
OBS90 = ("MID", "HALF_ONLY", "FULL_ONLY", "MID_PLUS_HALF", "MID_PLUS_FULL")
N_BOOT = 2000
BOOT_SEED = 2029112001


def need(x, msg):
    if not x:
        raise RuntimeError(msg)


def read(p):
    need(p.is_file(), f"Missing file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def hashfile(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def key(r):
    return (int(r["scene_id"]), r["band"], r["mode_PRIVATE"], r["axis_PRIVATE"], r["level_PRIVATE"])


def source_check(d):
    r88, a88, r89, r90, a90, r93 = (d[x] for x in ("R88", "A88", "R89", "R90", "A90", "R93"))
    need(r88["smoke"] is False and r90["smoke"] is False, "Formal reports required")
    need(r88["n_cases"] == r90["n_cases"] == 384, "Cohort size mismatch")
    need(r88["n_base_scenes"] == r90["n_base_scenes"] == 24, "Cluster size mismatch")
    need(len(a88) == len(a90) == 384, "Missing case audit rows")
    need(r88["E4_accessed"] is False and r90["E4_accessed"] is False and
         r93["E4_accessed"] is False, "E4 boundary violation")
    need(r93["new_fits"] == r93["new_simulations"] == 0, "P93 not read only")
    for fname, expected in r89["source_sha256"].items():
        if fname == "p4m2d1_fresh_report.json":
            need(hashfile(PATHS["R88"]) == expected, "R88 hash mismatch via P89")
        if fname == "p4m2d1_fresh_audit_PRIVATE.json":
            need(hashfile(PATHS["A88"]) == expected, "A88 hash mismatch via P89")
    expected90 = r90["source_provenance"]
    need(hashfile(PATHS["R88"]) == expected90["p4m2d1_fresh_report.json"],
         "P90 historic P88 source hash mismatch")
    closure = r93["source_sha256"]
    need(hashfile(PATHS["R90"]) == closure["R90"], "R90 hash mismatch via P93")
    need(hashfile(PATHS["A90"]) == closure["A90"], "A90 hash mismatch via P93")
    need(r90["action_used_for_fit"] is False and r90["threshold_tuning"] is False,
         "P90 protocol unexpectedly changed")
    need(r88["truth_used_for_fitting"] is False and r88["high_used_for_fitting"] is False,
         "P88 protocol unexpectedly changed")
    return {k: hashfile(p) for k, p in PATHS.items()}


def check_cohort(rows, name):
    ids = [key(r) for r in rows]
    need(len(ids) == len(set(ids)) == 384, f"{name}: duplicate case keys")
    need(set(k[0] for k in ids) == set(range(24)), f"{name}: missing scenes")
    need(all(sum(k[0] == s for k in ids) == 16 for s in range(24)),
         f"{name}: bad case count in scene")
    return sorted(rows, key=key)


def rms_to_mse(rms):
    need(rms is not None, "Missing RMS in apparently available candidate")
    x = float(rms)
    need(np.isfinite(x) and x >= 0, "Invalid RMS")
    return x*x


def loss88(row, cfg, obs):
    loss = {}
    for i, law in enumerate(LAWS):
        domain = "wide" if cfg[i] == "W" else "outer_box"
        mid = row["mid_rms_PRIVATE"][domain][law]
        high = row["high_rms_PRIVATE"][domain][law]
        val = None
        if (obs == "MID" and mid is not None) or (obs == "HIGH_ONLY" and high is not None) or (mid is not None and high is not None):
            if obs == "MID": val = rms_to_mse(mid)
            elif obs == "HIGH_ONLY": val = rms_to_mse(high)
            elif obs == "MID_PLUS_HIGH": val = (rms_to_mse(mid)+rms_to_mse(high))/2.0
            else: raise RuntimeError(f"Unknown P88 obs: {obs}")
        loss[law] = val
    return loss


def loss90(row, cfg, obs):
    loss = {}
    for i, law in enumerate(LAWS):
        domain = "wide" if cfg[i] == "W" else "outer_box"
        status = row["fit_status_PRIVATE"][domain][law]
        if not status["available"]:
            loss[law] = None
            continue
        mid = rms_to_mse(status["validation_rms_mm"])
        if obs == "MID": val = mid
        else:
            dose = "HALF" if "HALF" in obs else "FULL"
            held = rms_to_mse(row["heldout_rms_PRIVATE"][dose][domain][law])
            val = (mid+held)/2.0 if obs.startswith("MID_PLUS") else held
        loss[law] = val
    return loss


def diagnose(row, cfg, obs, dataset):
    losses = loss88(row,cfg,obs) if dataset=="P88" else loss90(row,cfg,obs)
    dec = (row["decisions"][obs][cfg] if dataset=="P88"
           else row["decisions"][cfg][obs])
    avail = all(v is not None for v in losses.values())
    need(avail == bool(dec["available"]), f"Availability mismatch: {dataset} {key(row)} {cfg} {obs}")
    truth = TRUE_LAW[row["mode_PRIVATE"]]
    if not avail:
        need(dec["winner"] is None, "Missing candidate was assigned a winner")
        return {"available": False, "winner": None, "gamma_mm2": None,
                "gaps_mm2_PRIVATE": None, "error_mm2_PRIVATE": losses, "correct_PRIVATE": False}
    winner = min(LAWS, key=lambda law: losses[law])
    need(winner == dec["winner"], f"Winner parity mismatch: {dataset} {key(row)} {cfg} {obs}")
    need((winner == truth) == bool(dec["correct_PRIVATE"]), "Correctness mismatch")
    gaps = {law: losses[law]-losses[truth] for law in LAWS if law!=truth}
    gamma = min(gaps.values())
    # Deterministic ties follow original candidate order. A zero margin may
    # be correct or incorrect depending on which law wins the tie.
    return {"available": True, "winner": winner, "gamma_mm2": float(gamma),
            "gaps_mm2_PRIVATE": {k: float(v) for k,v in gaps.items()},
            "error_mm2_PRIVATE": {k: float(v) for k,v in losses.items()},
            "correct_PRIVATE": winner==truth}


def finite_stats(values):
    vals = np.asarray(values, dtype=float)
    if not len(vals):
        return {"n": 0, "mean": None, "median": None, "positive": None, "negative": None, "zero": None}
    return {"n": len(vals), "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "positive": int((vals>1e-9).sum()), "negative": int((vals<-1e-9).sum()),
            "zero": int((abs(vals)<=1e-9).sum())}


def bootstrap_scene_mean(rows, field, seed):
    by_scene=defaultdict(list)
    for r in rows:
        if r.get(field) is not None:
            by_scene[r["scene_id"]].append(r[field])
    ids=sorted(by_scene)
    need(ids==list(range(24)), f"Missing scene groups for {field}")
    rng=np.random.default_rng(seed)
    dist=[]
    for _ in range(N_BOOT):
        chosen=rng.integers(0,24,size=24)
        vals=[v for sid in chosen for v in by_scene[int(sid)]]
        dist.append(float(np.mean(vals)))
    return [float(np.quantile(dist,0.025)),float(np.quantile(dist,0.975))]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true",help="Process one complete scene; no scientific inference")
    args=parser.parse_args()
    print("="*70)
    print("P4-T1A — FROZEN SQUARED-LOSS COMPETITION MARGIN AUDIT")
    print("="*70)
    d={k: read(p) for k,p in PATHS.items()}
    hashes=source_check(d)
    all88=check_cohort(d["A88"],"P88")
    all90=check_cohort(d["A90"],"P90")
    data88=[r for r in all88 if (not args.smoke or r["scene_id"]==0)]
    data90=[r for r in all90 if (not args.smoke or r["scene_id"]==0)]
    private=[]
    by88=[]
    by90=[]
    for dataset,rows,obs in (("P88",data88,OBS88),("P90",data90,OBS90)):
        for row in rows:
            e={"dataset":dataset,"key_PRIVATE":list(key(row)),"true_law_PRIVATE":TRUE_LAW[row["mode_PRIVATE"]],"cells":{}}
            for cfg in CONFIGS:
                e["cells"][cfg]={}
                for method in obs:
                    e["cells"][cfg][method]=diagnose(row,cfg,method,dataset)
            private.append(e)
            if dataset=="P88":
                def gamma(cfg,obs): return e["cells"][cfg][obs]["gamma_mm2"]
                v=[gamma(cfg,method) for cfg in ("WOO","OOO") for method in ("MID","MID_PLUS_HIGH")]
                j=(v[1]-v[3])-(v[0]-v[2]) if all(x is not None for x in v) else None
                def correct(cfg,method): return int(e["cells"][cfg][method]["correct_PRIVATE"])
                binary=((correct("WOO","MID_PLUS_HIGH")-correct("OOO","MID_PLUS_HIGH"))
                        -(correct("WOO","MID")-correct("OOO","MID")))
                by88.append({"scene_id":row["scene_id"],"mode_PRIVATE":row["mode_PRIVATE"],
                             "level_PRIVATE":row["level_PRIVATE"], "J_mm2_PRIVATE":j,
                             "binary_interaction_PRIVATE":binary})
            else:
                def gamma(method): return e["cells"]["WOO"][method]["gamma_mm2"]
                half=gamma("MID_PLUS_HALF"); full=gamma("MID_PLUS_FULL")
                delta=full-half if half is not None and full is not None else None
                correct_h=e["cells"]["WOO"]["MID_PLUS_HALF"]["correct_PRIVATE"]
                correct_f=e["cells"]["WOO"]["MID_PLUS_FULL"]["correct_PRIVATE"]
                sep=row["separation_mm"]["WOO"]
                normsep=(sep["FULL"]/2.0-sep["HALF"]) if all(sep[x] is not None for x in ("HALF","FULL")) else None
                by90.append({"scene_id":row["scene_id"],"mode_PRIVATE":row["mode_PRIVATE"],
                             "axis_PRIVATE":row["axis_PRIVATE"],"level_PRIVATE":row["level_PRIVATE"],
                             "delta_gamma_full_minus_half_mm2_PRIVATE":delta,
                             "correct_half_PRIVATE":bool(correct_h), "correct_full_PRIVATE":bool(correct_f),
                             "separation_full_over_2_minus_half_mm":normsep})
    print("[PASS] Exact winner parity for P88 and P90, all available methods/configurations")
    need(len(private)==(32 if args.smoke else 768),"Wrong audit size")
    source={k:hashes[k] for k in hashes}
    report={"protocol":"P4-T1A read-only theory/empirical margin-consistency audit",
            "smoke":bool(args.smoke),"cohorts_kept_separate":True,
            "n_cases_P88":len(data88),"n_cases_P90":len(data90),"input_sha256":source,
            "metric_definitions":{
                "loss_mm2":"(observed candidate prediction RMS in mm)^2; joint loss averages MID and held-out squared RMS",
                "gamma_mm2":"min_{j!=truth}(L_j-L_truth); offline truth labels only for auditing",
                "J_mm2":"[gamma(WOO,joint)-gamma(OOO,joint)]-[gamma(WOO,MID)-gamma(OOO,MID)]",
                "P90_delta_gamma_mm2":"gamma(WOO,MID_PLUS_FULL)-gamma(WOO,MID_PLUS_HALF)",
                "dose_normalized_separation_mm":"Elastic-vs-Cubic FULL separation / 2 - HALF separation; descriptive only",
                "abstention":"No margin when a required candidate fit is unavailable; never drop case from classification denominator"},
            "E4_accessed":False,"new_fits":0,"new_simulations":0,"threshold_tuning":False,
            "truth_used_for_model_choice":False,
            "limitations":[
                "P88 and P90 are separate cohorts; no pooled causal estimate",
                "True-law margins are offline explanatory metrics, unavailable to an actual agent",
                "Positive gamma is only a deterministic score comparison, not a calibrated error probability",
                "Squared-error margin differs from the earlier RMS Elastic-Cubic score gap",
                "Gaussian risk formula is not validated without independent noise/covariance information",
                "Increasing action amplitude changes signal scale and may change dynamics; no isolated information-theoretic gain is proven",
                "Any inferential summaries here are post-hoc descriptive; the registered P88/P90 contrasts retain priority",
                "No structural identifiability guarantee and no real-world transfer claim"]}
    if args.smoke:
        report["smoke_checks"]={"p88_row_count":len(data88),"p90_row_count":len(data90),
                                "description":"Parity and input checks only; no effect estimation"}
    else:
        vals88=[r["J_mm2_PRIVATE"] for r in by88 if r["J_mm2_PRIVATE"] is not None]
        binary88=sum(r["binary_interaction_PRIVATE"] for r in by88)/len(by88)
        need(abs(binary88-d["R88"]["comparisons"]["PRIMARY_action_prior_interaction_overall"])<1e-12,
             "P88 primary interaction reproduction mismatch")
        nsub=[r for r in by90 if r["mode_PRIVATE"]=="nonlinear" and r["level_PRIVATE"]=="M2"]
        need(len(nsub)==96,"P90 Nonlinear M2 wrong count")
        n_avail=sum(r["delta_gamma_full_minus_half_mm2_PRIVATE"] is not None for r in nsub)
        need(n_avail==96,"P90 primary gamma coverage incomplete")
        p90diff=sum(int(r["correct_full_PRIVATE"])-int(r["correct_half_PRIVATE"]) for r in nsub)/96
        need(abs(p90diff-d["R90"]["predeclared_effects"]["PRIMARY_WOO_nonlinear_M2_recall_full_minus_half"])<1e-12,
             "P90 registered primary effect mismatch")
        need(abs(p90diff-d["R93"]["primary_paired"]["FROZEN"]["full_minus_half"])<1e-12,
             "P93 closure primary effect mismatch")
        report["P88"]={"n":len(by88),"n_J_complete":len(vals88),
                       "registered_binary_interaction":binary88,
                       "exploratory_J_mm2":finite_stats(vals88),
                       "exploratory_J_scene_cluster_mean_CI95":(bootstrap_scene_mean(by88,"J_mm2_PRIVATE",BOOT_SEED) if len(vals88)==384 else None),
                       "J_by_mode_PRIVATE":{m:finite_stats([r["J_mm2_PRIVATE"] for r in by88 if r["mode_PRIVATE"]==m and r["J_mm2_PRIVATE"] is not None]) for m in TRUE_LAW}}
        gamma_delta=[r["delta_gamma_full_minus_half_mm2_PRIVATE"] for r in nsub]
        cls=Counter((r["correct_half_PRIVATE"],r["correct_full_PRIVATE"]) for r in nsub)
        norm=[r["separation_full_over_2_minus_half_mm"] for r in nsub if r["separation_full_over_2_minus_half_mm"] is not None]
        def group(mask):
            group_rows=[r for r in nsub if mask(r)]
            return {"n":len(group_rows),"delta_gamma":finite_stats([r["delta_gamma_full_minus_half_mm2_PRIVATE"] for r in group_rows])}
        report["P90_nonlinear_M2"]={"n":96,
           "registered_recall_full_minus_half":p90diff,
           "paired_counts_PRIVATE":{"both_correct":cls[(True,True)],"corrected_by_full":cls[(False,True)],
                                    "harmed_by_full":cls[(True,False)],"both_wrong":cls[(False,False)]},
           "exploratory_delta_gamma_full_minus_half_mm2":finite_stats(gamma_delta),
           "exploratory_delta_gamma_mean_scene_cluster_CI95":bootstrap_scene_mean(nsub,"delta_gamma_full_minus_half_mm2_PRIVATE",BOOT_SEED+1),
           "gamma_change_by_status_PRIVATE":{
               "corrected_by_full":group(lambda r:not r["correct_half_PRIVATE"] and r["correct_full_PRIVATE"]),
               "harmed_by_full":group(lambda r:r["correct_half_PRIVATE"] and not r["correct_full_PRIVATE"]),
               "unchanged_correct":group(lambda r:r["correct_half_PRIVATE"] and r["correct_full_PRIVATE"]),
               "unchanged_wrong":group(lambda r:not r["correct_half_PRIVATE"] and not r["correct_full_PRIVATE"])
           },
           "exploratory_dose_normalized_separation_mm":finite_stats(norm)}
        print("[PASS] P88 registered +8.33pp interaction reproduced")
        print("[PASS] P90 96-case +19/96 paired effect reproduced and P93 closure parity")
        print(f"[P88]  J (mm2) median={np.median(vals88):+.6f}; 24-scene mean CI={report['P88']['exploratory_J_scene_cluster_mean_CI95']}")
        print(f"[P90]  Delta gamma (mm2) median={np.median(gamma_delta):+.6f}; dose normalized separation median={np.median(norm):+.6f} mm")
    report["audit_rows"]={"P88":len(by88),"P90":len(by90),"all_cells_checked":int(len(data88)*len(CONFIGS)*len(OBS88)+len(data90)*len(CONFIGS)*len(OBS90))}
    out=OUT/("p4t1a_smoke_report.json" if args.smoke else "p4t1a_margin_report.json")
    private_out=OUT/("p4t1a_smoke_audit_PRIVATE.json" if args.smoke else "p4t1a_margin_audit_PRIVATE.json")
    write(out,report)
    write(private_out,{"case_cells_PRIVATE":private,"P88_exploratory_rows_PRIVATE":by88,"P90_exploratory_rows_PRIVATE":by90})
    print("Saved:",out)
    print("Saved:",private_out)
    print("P4-T1A COMPLETE")

if __name__=="__main__":
    main()
