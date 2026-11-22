r"""P4-T1B: read-only competitor transitions and scale-aware margin audit.

No MuJoCo, SLSQP, new fits/simulations, threshold tuning, or E4 access.
Run from D:/OnShouldersGiants/TwinWorld_P0 as:
    .\Giant\Scripts\python.exe .\experiments\p2d1\95_p4t1b_competitor_transition_scale_audit.py --smoke
    .\Giant\Scripts\python.exe .\experiments\p2d1\95_p4t1b_competitor_transition_scale_audit.py

P88 and P90 are distinct scene cohorts and are NEVER pooled.
The true model is used only in retrospective/offline auditing, not for selection.
All new analyses are post-hoc explanatory, not registered confirmatory tests.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "p4t1b"
PATHS = {
    "R94": RES / "p4t1a" / "p4t1a_margin_report.json",
    "A94": RES / "p4t1a" / "p4t1a_margin_audit_PRIVATE.json",
    "R88": RES / "p4m2d1" / "p4m2d1_fresh_report.json",
    "A88": RES / "p4m2d1" / "p4m2d1_fresh_audit_PRIVATE.json",
    "R89": RES / "p4m2d2a" / "p4m2d2a_fresh_report.json",
    "R90": RES / "p4m2d2b" / "p4m2d2b_v2_fresh_report.json",
    "A90": RES / "p4m2d2b" / "p4m2d2b_v2_fresh_audit_PRIVATE.json",
    "R93": RES / "p4s1c" / "p4s1c_solver_closure_report.json",
}
LAWS = ("elastic", "deadzone", "cubic")
TRUE_LAW = {"elastic": "elastic", "deadzone": "deadzone", "nonlinear": "cubic"}
OBS88 = ("MID", "HIGH_ONLY", "MID_PLUS_HIGH")
OBS90 = ("MID", "HALF_ONLY", "FULL_ONLY", "MID_PLUS_HALF", "MID_PLUS_FULL")
CONFIGS = ("WWW", "OOO", "WOO")
N_BOOT = 2000
BOOT_SEED = 2029112101
ATOL = 1e-7


def need(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def read(path):
    need(path.is_file(), f"Missing immutable source: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def close(a, b, desc):
    need(np.isfinite(float(a)) and np.isfinite(float(b)), f"Nonfinite {desc}")
    need(abs(float(a)-float(b)) <= ATOL*max(1.0, abs(float(a)), abs(float(b))),
         f"Mismatch {desc}: {a} versus {b}")


def case_key(entry):
    return (entry["dataset"], *entry["key_PRIVATE"])


def stat(values):
    v = np.asarray(values, dtype=float)
    need(np.isfinite(v).all(), "Nonfinite values in metric")
    return {
        "n": int(len(v)),
        "mean": float(v.mean()) if len(v) else None,
        "median": float(np.median(v)) if len(v) else None,
        "positive": int(np.count_nonzero(v > 1e-9)),
        "negative": int(np.count_nonzero(v < -1e-9)),
        "near_zero": int(np.count_nonzero(np.abs(v) <= 1e-9)),
    }


def ci_scene_mean(rows, field, seed):
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[int(row["scene_id"])].append(float(row[field]))
    need(set(by_scene) == set(range(24)), f"Missing scene for {field}")
    vals = {i: np.array(by_scene[i], dtype=float) for i in range(24)}
    rng = np.random.default_rng(seed)
    samples = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        scene_ids = rng.integers(0, 24, size=24)
        samples[i] = np.mean(np.concatenate([vals[int(j)] for j in scene_ids]))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def load_immutable_inputs():
    d = {k: read(p) for k, p in PATHS.items()}
    r94 = d["R94"]
    need(r94["smoke"] is False, "P94 full report required")
    need(r94["E4_accessed"] is False and r94["new_fits"] == 0 and
         r94["new_simulations"] == 0 and r94["threshold_tuning"] is False,
         "P94 protocol is not read-only / E4 sealed")
    need(r94["n_cases_P88"] == r94["n_cases_P90"] == 384, "Wrong source cohort size")
    need(r94["audit_rows"]["all_cells_checked"] == 9216, "Wrong source cell count")
    for k in ("R88", "A88", "R89", "R90", "A90", "R93"):
        need(sha(PATHS[k]) == r94["input_sha256"][k], f"Immutable source drift: {k}")
    need(d["R88"]["E4_accessed"] is False and d["R90"]["E4_accessed"] is False,
         "Historic E4 violation")
    need(d["R90"]["action_used_for_fit"] is False, "Heldout action entered training")
    need(d["R93"]["new_fits"] == 0 and d["R93"]["new_simulations"] == 0,
         "Closure provenance mismatch")
    rows = d["A94"]["case_cells_PRIVATE"]
    need(len(rows) == 768, "P94 private audit missing rows")
    keys = [case_key(r) for r in rows]
    need(len(set(keys)) == 768, "Duplicate P94 cases")
    by = {cohort: sorted((r for r in rows if r["dataset"] == cohort),
                         key=case_key) for cohort in ("P88", "P90")}
    for cohort in ("P88", "P90"):
        need(len(by[cohort]) == 384, f"Missing {cohort} cases")
        counts = Counter(int(x["key_PRIVATE"][0]) for x in by[cohort])
        need(counts == Counter({i:16 for i in range(24)}), f"{cohort} scene structure mismatch")
    return d, by, {k:sha(v) for k,v in PATHS.items()}


def check_cells(by):
    n_cells = 0
    n_available = 0
    for dataset, entries in by.items():
        required = OBS88 if dataset == "P88" else OBS90
        for r in entries:
            t = TRUE_LAW[r["key_PRIVATE"][2]]
            need(r["true_law_PRIVATE"] == t, "Truth label inconsistency")
            for cfg in CONFIGS:
                need(set(r["cells"][cfg]) == set(required), "Missing observation mode")
                for obs in required:
                    c = r["cells"][cfg][obs]
                    n_cells += 1
                    if not c["available"]:
                        need(c["winner"] is None and c["gamma_mm2"] is None,
                             "Unavailable case assigned winner/margin")
                        continue
                    n_available += 1
                    errors = c["error_mm2_PRIVATE"]
                    need(set(errors) == set(LAWS), "Missing candidate loss")
                    need(all(float(v)>=0 and np.isfinite(float(v)) for v in errors.values()),
                         "Invalid candidate losses")
                    winner = min(LAWS, key=lambda law:errors[law])
                    gamma = min(errors[l]-errors[t] for l in LAWS if l!=t)
                    need(winner == c["winner"] and (winner==t)==c["correct_PRIVATE"],
                         f"Frozen candidate decision mismatch: {dataset} {case_key(r)} {cfg} {obs}")
                    close(gamma, c["gamma_mm2"], "truth-conditioned min margin")
                    for l in LAWS:
                        if l != t:
                            close(errors[l]-errors[t], c["gaps_mm2_PRIVATE"][l], "pairwise margin")
    need(n_cells == 9216, f"Expected 9216 classification cells, got {n_cells}")
    return {"cells":n_cells, "available":n_available,
            "unavailable":n_cells-n_available}


def cell(r, cfg, obs):
    c = r["cells"][cfg][obs]
    need(c["available"], f"Required P88/P90 cell unavailable: {case_key(r)} {cfg} {obs}")
    return c


def rel_margin(r, cfg, obs):
    c = cell(r,cfg,obs)
    t = r["true_law_PRIVATE"]
    err = c["error_mm2_PRIVATE"]
    rival = min((l for l in LAWS if l != t), key=lambda l:err[l])
    ltrue = float(err[t]); lwrong = float(err[rival])
    need(ltrue+lwrong > 0, "Undefined scale-free ratio (both losses zero)")
    return (lwrong-ltrue)/(lwrong+ltrue), rival, ltrue, lwrong


def audit_p88(rows, expected_size=384):
    out=[]
    for r in rows:
        t=r["true_law_PRIVATE"]
        truth_mode=r["key_PRIVATE"][2]
        mid_o=cell(r,"OOO","MID")
        mid_w=cell(r,"WOO","MID")
        joint_o=cell(r,"OOO","MID_PLUS_HIGH")
        joint_w=cell(r,"WOO","MID_PLUS_HIGH")
        for obs in OBS88:
            o=cell(r,"OOO",obs)["error_mm2_PRIVATE"]
            w=cell(r,"WOO",obs)["error_mm2_PRIVATE"]
            for law in ("deadzone","cubic"):
                close(o[law], w[law], "OOO vs WOO non-elastic candidate loss equality")
        def corr(c):return int(c["correct_PRIVATE"])
        binary=(corr(joint_w)-corr(joint_o))-(corr(mid_w)-corr(mid_o))
        j=(joint_w["gamma_mm2"]-joint_o["gamma_mm2"])-(
            mid_w["gamma_mm2"]-mid_o["gamma_mm2"])
        elastic_mid=(mid_w["error_mm2_PRIVATE"]["elastic"]-
                     mid_o["error_mm2_PRIVATE"]["elastic"])
        elastic_joint=(joint_w["error_mm2_PRIVATE"]["elastic"]-
                       joint_o["error_mm2_PRIVATE"]["elastic"])
        row={"scene_id":int(r["key_PRIVATE"][0]),
             "case_key_PRIVATE":r["key_PRIVATE"],
             "mode_PRIVATE":truth_mode, "true_law_PRIVATE":t,
             "binary_interaction_PRIVATE":binary,
             "J_mm2_PRIVATE":float(j),
             "elastic_domain_loss_difference_mid_mm2_PRIVATE":float(elastic_mid),
             "elastic_domain_loss_difference_joint_mm2_PRIVATE":float(elastic_joint),
             "OOO_mid_correct_PRIVATE":bool(mid_o["correct_PRIVATE"]),
             "WOO_mid_correct_PRIVATE":bool(mid_w["correct_PRIVATE"]),
             "OOO_joint_correct_PRIVATE":bool(joint_o["correct_PRIVATE"]),
             "WOO_joint_correct_PRIVATE":bool(joint_w["correct_PRIVATE"]),
             "OOO_mid_winner_PRIVATE":mid_o["winner"],
             "WOO_mid_winner_PRIVATE":mid_w["winner"],
             "OOO_joint_winner_PRIVATE":joint_o["winner"],
             "WOO_joint_winner_PRIVATE":joint_w["winner"]}
        out.append(row)
    need(len(out)==expected_size, "P88 size wrong")
    return out


def pattern_counts(rows):
    patterns=Counter()
    for r in rows:
        key="".join("1" if r[k] else "0" for k in (
            "OOO_mid_correct_PRIVATE","WOO_mid_correct_PRIVATE",
            "OOO_joint_correct_PRIVATE","WOO_joint_correct_PRIVATE"))
        patterns[key]+=1
    return dict(sorted(patterns.items()))


def p88_summary(rows):
    by_mode={}
    for mode in TRUE_LAW:
        sub=[x for x in rows if x["mode_PRIVATE"]==mode]
        by_mode[mode]={
            "n":len(sub),
            "binary_interaction_sum":sum(r["binary_interaction_PRIVATE"] for r in sub),
            "J_mm2":stat([r["J_mm2_PRIVATE"] for r in sub]),
            "OOO_mid_correct":sum(r["OOO_mid_correct_PRIVATE"] for r in sub),
            "WOO_mid_correct":sum(r["WOO_mid_correct_PRIVATE"] for r in sub),
            "OOO_joint_correct":sum(r["OOO_joint_correct_PRIVATE"] for r in sub),
            "WOO_joint_correct":sum(r["WOO_joint_correct_PRIVATE"] for r in sub),
            "correctness_patterns_OM_WM_OJ_WJ":pattern_counts(sub),
        }
    total=sum(r["binary_interaction_PRIVATE"] for r in rows)
    return {"n":len(rows),"binary_interaction_sum":total,
            "binary_interaction_pp":100*total/len(rows),
            "J_mm2":stat([r["J_mm2_PRIVATE"] for r in rows]),
            "J_scene_cluster_mean_CI95_exploratory":ci_scene_mean(rows,"J_mm2_PRIVATE",BOOT_SEED),
            "by_mode_PRIVATE":by_mode,
            "pattern_bit_order":"OOO-MID,WOO-MID,OOO-MID_PLUS_HIGH,WOO-MID_PLUS_HIGH",
            "only_elastic_candidate_differs_between_OOO_and_WOO":True}


def audit_p90(rows):
    out=[]
    checked=0
    for r in rows:
        checked+=1
        for cfg in CONFIGS:
            for obs in OBS90:
                c=r["cells"][cfg][obs]
                if c["available"]:
                    rel_margin(r,cfg,obs)
        if not(r["key_PRIVATE"][2] == "nonlinear" and r["key_PRIVATE"][4] == "M2"):
            continue
        a=cell(r,"WOO","MID_PLUS_HALF")
        b=cell(r,"WOO","MID_PLUS_FULL")
        rh,wh,lh,wrh=rel_margin(r,"WOO","MID_PLUS_HALF")
        rf,wf,lf,wrf=rel_margin(r,"WOO","MID_PLUS_FULL")
        delta=b["gamma_mm2"]-a["gamma_mm2"]
        close(delta,(wrf-wrh)-(lf-lh),"P90 loss-gap decomposition")
        out.append({
            "scene_id":int(r["key_PRIVATE"][0]),
            "case_key_PRIVATE":r["key_PRIVATE"],
            "axis_PRIVATE":r["key_PRIVATE"][3],
            "correct_half_PRIVATE":bool(a["correct_PRIVATE"]),
            "correct_full_PRIVATE":bool(b["correct_PRIVATE"]),
            "best_wrong_half_PRIVATE":wh,
            "best_wrong_full_PRIVATE":wf,
            "gamma_half_mm2_PRIVATE":float(a["gamma_mm2"]),
            "gamma_full_mm2_PRIVATE":float(b["gamma_mm2"]),
            "delta_gamma_mm2_PRIVATE":float(delta),
            "relative_margin_half_PRIVATE":float(rh),
            "relative_margin_full_PRIVATE":float(rf),
            "delta_relative_margin_PRIVATE":float(rf-rh),
            "delta_true_model_loss_mm2_PRIVATE":float(lf-lh),
            "delta_best_wrong_model_loss_mm2_PRIVATE":float(wrf-wrh),
        })
    need(checked==384 and len(out)==96,"P90 384/96 coverage failed")
    return out


def p90_summary(rows):
    cls=Counter((r["correct_half_PRIVATE"],r["correct_full_PRIVATE"]) for r in rows)
    statuses={"both_correct":cls[(True,True)], "corrected_by_full":cls[(False,True)],
              "harmed_by_full":cls[(True,False)], "both_wrong":cls[(False,False)]}
    switches=Counter(f"{r['best_wrong_half_PRIVATE']}->{r['best_wrong_full_PRIVATE']}" for r in rows)
    by_axis={}
    for axis in sorted(set(r["axis_PRIVATE"] for r in rows)):
        sub=[r for r in rows if r["axis_PRIVATE"]==axis]
        by_axis[axis]={"n":len(sub),
                       "half_correct":sum(r["correct_half_PRIVATE"] for r in sub),
                       "full_correct":sum(r["correct_full_PRIVATE"] for r in sub),
                       "delta_gamma_mm2":stat([r["delta_gamma_mm2_PRIVATE"] for r in sub]),
                       "delta_relative_margin":stat([r["delta_relative_margin_PRIVATE"] for r in sub])}
    return {
        "n":len(rows),"half_correct":sum(r["correct_half_PRIVATE"] for r in rows),
        "full_correct":sum(r["correct_full_PRIVATE"] for r in rows),
        "paired_status":statuses,
        "delta_gamma_mm2":stat([r["delta_gamma_mm2_PRIVATE"] for r in rows]),
        "delta_relative_margin":stat([r["delta_relative_margin_PRIVATE"] for r in rows]),
        "delta_relative_margin_mean_scene_cluster_CI95_exploratory":
            ci_scene_mean(rows,"delta_relative_margin_PRIVATE",BOOT_SEED+1),
        "delta_best_wrong_model_loss_mm2":stat([r["delta_best_wrong_model_loss_mm2_PRIVATE"] for r in rows]),
        "delta_true_model_loss_mm2":stat([r["delta_true_model_loss_mm2_PRIVATE"] for r in rows]),
        "best_wrong_rival_switches_PRIVATE":dict(sorted(switches.items())),
        "by_axis_PRIVATE":by_axis,
        "relative_margin_definition":"(L_best_wrong - L_truth)/(L_best_wrong + L_truth), dimensionless; scale-invariant only under common loss multiplier",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true",help="One scene per cohort; check interfaces only")
    args=p.parse_args()
    print("="*74)
    print("P4-T1B — FROZEN COMPETITOR-TRANSITION / RELATIVE-MARGIN AUDIT")
    print("="*74)
    d,cohorts,hashes=load_immutable_inputs()
    check=check_cells(cohorts)  # checks every one of the 9216 inputs even in smoke
    print(f"[PASS] Immutable SHA lineage, {check['cells']} candidate decisions revalidated")
    if args.smoke:
        data88=[r for r in cohorts["P88"] if r["key_PRIVATE"][0]==0]
        data90=[r for r in cohorts["P90"] if r["key_PRIVATE"][0]==0]
    else:
        data88=cohorts["P88"]
        data90=cohorts["P90"]
    p88=audit_p88(data88, expected_size=(16 if args.smoke else 384))
    # In smoke P90 has four nonlinear M2 rows from one scene, not 96.
    if args.smoke:
        p90=[]
        for r in data90:
            if r["key_PRIVATE"][2]=="nonlinear" and r["key_PRIVATE"][4]=="M2":
                a=cell(r,"WOO","MID_PLUS_HALF");b=cell(r,"WOO","MID_PLUS_FULL")
                rh,_,_,_=rel_margin(r,"WOO","MID_PLUS_HALF")
                rf,_,_,_=rel_margin(r,"WOO","MID_PLUS_FULL")
                p90.append({"scene_id":0,"case_key_PRIVATE":r["key_PRIVATE"],
                            "delta_gamma_mm2_PRIVATE":b["gamma_mm2"]-a["gamma_mm2"],
                            "delta_relative_margin_PRIVATE":rf-rh})
        need(len(p88)==16 and len(p90)==4,"Smoke cohort shape mismatch")
    else:
        p90=audit_p90(data90)
    report={
        "protocol":"P4-T1B: immutable P88/P90 competitor and scale-sensitive margin audit",
        "smoke":bool(args.smoke),
        "n_p88":len(p88),"n_p90_nonlinear_m2":len(p90),
        "full_immutable_decisions_checked":check,
        "source_sha256":hashes,
        "self_sha256":sha(Path(__file__).resolve()),
        "E4_accessed":False,"new_fits":0,"new_simulations":0,
        "new_actions":0,"threshold_tuning":False,"truth_used_for_fit_or_selection":False,
        "different_cohorts_not_pooled":True,
        "posthoc_exploratory_not_predeclared":True,
        "limitations":[
            "Truth-conditioned margins are retrospective diagnostics, not operational agent scores",
            "Mean-margins and correctness are distinct statistics; neither implies the other's sign",
            "Relative margin is scale-free only for a common multiplicative scaling of all candidate losses",
            "The Full and Half trajectories need not be related by simple physical scaling",
            "A positive relative-margin shift is not a proof of action information gain or structural identifiability",
            "No noise covariance / Bayesian error probability is estimated",
            "P88/P90 independent scenes are separate cohorts and cannot be pooled for causal inference",
            "No optimizer-independent or hardware generalization claim",
        ],
    }
    if not args.smoke:
        report["P88"] = p88_summary(p88)
        report["P90_nonlinear_M2"] = p90_summary(p90)
        close(report["P88"]["binary_interaction_sum"]/384,
              d["R94"]["P88"]["registered_binary_interaction"],"P88 registered interaction")
        close(report["P90_nonlinear_M2"]["full_correct"]-report["P90_nonlinear_M2"]["half_correct"],
              19,"P90 +19 net corrected")
        close(report["P90_nonlinear_M2"]["delta_gamma_mm2"]["mean"],
              d["R94"]["P90_nonlinear_M2"]["exploratory_delta_gamma_full_minus_half_mm2"]["mean"],
              "P90 absolute gap mean")
        close(report["P88"]["J_mm2"]["mean"],
              d["R94"]["P88"]["exploratory_J_mm2"]["mean"],"P88 J mean")
        print("[PASS] P88 +32/384 interaction and mechanism-stratified decomposition")
        print("[PASS] P90 68->87/96, relative margins, and two-loss decomposition")
        print("[P88] binary contributions:", {k:v["binary_interaction_sum"] for k,v in report["P88"]["by_mode_PRIVATE"].items()})
        print("[P88] mean J mm2:", {k:round(v["J_mm2"]["mean"],5) for k,v in report["P88"]["by_mode_PRIVATE"].items()})
        print("[P90] delta relative margin mean/median:",
              report["P90_nonlinear_M2"]["delta_relative_margin"]["mean"],
              report["P90_nonlinear_M2"]["delta_relative_margin"]["median"])
        print("[P90] relative-margin cluster CI:",
              report["P90_nonlinear_M2"]["delta_relative_margin_mean_scene_cluster_CI95_exploratory"])
    else:
        report["smoke_checks"]="Source lineage, all decision cells, one P88/P90 scene, no inference"
        print("[PASS] Smoke interface checks, one scene per cohort, no statistical conclusion")
    name="p4t1b_smoke" if args.smoke else "p4t1b_competitor"
    pr=OUT/(name+"_report.json")
    pa=OUT/(name+"_audit_PRIVATE.json")
    save(pr,report)
    save(pa,{"P88_rows_PRIVATE":p88,"P90_nonlinear_M2_rows_PRIVATE":p90})
    print("Saved:",pr)
    print("Saved:",pa)
    print("P4-T1B COMPLETE")

if __name__=="__main__":
    main()
