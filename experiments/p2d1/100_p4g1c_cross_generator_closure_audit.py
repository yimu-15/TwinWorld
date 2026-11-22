"""P4-G1C (#100): read-only independent closure audit of P4-G1B (#99).

Verify all 384 paired ODE cases, the 24 saved checkpoints, source hashes,
all 84 aggregate classifier cells, candidate-score parity, true-law paired
contrasts and the original 2,000x24-scene bootstrap. Summarize mechanism-
specific cross-arm changes as POST-HOC DESCRIPTION, not causal or predictive
confirmation. Does not invoke MuJoCo, RK4 simulations, SLSQP, or E4.

Windows Giant (from repository root):
  .\\Giant\\Scripts\\python.exe experiments\\p2d1\\100_p4g1c_cross_generator_closure_audit.py --smoke
  .\\Giant\\Scripts\\python.exe experiments\\p2d1\\100_p4g1c_cross_generator_closure_audit.py

For testing on materialized uploads outside the repository:
  python 100_p4g1c_cross_generator_closure_audit.py \\
    --input-dir /mnt/data --source-dir /mnt/data \\
    --checkpoint-zip '/mnt/data/checkpoints(2).zip'
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = "p4g1b_fresh_report.json"
AUDIT = "p4g1b_fresh_audit_PRIVATE.json"
MANIFEST = "p4g1b_source_manifest.json"
SMOKE_REPORT = "p4g1b_smoke_report.json"
LAWS = ("elastic", "deadzone", "cubic")
TRUTH = {"elastic": "elastic", "deadzone": "deadzone", "nonlinear": "cubic"}
GENERATORS = ("IDEAL_ODE", "SHIFTED_ODE")
BUDGETS = ("FROZEN2", "MATCHED5")
CONFIGS = ("WWW", "OOO", "WOO")
METHODS = ("MID", "HALF_ONLY", "FULL_ONLY", "MID_PLUS_HALF", "MID_PLUS_FULL",
           "TRAIN_PEM", "PROFILE_SIGMA_BIC")
N_SCENES = 24
N_RESIDUALS = 204
PARAMETERS = {"elastic": 2, "deadzone": 3, "cubic": 3}
BOOT_SEED = 2029122702
N_BOOT = 2000
GAP_TOL = 1e-6
SOURCE_NAMES = {
    "SELF": "99_p4g1b_independent_generator_identification.py",
    "P16": "16_p2d1b_probe_diversity.py",
    "P22": "22_p3a1b_structural_revision.py",
    "P77": "77_p4m2b2a_matched_domain_fit.py",
    "P92": "92_p4s1b_full_solver_budget_audit.py",
    "P97": "97_p4b1b_fresh_classical_baseline_confirmation_v2.py",
    "P98": "98_p4g1a_independent_physics_generator_feasibility.py",
    "R97": "p4b1b_v2_fresh_report.json",
    "R98": "p4g1a_feasibility_report.json",
    "R98_SMOKE": "p4g1a_smoke_report.json",
}


def need(test, message):
    if not test:
        raise RuntimeError(message)


def load_json(path):
    need(path.is_file(), f"Missing read-only prerequisite: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def object_hash(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def get_source(srcdir, key, data_dir):
    """Canonical repo paths; accept hash-verified upload aliases in test runtime."""
    name = SOURCE_NAMES[key]
    choices = [srcdir / name]
    if key == "R97":
        choices.insert(0, srcdir / "results" / "p4b1b_v2" / name)
    elif key in ("R98", "R98_SMOKE"):
        choices.insert(0, srcdir / "results" / "p4g1a" / name)
    if key in ("R97", "R98", "R98_SMOKE"):
        choices.append(data_dir / name)
    else:
        choices.extend(sorted(srcdir.glob(Path(name).stem + "(*).py")))
    return [p for p in choices if p.is_file()]


def verify_sources(r, manifest, smoke_report, srcdir, data_dir):
    need(manifest["source_sha256"] == r["source_sha256"], "Source lineage differs")
    need(manifest["scene_fingerprints"] == r["scene_fingerprints_PRIVATE"],
         "Scene fingerprints/manifest conflict")
    expected = r["source_sha256"]
    checked = {}
    for key in SOURCE_NAMES:
        candidates = get_source(srcdir, key, data_dir)
        matches = [p for p in candidates if file_hash(p) == expected[key]]
        need(bool(matches), f"Missing/mutated frozen source {key}: {SOURCE_NAMES[key]}")
        checked[key] = str(matches[0].name)
    need(smoke_report["source_sha256"] == expected, "Smoke vs formal source lineage differs")
    need(smoke_report["smoke"] is True and smoke_report["formal_case_results_observed"] is False,
         "G1B smoke must contain no formal outcomes")
    need(r["smoke"] is False and r["n_scenes"] == N_SCENES and
         r["n_cases_per_generator"] == 384 and r["n_total_case_arm_pairs"] == 768 and
         r["n_fit_units"] == 4608, "Frozen cohort size changed")
    need(r["n_training_scalar_residuals"] == N_RESIDUALS,
         "Training residual count changed")
    need(r["fit_budget"] == {"original":2,"extra":3,"maxiter":150,"ftol":1e-9},
         "Frozen solver protocol changed")
    for field in ("E4_accessed", "truth_used_for_fit_or_selection",
                  "heldout_actions_used_for_fit", "threshold_tuning",
                  "is_zero_shot_parameter_transfer", "is_independent_hardware",
                  "has_two_independent_ODE_engines"):
        need(r[field] is False, f"Unexpected reporting flag: {field}")
    need(r["new_fits"] == 4608, "New-fit count differs")
    need(len(r["scene_fingerprints_PRIVATE"]) == 48 and
         len(set(r["scene_fingerprints_PRIVATE"])) == 48, "Scene fingerprint issue")
    return checked


def parse_checkpoint(name, folder, zf):
    if zf is None:
        return load_json(folder / f"scene_{name:02d}.json")
    with zf.open(f"checkpoints/scene_{name:02d}.json") as f:
        return json.load(f)


def verify_cases(rows, r, folder, zf, smoke):
    count = 16 if smoke else 384
    sample = rows[:count]
    need(len(rows) == 384, "Full case audit must include 384 rows")
    keys = [tuple(x["key_PRIVATE"]) for x in rows]
    need(len(set(keys)) == 384, "Duplicated 5-part case keys")
    for sid in range(24):
        sub = [row for row in rows if row["scene_id"] == sid]
        need(len(sub) == 16 and all(row["key_PRIVATE"][0] == sid for row in sub),
             f"Case distribution invalid at scene {sid}")
    for sid in range(1 if smoke else 24):
        cp = parse_checkpoint(sid, folder, zf)
        block = [row for row in rows if row["scene_id"] == sid]
        need(cp["scene_id"] == sid and cp["source_sha256"] == r["source_sha256"] and
             cp["base_fingerprints"] == r["scene_fingerprints_PRIVATE"][2*sid:2*sid+2] and
             cp["rows_sha256"] == object_hash(cp["rows"]) and cp["rows"] == block,
             f"Checkpoint content differs: scene {sid}")
    return sample


def dom(cfg, law):
    return "wide" if cfg[LAWS.index(law)] == "W" else "outer_box"


def approximate(a, b, tol=1e-9):
    return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=tol)


def verify_row_scores(row):
    n = 0
    for gen in GENERATORS:
        for arm in BUDGETS:
            fits = row["fit_results_PRIVATE"][gen][arm]
            for cfg in CONFIGS:
                chosen = {law:fits[dom(cfg,law)][law] for law in LAWS}
                valid = all(x is not None for x in chosen.values())
                for method in METHODS:
                    d = row["decisions"][gen][arm][cfg][method]
                    need(bool(d["available"]) == valid, "Decision availability/fit mismatch")
                    if not valid:
                        need(d["winner"] is None, "Unavailable candidate must abstain")
                        continue
                    scores = d.get("scores_mm_PRIVATE",d.get("scores_PRIVATE"))
                    need(isinstance(scores,dict) and set(scores) == set(LAWS),
                         "Missing candidate score")
                    for law in LAWS:
                        v = chosen[law]
                        if method == "MID": expected = v["mid"]
                        elif method == "HALF_ONLY": expected = v["half"]
                        elif method == "FULL_ONLY": expected = v["full"]
                        elif method == "MID_PLUS_HALF":
                            expected = math.sqrt((v["mid"]**2 + v["half"]**2)/2)
                        elif method == "MID_PLUS_FULL":
                            expected = math.sqrt((v["mid"]**2 + v["full"]**2)/2)
                        elif method == "TRAIN_PEM": expected = v["mse"]
                        else:
                            expected = N_RESIDUALS * math.log(v["mse"]) + \
                                       math.log(N_RESIDUALS) * PARAMETERS[law]
                        need(approximate(scores[law], expected),
                             f"Candidate score corrupted: {gen}/{arm}/{cfg}/{method}/{law}")
                    winner = min(LAWS, key=lambda law: (scores[law], LAWS.index(law)))
                    need(d["winner"] == winner, "Wrong argmin candidate")
                    n += 1
        for law in LAWS:
            w = row["fit_results_PRIVATE"][gen]["MATCHED5"]["wide"][law]
            o = row["fit_results_PRIVATE"][gen]["MATCHED5"]["outer_box"][law]
            gap = None if w is None or o is None else w["mse"] - o["mse"]
            diagnostic = row["nested_audit_PRIVATE"][gen][law]
            need((gap is None and diagnostic["gap"] is None) or
                 (gap is not None and approximate(gap,diagnostic["gap"])),
                 "Incorrect nested-domain gap")
            need(diagnostic["violation"] == (gap is not None and gap>GAP_TOL),
                 "Wrong nested-domain gap label")
    return n


def report_stats(rows, r):
    result = {}
    for gen in GENERATORS:
        result[gen]={}
        for arm in BUDGETS:
            result[gen][arm]={}
            for cfg in CONFIGS:
                result[gen][arm][cfg]={}
                for method in METHODS:
                    ds = [row["decisions"][gen][arm][cfg][method] for row in rows]
                    available=sum(bool(d["available"]) for d in ds)
                    correct=sum(bool(d["available"] and d["winner"]==row["true_law_PRIVATE"])
                                for row,d in zip(rows,ds))
                    src = r["stats"][gen][arm][cfg][method]
                    need(src["correct"]==correct and src["available"]==available and
                         src["total"]==len(rows),"Aggregate statistic inconsistent")
                    for mode in TRUTH:
                        group=[(row,d) for row,d in zip(rows,ds) if row["mode_PRIVATE"]==mode]
                        c=sum(bool(d["available"] and d["winner"]==row["true_law_PRIVATE"])
                              for row,d in group)
                        need(src["by_mode_PRIVATE"][mode]=={"n":len(group),"correct":c},
                             "By-mechanism statistic inconsistent")
                    result[gen][arm][cfg][method] = {"correct":correct,"available":available,
                                                      "total":len(rows)}
    return result


def primary_and_bootstrap(rows, r):
    d = {}; pair_counts={}
    for gen in GENERATORS:
        observations=[]; pairs=Counter()
        for row in rows:
            if row["mode_PRIVATE"]!="nonlinear" or row["condition_PRIVATE"]!="M2":
                continue
            de=row["decisions"][gen]["MATCHED5"]["WOO"]
            a=int(de["MID_PLUS_HALF"]["available"] and de["MID_PLUS_HALF"]["winner"]=="cubic")
            b=int(de["MID_PLUS_FULL"]["available"] and de["MID_PLUS_FULL"]["winner"]=="cubic")
            pairs[(a,b)]+=1
            observations.append((row["scene_id"], b-a))
        need(len(observations)==96,"Primary cohort size differs")
        test={"n":96,"half_correct":pairs[(1,0)]+pairs[(1,1)],
              "full_correct":pairs[(0,1)]+pairs[(1,1)],
              "net":sum(v for _,v in observations),
              "effect":sum(v for _,v in observations)/96}
        need(test==r["primary_contrasts"][gen],"Primary effect differs")
        counts={"both_correct":pairs[(1,1)],"corrected_by_full":pairs[(0,1)],
                "harmed_by_full":pairs[(1,0)],"both_wrong":pairs[(0,0)]}
        need(counts==r["primary_pair_status"][gen],"Paired status differs")
        pair_counts[gen]=counts
        d[gen]=observations
    dd=[(p[0],q[1]-p[1]) for p,q in zip(d["IDEAL_ODE"], d["SHIFTED_ODE"])]
    need(all(p[0]==q[0] for p,q in zip(d["IDEAL_ODE"],d["SHIFTED_ODE"])),
         "Physical-arm pairing mismatch")
    need(sum(x for _,x in dd)==r["primary_contrasts"]["SHIFT_MINUS_IDEAL_DD"]["net"],
         "Difference of effects differs")
    all_d={**d,"SHIFT_MINUS_IDEAL_DD":dd}
    bs=np.random.default_rng(BOOT_SEED).integers(0,N_SCENES,size=(N_BOOT,N_SCENES))
    cis={}
    for key,pairs in all_d.items():
        byscene=defaultdict(list)
        for sid,value in pairs: byscene[sid].append(value)
        need(set(byscene)==set(range(N_SCENES)) and
             all(len(byscene[sid])==4 for sid in range(N_SCENES)),
             "Bootstrap cluster constitution differs")
        values=np.asarray([np.mean(byscene[sid]) for sid in range(N_SCENES)])
        ci=np.quantile(values[bs].mean(axis=1),[.025,.975]).tolist()
        need(np.allclose(ci,r["bootstrap"]["CI95"][key],atol=1e-14),
             f"Frozen CI differs: {key}")
        cis[key]=ci
    return pair_counts,cis


def posthoc_descriptives(rows, r):
    """No hypothesis tuning; paired subgroup summaries only; exploratory."""
    gen_modes={}; totals={}
    for gen in GENERATORS:
        gen_modes[gen]={}
        for mode in TRUTH:
            sub=[row for row in rows if row["mode_PRIVATE"]==mode]
            cell={"n":len(sub)}
            for m in ("MID_PLUS_HALF","MID_PLUS_FULL","TRAIN_PEM","PROFILE_SIGMA_BIC"):
                cell[m]=sum(bool(row["decisions"][gen]["MATCHED5"]["WOO"][m]["available"]
                                 and row["decisions"][gen]["MATCHED5"]["WOO"][m]["winner"]
                                 ==row["true_law_PRIVATE"]) for row in sub)
            gen_modes[gen][mode]=cell
        totals[gen]=sum(gen_modes[gen][mode]["MID_PLUS_FULL"] for mode in TRUTH)
        need(totals[gen]==r["stats"][gen]["MATCHED5"]["WOO"]["MID_PLUS_FULL"]["correct"],
             "Cross-mode totals do not add up")
    shifts={mode:gen_modes["SHIFTED_ODE"][mode]["MID_PLUS_FULL"]-
           gen_modes["IDEAL_ODE"][mode]["MID_PLUS_FULL"] for mode in TRUTH}
    need(sum(shifts.values())==totals["SHIFTED_ODE"]-totals["IDEAL_ODE"],
         "Cross-arm case shift differs")
    nested={}
    nfev={}
    for gen in GENERATORS:
        nested[gen]={law:sum(row["nested_audit_PRIVATE"][gen][law]["violation"]
                             for row in rows) for law in LAWS}
        need({k:v for k,v in nested[gen].items() if v} ==
             r["nested_solver_violations"][gen],"Nested count inconsistent")
        nfev[gen]={}
        for domain in ("wide","outer_box"):
            org=sum(row["fit_audit_PRIVATE"][gen][domain][law]["original"]["nfev"]
                    for row in rows for law in LAWS)
            extra=sum(s["nfev"] for row in rows for law in LAWS
                      for s in row["fit_audit_PRIVATE"][gen][domain][law]["extra"])
            nfev[gen][domain]={"original":org,"extra":extra}
        need(nfev[gen]==r["solver_nfev"][gen],"Function evaluation totals differ")
        v=float(np.median([row["oracle_after_selection_PRIVATE"][gen]
                           ["oracle_law_rms_to_clean_B_mm"] for row in rows]))
        need(approximate(v,r["privileged_postdecision_diagnostic"][gen]
                         ["true_model_clean_B_rms_mm_median"]),"Oracle mismatch")
    # Exploratory, paired across physical arms, cluster-resampled. NOT a
    # registered confirmatory test and not a MuJoCo-to-ODE causal contrast.
    rng=np.random.default_rng(2029122703)
    idx=rng.integers(0,N_SCENES,size=(2000,N_SCENES))
    exploratory={}
    for mode in (*TRUTH,"all"):
        records=rows if mode=="all" else [x for x in rows if x["mode_PRIVATE"]==mode]
        grouped=defaultdict(list)
        for row in records:
            d=[]
            for gen in GENERATORS:
                c=row["decisions"][gen]["MATCHED5"]["WOO"]["MID_PLUS_FULL"]
                d.append(int(c["available"] and c["winner"]==row["true_law_PRIVATE"]))
            grouped[row["scene_id"]].append(d[1]-d[0])
        need(len(grouped)==24,"Missing exploratory scene")
        values=np.array([np.mean(grouped[sid]) for sid in range(N_SCENES)])
        ci=np.quantile(values[idx].mean(axis=1),[.025,.975]).tolist()
        exploratory[mode]={"n":len(records),"net_shift_minus_ideal":int(sum(sum(v) for v in grouped.values())),
                           "pp_shift_minus_ideal":100*sum(sum(v) for v in grouped.values())/len(records),
                           "scene_cluster_bootstrap_CI95_pp_posthoc":[100*v for v in ci]}
    return gen_modes,shifts,nested,nfev,exploratory


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--smoke",action="store_true")
    pa.add_argument("--input-dir",type=Path,default=HERE/"results"/"p4g1b")
    pa.add_argument("--source-dir",type=Path,default=HERE)
    pa.add_argument("--checkpoint-zip",type=Path,default=None)
    args=pa.parse_args()
    dat=args.input_dir
    report=load_json(dat/REPORT)
    rows=load_json(dat/AUDIT)
    manifest=load_json(dat/MANIFEST)
    smoke_report=load_json(dat/SMOKE_REPORT)
    print("="*74)
    print("P4-G1C — READ-ONLY CROSS-GENERATOR IDENTIFICATION CLOSURE")
    print("="*74)
    print("Smoke:",args.smoke,"| no MuJoCo | no fits | no E4")
    sources=verify_sources(report,manifest,smoke_report,args.source_dir,dat)
    if args.checkpoint_zip:
        need(args.checkpoint_zip.is_file(),"Checkpoint zip missing")
        zf=zipfile.ZipFile(args.checkpoint_zip)
    else:zf=None
    try:sample=verify_cases(rows,report,dat/"checkpoints",zf,args.smoke)
    finally:
        if zf is not None:zf.close()
    score_checks=sum(verify_row_scores(row) for row in sample)
    print("[PASS] SHA lineage, fixed cohort, checkpoint(s), candidate score parity")
    print("[PASS] Verified",score_checks,"available decision cells in subset")
    out=HERE/"results"/"p4g1c"
    if args.smoke:
        result={"protocol":"P4-G1C read-only smoke, no inferential claims",
                "n_rows_rechecked":len(sample),"source_files_checked":sources,
                "n_available_decisions_rechecked":score_checks,"E4_accessed":False,
                "new_fits":0,"new_simulations":0}
        output=out/"p4g1c_smoke_report.json"
        save_json(output,result)
        print("[PASS] Smoke checks only. No scientific conclusions.")
        print("Saved:",output)
        return
    stats=report_stats(rows,report)
    pairs,cis=primary_and_bootstrap(rows,report)
    modes,shifts,nested,nfev,exploratory=posthoc_descriptives(rows,report)
    decision={"p4g1b_numerical_reconstruction":"PASS",
              "p4g1b_primary_shifted_effect_confirmed_positive":"NOT ESTABLISHED (95% CI includes zero)",
              "p4g1b_primary_ideal_effect_confirmed_positive":"NOT ESTABLISHED (95% CI includes zero)",
              "generalization_claim":"conditional ODE structural transfer with re-fitting; not two independent engines or hardware",
              "solver_global_optimum":"NOT ESTABLISHED; deadzone nested violations present",
              "E4_sealed":True}
    out_report={"protocol":"P4-G1C frozen G1B consistency and limitations audit",
                "status":"PASS_WITH_SCIENTIFIC_LIMITATIONS",
                "n_cases":384,"n_generator_case_pairs":768,
                "n_base_scenes":24,"n_stats_cells":84,
                "source_files_verified":sources,
                "n_score_cells_rechecked":score_checks,
                "checkpoint_scene_count":24,
                "primary_contrasts":report["primary_contrasts"],
                "primary_paired_status_recomputed":pairs,
                "primary_scene_cluster_CI95_recomputed":cis,
                "matched5_WOO_mode_accuracy_counts_posthoc":modes,
                "full_ode_shift_minus_ideal_by_mode_posthoc":shifts,
                "matched5_WOO_mid_plus_full_total":{gen:stats[gen]["MATCHED5"]["WOO"]["MID_PLUS_FULL"] for gen in GENERATORS},
                "nested_violations":nested,"function_evaluations":nfev,
                "cross_arm_shift_exploratory":exploratory,
                "scientific_interpretation_gates":decision,
                "limitations":report["limitations"],
                "no_new_fits":True,"no_new_simulations":True,"E4_accessed":False}
    audit_summary={"rows":[{"key_PRIVATE":row["key_PRIVATE"],
                "truth_PRIVATE":row["true_law_PRIVATE"],
                "WOO_MATCHED5":{gen:{m:row["decisions"][gen]["MATCHED5"]["WOO"][m]["winner"]
                                   for m in ("MID_PLUS_HALF","MID_PLUS_FULL","TRAIN_PEM","PROFILE_SIGMA_BIC")}
                                for gen in GENERATORS},
                "nested_gaps_PRIVATE":{gen:{law:row["nested_audit_PRIVATE"][gen][law]["gap"]
                                       for law in LAWS} for gen in GENERATORS}}
                              for row in rows],
                   "privacy_note":"Contains truth labels; retrospective audit only; no estimator may use these data"}
    rp=out/"p4g1c_closure_report.json"; ap=out/"p4g1c_closure_audit_PRIVATE.json"
    save_json(rp,out_report);save_json(ap,audit_summary)
    print("[PASS] 84 aggregate cells, candidate scores, 24 checkpoints, 2,000 cluster resamples")
    print("[PRIMARY]",report["primary_contrasts"])
    print("[POSTHOC MODE SHIFT]",shifts)
    print("[NESTED]",nested)
    print("[SCIENTIFIC STATUS] ODE dose effect CI crosses zero; model shift is mechanism dependent")
    print("Saved:",rp,"\nSaved:",ap)
    print("P4-G1C COMPLETE")


if __name__=="__main__":
    main()
