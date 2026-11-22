"""#104 P4-G2D: independent read-only closure of P4-G2C fresh confirmation.

From TwinWorld_P0 root:
  .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\104_p4g2d_fresh_confirmation_closure_audit.py --smoke
  .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\104_p4g2d_fresh_confirmation_closure_audit.py

No ODE, MuJoCo, optimizer, source/target refitting, or E4 access. Never
modifies any P98-P103 artifacts. Recompute paired counts, 24-cluster
bootstrap, every stored winner/relative margin, nested-domain inequalities,
function evaluations, and 24 scene checkpoints.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE/'results'
SRC = HERE/'103_p4g2c_fresh_frozen_vs_refit_confirmation.py'
R = BASE/'p4g2c'/'p4g2c_fresh_report.json'
A = BASE/'p4g2c'/'p4g2c_fresh_audit_PRIVATE.json'
M = BASE/'p4g2c'/'p4g2c_source_manifest.json'
CP = BASE/'p4g2c'/'checkpoints'
OUT = BASE/'p4g2d'
SOURCES = {
 'P98': HERE/'98_p4g1a_independent_physics_generator_feasibility.py',
 'P99': HERE/'99_p4g1b_independent_generator_identification.py',
 'P102': HERE/'102_p4g2b_frozen_vs_refit_competition_audit.py',
 'R99': BASE/'p4g1b'/'p4g1b_fresh_report.json',
 'R101': BASE/'p4g2a'/'p4g2a_diagnostic_report.json',
 'R102': BASE/'p4g2b'/'p4g2b_closure_report.json',
 'M102': BASE/'p4g2b'/'p4g2b_source_manifest.json',
 'SELF': SRC,
}
ARMS=('PAIRED_IDEAL_FIXED','SHIFTED_REFIT')
PHYS=('IDEAL_ODE','SHIFTED_ODE')
LAWS=('elastic','deadzone','cubic')
METHODS=('MID_PLUS_HALF','MID_PLUS_FULL')
BOOT_SEED=2030011702
N_BOOT=2000
N_SCENES=24

def must(ok,message):
    if not ok: raise RuntimeError('[FAIL] '+message)

def read(path):
    must(path.is_file(),f'Missing frozen input: {path}')
    with path.open(encoding='utf-8') as f: return json.load(f)

def filehash(path):
    must(path.is_file(),f'Missing frozen source: {path}')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def objhash(x):
    s=json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def near(a,b,tol=1e-9):
    return math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol)

def output(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/name; tmp=p.with_suffix(p.suffix+'.writing')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,p)
    print('Saved:',p)

def verify_lineage(report,manifest):
    must(report['smoke'] is False and report['n_cases']==384
        and report['n_base_scenes']==24 and report['n_generator_case_pairs']==768
        and report['n_fit_units']==4608,'Original experiment size')
    must(report['E4_accessed'] is False and manifest['E4_accessed'] is False,
         'E4 seal')
    must(report['truth_used_for_fit_or_selection'] is False and
         report['heldout_used_for_fit'] is False and
         report['adaptive_tuning'] is False,'Leakage guard')
    must(report['protocol_fixed_before_fresh_outcome'] is True and
         report['primary_is_privileged_paired_source_ablation'] is True and
         manifest['paired_source_privileged'] is True,'Privileged-source disclosure')
    must(report['fit_budget']['n_training_B_residuals']==204
         and manifest['n_training_scalar_residuals']==204,'204-sample training convention')
    must(report['scene_seed']==manifest['scene_seed']==20300117
         and report['noise_seed']==manifest['noise_seed']==2030011701,
         'Immutable seeds')
    must(report['bootstrap']=={'unit':'base_scene','n_clusters':24,
          'resamples':2000,'seed':BOOT_SEED,
          'CI_method':'paired scene-cluster percentile; two-sided'},
         'Bootstrap protocol changed')
    must(report['sources']==manifest['source_sha256'],'Report and manifest sources mismatch')
    for name,path in SOURCES.items():
        must(filehash(path)==report['sources'][name],f'{name} source SHA mismatch')
    return True

def check_rows(rows,report,manifest,smoke):
    n=16 if smoke else 384
    must(isinstance(rows,list) and len(rows)==n,'Case count wrong')
    keys=set(); stats=Counter(); score_count=0; fit_count=0
    fingerprints=report['scene_fingerprints_PRIVATE']
    must(fingerprints==manifest['scene_fingerprints_PRIVATE'] and
         len(fingerprints)==48 and len(set(fingerprints))==48,
         'Base-scene fingerprint invariant')
    for q in rows:
        key=tuple(q['key_PRIVATE']); sid=q['scene_id']; truth=q['true_law_PRIVATE']
        must(len(key)==5 and key[0]==sid and key not in keys,'Unique case key')
        keys.add(key); stats[(sid,key[1],truth)]+=1
        must(truth=={'elastic':'elastic','deadzone':'deadzone','nonlinear':'cubic'}[key[2]],
             'Truth-mode mapping')
        must(q['condition_PRIVATE']==key[4],'M0/M2 condition mapping')
        must(q['physical_scene_sha256_PRIVATE'], 'Scene SHA absent')
        must(set(q['clean_fingerprints_PRIVATE'])==set(PHYS),'Two generator clean fingerprints')
        for ga in PHYS:
            vals=q['clean_fingerprints_PRIVATE'][ga]
            must(set(vals)=={'e1','e2','weak','strong','mid','HALF','FULL'},
                 'Seven clean probes per generator')
            must(all(len(v)==64 for v in vals.values()),'Probe SHA256 format')
        for arm in ARMS:
            for meth in METHODS:
                d=q['decisions'][arm][meth]
                must(isinstance(d['available'],bool),'Availability type')
                diag=q['post_selection_diagnostics_PRIVATE'][arm][meth]
                if d['available']:
                    scores=d['scores_mm_PRIVATE']
                    must(set(scores)==set(LAWS) and all(math.isfinite(float(s)) and float(s)>=0 for s in scores.values()),'Finite candidate RMS')
                    winner=min(LAWS,key=lambda law:(scores[law],LAWS.index(law)))
                    must(d['winner']==winner,'Stored winner disagrees with scores')
                    good=scores[truth]**2
                    wrong=min(scores[law]**2 for law in LAWS if law!=truth)
                    denom=good+wrong
                    must(bool(diag['correct'])==(winner==truth),'Truth diagnostic inconsistency')
                    must(near(diag['true_law_rms_mm'],scores[truth]),'True RMS mismatch')
                    must(near(diag['squared_margin_mm2'],wrong-good,1e-8), 'Squared margin mismatch')
                    must(near(diag['relative_margin'],(wrong-good)/denom if denom else 0), 'Relative margin mismatch')
                    score_count+=len(LAWS)
                else:
                    must(d['winner'] is None and not diag['correct'],'Abstention handling')
        # Check independent target-refit scoring against stored target fit holdouts.
        for law in LAWS:
            dom='wide' if law=='elastic' else 'outer_box'
            fit=q['candidate_fits_PRIVATE']['SHIFTED_ODE']['MATCHED5'][dom][law]
            if fit is not None:
                for meth,other in [('MID_PLUS_HALF','half'),('MID_PLUS_FULL','full')]:
                    target=q['decisions']['SHIFTED_REFIT'][meth]
                    if target['available']:
                        predict=math.hypot(fit['mid'],fit[other])/math.sqrt(2)
                        must(near(predict,target['scores_mm_PRIVATE'][law],1e-7),
                             'SHIFTED holdout scoring not consistent with selected fit')
        for ga in PHYS:
            for law in LAWS:
                w=q['candidate_fits_PRIVATE'][ga]['MATCHED5']['wide'][law]
                o=q['candidate_fits_PRIVATE'][ga]['MATCHED5']['outer_box'][law]
                gap=None if w is None or o is None else float(w['mse']-o['mse'])
                entry=q['nested_PRIVATE'][ga][law]
                must((gap is None and entry['gap'] is None) or
                    (gap is not None and near(gap,entry['gap'],1e-8)),
                    'Nested MSE gap formula')
                must(entry['violation']==bool(gap is not None and gap>1e-6),
                     'Nested violation threshold')
                for dom in ('wide','outer_box'):
                    logs=q['fit_logs_PRIVATE'][ga][dom][law]
                    must(len(logs['extra'])==3,'Three predeclared extra starts')
                    must(int(logs['original']['nfev'])>=0 and all(int(x['nfev'])>=0 for x in logs['extra']),
                         'Negative function eval counts')
                    fit_count+=1
    return {'cases':n,'candidate_score_cells_available':score_count,'fit_units':fit_count,'case_sha256':objhash(rows)}

def verify_checkpoints(rows,report,manifest):
    for sid in range(N_SCENES):
        path=CP/f'scene_{sid:02d}.json'
        pack=read(path)
        part=rows[sid*16:(sid+1)*16]
        must(pack['scene_id']==sid and pack['source_sha256']==report['sources'],
             f'Checkpoint {sid}: provenance mismatch')
        must(pack['base_fingerprints']==manifest['scene_fingerprints_PRIVATE'][sid*2:sid*2+2],
             f'Checkpoint {sid}: scene fingerprint mismatch')
        must(pack['rows_sha256']==objhash(pack['rows']) and pack['rows']==part,
             f'Checkpoint {sid}: rows or CRC mismatch')
        must(len(part)==16 and all(r['scene_id']==sid for r in part),
             f'Checkpoint {sid}: incomplete cluster')

def check_aggregate(rows,report):
    counts=report['aggregate_classification']
    output_counts={}
    for arm in ARMS:
        output_counts[arm]={}
        for meth in METHODS:
            ds=[r['decisions'][arm][meth] for r in rows]
            observed=sum(d['available'] and d['winner']==r['true_law_PRIVATE'] for r,d in zip(rows,ds))
            avail=sum(int(d['available']) for d in ds)
            expected=counts[arm][meth]
            must((observed,avail,len(rows))==(expected['correct'],expected['available'],expected['n']),
                 'Aggregate classification discrepancy')
            for law in LAWS:
                subset=[(r,d) for r,d in zip(rows,ds) if r['true_law_PRIVATE']==law]
                nlaw=sum(d['available'] and d['winner']==law for r,d in subset)
                must(expected['by_law'][law]=={'n':len(subset),'correct':nlaw},
                     'Per-mechanism classification discrepancy')
            output_counts[arm][meth]={'correct':observed,'available':avail,'total':len(rows)}
    for ga in PHYS:
        viol=Counter(law for r in rows for law in LAWS if r['nested_PRIVATE'][ga][law]['violation'])
        must(dict(viol)==report['nested_violations'][ga],f'{ga} nested counts')
        for dom in ('wide','outer_box'):
            old=sum(int(r['fit_logs_PRIVATE'][ga][dom][law]['original']['nfev']) for r in rows for law in LAWS)
            extra=sum(int(x['nfev']) for r in rows for law in LAWS
                      for x in r['fit_logs_PRIVATE'][ga][dom][law]['extra'])
            must({'original':old,'extra':extra}==report['function_evaluations'][ga][dom],
                 'Function evaluation count discrepancy')
    return output_counts

def independent_pair(rows,pred,arm_a,meth_a,arm_b,meth_b):
    values=defaultdict(list); trans=Counter()
    for r in rows:
        if not pred(r): continue
        tru=r['true_law_PRIVATE']; ds=r['decisions']
        da=ds[arm_a][meth_a];db=ds[arm_b][meth_b]
        ca=int(da['available'] and da['winner']==tru)
        cb=int(db['available'] and db['winner']==tru)
        trans[(ca,cb)]+=1;values[r['scene_id']].append(ca-cb)
    must(set(values)==set(range(24)) and len(set(map(len,values.values())))==1,
         'Missing scene-cluster for paired effect')
    scene=np.array([np.mean(values[s]) for s in range(24)])
    return dict(n=sum(trans.values()),a_correct=sum(a*c for (a,b),c in trans.items()),
          b_correct=sum(b*c for (a,b),c in trans.items()),
          both_correct=trans[(1,1)],a_only=trans[(1,0)],b_only=trans[(0,1)],
          both_wrong=trans[(0,0)],net=trans[(1,0)]-trans[(0,1)],
          effect=float(scene.mean())),scene

def verify_pairs(rows,report):
    is_nm2=lambda r:r['true_law_PRIVATE']=='cubic' and r['condition_PRIVATE']=='M2'
    spec={
      'PRIMARY_SHIFTED_FIXED_MINUS_REFIT_ALL_FULL':(lambda r:True,'PAIRED_IDEAL_FIXED','MID_PLUS_FULL','SHIFTED_REFIT','MID_PLUS_FULL'),
      'SECONDARY_ELASTIC_FIXED_MINUS_REFIT_FULL':(lambda r:r['true_law_PRIVATE']=='elastic','PAIRED_IDEAL_FIXED','MID_PLUS_FULL','SHIFTED_REFIT','MID_PLUS_FULL'),
      'SECONDARY_NM2_FIXED_FULL_MINUS_HALF':(is_nm2,'PAIRED_IDEAL_FIXED','MID_PLUS_FULL','PAIRED_IDEAL_FIXED','MID_PLUS_HALF'),
      'SECONDARY_NM2_REFIT_FULL_MINUS_HALF':(is_nm2,'SHIFTED_REFIT','MID_PLUS_FULL','SHIFTED_REFIT','MID_PLUS_HALF'),
    }
    draw=np.random.default_rng(BOOT_SEED).integers(0,24,size=(N_BOOT,24))
    audited={}; pieces={}
    for key,config in spec.items():
        obj,scene=independent_pair(rows,*config);pieces[key]=scene
        ci=np.quantile(scene[draw].mean(axis=1),[.025,.975]).tolist()
        expected=report['primary_and_secondary_paired_contrasts'][key]
        for k,v in obj.items(): must(near(v,expected[k]) if k=='effect' else v==expected[k],f'{key} {k}')
        must(all(near(x,y,1e-12) for x,y in zip(ci,expected['scene_cluster_CI95'])),
             f'{key} 24-scene bootstrap mismatch')
        audited[key]={**obj,'scene_cluster_CI95':ci}
    dd=pieces['SECONDARY_NM2_FIXED_FULL_MINUS_HALF']-pieces['SECONDARY_NM2_REFIT_FULL_MINUS_HALF']
    ci=np.quantile(dd[draw].mean(axis=1),[.025,.975]).tolist()
    o={'n':96,'net':int(round(dd.sum()*4)), 'effect':float(dd.mean()), 'scene_cluster_CI95':ci}
    key='SECONDARY_NM2_FIXED_MINUS_REFIT_DOSE_DD'; ref=report['primary_and_secondary_paired_contrasts'][key]
    must(o['n']==ref['n'] and o['net']==ref['net'] and near(o['effect'],ref['effect']) and
        all(near(x,y,1e-12) for x,y in zip(ci,ref['scene_cluster_CI95'])),
        'Nonlinear-M2 paired difference-of-effects changed')
    audited[key]=o
    return audited

def verify_scene_disjoint(manifest):
    current=set(manifest['scene_fingerprints_PRIVATE'])
    for rpath,key in ((BASE/'p4g1b'/'p4g1b_fresh_report.json','scene_fingerprints_PRIVATE'),
                      (BASE/'p4g1a'/'p4g1a_feasibility_report.json','scene_fingerprints_PRIVATE')):
        hist=read(rpath)
        if key in hist:
            must(not current.intersection(hist[key]),f'Historical scene overlap: {rpath}')
    # Original P103 checks additional smoke and pilot seeds before any outcome.
    return True

def run(smoke):
    report,rows,manifest=read(R),read(A),read(M)
    verify_lineage(report,manifest)
    if smoke:
        checked=check_rows(rows[:16],report,manifest,True)
        must(len(read(CP/'scene_00.json')['rows'])==16,'Smoke checkpoint interface')
        result={'protocol':'P4-G2D smoke read-only audit','checked_cases':16,
                'score_cells_available':checked['candidate_score_cells_available'],
                'no_fits':True,'no_simulations':True,'E4_accessed':False,
                'scientific_effect_estimate':False}
        print('[PASS] Immutable lineage, 16 scored cases; smoke-only interface')
        output('p4g2d_smoke_report.json',result)
        return
    checked=check_rows(rows,report,manifest,False)
    must(checked['fit_units']==4608 and checked['candidate_score_cells_available']==4608,
         'All 4608 fit units and 4608 candidate scoring cells present')
    verify_checkpoints(rows,report,manifest)
    verify_scene_disjoint(manifest)
    classification=check_aggregate(rows,report)
    pair=verify_pairs(rows,report)
    result={'protocol':'P4-G2D independent frozen confirmation closure',
      'status':'PASS_WITH_PRIVILEGED_SOURCE_AND_NUMERICAL_LIMITATIONS',
      'source_sha256':filehash(SRC),
      'input_sha256':{k:filehash(p) for k,p in [('R103',R),('A103',A),('M103',M)]},
      'n_base_scenes':24,'n_cases':384,'n_generator_case_pairs':768,
      'n_fit_units_rechecked':checked['fit_units'],
      'candidate_score_cells_rechecked':checked['candidate_score_cells_available'],
      'checkpoint_scene_count':24,'classification':classification,
      'paired_effects':pair,
      'nested_violations':report['nested_violations'],
      'function_evaluations':report['function_evaluations'],
      'interpretation':{
       'main_effect':'Confirmed on new scenes for paired privileged source-versus-refit contrast',
       'deployable_zero_shot_superiority':'NOT_ESTABLISHED',
       'same_scene_ideal_source_privilege':'YES',
       'dose_effect_global':'CONDITIONAL; Nonlinear-M2 fixed negative and refit positive',
       'two_independent_engines':'NO; one RK4 implementation two physical arms',
       'global_SLSQP_optimum':'NOT_ESTABLISHED',
       'E4_sealed':True},
      'no_new_fits':True,'no_new_simulations':True,'E4_accessed':False}
    private={'case_sha256':checked['case_sha256'],
             'scene_id_coverage':{str(sid):sum(r['scene_id']==sid for r in rows) for sid in range(24)},
             'base_scene_fingerprints_PRIVATE':manifest['scene_fingerprints_PRIVATE'],
             'paired_comparison_recomputed_PRIVATE':pair,
             'note':'No fresh outcomes or revised hyperparameters; strictly retrospective closure'}
    print('[PASS] All 384 aligned cases, 4608 fit/scoring cells, 24 exact checkpoints')
    print('[PASS] Candidate winners, target fit scores, nested gaps, nfev, 2000 cluster draws')
    print('[PRIMARY]',pair['PRIMARY_SHIFTED_FIXED_MINUS_REFIT_ALL_FULL'])
    print('[SECONDARY ELASTIC]',pair['SECONDARY_ELASTIC_FIXED_MINUS_REFIT_FULL'])
    print('[NM2 FIXED/REFIT]',pair['SECONDARY_NM2_FIXED_FULL_MINUS_HALF']['effect'],
          pair['SECONDARY_NM2_REFIT_FULL_MINUS_HALF']['effect'])
    print('[CAUTION] Paired same-scene IDEAL source is privileged; E4 remains sealed')
    output('p4g2d_closure_report.json',result)
    output('p4g2d_closure_audit_PRIVATE.json',private)
    print('P4-G2D COMPLETE')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    print('='*76)
    print('P4-G2D — READ-ONLY FRESH CONFIRMATION CLOSURE')
    print('='*76)
    print('Smoke:',args.smoke,'| no MuJoCo | no ODE | no SLSQP | no E4')
    run(args.smoke)
