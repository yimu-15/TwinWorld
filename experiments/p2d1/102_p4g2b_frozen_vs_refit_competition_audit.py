"""Experiment #102 / P4-G2B — post-hoc read-only selection-inversion audit.

Purpose: independently reconstruct #101 fixed IDEAL parameter counterfactual
and #99 SHIFTED re-fit winner/score comparisons on the SAME 384 physical cases.

Strictly retrospective. Reads existing JSON and checkpoint files; no MuJoCo,
ODE integration, SLSQP fitting, E4, new parameter selection or tuning.

Run from project root:
 .\\Giant\\Scripts\\python.exe experiments\\p2d1\\102_p4g2b_frozen_vs_refit_competition_audit.py --smoke
 .\\Giant\\Scripts\\python.exe experiments\\p2d1\\102_p4g2b_frozen_vs_refit_competition_audit.py

Offline audit of uploaded flat files:
 python 102_p4g2b_frozen_vs_refit_competition_audit.py --input-dir /mnt/data --checkpoints-zip '/mnt/data/checkpoints(3).zip'
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import statistics
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE / 'results'
OUT=ROOT / 'p4g2b'
METHODS=('MID_PLUS_HALF','MID_PLUS_FULL')
LAWS=('elastic','deadzone','cubic')
ARMS=('IDEAL_ODE','VISCOUS_ONLY','DRY_ONLY','LAG_ONLY','SHIFTED_ODE')
SEED=2029122901
RESAMPLES=2000


def need(cond,msg):
    if not cond: raise RuntimeError(msg)

def read(path):
    p=Path(path)
    need(p.is_file(),f'Missing required input: {p}')
    with p.open(encoding='utf-8') as f: return json.load(f)

def hashfile(p):
    p=Path(p); need(p.is_file(),f'Absent provenance file: {p}')
    return sha256(p.read_bytes()).hexdigest()

def canonical_hash(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def paths(input_dir):
    if input_dir:
        p=Path(input_dir).resolve()
        return {k:p / fn for k,fn in {
            'R99':'p4g1b_fresh_report.json',
            'A99':'p4g1b_fresh_audit_PRIVATE.json',
            'R100':'p4g1c_closure_report.json',
            'R101':'p4g2a_diagnostic_report.json',
            'A101':'p4g2a_diagnostic_audit_PRIVATE.json',
            'M101':'p4g2a_source_manifest.json',
        }.items()}
    return {'R99':ROOT/'p4g1b'/'p4g1b_fresh_report.json',
            'A99':ROOT/'p4g1b'/'p4g1b_fresh_audit_PRIVATE.json',
            'R100':ROOT/'p4g1c'/'p4g1c_closure_report.json',
            'R101':ROOT/'p4g2a'/'p4g2a_diagnostic_report.json',
            'A101':ROOT/'p4g2a'/'p4g2a_diagnostic_audit_PRIVATE.json',
            'M101':ROOT/'p4g2a'/'p4g2a_source_manifest.json'}


def verify_checkpoints(rows,source,zip_path,skip):
    if skip: return 'NOT_CHECKED_SMOKE'
    if zip_path:
        z=zipfile.ZipFile(zip_path)
        need(z.testzip() is None,'Corrupt checkpoint archive')
        names=[n for n in z.namelist() if n.endswith('.json')]
        need(len(names)==24,'Checkpoint ZIP does not contain 24 scene JSON files')
        def obtain(sid):
            suffix=f'scene_{sid:02d}.json'
            matches=[n for n in names if n.endswith('/'+suffix) or n==suffix]
            need(len(matches)==1,f'Ambiguous checkpoint name {suffix}')
            return json.loads(z.read(matches[0]))
    else:
        folder=ROOT/'p4g2a'/'checkpoints'
        need(folder.is_dir(),'P101 checkpoints directory missing; use --checkpoints-zip when auditing uploads')
        def obtain(sid): return read(folder / f'scene_{sid:02d}.json')
    for sid in range(24):
        checkpoint=obtain(sid)
        slice_rows=rows[sid*16:(sid+1)*16]
        need(checkpoint['scene_id']==sid,f'Checkpoint scene_id mismatch: {sid}')
        need(checkpoint['rows_sha256']==canonical_hash(checkpoint['rows']),f'Checkpoint row hash mismatch: {sid}')
        need(checkpoint['source_sha256']==source,f'Checkpoint source mismatch: {sid}')
        need(checkpoint['input_case_keys']==[x['key_PRIVATE'] for x in slice_rows],f'Checkpoint keys mismatch: {sid}')
        need(checkpoint['rows']==slice_rows,f'Checkpoint case values mismatch: {sid}')
    return 'PASS_24_EXACT'


def squared_scores(row,old,arm,method,tag):
    if tag=='fixed':
        what='HALF' if method=='MID_PLUS_HALF' else 'FULL'
        return {law:(row['arms'][arm]['scores_rms_mm_PRIVATE'][law]['mid']**2+
                      row['arms'][arm]['scores_rms_mm_PRIVATE'][law][what]**2)/2
                for law in LAWS}
    need(tag=='refit' and arm=='SHIFTED_ODE','Invalid comparator')
    data=old['decisions']['SHIFTED_ODE']['MATCHED5']['WOO'][method]
    need(data['available'] is True,'P99 refit abstained; cannot assume coverage')
    return {law:data['scores_mm_PRIVATE'][law]**2 for law in LAWS}


def assess(scores,truth):
    vals=[(scores[l],i,l) for i,l in enumerate(LAWS)]
    need(all(math.isfinite(z[0]) and z[0]>=0 for z in vals),'Invalid square RMS')
    selected=min(vals)[2]
    wrong=min(scores[l] for l in LAWS if l!=truth)
    right=scores[truth]
    den=wrong+right
    need(den>0,'Cannot normalize all-zero losses')
    return {'winner':selected,'correct':selected==truth,'true_law_rms_mm':math.sqrt(right),
            'best_wrong_rms_mm':math.sqrt(wrong),'squared_margin_mm2':wrong-right,
            'relative_margin':(wrong-right)/den,
            'best_wrong_law':min((scores[l],i,l) for i,l in enumerate(LAWS) if l!=truth)[2]}


def cluster_ci(rows,fn,seed):
    scene_scores=[]
    for sid in range(24):
        subs=[x for x in rows if x['scene_id']==sid and fn(x) is not None]
        need(len(subs)>0,'Missing scene subgroup')
        scene_scores.append([fn(x) for x in subs])
    # All defined endpoints have an identical number of conditions in every scene.
    counts={len(s) for s in scene_scores}
    need(len(counts)==1,f'Unequal scene contributions: {counts}')
    rng=random.Random(seed)
    draws=[]
    for _ in range(RESAMPLES):
        picks=[rng.randrange(24) for _ in range(24)]
        vals=[v for j in picks for v in scene_scores[j]]
        draws.append(sum(vals)/len(vals))
    draws.sort()
    def perc(p):
        x=(len(draws)-1)*p
        lo=int(x);hi=min(lo+1,len(draws)-1)
        return draws[lo]*(hi-x)+draws[hi]*(x-lo)
    return [perc(.025),perc(.975)]


def transition(rows,a,b):
    out={'both_correct':0,'a_only_correct':0,'b_only_correct':0,'both_wrong':0}
    for q in rows:
        x=q[a]['correct'];y=q[b]['correct']
        out['both_correct' if x and y else 'a_only_correct' if x else 'b_only_correct' if y else 'both_wrong']+=1
    out['n']=len(rows);out['net_a_minus_b']=out['a_only_correct']-out['b_only_correct']
    return out


def mean_median(values):
    return {'mean':sum(values)/len(values),'median':statistics.median(values)}


def run(args):
    files=paths(args.input_dir)
    R99,A99,R100,R101,A101,M101=(read(files[x]) for x in ('R99','A99','R100','R101','A101','M101'))
    old=A99;data=A101['rows_PRIVATE']
    need(R99['smoke'] is False and R99['n_cases_per_generator']==384 and R99['n_scenes']==24,
         'P99 formal source cohort unavailable')
    need(R100['status']=='PASS_WITH_SCIENTIFIC_LIMITATIONS' and R100['E4_accessed'] is False,
         'P100 closure status changed')
    need(R101['smoke'] is False and R101['n_cases']==384 and R101['n_base_scenes']==24
         and not R101['E4_accessed'] and not R101['refit_performed'] and
         R101['is_posthoc_exploratory'],'P101 scientific protocol changed')
    need(len(data)==len(old)==384,'384 aligned cases required')
    need(R101['sources']==M101['source_sha256'],'P101 manifest/report mismatch')
    for key,fn in [('P99','99_p4g1b_independent_generator_identification.py'),
                   ('P101','101_p4g2a_frozen_parameter_physics_factor_diagnostic.py')]:
        expected=R101['sources']['SELF' if key=='P101' else key]
        need(hashfile(HERE/fn)==expected,f'{key} source SHA mismatch')
    hashes={name:hashfile(path) for name,path in files.items()}
    # P101 directly incorporates exactly the frozen R99/A99/R100 inputs.
    for key in ('R99','A99','R100'):
        need(hashes[key]==R101['sources'][key],f'{key} frozen JSON SHA mismatch')
    keys=[tuple(q['key_PRIVATE']) for q in data]
    need(len(set(keys))==384,'Case keys duplicated')
    for sid in range(24):
        need(all(q['scene_id']==sid for q in data[sid*16:(sid+1)*16]),
             f'Unexpected case order in scene {sid}')
    cp_status=verify_checkpoints(data,R101['sources'],args.checkpoints_zip,args.smoke)
    audit=[]
    cohort=range(16) if args.smoke else range(384)
    for i in cohort:
        row,older=data[i],old[i]
        truth=row['true_law_PRIVATE'];key=row['key_PRIVATE']
        need(key==older['key_PRIVATE'] and truth==older['true_law_PRIVATE'],f'P99/P101 key mismatch {i}')
        need(truth in LAWS,'Unexpected physical mechanism')
        for law in LAWS:
            fit=older['fit_results_PRIVATE']['IDEAL_ODE']['MATCHED5']['wide' if law=='elastic' else 'outer_box'][law]
            need(fit and row['frozen_ideal_fit_origin_PRIVATE'][law]==fit['origin'],
                 'Frozen IDEAL fit provenance differs')
        comparators={}
        for arm in ARMS:
            comparators[arm]={}
            for method in METHODS:
                trial=assess(squared_scores(row,older,arm,method,'fixed'),truth)
                need(trial['winner']==row['arms'][arm]['winner'][method],
                     f'P101 decision recomputation mismatch at {key}/{arm}/{method}')
                if arm=='IDEAL_ODE':
                    source=older['decisions']['IDEAL_ODE']['MATCHED5']['WOO'][method]
                    need(source['available'] is True and source['winner']==trial['winner'],
                         'P99 IDEAL decision parity failed')
                    now=squared_scores(row,older,arm,method,'fixed')
                    for law in LAWS:
                        expected=source['scores_mm_PRIVATE'][law]**2
                        need(math.isclose(now[law],expected,rel_tol=1e-8,abs_tol=1e-6),
                             f'P99 IDEAL score parity failed {key}/{law}')
                comparators[arm][method]=trial
        comparators['SHIFTED_REFIT']={}
        for method in METHODS:
            trial=assess(squared_scores(row,older,'SHIFTED_ODE',method,'refit'),truth)
            need(row['P99_shift_refit_winner'][method]==trial['winner'],
                 'P99 SHIFTED refit winner parity failed')
            comparators['SHIFTED_REFIT'][method]=trial
        audit.append({'key_PRIVATE':key,'scene_id':row['scene_id'],'truth_PRIVATE':truth,
                      'mode_PRIVATE':row['mode_PRIVATE'],'evaluated':comparators})
    print(f'[PASS] {len(audit)} aligned cases, scored all five frozen arms and SHIFTED refit')
    if args.smoke:
        report={'protocol':'P4-G2B retrospective selection inversion audit',
                'smoke':True,'n_rows_checked':len(audit),'checkpoints':cp_status,
                'no_fits':True,'no_simulations':True,'E4_accessed':False}
        return report,{'rows_PRIVATE':audit}
    stats={}
    for method in METHODS:
        stats[method]={}
        for comp in list(ARMS)+['SHIFTED_REFIT']:
            group=audit
            vals=[q['evaluated'][comp][method] for q in group]
            stat={'n':len(vals),'correct':sum(q['correct'] for q in vals),
                  'by_truth':{law:{'n':sum(q['truth_PRIVATE']==law for q in group),
                       'correct':sum(q['truth_PRIVATE']==law and q['evaluated'][comp][method]['correct']
                                     for q in group)} for law in LAWS},
                  'winners_by_truth':{law:dict(Counter(q['evaluated'][comp][method]['winner']
                      for q in group if q['truth_PRIVATE']==law)) for law in LAWS},
                  'relative_margin':mean_median([q['relative_margin'] for q in vals]),
                  'squared_margin_mm2':mean_median([q['squared_margin_mm2'] for q in vals])}
            stats[method][comp]=stat
            if comp in ARMS:
                expected=R101['summary']['classification_frozen_IDEAL_parameters'][comp][method]
                need(stat['correct']==expected['correct'],'P101 aggregate score mismatch')
                for law in LAWS:
                    need(stat['by_truth'][law]==expected['by_law'][law],
                         f'P101 law score mismatch: {method}/{comp}/{law}')
        if method in R101['summary']['historical_P99_SHIFTED_refitted_classification']:
            exp=R101['summary']['historical_P99_SHIFTED_refitted_classification'][method]
            need(stats[method]['SHIFTED_REFIT']['correct']==exp['correct'],
                 'P99 refit score mismatch')
    transitions={}
    for method in METHODS:
        transitions[method]={}
        for law in ('ALL',)+LAWS:
            group=[q for q in audit if law=='ALL' or q['truth_PRIVATE']==law]
            baseA=[{'fixed':q['evaluated']['SHIFTED_ODE'][method],
                    'refit':q['evaluated']['SHIFTED_REFIT'][method]} for q in group]
            tr=transition(baseA,'fixed','refit')
            fr=[q['fixed']['true_law_rms_mm'] for q in baseA]
            rr=[q['refit']['true_law_rms_mm'] for q in baseA]
            mr=[q['fixed']['relative_margin'] for q in baseA]
            ms=[q['refit']['relative_margin'] for q in baseA]
            tr['fixed_true_model_rms_mm']=mean_median(fr)
            tr['refit_true_model_rms_mm']=mean_median(rr)
            tr['fixed_relative_margin']=mean_median(mr)
            tr['refit_relative_margin']=mean_median(ms)
            tr['accuracy_effect_fixed_minus_refit']=tr['net_a_minus_b']/len(group)
            transitions[method][law]=tr
    # Post-hoc scene-cluster resampling; scene is the only independent unit.
    ci={}
    for method in METHODS:
        ci[method]={}
        for law in ('ALL',)+LAWS:
            fn=lambda q,law=law,method=method: (None if law!='ALL' and q['truth_PRIVATE']!=law else
                    int(q['evaluated']['SHIFTED_ODE'][method]['correct'])-
                    int(q['evaluated']['SHIFTED_REFIT'][method]['correct']))
            ci[method][law]=cluster_ci(audit,fn,SEED+(0 if method=='MID_PLUS_FULL' else 100)+
                                       {'ALL':0,'elastic':1,'deadzone':2,'cubic':3}[law])
    # Original primary endpoint is NOT these post-hoc regime comparisons.
    dose={}
    for comp in ('IDEAL_ODE','VISCOUS_ONLY','DRY_ONLY','LAG_ONLY','SHIFTED_ODE','SHIFTED_REFIT'):
        group=[q for q in audit if q['truth_PRIVATE']=='cubic' and q['key_PRIVATE'][-1]=='M2']
        need(len(group)==96,'96-case Nonlinear-M2 subgroup not recovered')
        h=sum(q['evaluated'][comp]['MID_PLUS_HALF']['correct'] for q in group)
        f=sum(q['evaluated'][comp]['MID_PLUS_FULL']['correct'] for q in group)
        dose[comp]={'n':96,'half_correct':h,'full_correct':f,'net_full_minus_half':f-h,
                    'effect_full_minus_half':(f-h)/96}
    need(stats['MID_PLUS_FULL']['SHIFTED_ODE']['correct']==346 and
         stats['MID_PLUS_FULL']['SHIFTED_REFIT']['correct']==313 and
         stats['MID_PLUS_FULL']['IDEAL_ODE']['correct']==331,'Frozen historical headline scores changed')
    need(dose['SHIFTED_REFIT']['half_correct']==80 and dose['SHIFTED_REFIT']['full_correct']==85,
         'Registered P99 primary effect parity failed')
    need(dose['SHIFTED_ODE']['half_correct']==87 and dose['SHIFTED_ODE']['full_correct']==78,
         'P101 posthoc Nonlinear-M2 counts failed')
    report={'protocol':'P4-G2B posthoc frozen-versus-refit score and mechanism competition closure',
            'smoke':False,'scientific_status':'RETROSPECTIVE_DIAGNOSTIC_ONLY',
            'n_base_scenes':24,'n_cases':384,'checkpoint_verification':cp_status,
            'source_sha256':hashes,'original_source_sha256':R101['sources'],
            'score_cells_recomputed':len(audit)*(len(ARMS)+1)*len(METHODS),
            'classification':stats,'paired_fixed_vs_refit':transitions,
            'paired_effect_scene_bootstrap_CI95_POSTHOC':ci,
            'nonlinear_M2_dose_endpoint':dose,
            'bootstrap':{'scene_clusters':24,'resamples':RESAMPLES,'seed':SEED,
                         'status':'post-hoc descriptive interval; not newly preregistered confirmation'},
            'critical_limits':[
               'P101 fixed IDEAL parameters and P99 refit SHIFTED parameters are not random treatment assignments',
               'The same SHIFTED observations are compared; no new scenes and no out-of-sample confirmation',
               'Lower true-model prediction RMS does not imply a larger model-selection margin',
               'True mechanism is used only for post-hoc diagnostic margins and accuracy labels',
               'The P99 registered SHIFTED dose contrast remains +5/96 with CI crossing zero',
               'SHIFTED-only frozen dose contrast for Nonlinear-M2 is -9/96, not a general benefit',
               'No claim that freezing is universally better, nor that re-fitting causes a decrease',
               'No MuJoCo, ODE integration, SLSQP, hardware, new observations, E4 or parameter search'],
            'new_fits':0,'new_simulations':0,'E4_accessed':False}
    print('[PASS] 24-scene posthoc paired comparisons, all historical score and result parity')
    return report,{'rows_PRIVATE':audit}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--input-dir',type=str,default=None)
    p.add_argument('--checkpoints-zip',type=str,default=None)
    args=p.parse_args()
    print('='*80)
    print('P4-G2B — READ-ONLY FROZEN-vs-REFIT SELECTION-INVERSION AUDIT')
    print('='*80)
    print('Smoke:',args.smoke,'| no fits/simulations | no E4 | post-hoc exploratory')
    report,audit=run(args)
    prefix='smoke' if args.smoke else 'closure'
    rp=OUT/f'p4g2b_{prefix}_report.json'
    ap=OUT/f'p4g2b_{prefix}_audit_PRIVATE.json'
    if not args.smoke:
        manifest=OUT/'p4g2b_source_manifest.json'
        pack={'input_sha256':report['source_sha256'],'current_script_sha256':hashfile(__file__),
              'analysis':'posthoc_retrospective','bootstrap_seed':SEED}
        if manifest.exists():need(read(manifest)==pack,'P4-G2B manifest changed, do not mix outputs')
        else:save(manifest,pack)
        print('[MAIN] SHIFTED fixed/refit Mid+Full:',report['classification']['MID_PLUS_FULL']['SHIFTED_ODE']['correct'],
              report['classification']['MID_PLUS_FULL']['SHIFTED_REFIT']['correct'])
        print('[TRANSITIONS] Fixed-only/refit-only:',report['paired_fixed_vs_refit']['MID_PLUS_FULL']['ALL']['a_only_correct'],
              report['paired_fixed_vs_refit']['MID_PLUS_FULL']['ALL']['b_only_correct'])
        print('[EXPLORATORY CI] Fixed-refit:',report['paired_effect_scene_bootstrap_CI95_POSTHOC']['MID_PLUS_FULL']['ALL'])
        print('[NONLINEAR-M2] SHIFTED fixed/refit Full-Half:',report['nonlinear_M2_dose_endpoint']['SHIFTED_ODE']['net_full_minus_half'],
              report['nonlinear_M2_dose_endpoint']['SHIFTED_REFIT']['net_full_minus_half'])
    save(rp,report);save(ap,audit)
    print('Saved:',rp);print('Saved:',ap)
    print('P4-G2B COMPLETE')

if __name__=='__main__': main()
