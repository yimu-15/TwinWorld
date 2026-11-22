"""Experiment #101 / P4-G2A: exploratory one-factor physics attribution.

POST-HOC relative to #99/100 results, not confirmatory validation. Generate
counterfactual physical observations using #98's unchanged RK4 simulator and
#99's exact 384 physical cases and shared observation-noise streams. The
IDEAL_ODE MATCHED5/WOO candidate parameters from #99 are *held fixed* for
all five physics arms; never fit new parameters, adjust the perturbation, or
access E4. Reproduce both original IDEAL_ODE and SHIFTED_ODE observations by
SHA-256 before interpreting the new arms.

The factor effects here are controlled GENERATOR perturbations evaluated with
frozen IDEAL parameters. They do not decompose #99's SHIFTED accuracy under
separately re-fitted parameters, nor prove a unique causal physical mechanism.

Windows Giant:
  .\\Giant\\Scripts\\python.exe experiments\\p2d1\\101_p4g2a_frozen_parameter_physics_factor_diagnostic.py --smoke
  .\\Giant\\Scripts\\python.exe experiments\\p2d1\\101_p4g2a_frozen_parameter_physics_factor_diagnostic.py

For local audit of uploaded files (non-MuJoCo):
  python 101_p4g2a_frozen_parameter_physics_factor_diagnostic.py --smoke --input-dir /mnt/data
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from runpy import run_path
import sys

import numpy as np

SELF = Path(__file__).resolve()
HERE = SELF.parent
OUT = HERE / 'results' / 'p4g2a'
P98 = HERE / '98_p4g1a_independent_physics_generator_feasibility.py'
P99 = HERE / '99_p4g1b_independent_generator_identification.py'
P22 = HERE / '22_p3a1b_structural_revision.py'
P20 = HERE / '20_p3a0_parameter_refit.py'
P100 = HERE / '100_p4g1c_cross_generator_closure_audit.py'
ENDPOINTS = ('mid', 'HALF', 'FULL')
LAWS = ('elastic', 'deadzone', 'cubic')
ARMS = ('IDEAL_ODE', 'VISCOUS_ONLY', 'DRY_ONLY', 'LAG_ONLY', 'SHIFTED_ODE')
N_SCENES = 24
N_CASES = 384
CONDITIONS_PER_SCENE = 16
NOISE_SEED = 2029122701
SIGMA = .005


def need(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def hash_file(path: Path) -> str:
    need(path.is_file(), f'Missing frozen file: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def obj_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def load(path: Path):
    need(path.is_file(), f'Missing input: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n',
                   encoding='utf-8')
    os.replace(tmp, path)


def as_rms_mm(x, y):
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    need(xx.shape == yy.shape and np.isfinite(xx).all() and np.isfinite(yy).all(),
         'Nonfinite or incompatible arrays in RMS')
    return float(np.sqrt(np.mean((xx-yy)**2)) * 1000)


def source_inputs(input_dir: str | None):
    if input_dir:
        root = Path(input_dir).resolve()
        return (root/'p4g1b_fresh_report.json', root/'p4g1b_fresh_audit_PRIVATE.json',
                root/'p4g1b_source_manifest.json', root/'p4g1c_closure_report.json',
                root/'p4g1c_closure_audit_PRIVATE.json')
    return (HERE/'results'/'p4g1b'/'p4g1b_fresh_report.json',
            HERE/'results'/'p4g1b'/'p4g1b_fresh_audit_PRIVATE.json',
            HERE/'results'/'p4g1b'/'p4g1b_source_manifest.json',
            HERE/'results'/'p4g1c'/'p4g1c_closure_report.json',
            HERE/'results'/'p4g1c'/'p4g1c_closure_audit_PRIVATE.json')


def preflight(input_dir):
    R99, A99, M99, R100, A100 = source_inputs(input_dir)
    manifest = {str(k):hash_file(v) for k,v in {
        'P98': P98, 'P99': P99, 'P20': P20, 'P22': P22, 'P100': P100,
        'R99': R99, 'A99': A99, 'M99': M99, 'R100': R100, 'A100': A100,
        'SELF': SELF}.items()}
    r99, rows, m99, r100, a100 = [load(p) for p in (R99,A99,M99,R100,A100)]
    need(r99.get('smoke') is False and r99['n_scenes']==24 and
         r99['n_cases_per_generator']==384 and not r99['E4_accessed'],
         'P99 not the frozen formal cohort')
    need(r99['source_sha256']['P98']==manifest['P98'] and
         r99['source_sha256']['P22']==manifest['P22'] and
         r99['source_sha256']['SELF']==manifest['P99'],
         'Frozen P99/ODE/Probe source mismatch')
    need(m99['source_sha256']==r99['source_sha256'] and
         r99['scene_fingerprints_PRIVATE']==m99['scene_fingerprints'],
         'P99 source manifest mismatch')
    need(r100['status']=='PASS_WITH_SCIENTIFIC_LIMITATIONS' and
         r100['n_cases']==384 and r100['n_score_cells_rechecked']==32242 and
         r100['no_new_fits'] is True and r100['E4_accessed'] is False,
         'P100 closure not in expected state')
    need(isinstance(rows,list) and len(rows)==384 and
         isinstance(a100,dict) and len(a100['rows'])==384,
         'P99/P100 row counts invalid')
    all_keys = [tuple(x['key_PRIVATE']) for x in rows]
    need(len(set(all_keys))==384, 'Duplicate P99 keys')
    need(all(a['key_PRIVATE']==b['key_PRIVATE'] for a,b in zip(rows,a100['rows'])),
         'P100 rows misaligned to P99')
    for scene in range(N_SCENES):
        block=[x for x in rows if x['scene_id']==scene]
        need(len(block)==CONDITIONS_PER_SCENE, f'Incomplete scene {scene}')
    gen=run_path(str(P98))
    p99=run_path(str(P99))
    p22=run_path(str(P22))
    need(gen['ARMS']['SHIFTED_ODE']==r99['ODE_arms']['SHIFTED_ODE'] and
         gen['ARMS']['IDEAL_ODE']==r99['ODE_arms']['IDEAL_ODE'],
         'Frozen physics changed')
    need(r99['noise_seed']==NOISE_SEED and r99['n_training_scalar_residuals']==204,
         'Frozen noise/training protocol changed')
    dyn = {
        'IDEAL_ODE': dict(gen['ARMS']['IDEAL_ODE']),
        'VISCOUS_ONLY': dict(gen['ARMS']['IDEAL_ODE'],viscous_rate_s_inv=.08),
        'DRY_ONLY': dict(gen['ARMS']['IDEAL_ODE'],dry_acc_ms2=.02),
        'LAG_ONLY': dict(gen['ARMS']['IDEAL_ODE'],actuator_tau_s=.025),
        'SHIFTED_ODE': dict(gen['ARMS']['SHIFTED_ODE']),
    }
    need(dyn['VISCOUS_ONLY']['viscous_rate_s_inv']==dyn['SHIFTED_ODE']['viscous_rate_s_inv'] and
         dyn['DRY_ONLY']['dry_acc_ms2']==dyn['SHIFTED_ODE']['dry_acc_ms2'] and
         dyn['LAG_ONLY']['actuator_tau_s']==dyn['SHIFTED_ODE']['actuator_tau_s'],
         'Factor magnitude not identical to original SHIFTED')
    return gen,p99,p22,rows,manifest,dyn,r99,r100


def noise_and_probe(p99, Probe, pa, ac, key, name):
    sid, band, mode, axis, level = key
    tag = 'HALF_FULL_PAIR' if name in ('HALF','FULL') else name
    rng = p99['stable_rng'](NOISE_SEED,sid,band,mode,axis,level,tag)
    pn = rng.normal(0,SIGMA,(51,2))
    an = rng.normal(0,SIGMA,(51,2))
    probe = Probe(pa+pn,ac+an)
    need(np.asarray(probe.y_b).shape==(51,) and np.asarray(probe.a_half).shape==(101,),
         'RevisionProbe API changed')
    return probe


def classify(scores):
    return min(LAWS,key=lambda l:(scores[l],LAWS.index(l)))


def process_case(record, gen,p99,Probe,dyn,commands):
    key=record['key_PRIVATE']
    need(len(key)==5,'Invalid case key')
    sid,band,mode,axis,level=key
    need(record['scene_id']==sid,'Case/scene mismatch')
    scene=record['physical_scene_parameters_PRIVATE']
    need(record['physical_scene_sha256_PRIVATE']==gen['digest_object'](scene),
         'Case parameter fingerprint mismatch')
    fit=record['fit_results_PRIVATE']['IDEAL_ODE']['MATCHED5']
    chosen={law:fit['wide' if law=='elastic' else 'outer_box'][law] for law in LAWS}
    need(all(v is not None and len(v['params_PRIVATE'])==(2 if law=='elastic' else 3)
                 for law,v in chosen.items()),'Missing frozen IDEAL WOO candidate fit')
    params={law:v['params_PRIVATE'] for law,v in chosen.items()}
    outputs={}
    ideal_active={}
    for ga in ARMS:
        clean={}
        for action in ENDPOINTS:
            passive,active=gen['simulate'](scene,mode,commands[action],dyn[ga])
            need(passive.shape==active.shape==(51,2), 'ODE observation shape changed')
            clean[action]=(passive,active)
            if ga in ('IDEAL_ODE','SHIFTED_ODE'):
                digest=gen['digest_object']({'p':passive.tolist(),'a':active.tolist()})
                need(digest==record['clean_observation_fingerprints_PRIVATE'][ga][action],
                     f'P99 frozen replay differs at {key}, {ga}, {action}')
        if ga=='IDEAL_ODE':
            ideal_active={action:clean[action][1] for action in ENDPOINTS}
        scores={law:{} for law in LAWS}
        for action in ENDPOINTS:
            pa,ac=clean[action]
            probe=noise_and_probe(p99,Probe,pa,ac,key,action)
            for law in LAWS:
                predicted=np.asarray(probe.predict(law,params[law]),dtype=float)
                need(predicted.shape==(51,) and np.isfinite(predicted).all(),
                     f'Nonfinite frozen predictor: {ga}/{law}/{key}')
                scores[law][action]=as_rms_mm(predicted,probe.y_b)
        selection={}
        for method,other in (('MID_PLUS_HALF','HALF'),('MID_PLUS_FULL','FULL')):
            sc={law:math.sqrt((scores[law]['mid']**2+scores[law][other]**2)/2)
                for law in LAWS}
            selection[method]=classify(sc)
        if ga=='IDEAL_ODE':
            original=record['decisions']['IDEAL_ODE']['MATCHED5']['WOO']
            for method in selection:
                need(original[method]['available'] is True and
                     original[method]['winner']==selection[method],
                     f'P99 decision parity mismatch on {key}/{method}')
                for law in LAWS:
                    score_ref=float(original[method]['scores_mm_PRIVATE'][law])
                    score_now=math.sqrt((scores[law]['mid']**2+scores[law][
                        'HALF' if method=='MID_PLUS_HALF' else 'FULL']**2)/2)
                    need(math.isclose(score_now,score_ref,rel_tol=1e-9,abs_tol=1e-7),
                         f'P99 score parity mismatch on {key}/{law}/{method}')
        differences={}
        for action in ENDPOINTS:
            differences[action]={
                'A_vs_ideal_mm':as_rms_mm(clean[action][1][:,0],ideal_active[action][:,0]),
                'B_vs_ideal_mm':as_rms_mm(clean[action][1][:,1],ideal_active[action][:,1])}
        outputs[ga]={'winner':selection,'true_law_rms_mm_PRIVATE':scores[record['true_law_PRIVATE']],
                     'motion_shift_mm':differences,
                     'scores_rms_mm_PRIVATE':scores}
    return {'key_PRIVATE':key,'scene_id':sid,'true_law_PRIVATE':record['true_law_PRIVATE'],
            'mode_PRIVATE':mode,'frozen_ideal_fit_origin_PRIVATE':{
                law:chosen[law]['origin'] for law in LAWS},
            'arms':outputs,
            'P99_shift_refit_winner':{
                m:record['decisions']['SHIFTED_ODE']['MATCHED5']['WOO'][m]['winner']
                for m in ('MID_PLUS_HALF','MID_PLUS_FULL')}}


def summarize(rows):
    stats={}
    for ga in ARMS:
        stats[ga]={}
        for meth in ('MID_PLUS_HALF','MID_PLUS_FULL'):
            counts={}
            for law in LAWS:
                subset=[r for r in rows if r['true_law_PRIVATE']==law]
                correct=sum(x['arms'][ga]['winner'][meth]==law for x in subset)
                counts[law]={'n':len(subset),'correct':int(correct)}
            total=sum(x['correct'] for x in counts.values())
            stats[ga][meth]={'correct':total,'total':len(rows),
                             'accuracy':total/len(rows),'by_law':counts}
    refit={meth:{'correct':sum(r['P99_shift_refit_winner'][meth]==r['true_law_PRIVATE']
                                  for r in rows), 'total':len(rows)}
            for meth in ('MID_PLUS_HALF','MID_PLUS_FULL')}
    metrics={}
    for ga in ARMS:
        diffs={a:{part:[] for part in ('A_vs_ideal_mm','B_vs_ideal_mm')}
               for a in ENDPOINTS}
        rms={a:[] for a in ENDPOINTS}
        for r in rows:
            for action in ENDPOINTS:
                for part in diffs[action]:
                    diffs[action][part].append(r['arms'][ga]['motion_shift_mm'][action][part])
                rms[action].append(r['arms'][ga]['true_law_rms_mm_PRIVATE'][action])
        metrics[ga]={action:{part:{'median':float(np.median(v)),'mean':float(np.mean(v))}
                             for part,v in diffs[action].items()}|
                        {'frozen_true_law_noisy_RMS_mm':{
                            'median':float(np.median(rms[action])),
                            'mean':float(np.mean(rms[action]))}}
                     for action in ENDPOINTS}
    transitions={}
    for ga in ARMS[1:]:
        trans={}
        for law in LAWS:
            ss=[r for r in rows if r['true_law_PRIVATE']==law]
            lost=gained=both=neither=0
            for r in ss:
                i=(r['arms']['IDEAL_ODE']['winner']['MID_PLUS_FULL']==law)
                j=(r['arms'][ga]['winner']['MID_PLUS_FULL']==law)
                if i and j:both+=1
                elif i and not j:lost+=1
                elif not i and j:gained+=1
                else:neither+=1
            trans[law]={'n':len(ss),'both_correct':both,'lost':lost,
                        'gained':gained,'both_wrong':neither,
                        'net_change':gained-lost}
        transitions[ga]=trans
    return {'classification_frozen_IDEAL_parameters':stats,
            'historical_P99_SHIFTED_refitted_classification':refit,
            'motion_and_true_law_diagnostics':metrics,
            'IDEAL_to_factor_classification_transitions':transitions}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true',help='One case, no statistical interpretation')
    parser.add_argument('--input-dir',default=None,
                        help='Optional location of flat, hash-verified uploaded audit files')
    args=parser.parse_args()
    print('='*76)
    print('P4-G2A — POST-HOC PHYSICS COMPONENTS / FROZEN-PARAMETER DIAGNOSTIC')
    print('='*76)
    print('Smoke:',args.smoke,'| New fits: 0 | E4 sealed | exploratory')
    gen,p99,p22,original,manifest,dyn,r99,r100=preflight(args.input_dir)
    commands=gen['command_profiles']()
    need(set(ENDPOINTS)<=set(commands),'Frozen action profiles unavailable')
    print('[PASS] P98/P99/P100 source lineage, 384 cases, fixed physics factors')
    print('[PROTOCOL] 1 case smoke, otherwise 384; 5 physical arms; 3 heldout actions')
    print('[PROTOCOL] IDEAL matched-five WOO parameters frozen across ALL arms')
    this_meta={'source_sha256':manifest,'factors':dyn,'noise_seed':NOISE_SEED,
               'reference_cohort_sha256':manifest['A99'],
               'scenes':r99['scene_fingerprints_PRIVATE'],
               'methods':['MID_PLUS_HALF','MID_PLUS_FULL'],
               'status':'posthoc exploratory; no causal classification attribution proof'}
    if not args.smoke:
        mp=OUT/'p4g2a_source_manifest.json'
        if mp.exists():need(load(mp)==this_meta,'Input/source changed, do not mix checkpoints')
        else:write(mp,this_meta)
    rows=[]
    case_iter=original[:1] if args.smoke else original
    for sid in (range(1) if args.smoke else range(N_SCENES)):
        block=case_iter if args.smoke else case_iter[sid*16:(sid+1)*16]
        need(len(block)==(1 if args.smoke else 16),'Case count per scene changed')
        checkpoint=OUT/'checkpoints'/f'scene_{sid:02d}.json'
        if not args.smoke and checkpoint.exists():
            existing=load(checkpoint)
            need(existing['source_sha256']==manifest and
                 existing['scene_id']==sid and
                 existing['input_case_keys']==[r['key_PRIVATE'] for r in block] and
                 existing['rows_sha256']==obj_hash(existing['rows']),
                 f'Checkpoint changed: {checkpoint}')
            new=existing['rows']
            print(f'[RESUME] scene {sid+1}/{N_SCENES}: 16 cases',flush=True)
        else:
            new=[process_case(r,gen,p99,p22['RevisionProbe'],dyn,commands) for r in block]
            if not args.smoke:
                write(checkpoint,{'source_sha256':manifest,'scene_id':sid,
                      'input_case_keys':[r['key_PRIVATE'] for r in block],
                      'rows_sha256':obj_hash(new),'rows':new})
            print(f'[DONE] scene {sid+1}/{1 if args.smoke else N_SCENES}: {len(new)} cases',flush=True)
        rows+=new
    need(len(rows)==len(case_iter),'Missing output rows')
    stats=summarize(rows)
    if not args.smoke:
        need(stats['historical_P99_SHIFTED_refitted_classification']['MID_PLUS_FULL']['correct']==313,
             'P99 refitted SHIFTED published result changed')
        need(stats['classification_frozen_IDEAL_parameters']['IDEAL_ODE']['MID_PLUS_FULL']['correct']==331,
             'P99 IDEAL published result changed')
        need(stats['classification_frozen_IDEAL_parameters']['IDEAL_ODE']['MID_PLUS_HALF']['correct']==316,
             'P99 IDEAL half published result changed')
        for law,n in (('elastic',96),('deadzone',144),('cubic',144)):
            need(stats['classification_frozen_IDEAL_parameters']['IDEAL_ODE']['MID_PLUS_FULL']['by_law'][law]['n']==n,
                 'Mechanism denominator changed')
    report={'protocol':'P4-G2A frozen ideal-fit one-factor generator diagnostic',
            'smoke':args.smoke,'n_cases':len(rows),'n_base_scenes':(1 if args.smoke else 24),
            'physics_factors_FIXED':dyn,'sources':manifest,
            'refit_performed':False,'true_parameters_used_for_selection':False,
            'is_posthoc_exploratory':True,'E4_accessed':False,'summary':stats,
            'cautions':[
                'Outcome-informed motivation after P99/100: post-hoc exploratory, not preregistered confirmation',
                'Candidate parameters fixed from P99 IDEAL_ODE MATCHED5/WOO for every factor; not SHIFTED re-fitting',
                'All physical factors computed with same ODE engine; not separate physics engines',
                'Single-factor effects on classification are not additive under nonlinear dynamics',
                'Fixed noise stream is recreated identically for every arm and HALF/FULL pair',
                'Truth labels used only for post-decision grouping; no fitting performed',
                'No real hardware, no global optimality claim; E4 remains sealed',
                'For causally confirmed classifier effect, require independent new scenes and frozen rule before observing outcomes',
            ]}
    tag='smoke' if args.smoke else 'diagnostic'
    rp=OUT/f'p4g2a_{tag}_report.json'
    ap=OUT/f'p4g2a_{tag}_audit_PRIVATE.json'
    write(rp,report);write(ap,{'rows_PRIVATE':rows})
    print('[PASS] Exact IDEAL/SHIFTED source observation SHA replay and IDEAL decision parity')
    if not args.smoke:
        for ga in ARMS:
            data=stats['classification_frozen_IDEAL_parameters'][ga]['MID_PLUS_FULL']
            print(f'{ga}/fixed-ideal Mid+Full: {data["correct"]}/{data["total"]};'
                  f' Elastic {data["by_law"]["elastic"]["correct"]}/96')
        print('[REFERENCE] P99 SHIFTED refit Mid+Full:',
              stats['historical_P99_SHIFTED_refitted_classification']['MID_PLUS_FULL'])
    else:
        print('[PASS] Smoke only; not an estimate of physics-factor effects')
    print('Saved:',rp)
    print('Saved:',ap)
    print('P4-G2A COMPLETE')

if __name__=='__main__':
    main()
