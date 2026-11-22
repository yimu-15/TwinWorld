"""Experiment #106 / P4-B2B: FRESH-scene comparison of classical identification methods.

PRE-OUTCOME FROZEN PROTOCOL. On identical SHIFTED_ODE observations:
 - SLSQP_MATCHED5: reference 2+3-start output-error parameter optimization;
 - TRF_OUT: independent trust-region reflective output-error minimizer, 2+3 starts;
 - EQERR_SG: independent derivative/equation-error parameter regression,
             SG(11,3), trim 5 samples each side, 21 gap grid points.
All receive the same four public (A,B) training trajectories, 204 observed
B-position samples, same three candidate structures and WOO bounds, and use
exactly the same untouched heldout MID/HALF/FULL observations. SLSQP and TRF
optimize the same output-error objective. EQERR optimizes a different physical
objective with 164 derived acceleration samples; it has different cost and
noise sensitivity. This is equal OBSERVATION permission, NOT equal compute.

Primary endpoint was fixed before seeing fresh outcomes: EQERR_SG minus
SLSQP_MATCHED5 in WOO MID_PLUS_FULL identification accuracy across all
384 cases (24 base scenes x 2 bands x 8 physical conditions), 24-scene
paired cluster bootstrap (2000 resamples). TRF-SLSQP secondary; other
configs, law strata, dose responses, solver failures and costs diagnostic.
Do NOT choose a winning algorithm from the study and re-label E4 as unseen.
No fitting on heldout actions, no access to E4, no outcome-driven tuning.

Usage from TwinWorld_P0 root:
 .\\Giant\\Scripts\\python.exe -m py_compile .\\experiments\\p2d1\\106_p4b2b_fresh_classical_identification_comparison.py
 .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\106_p4b2b_fresh_classical_identification_comparison.py --smoke
 .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\106_p4b2b_fresh_classical_identification_comparison.py
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from runpy import run_path
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import scipy

SELF = Path(__file__).resolve()
HERE = SELF.parent
BASE = HERE / 'results'
OUT = BASE / 'p4b2b'
P105 = HERE / '105_p4b2a_independent_identification_baseline_pilot.py'
R105 = BASE / 'p4b2a' / 'p4b2a_pilot_report.json'
A105 = BASE / 'p4b2a' / 'p4b2a_pilot_audit_PRIVATE.json'
R103 = BASE / 'p4g2c' / 'p4g2c_fresh_report.json'
M103 = BASE / 'p4g2c' / 'p4g2c_source_manifest.json'
R99 = BASE / 'p4g1b' / 'p4g1b_fresh_report.json'
R98 = BASE / 'p4g1a' / 'p4g1a_feasibility_report.json'
S98 = BASE / 'p4g1a' / 'p4g1a_smoke_report.json'

SCENE_SEED = 20300223
SMOKE_SCENE_SEED = 20310223
NOISE_SEED = 2030022301
SMOKE_NOISE_SEED = 2031022301
BOOT_SEED = 2030022302
N_SCENES = 24
N_BOOT = 2000
N_CASES = 384
TRAIN = ('e1','e2','weak','strong')
LAWS = ('elastic','deadzone','cubic')
DOMAINS = ('wide','outer_box')
CONFIGS = ('WWW','OOO','WOO')
METHODS = ('MID_PLUS_HALF','MID_PLUS_FULL')
ARMS = ('EQERR_SG','TRF_OUT','SLSQP_MATCHED5_REFERENCE')
PRIMARY_CFG = 'WOO'
PRIMARY_METHOD = 'MID_PLUS_FULL'
N_B_POSITIONS = 204
MIN_COVERAGE = .90
GAP_TOL = 1e-6


def need(cond, message):
    if not cond:
        raise RuntimeError(message)


def sha(path):
    need(path.is_file(), f'Missing immutable source/evidence: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    need(path.is_file(), f'Missing JSON: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, separators=(',', ':'), sort_keys=True,
              allow_nan=False).encode('utf-8')).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                              allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def preflight():
    # P105's own preflight already validates P98,P99,P103,P104 and historical
    # source hashes. Here we additionally freeze the exact P105 pilot release.
    files={'P105':P105, 'R105':R105,'A105':A105,'R103':R103,
           'M103':M103,'R99':R99,'R98':R98,'S98':S98,'SELF':SELF}
    hashes={k:sha(v) for k,v in files.items()}
    pilot=read(R105)
    need(pilot['smoke'] is False and pilot['n_cases']==3 and
         pilot['SELF_sha256']==hashes['P105'] and
         pilot['new_fresh_scenes'] is False and
         pilot['truth_used_during_fit'] is False and
         pilot['heldout_used_for_fit'] is False and
         pilot['E4_accessed'] is False,
         'Historical #105 feasibility pilot/source is not frozen')
    p105=run_path(str(P105))
    (p103,p99,gen,p77,p92,p22,commands,starts,rmap,
        old103, man103, source105) = p105['preflight']()
    need(old103['E4_accessed'] is False and old103['n_cases']==384,
         'P103 data validation changed')
    need(read(R99)['E4_accessed'] is False and read(R98)['E4_accessed'] is False,
         'P99 or P98 E4 flag changed')
    need(p105['TRAIN']==TRAIN and p105['N_SAMPLES']==N_B_POSITIONS and
         p105['LAWS']==LAWS and p105['DOMAINS']==DOMAINS and
         p105['SG_WINDOW']==11 and p105['SG_POLY']==3 and
         p105['SG_EDGE']==5 and p105['DEADZONE_GRID']==21 and
         p105['TRF_MAX_NFEV']==150,
         'P105 estimator implementation/protocol changed')
    need(all(len(starts[l])==3 and len(p77['initial_points'](l))==2
             for l in LAWS), 'Reference 2+3 starts differ')
    return p105,p103,p99,gen,p77,p92,p22,commands,starts,hashes,old103,man103


def fresh_scenes(gen,p99,smoke,old103,man103):
    seed = SMOKE_SCENE_SEED if smoke else SCENE_SEED
    rng=np.random.default_rng(seed)
    n=1 if smoke else N_SCENES
    scenes=[{b:gen['sample_scene'](rng,b) for b in gen['BANDS']}
            for _ in range(n)]
    fps=[gen['digest_object'](s[b]) for s in scenes for b in gen['BANDS']]
    need(len(fps)==len(set(fps))==2*n,'Duplicate P4-B2B base scene')
    old=set(read(R98)['scene_fingerprints_PRIVATE']) | \
        set(read(S98)['scene_fingerprints_PRIVATE']) | \
        set(read(R99)['scene_fingerprints_PRIVATE']) | \
        set(man103['scene_fingerprints_PRIVATE'])
    # P99 and P103 interface smoke seeds have never been used for inference;
    # still exclude them so B2B cannot overlap any earlier smoke / pilot.
    earlier_seeds=[p99['SMOKE_SCENE_SEED'],20300107,20310107]
    if not smoke:
        earlier_seeds.append(SMOKE_SCENE_SEED)
    for x in earlier_seeds:
        rr=np.random.default_rng(x)
        old.update(gen['digest_object'](gen['sample_scene'](rr,b))
                   for b in gen['BANDS'])
    need(not (old & set(fps)), 'Fresh B2B scene overlaps historical/smoke/pilot')
    return scenes,fps


def record_decisions(fit_sets, target, p103):
    out={}
    for arm,fits in fit_sets.items():
        out[arm]={cfg:p103['scored_decision'](fits,target,cfg)
                  for cfg in CONFIGS}
    return out


def one_case(sid,band,mode,axis,level,base,p105,p103,p99,gen,p77,p92,
             Probe,commands,starts,noise_seed,smoke=False):
    scene=gen['apply_condition'](base,mode,axis,level)
    observed,clean,fps=p99['sample_noisy_probes'](
        gen,Probe,scene,mode,commands,sid,band,axis,level,noise_seed)
    target=observed['SHIFTED_ODE']
    need(len(target)==7 and all(target[k].y_b.shape==(51,) for k in TRAIN),
         'Unexpected training B-position vector shape')
    fits,logs=p105['fit_methods'](target,p77,p103,starts)
    t0=time.perf_counter()
    baseline,baseline_logs=p99['fit_all'](target,p77,p92,starts)
    slsqp_wall=float(time.perf_counter()-t0)
    choices=record_decisions({**{m:{'MATCHED5':fits[m]} for m in ARMS[:2]},
                              'SLSQP_MATCHED5_REFERENCE':baseline},target,p103)
    # Original selection implementation must agree with independently
    # assembled SLSQP decisions (same reference parameters/observations).
    original=p99['select_results'](baseline,
                     lambda mse,law,meth: 0.0)
    for meth in METHODS:
        a=original['MATCHED5']['WOO'][meth]
        b=choices['SLSQP_MATCHED5_REFERENCE']['WOO'][meth]
        need(a['available']==b['available'] and a['winner']==b['winner'],
             'SLSQP old/new selector disagreement')
        if a['available']:
            need(all(math.isclose(a['scores_mm_PRIVATE'][l],
                        b['scores_mm_PRIVATE'][l],rel_tol=1e-9,abs_tol=1e-7)
                     for l in LAWS),'SLSQP candidate scores changed')
    if smoke:
        # A smoke is an interface test. Do not inspect classification truth.
        return {'smoke_shapes':{'training_position_samples':sum(len(target[a].y_b) for a in TRAIN),
             'heldout_probe_B':len(target['FULL'].y_b),
             'heldout_probe_A':len(target['FULL'].a_half)},
             'finite_score_checks':sum(int(choices[a]['WOO'][m]['available'])
                                       for a in ARMS for m in METHODS),
             'fit_availability':{a:sum(fits[a][d][l] is not None
                               for d in DOMAINS for l in LAWS) for a in ARMS[:2]}}
    nested={}
    for alg,all_fits in ([(m,fits[m]) for m in ARMS[:2]] +
                         [('SLSQP_MATCHED5_REFERENCE',baseline['MATCHED5'])]):
        nested[alg]={}
        for l in LAWS:
            w,o=all_fits['wide'][l],all_fits['outer_box'][l]
            # EQERR MSE measures ACCELERATION equation error, while the
            # other two use normalized B-position output error. Do not mix.
            gap = None if w is None or o is None else float(w['mse']-o['mse'])
            nested[alg][l]={'gap':gap,'violation':bool(gap is not None and gap>GAP_TOL)}
    truth=p99['TRUTH'][mode] # AFTER ALL fits and decisions; never to optimizers.
    return {'key_PRIVATE':[sid,band,mode,axis,level], 'scene_id':sid,
            'mode_PRIVATE':mode,'true_law_PRIVATE':truth,
            'condition_PRIVATE':level,
            'scene_sha256_PRIVATE':gen['digest_object'](scene),
            'clean_fingerprints_PRIVATE':fps,
            'decisions':choices,
            'fits_PRIVATE':{**fits,'SLSQP_MATCHED5_REFERENCE':baseline['MATCHED5']},
            'fit_logs_PRIVATE':{**logs,'SLSQP_MATCHED5_REFERENCE':baseline_logs},
            'SLSQP_fit_plus_internal_scoring_wall_seconds':slsqp_wall,
            'nested_PRIVATE':nested}


def correct(row, arm, meth, cfg='WOO'):
    d=row['decisions'][arm][cfg][meth]
    return int(bool(d['available'] and d['winner']==row['true_law_PRIVATE']))


def scene_cluster_pair(rows,arm_a,meth_a,arm_b,meth_b,predicate=lambda r:True):
    selected=[r for r in rows if predicate(r)]
    need(selected and len({r['scene_id'] for r in selected})==N_SCENES,
         'Paired analysis missing scene clusters')
    per=defaultdict(list)
    status=Counter()
    for row in selected:
        x,y=correct(row,arm_a,meth_a),correct(row,arm_b,meth_b)
        per[row['scene_id']].append(x-y)
        status[(x,y)]+=1
    need(len({len(v) for v in per.values()})==1,
         'Paired per-scene condition count differs')
    vv=np.asarray([np.mean(per[s]) for s in range(N_SCENES)],dtype=float)
    return {'n':len(selected),'a_correct':status[(1,0)]+status[(1,1)],
            'b_correct':status[(0,1)]+status[(1,1)],
            'both_correct':status[(1,1)],'a_only':status[(1,0)],
            'b_only':status[(0,1)],'both_wrong':status[(0,0)],
            'net':status[(1,0)]-status[(0,1)],'effect':float(vv.mean()),
            '_cluster_values':vv}


def summarize(rows):
    need(len(rows)==N_CASES and len({tuple(x['key_PRIVATE']) for x in rows})==N_CASES,
         'Invalid fresh case coverage')
    for sid in range(N_SCENES):
        need(sum(x['scene_id']==sid for x in rows)==16, 'Missing 16-case scene')
    stats={}
    for arm in ARMS:
        stats[arm]={}
        for cfg in CONFIGS:
            stats[arm][cfg]={}
            for meth in METHODS:
                ds=[r['decisions'][arm][cfg][meth] for r in rows]
                stats[arm][cfg][meth]={
                    'n':N_CASES,'correct':sum(correct(r,arm,meth,cfg) for r in rows),
                    'available':sum(int(x['available']) for x in ds),
                    'by_truth':{law:{'n':sum(r['true_law_PRIVATE']==law for r in rows),
                        'correct':sum(correct(r,arm,meth,cfg) for r in rows
                                      if r['true_law_PRIVATE']==law)} for law in LAWS}}
    prim=scene_cluster_pair(rows,'EQERR_SG','MID_PLUS_FULL',
                              'SLSQP_MATCHED5_REFERENCE','MID_PLUS_FULL')
    trs=scene_cluster_pair(rows,'TRF_OUT','MID_PLUS_FULL',
                              'SLSQP_MATCHED5_REFERENCE','MID_PLUS_FULL')
    eqtr=scene_cluster_pair(rows,'EQERR_SG','MID_PLUS_FULL',
                              'TRF_OUT','MID_PLUS_FULL')
    pairs={'PRIMARY_EQERR_MINUS_SLSQP_WOO_FULL':prim,
           'SECONDARY_TRF_MINUS_SLSQP_WOO_FULL':trs,
           'SECONDARY_EQERR_MINUS_TRF_WOO_FULL':eqtr}
    for arm in ARMS:
        pairs['DIAGNOSTIC_'+arm+'_NM2_FULL_MINUS_HALF']=scene_cluster_pair(
            rows,arm,'MID_PLUS_FULL',arm,'MID_PLUS_HALF',
            lambda r:r['true_law_PRIVATE']=='cubic' and r['condition_PRIVATE']=='M2')
    draw=np.random.default_rng(BOOT_SEED).integers(0,N_SCENES,size=(N_BOOT,N_SCENES))
    for item in pairs.values():
        values=item.pop('_cluster_values')
        item['scene_cluster_CI95']=[float(z) for z in np.quantile(values[draw].mean(1),[.025,.975])]
    cost={}
    for arm in ARMS:
        cost[arm]={}
        if arm=='SLSQP_MATCHED5_REFERENCE':
            cost[arm]['total_optimizer_objective_calls']=sum(
                int(r['fit_logs_PRIVATE'][arm][d][l]['nfev_total'])
                for r in rows for d in DOMAINS for l in LAWS)
            cost[arm]['total_fit_plus_internal_scoring_wall_seconds']=float(sum(
                r['SLSQP_fit_plus_internal_scoring_wall_seconds'] for r in rows))
            cost[arm]['cost_unit']='SLSQP objective nfev (2 original + 3 extra starts); wall includes internal heldout scoring'
        elif arm=='TRF_OUT':
            cost[arm]['total_optimizer_objective_calls']=sum(
                int(r['fit_logs_PRIVATE'][arm][d][l]['residual_calls_ALL'])
                for r in rows for d in DOMAINS for l in LAWS)
            cost[arm]['total_fit_wall_seconds']=float(sum(
                r['fit_logs_PRIVATE'][arm][d][l]['elapsed_s']
                for r in rows for d in DOMAINS for l in LAWS))
            cost[arm]['cost_unit']='ALL TRF residual calls including numerical derivatives'
        else:
            cost[arm]['total_regression_grid_evaluations']=sum(
                int(r['fit_logs_PRIVATE'][arm][d][l]['attempts'])
                for r in rows for d in DOMAINS for l in LAWS)
            cost[arm]['total_fit_wall_seconds']=float(sum(
                r['fit_logs_PRIVATE'][arm][d][l]['elapsed_s']
                for r in rows for d in DOMAINS for l in LAWS))
            cost[arm]['cost_unit']='EQERR constrained linear regressions / grid points, NOT residual nfev'
        cost[arm]['fit_law_domain_units']=N_CASES*6
    nested={arm:dict(Counter(l for r in rows for l in LAWS
                         if r['nested_PRIVATE'][arm][l]['violation'])) for arm in ARMS}
    coverage={a:{m:stats[a]['WOO'][m]['available']/N_CASES for m in METHODS}
              for a in ARMS}
    return stats,pairs,cost,nested,coverage


def smoke_run(scene,p105,p103,p99,gen,p77,p92,Probe,commands,starts,hashes):
    res=one_case(0,'original','elastic','matched','M0',scene['original'],
                 p105,p103,p99,gen,p77,p92,Probe,commands,starts,
                 SMOKE_NOISE_SEED,smoke=True)
    need(res['smoke_shapes']=={'training_position_samples':204,
         'heldout_probe_B':51,'heldout_probe_A':101},'Smoke probe shape mismatch')
    need(res['finite_score_checks']==6, 'Smoke candidate scoring unavailable')
    doc={'protocol':'P4-B2B independent out-of-cohort smoke (not effect estimate)',
         'smoke':True,'source_sha256':hashes,'test_conditions':1,
         'interface':res,'new_scientific_results':0,'E4_accessed':False}
    path=OUT/'p4b2b_smoke_report.json'
    save(path,doc)
    print('[PASS] Original SLSQP, TRF and EQERR complete one out-of-cohort interface case')
    print('[PASS] Same 204 training B positions, 51-B / 101-A heldouts, finite scores')
    print('[NOTE] Smoke truth/outcomes not inspected; NO scientific effect estimate')
    print('Saved:',path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    arg=parser.parse_args()
    print('='*76)
    print('P4-B2B — FRESH CLASSICAL IDENTIFICATION BASELINE COMPARISON')
    print('='*76)
    print(f'Smoke: {arg.smoke} | same TRAIN observations | E4 sealed')
    (p105,p103,p99,gen,p77,p92,p22,commands,starts,
     hashes,old103,man103)=preflight()
    print('[PASS] P105 pilot, P103/P104 historical lineage and original estimators unchanged')
    scenes, fps=fresh_scenes(gen,p99,arg.smoke,old103,man103)
    print('[PASS] New scene fingerprints disjoint from P98/P99/P103 and pilot/smoke')
    print('[PROTOCOL] 24 scenes x 2 bands x 8 conditions, SHIFTED_ODE observations')
    print('[PROTOCOL] All methods four TRAIN actions; EQERR different objective/compute cost')
    if arg.smoke:
        smoke_run(scenes[0],p105,p103,p99,gen,p77,p92,p22['RevisionProbe'],
                  commands,starts,hashes)
        print('P4-B2B SMOKE COMPLETE')
        return
    manifest={'source_sha256':hashes,'scene_seed':SCENE_SEED,
        'smoke_scene_seed':SMOKE_SCENE_SEED,'noise_seed':NOISE_SEED,
        'bootstrap_seed':BOOT_SEED,'scene_fingerprints_PRIVATE':fps,
        'protocol':'P4-B2B 24 fresh scenes, SHIFTED_ODE only, 3 estimators',
        'primary':'WOO MID_PLUS_FULL EQERR_SG minus SLSQP_MATCHED5_REFERENCE',
        'n_training_B_positions':N_B_POSITIONS,
        'E4_accessed':False}
    mf=OUT/'p4b2b_source_manifest.json'
    if mf.exists():
        need(read(mf)==manifest,'Manifest differs: do not combine old checkpoints')
    else:
        save(mf,manifest)
    rows=[]
    for sid,base_pair in enumerate(scenes):
        cp=OUT/'checkpoints'/f'scene_{sid:02d}.json'
        if cp.exists():
            pack=read(cp)
            need(pack['scene_id']==sid and pack['source_sha256']==hashes and
                 pack['base_fingerprints']==fps[2*sid:2*sid+2] and
                 len(pack['rows'])==16 and pack['rows_sha256']==digest(pack['rows']),
                 f'Checkpoint mismatch in scene {sid}')
            block=pack['rows']
            print(f'[RESUME] scene {sid+1}/{N_SCENES}: 16 cases',flush=True)
        else:
            block=[]
            for band in gen['BANDS']:
                for mode in p99['MODES']:
                    for axis,level in p99['CONDITIONS'][mode]:
                        block.append(one_case(sid,band,mode,axis,level,
                             base_pair[band],p105,p103,p99,gen,p77,p92,
                             p22['RevisionProbe'],commands,starts,NOISE_SEED))
            need(len(block)==16,'Incomplete scene block')
            save(cp,{'scene_id':sid,'source_sha256':hashes,
                     'base_fingerprints':fps[2*sid:2*sid+2],
                     'rows_sha256':digest(block),'rows':block})
            print(f'[DONE] scene {sid+1}/{N_SCENES}: 16 cases',flush=True)
        rows.extend(block)
    stats,pairs,cost,nested,coverage=summarize(rows)
    report={'protocol':'P4-B2B fresh-scene classical identification estimator comparison',
       'smoke':False,'protocol_frozen_before_outcome':True,
       'n_base_scenes':N_SCENES,'n_cases':N_CASES,'scene_seed':SCENE_SEED,
       'noise_seed':NOISE_SEED,'generator':'SHIFTED_ODE',
       'source_sha256':hashes,'methods':list(ARMS),
       'training_actions':list(TRAIN),'heldout_actions':['mid','HALF','FULL'],
       'n_training_B_positions':N_B_POSITIONS,
       'EQERR':{'SG_window':11,'SG_poly':3,'trim_each_end':5,'derived_acceleration_samples':164,
                'deadzone_gap_grid':21},
       'TRF':{'starts':5,'max_nfev_scipy_per_start':150,
              'all_residual_calls_including_numerical_jacobians':True},
       'SLSQP':{'starts':5,'maxiter_per_start':150},
       'primary_and_secondary':pairs,
       'classification':stats,'compute_cost':cost,'nested_violations':nested,
       'coverage':coverage,
       'coverage_guard_90pct_pass':all(x>=MIN_COVERAGE for a in coverage.values() for x in a.values()),
       'bootstrap':{'unit':'base_scene','n_clusters':N_SCENES,
                    'resamples':N_BOOT,'seed':BOOT_SEED,
                    'method':'paired percentile 95% CI'},
       'limitations':[
          'Equal observational access does not imply equal computational budget',
          'SLSQP wall time includes embedded heldout scoring in original P99 fit_all; wall timings across methods are NOT strictly comparable',
          'EQERR differentiates noisy position with SG and minimizes acceleration equation error; its MSE is not B-position MSE',
          'TRF and SLSQP share 204 B-output residual objective and starts; nfev and wall times differ',
          'Finite-start local optimizers cannot guarantee global optimum',
          'SHIFTED ODE is one RK4 generator with unmodeled drag, friction and actuator lag; not hardware generalization',
          'Truth only for retrospective aggregation, never fitting or method choice',
          'All abstentions counted incorrect, no adaptive tuning or result-based exclusions',
          'Only 24 base scene clusters are independent; the 384 rows are not IID',
          'Pilot #105 was retrospective; P4-B2B fresh outcomes are new',
          'E4 remains sealed'],
       'new_e4_accessed':False,'truth_used_during_fit':False,
       'heldout_used_for_fit':False,'adaptive_tuning':False,
       'environment':{'python':sys.version,'numpy':np.__version__,
           'scipy':scipy.__version__,'platform':platform.platform()}}
    rp=OUT/'p4b2b_fresh_report.json'
    ap=OUT/'p4b2b_fresh_audit_PRIVATE.json'
    save(rp,report)
    save(ap,{'rows_PRIVATE':rows})
    print('[PRIMARY]',pairs['PRIMARY_EQERR_MINUS_SLSQP_WOO_FULL'])
    print('[SECONDARY TRF]',pairs['SECONDARY_TRF_MINUS_SLSQP_WOO_FULL'])
    print('[COVERAGE]',coverage)
    print('[NESTED]',nested)
    print('[COMPUTE]',cost)
    print('Saved:',rp,'\nSaved:',ap)
    print('P4-B2B COMPLETE')

if __name__=='__main__':
    main()