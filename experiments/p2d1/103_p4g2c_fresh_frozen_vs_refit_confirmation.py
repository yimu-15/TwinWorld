"""Experiment #103 / P4-G2C — FRESH-SCENE paired parameter-source confirmation.

Prospective protocol frozen BEFORE first fresh outcome. New independent base
scenes; unchanged P98 two-arm RK4 generator; unchanged P22 A-conditioned B
predictor and P77/P92 2+3-start SLSQP. IDEAL_ODE and SHIFTED_ODE observations
are GENERATED FOR THE SAME physical scene with paired observation noise.

Comparison on IDENTICAL, held-out SHIFTED observations:
 A. PAIRED-IDEAL-FROZEN: fit candidates on same-scene IDEAL_ODE four training
    actions, freeze every fitted candidate parameter, predict SHIFTED heldouts.
 B. SHIFTED-REFIT: fit candidates on same-scene SHIFTED_ODE four training
    actions, predict those same SHIFTED heldouts.

CRITICAL: Arm A uses PRIVILEGED same-scene IDEAL training observations.
This is a controlled parameter-source/selection-inversion experiment, NOT
zero-shot transfer or a deployable head-to-head benchmark. Both arms use
matched five-start SLSQP fitting, although real function-evaluation costs
need not match. True law is never given to fitting or model selection.

Primary predeclared endpoint: A-B difference in MID_PLUS_FULL mechanism
accuracy, all 384 conditions, MATCHED5/WOO. Secondary: Elastic-only
MID_PLUS_FULL difference; Nonlinear-M2 MID_PLUS_FULL vs MID_PLUS_HALF dose
for both A and B; other law strata and margins exploratory. Unavailable
fit results are errors in full denominators. Bootstrap 24 scene clusters,
2000 resamples; two-sided percentile intervals. E4 NEVER READ.

From the root of TwinWorld_P0:
 .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\103_p4g2c_fresh_frozen_vs_refit_confirmation.py --smoke
 .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\103_p4g2c_fresh_frozen_vs_refit_confirmation.py
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
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
BASE = HERE / 'results'
OUT = BASE / 'p4g2c'
P98 = HERE / '98_p4g1a_independent_physics_generator_feasibility.py'
P99 = HERE / '99_p4g1b_independent_generator_identification.py'
P102 = HERE / '102_p4g2b_frozen_vs_refit_competition_audit.py'
R99 = BASE / 'p4g1b' / 'p4g1b_fresh_report.json'
R101 = BASE / 'p4g2a' / 'p4g2a_diagnostic_report.json'
R102 = BASE / 'p4g2b' / 'p4g2b_closure_report.json'
M102 = BASE / 'p4g2b' / 'p4g2b_source_manifest.json'

# All seeds and endpoints frozen before generating fresh data.
SCENE_SEED = 20300117
SMOKE_SCENE_SEED = 20310107
NOISE_SEED = 2030011701
SMOKE_NOISE_SEED = 2031010701
BOOT_SEED = 2030011702
N_SCENES = 24
N_BOOT = 2000
LAWS = ('elastic', 'deadzone', 'cubic')
METHODS = ('MID_PLUS_HALF', 'MID_PLUS_FULL')
TRAIN = ('e1', 'e2', 'weak', 'strong')
N_TRAIN_RESIDUALS = 204
GAP_TOL = 1e-6
PRIMARY_CONFIG = 'WOO'
PRIMARY_BUDGET = 'MATCHED5'
MIN_COVERAGE = 0.90  # interpretability flag, no exclusion


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def digest_file(path):
    need(path.is_file(), f'Required frozen source absent: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    need(path.is_file(), f'Required input absent: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def objhash(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False,
                                  allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def preflight():
    paths = {'P98': P98, 'P99': P99, 'P102': P102, 'R99': R99,
             'R101': R101, 'R102': R102, 'M102': M102, 'SELF': SELF}
    lineage = {k: digest_file(v) for k, v in paths.items()}
    r99, r101, r102, m102 = read(R99), read(R101), read(R102), read(M102)
    need(r99['n_scenes'] == 24 and r99['n_cases_per_generator'] == 384 and
         r99['E4_accessed'] is False and r99['source_sha256']['SELF'] == lineage['P99'] and
         r99['source_sha256']['P98'] == lineage['P98'], 'P99 provenance changed')
    need(r101['n_cases'] == 384 and r101['refit_performed'] is False and
         r101['E4_accessed'] is False and r101['sources']['P99'] == lineage['P99'],
         'P101 exploratory source changed')
    need(r102['smoke'] is False and r102['n_cases'] == 384 and
         r102['scientific_status'] == 'RETROSPECTIVE_DIAGNOSTIC_ONLY' and
         r102['new_fits'] == 0 and r102['E4_accessed'] is False and
         r102['score_cells_recomputed'] == 4608,
         'P102 retrospective audit not finalized')
    need(m102['current_script_sha256'] == lineage['P102'] and
         r102['source_sha256']['R101'] == lineage['R101'] and
         r102['source_sha256']['R99'] == lineage['R99'],
         'P102 source or input hash mismatch')
    # Reuse the full validated historical interface and laws; no branch selection
    # based on old outcomes or test truths.
    p99 = run_path(str(P99))
    gen, p77, p92, p22, commands, src99, historical = p99['preflight']()
    need(gen['ARMS'] == r99['ODE_arms'] and
         p99['N_TRAIN_RESIDUALS'] == N_TRAIN_RESIDUALS and
         p99['TRAIN'] == TRAIN and p99['LAWS'] == LAWS and
         p99['SIGMA'] == 0.005 and p99['N_SCENES'] == N_SCENES,
         'Frozen identification contract differs')
    starts = {law: p92['common_starts'](p77, law) for law in LAWS}
    need(all(len(starts[l]) == 3 and len(p77['initial_points'](l)) == 2 for l in LAWS),
         'Two-plus-three optimizer start count changed')
    return p99, gen, p77, p92, p22, commands, starts, lineage, historical


def generate_scenes(gen, historical, smoke, p99):
    rng = np.random.default_rng(SMOKE_SCENE_SEED if smoke else SCENE_SEED)
    n = 1 if smoke else N_SCENES
    scenes = [{b: gen['sample_scene'](rng, b) for b in gen['BANDS']} for _ in range(n)]
    fps = [gen['digest_object'](pair[band]) for pair in scenes for band in gen['BANDS']]
    need(len(fps) == len(set(fps)) == 2*n, 'Duplicate fresh base scene')
    r98, smoke98, r97 = historical
    r99 = read(R99)
    previous = set(r98['scene_fingerprints_PRIVATE']) | \
               set(smoke98['scene_fingerprints_PRIVATE']) | \
               set(r99['scene_fingerprints_PRIVATE']) | \
               set(r97['scene_fingerprints'])
    # P99 smoke input was not the formal cohort; regenerate only its scene
    # fingerprints, never fitted results, and require disjointness as well.
    old_smoke_rng = np.random.default_rng(p99['SMOKE_SCENE_SEED'])
    p99_smoke = [gen['sample_scene'](old_smoke_rng, band)
                 for band in gen['BANDS']]
    previous.update(gen['digest_object'](x) for x in p99_smoke)
    # An internal full-budget one-case interface pilot used this seed; its
    # result was inspected, so the entire pilot-seed cohort is excluded.
    excluded_pilot_rng = np.random.default_rng(20300107)
    excluded_pilot_bases = [gen['sample_scene'](excluded_pilot_rng, band)
                            for band in gen['BANDS']]
    previous.update(gen['digest_object'](x) for x in excluded_pilot_bases)
    need(not previous.intersection(fps), 'Fresh scenes overlap P98/P99/pilot sources')
    # Exclude both smoke sets from formal cohort without fitting on them.
    if not smoke:
        smoke_rng = np.random.default_rng(SMOKE_SCENE_SEED)
        smoke_bases = [gen['sample_scene'](smoke_rng, band) for band in gen['BANDS']]
        need(not set(fps).intersection(gen['digest_object'](s) for s in smoke_bases),
             'Fresh cohort shares a base scene with smoke')
    return scenes, fps


def winner_from_scores(scores):
    need(set(scores) == set(LAWS) and all(math.isfinite(float(v)) and v >= 0
                                           for v in scores.values()),
         'Invalid candidate model selection scores')
    return min(LAWS, key=lambda law: (scores[law], LAWS.index(law)))


def scored_decision(fitted, probe_dict, config='WOO'):
    """Candidate scores on external observation; never refit the parameters."""
    chosen = {law: fitted['MATCHED5']['wide' if config[LAWS.index(law)] == 'W'
                                    else 'outer_box'][law] for law in LAWS}
    if any(f is None for f in chosen.values()):
        return {m: {'available': False, 'winner': None} for m in METHODS}
    singles = {law: {} for law in LAWS}
    for law, fit in chosen.items():
        params = fit['params_PRIVATE']
        for action in ('mid', 'HALF', 'FULL'):
            pred = np.asarray(probe_dict[action].predict(law, params), dtype=float)
            obs = np.asarray(probe_dict[action].y_b, dtype=float)
            need(pred.shape == obs.shape == (51,) and np.isfinite(pred).all(),
                 'Invalid 51-sample prediction in heldout actions')
            singles[law][action] = float(1000*np.sqrt(np.mean((pred-obs)**2)))
    ans = {}
    for method, other in (('MID_PLUS_HALF', 'HALF'), ('MID_PLUS_FULL', 'FULL')):
        scores = {law: float(math.hypot(singles[law]['mid'], singles[law][other]) /
                             math.sqrt(2)) for law in LAWS}
        ans[method] = {'available': True, 'winner': winner_from_scores(scores),
                       'scores_mm_PRIVATE': scores}
    return ans


def decision_correct(d, truth):
    return int(d['available'] and d['winner'] == truth)


def diagnostics(decisions, true_law):
    result = {}
    for name in METHODS:
        d = decisions[name]
        if not d['available']:
            result[name] = {'correct': False, 'true_law_rms_mm': None,
                            'relative_margin': None, 'squared_margin_mm2': None}
            continue
        losses = {law: d['scores_mm_PRIVATE'][law]**2 for law in LAWS}
        right = losses[true_law]
        wrong = min(losses[law] for law in LAWS if law != true_law)
        denom = right + wrong
        result[name] = {'correct': bool(d['winner'] == true_law),
                        'true_law_rms_mm': math.sqrt(right),
                        'relative_margin': (wrong-right)/denom if denom else 0,
                        'squared_margin_mm2': wrong-right}
    return result


def run_case(p99, gen, p77, p92, Probe, commands, starts, base,
             sid, band, mode, axis, level, noise_seed):
    scene = gen['apply_condition'](base, mode, axis, level)
    probes, clean, fingerprints = p99['sample_noisy_probes'](
        gen, Probe, scene, mode, commands, sid, band, axis, level, noise_seed)
    # Fit all original law/domain candidates for numerical parity and bounds
    # auditing; neither optimization receives the true law.
    ideal_fits, ideal_logs = p99['fit_all'](probes['IDEAL_ODE'], p77, p92, starts)
    target_fits, target_logs = p99['fit_all'](probes['SHIFTED_ODE'], p77, p92, starts)
    source_dec = scored_decision(ideal_fits, probes['SHIFTED_ODE'])
    target_dec = scored_decision(target_fits, probes['SHIFTED_ODE'])
    # Independently compare results to frozen P99 selector on same test data.
    refit_dec = p99['select_results'](target_fits,
                lambda mse,law,meth: 0.0 if meth == 'TRAIN_PEM' else 0.0)
    for method in METHODS:
        old = refit_dec['MATCHED5']['WOO'][method]
        new = target_dec[method]
        need(old['available'] == new['available'] and old['winner'] == new['winner'],
             f'P99 selector parity failed: {method}')
        if new['available']:
            for law in LAWS:
                need(math.isclose(old['scores_mm_PRIVATE'][law],
                                  new['scores_mm_PRIVATE'][law],rel_tol=1e-9,abs_tol=1e-7),
                     'Target refit score parity failed')
    nested = {}
    for ga, fits in (('IDEAL_ODE', ideal_fits),('SHIFTED_ODE',target_fits)):
        nested[ga] = {}
        for law in LAWS:
            w = fits['MATCHED5']['wide'][law]
            o = fits['MATCHED5']['outer_box'][law]
            gap = None if w is None or o is None else float(w['mse']-o['mse'])
            nested[ga][law] = {'gap':gap, 'violation':bool(gap is not None and gap>GAP_TOL)}
    truth = p99['TRUTH'][mode]  # used AFTER fitting/selection only
    return {'key_PRIVATE':[sid,band,mode,axis,level], 'scene_id':sid,
            'true_law_PRIVATE':truth, 'mode_PRIVATE':mode,
            'condition_PRIVATE':level,
            'physical_scene_sha256_PRIVATE':gen['digest_object'](scene),
            'clean_fingerprints_PRIVATE':fingerprints,
            'decisions':{'PAIRED_IDEAL_FIXED':source_dec,'SHIFTED_REFIT':target_dec},
            'post_selection_diagnostics_PRIVATE':{
                 'PAIRED_IDEAL_FIXED': diagnostics(source_dec, truth),
                 'SHIFTED_REFIT': diagnostics(target_dec, truth)},
            'candidate_fits_PRIVATE':{'IDEAL_ODE':ideal_fits,'SHIFTED_ODE':target_fits},
            'fit_logs_PRIVATE':{'IDEAL_ODE':ideal_logs,'SHIFTED_ODE':target_logs},
            'nested_PRIVATE':nested}


def paired_effect(rows, predicate, arm_a, method_a, arm_b, method_b):
    group = [r for r in rows if predicate(r)]
    need(group and all('scene_id' in r for r in group), 'Empty paired subgroup')
    per_scene = defaultdict(list)
    status = Counter()
    for row in group:
        truth = row['true_law_PRIVATE']
        a = decision_correct(row['decisions'][arm_a][method_a],truth)
        b = decision_correct(row['decisions'][arm_b][method_b],truth)
        status[(a,b)] += 1
        per_scene[row['scene_id']].append(a-b)
    # Each of the 24 base scenes contributes equally to every predeclared
    # endpoint. The bootstrap never treats 384 within-scene cases as IID.
    need(len(per_scene)==N_SCENES and len({len(v) for v in per_scene.values()})==1,
         'Cluster sizes or completeness changed')
    values = np.asarray([np.mean(per_scene[i]) for i in range(N_SCENES)])
    return {'n':len(group), 'a_correct':sum(k[0]*v for k,v in status.items()),
            'b_correct':sum(k[1]*v for k,v in status.items()),
            'both_correct':status[(1,1)], 'a_only':status[(1,0)],
            'b_only':status[(0,1)], 'both_wrong':status[(0,0)],
            'net':status[(1,0)]-status[(0,1)],
            'effect':float(np.mean(values)), '_scene_values':values}


def summarize(rows):
    any_case=lambda x: True
    is_elastic=lambda x:x['true_law_PRIVATE']=='elastic'
    is_nm2=lambda x:x['true_law_PRIVATE']=='cubic' and x['condition_PRIVATE']=='M2'
    pairs = {
      'PRIMARY_SHIFTED_FIXED_MINUS_REFIT_ALL_FULL':paired_effect(rows,any_case,
        'PAIRED_IDEAL_FIXED','MID_PLUS_FULL','SHIFTED_REFIT','MID_PLUS_FULL'),
      'SECONDARY_ELASTIC_FIXED_MINUS_REFIT_FULL':paired_effect(rows,is_elastic,
        'PAIRED_IDEAL_FIXED','MID_PLUS_FULL','SHIFTED_REFIT','MID_PLUS_FULL'),
      'SECONDARY_NM2_FIXED_FULL_MINUS_HALF':paired_effect(rows,is_nm2,
        'PAIRED_IDEAL_FIXED','MID_PLUS_FULL','PAIRED_IDEAL_FIXED','MID_PLUS_HALF'),
      'SECONDARY_NM2_REFIT_FULL_MINUS_HALF':paired_effect(rows,is_nm2,
        'SHIFTED_REFIT','MID_PLUS_FULL','SHIFTED_REFIT','MID_PLUS_HALF'),
    }
    # Predeclared paired difference of dose effects on same 96 NM2 cases.
    grouped=defaultdict(list)
    for r in rows:
        if is_nm2(r):
            t = r['true_law_PRIVATE']
            ds = r['decisions']
            fixed = decision_correct(ds['PAIRED_IDEAL_FIXED']['MID_PLUS_FULL'],t) - \
                    decision_correct(ds['PAIRED_IDEAL_FIXED']['MID_PLUS_HALF'],t)
            refit = decision_correct(ds['SHIFTED_REFIT']['MID_PLUS_FULL'],t) - \
                    decision_correct(ds['SHIFTED_REFIT']['MID_PLUS_HALF'],t)
            grouped[r['scene_id']].append(fixed-refit)
    need(len(grouped)==24 and all(len(x)==4 for x in grouped.values()),
         'Nonlinear M2 difference-of-effects denominator changed')
    dd=np.asarray([np.mean(grouped[s]) for s in range(24)])
    pairs['SECONDARY_NM2_FIXED_MINUS_REFIT_DOSE_DD']={
        'n':96,'net':int(round(float(np.sum(dd)*4))),
        'effect':float(np.mean(dd)), '_scene_values':dd}
    draw=np.random.default_rng(BOOT_SEED).integers(0,N_SCENES,size=(N_BOOT,N_SCENES))
    for data in pairs.values():
        v=data.pop('_scene_values')
        data['scene_cluster_CI95']=[float(x) for x in np.quantile(v[draw].mean(1),[.025,.975])]
    counts={}
    for arm in ('PAIRED_IDEAL_FIXED','SHIFTED_REFIT'):
        counts[arm]={}
        for m in METHODS:
            decisions=[r['decisions'][arm][m] for r in rows]
            counts[arm][m]={
                'correct':sum(decision_correct(d,r['true_law_PRIVATE']) for d,r in zip(decisions,rows)),
                'available':sum(int(d['available']) for d in decisions),
                'n':len(rows),
                'by_law':{law:{'n':sum(r['true_law_PRIVATE']==law for r in rows),
                      'correct':sum(decision_correct(d,r['true_law_PRIVATE']) for d,r in zip(decisions,rows)
                                    if r['true_law_PRIVATE']==law)} for law in LAWS}}
    nested={ga:dict(Counter(law for row in rows for law in LAWS
               if row['nested_PRIVATE'][ga][law]['violation']))
               for ga in ('IDEAL_ODE','SHIFTED_ODE')}
    nfev={ga:{domain:{'original':sum(row['fit_logs_PRIVATE'][ga][domain][law]['original']['nfev']
                                         for row in rows for law in LAWS),
                       'extra':sum(sum(int(v['nfev']) for v in row['fit_logs_PRIVATE'][ga][domain][law]['extra'])
                                     for row in rows for law in LAWS)}
                for domain in ('wide','outer_box')}
          for ga in ('IDEAL_ODE','SHIFTED_ODE')}
    # Post-decision truth-conditioned model-error summaries: do not inform
    # runtime model selection. No arbitrary selection of favorable mechanism.
    fidelity={}
    for arm in ('PAIRED_IDEAL_FIXED','SHIFTED_REFIT'):
        fidelity[arm]={}
        for law in LAWS:
            points=[r['post_selection_diagnostics_PRIVATE'][arm]['MID_PLUS_FULL']
                    for r in rows if r['true_law_PRIVATE']==law]
            arr=np.array([z['true_law_rms_mm'] for z in points
                         if z['true_law_rms_mm'] is not None])
            margins=np.array([z['relative_margin'] for z in points
                              if z['relative_margin'] is not None])
            fidelity[arm][law]={'available':len(arr),'median_true_law_rms_mm':
                                   float(np.median(arr)) if len(arr) else None,
                               'median_relative_margin':
                                   float(np.median(margins)) if len(margins) else None}
    return pairs,counts,nested,nfev,fidelity


def smoke_run(p99,gen,p77,p92,Probe,commands,starts,scenes,lineage):
    sid=0;band='original';mode='elastic';axis='matched';level='M0'
    scene=gen['apply_condition'](scenes[0][band],mode,axis,level)
    probes, clean, hashes=p99['sample_noisy_probes'](
        gen, Probe, scene, mode, commands, sid,band,axis,level,SMOKE_NOISE_SEED)
    evaluations={}
    for ga in ('IDEAL_ODE','SHIFTED_ODE'):
        training=[probes[ga][a] for a in TRAIN]
        old=p77['fit_domain'](training,'elastic','wide')
        obj=p92['objective'](training,'elastic',p99['SIGMA'])
        one=p92['extra_fit'](obj,p77,'elastic','wide',starts['elastic'][:1])
        new=p92['selected_fit'](old,one,obj,p77,'elastic','wide')
        need(len(one)==1 and int(old['nfev'])>=0 and int(one[0]['nfev'])>=0,
             'Small optimizer-call smoke failed')
        need(probes[ga]['mid'].a_half.shape==(101,) and probes[ga]['mid'].y_b.shape==(51,),
             'Probe shape regression')
        evaluations[ga]={'original_converged':bool(old['converged']),
                         'extra_converged':bool(one[0]['converged']),
                         'available':new is not None,
                         'original_nfev':int(old['nfev']),
                         'extra_nfev':int(one[0]['nfev'])}
    rpt={'protocol':'P4-G2C out-of-cohort interface smoke; NOT a scientific result',
         'smoke':True,'n_base_scenes':1,'n_test_cases':1,
         'source_sha256':lineage,'E4_accessed':False,
         'new_classification_outcomes':0,'full_matched5_budget_used':False}
    audit={'clean_ODE_sha256_PRIVATE':hashes,'optimizer_interface_PRIVATE':evaluations}
    rp=OUT/'p4g2c_smoke_report.json';ap=OUT/'p4g2c_smoke_audit_PRIVATE.json'
    write(rp,rpt);write(ap,audit)
    print('[PASS] Two ODE arms, paired noise, B-51/A-101 probe, reduced optimizer smoke')
    print('[NOTE] Smoke uses 2 original + 1 extra start in Elastic/Wide; no scientific estimate')
    print('Saved:',rp,'\nSaved:',ap)
    print('P4-G2C SMOKE COMPLETE')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    print('='*78)
    print('P4-G2C — FRESH PAIRED IDEAL-FROZEN vs SHIFTED-REFIT CONFIRMATION')
    print('='*78)
    print('Smoke:',args.smoke,'| E4 sealed | same-scene IDEAL source is PRIVILEGED')
    p99,gen,p77,p92,p22,commands,starts,lineage,historical=preflight()
    scenes, fingerprints=generate_scenes(gen,historical,args.smoke,p99)
    print('[PASS] P98/P99/P101/P102 hashes, 204 B training residuals, nested bounds')
    print('[PASS] Independent fresh scenes vs P98/P99 + smoke, no E4')
    print('[PROTOCOL] 24x2x8 paired conditions; two ODE branches; each law/domain uses 2+3 starts')
    print('[CRITICAL] Same-scene IDEAL source training is privileged; no deployable zero-shot claim')
    if args.smoke:
        smoke_run(p99,gen,p77,p92,p22['RevisionProbe'],commands,starts,scenes,lineage)
        return
    meta={'source_sha256':lineage,'scene_seed':SCENE_SEED,
          'smoke_scene_seed':SMOKE_SCENE_SEED,
          'noise_seed':NOISE_SEED,'boot_seed':BOOT_SEED,
          'scene_fingerprints_PRIVATE':fingerprints,
          'protocol':'P4-G2C 24 new scenes; paired IDEAL source vs SHIFTED refit',
          'primary':'MATCHED5 WOO MID_PLUS_FULL fixed-ideal minus shifted-refit all 384',
          'paired_source_privileged':True,
          'n_training_scalar_residuals':N_TRAIN_RESIDUALS,
          'E4_accessed':False}
    mf=OUT/'p4g2c_source_manifest.json'
    if mf.exists():
        need(read(mf)==meta,'Formal manifest changed; do not combine checkpoints')
    else:
        write(mf,meta)
    rows=[]
    for sid,base_pair in enumerate(scenes):
        cp=OUT/'checkpoints'/f'scene_{sid:02d}.json'
        if cp.exists():
            pack=read(cp)
            need(pack['scene_id']==sid and pack['source_sha256']==lineage and
                 pack['base_fingerprints']==fingerprints[2*sid:2*sid+2] and
                 pack['rows_sha256']==objhash(pack['rows']) and len(pack['rows'])==16,
                 f'Checkpoint source/identity/row hash changed: scene {sid}')
            block=pack['rows']
            print(f'[RESUME] scene {sid+1}/24: 16 paired cases',flush=True)
        else:
            block=[]
            for band in p99['BANDS']:
                for mode in p99['MODES']:
                    for axis,level in p99['CONDITIONS'][mode]:
                        block.append(run_case(p99,gen,p77,p92,p22['RevisionProbe'],
                          commands,starts,base_pair[band],sid,band,mode,axis,level,
                          NOISE_SEED))
            need(len(block)==16,'Wrong physical conditions per scene')
            write(cp,{'scene_id':sid,'source_sha256':lineage,
                      'base_fingerprints':fingerprints[2*sid:2*sid+2],
                      'rows_sha256':objhash(block),'rows':block})
            print(f'[DONE] scene {sid+1}/24: 16 paired cases',flush=True)
        rows.extend(block)
    need(len(rows)==384 and len({tuple(q['key_PRIVATE']) for q in rows})==384,
         'Wrong total cohort size or duplicate case')
    for sid in range(N_SCENES):
        need(sum(x['scene_id']==sid for x in rows)==16,'Incomplete 16-case scene')
    pairs,counts,nested,nfev,fidelity=summarize(rows)
    coverage={arm:{method:counts[arm][method]['available']/384 for method in METHODS}
              for arm in ('PAIRED_IDEAL_FIXED','SHIFTED_REFIT')}
    coverage_ok=all(coverage[arm][method]>=MIN_COVERAGE for arm in coverage
                    for method in coverage[arm])
    report={'protocol':'P4-G2C pre-outcome frozen paired parameter-source fresh confirmation',
            'smoke':False,'n_base_scenes':N_SCENES,'n_cases':384,
            'n_generator_case_pairs':768,
            'n_fit_units':384*2*3*2,
            'sources':lineage,'scene_seed':SCENE_SEED,'noise_seed':NOISE_SEED,
            'scene_fingerprints_PRIVATE':fingerprints,
            'generator_arms':gen['ARMS'],
            'inference':'P22 A-conditioned B, P77/P92 five-start SLSQP',
            'fit_budget':{'original_starts':2,'extra_common_starts':3,
                         'maxiter':150,'n_training_B_residuals':204},
            'protocol_fixed_before_fresh_outcome':True,
            'primary_is_privileged_paired_source_ablation':True,
            'primary_and_secondary_paired_contrasts':pairs,
            'aggregate_classification':counts,
            'full_denominator_coverage':coverage,
            'passes_90pct_interpretability_guard':coverage_ok,
            'nested_violations':nested,
            'function_evaluations':nfev,
            'postdecision_true_law_diagnostics':fidelity,
            'bootstrap':{'unit':'base_scene','n_clusters':24,
                         'resamples':N_BOOT,'seed':BOOT_SEED,
                         'CI_method':'paired scene-cluster percentile; two-sided'},
            'E4_accessed':False,'truth_used_for_fit_or_selection':False,
            'heldout_used_for_fit':False,'adaptive_tuning':False,
            'limitations':[
               'Fixed-IDEAL uses privileged same-scene IDEAL observations for source fitting: NOT a deployable transfer baseline',
               'Two branches of one standalone RK4 implementation; no hardware or independent second engine',
               'Both source and target fits use all four source-specific training actions; evaluation on same SHIFTED heldouts',
               'This confirms a controlled parameter-source contrast, not a causal effect of parameter freezing',
               'Original candidate family omits SHIFTED ground drag/actuator lag',
               'MATCHED5 matches optimizer starts/maxiter, not actual function evaluations',
               'SLSQP local minima/nested-domain violations possible',
               '24 scene-cluster Bootstrap; case-level confidence bounds are inappropriate',
               'Postdecision true-law margins are retrospective only',
               'All abstentions retained as errors; no E4 data used'],
            'environment':{'python':sys.version,'numpy':np.__version__,
                           'scipy':scipy.__version__,'platform':platform.platform()}}
    rp=OUT/'p4g2c_fresh_report.json';ap=OUT/'p4g2c_fresh_audit_PRIVATE.json'
    write(rp,report);write(ap,rows)
    print('[PRIMARY]',pairs['PRIMARY_SHIFTED_FIXED_MINUS_REFIT_ALL_FULL'])
    print('[SECONDARY ELASTIC]',pairs['SECONDARY_ELASTIC_FIXED_MINUS_REFIT_FULL'])
    print('[SECONDARY NM2]',pairs['SECONDARY_NM2_FIXED_FULL_MINUS_HALF'],
          pairs['SECONDARY_NM2_REFIT_FULL_MINUS_HALF'])
    print('[COVERAGE]',coverage,'passes_guard:',coverage_ok)
    print('[NESTED]',nested,'[E4] sealed')
    print('Saved:',rp,'\nSaved:',ap)
    print('P4-G2C COMPLETE')


if __name__=='__main__':
    main()
