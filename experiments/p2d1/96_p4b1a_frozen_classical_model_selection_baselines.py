"""P4-B1A / Experiment 96 — frozen classical model-selection baseline pilot.

Read-only, NO MuJoCo, NO SLSQP, NO new fit/action/scene, NO E4.
Uses identical previously fitted candidate residuals under each configuration.
This is a retrospective baseline feasibility/audit on a previously observed
P90 cohort, NOT independent confirmation. B1B must use new independent scenes.

Run:
  .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\96_p4b1a_frozen_classical_model_selection_baselines.py --smoke
  .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\96_p4b1a_frozen_classical_model_selection_baselines.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
OUT = RES / 'p4b1a'
PATHS = {
    'R90': RES / 'p4m2d2b' / 'p4m2d2b_v2_fresh_report.json',
    'A90': RES / 'p4m2d2b' / 'p4m2d2b_v2_fresh_audit_PRIVATE.json',
    'R92': RES / 'p4s1b' / 'p4s1b_fresh_report.json',
    'A92': RES / 'p4s1b' / 'p4s1b_fresh_audit_PRIVATE.json',
    'R93': RES / 'p4s1c' / 'p4s1c_solver_closure_report.json',
    'R95': RES / 'p4t1b' / 'p4t1b_competitor_report.json',
    'A95': RES / 'p4t1b' / 'p4t1b_competitor_audit_PRIVATE.json',
    'S90': HERE / '90_p4m2d2b_amplitude_dose_response_v2.py',
    'S92': HERE / '92_p4s1b_full_solver_budget_audit.py',
    'S95': HERE / '95_p4t1b_competitor_transition_scale_audit.py',
}
LAWS = ('elastic', 'deadzone', 'cubic')
CONFIGS = ('WWW', 'OOO', 'WOO')
ARMS = ('FROZEN', 'MATCHED5')
MODE_TO_LAW = {'elastic': 'elastic', 'deadzone': 'deadzone', 'nonlinear': 'cubic'}
K = {'elastic': 2, 'deadzone': 3, 'cubic': 3}
SELECTORS = ('TRAIN_PEM', 'KNOWN_SIGMA_AIC', 'KNOWN_SIGMA_BIC',
             'PROFILE_SIGMA_AIC', 'PROFILE_SIGMA_BIC')
TRAIN_ACTIONS = ('e1', 'e2', 'weak', 'strong')
# Each RevisionProbe.y_b contains 51 time steps x 2 displacement channels;
# the original training objective takes mean of all normalized residuals.
N_PER_ACTION = 51 * 2
N_TRAIN = len(TRAIN_ACTIONS) * N_PER_ACTION  # 408
N_BOOT = 2000
BOOT_SEED = 2029112201
GAP_EPS = 1e-15  # numeric guard only; never used to select thresholds


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def load(path):
    need(path.is_file(), f'Missing frozen input: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')


def key90(row):
    return (int(row['scene_id']), row['band'], row['mode_PRIVATE'],
            row['axis_PRIVATE'], row['level_PRIVATE'])


def key92(row):
    return tuple(row['key_PRIVATE'])


def truth(row):
    return MODE_TO_LAW[row['mode_PRIVATE']]


def domain(config, idx):
    return 'wide' if config[idx] == 'W' else 'outer_box'


def check_provenance():
    d = {k:load(PATHS[k]) for k in ('R90','A90','R92','A92','R93','R95','A95')}
    for src in ('R90', 'R92', 'R95'):
        need(d[src].get('smoke') is False, f'Formal report required: {src}')
    for src in ('R90', 'R92', 'R93', 'R95'):
        need(d[src].get('E4_accessed') is False, f'E4 flag: {src}')
    need(d['R90']['n_cases'] == d['R92']['n_cases'] == 384,
         'Not a complete 384-case cohort')
    need(d['R95']['n_p88']==384 and d['R95']['n_p90_nonlinear_m2']==96,
         'P95 source dimensions changed')
    need(d['R95']['new_fits'] == d['R95']['new_simulations'] == d['R95']['new_actions'] == 0,
         'P95 must be read-only')
    need(d['R93']['new_fits'] == d['R93']['new_simulations'] == 0,
         'P93 must be read-only')
    need(d['R90']['action_used_for_fit'] is False and
         d['R92']['heldout_used_for_fitting'] is False and
         d['R92']['truth_used_for_fitting'] is False,
         'Leakage detected in frozen history')
    links = (
        ('R90', 'R95', 'R90'), ('A90', 'R95', 'A90'),
        ('R93', 'R95', 'R93'), ('S95', 'R95', 'self_sha256'),
        ('S92', 'R92', 'SELF'), ('S90', 'R92', 'P90'),
        ('R90', 'R92', 'R90'), ('A90', 'R92', 'A90'),
        ('S92', 'R93', 'P92'),
    )
    for file_key, report_key, hash_key in links:
        r=d[report_key]
        expected=r['self_sha256'] if hash_key == 'self_sha256' else r['source_sha256'][hash_key]
        need(sha(PATHS[file_key]) == expected,
             f'SHA mismatch: {file_key} in {report_key}, expected {expected}')
    a90,a92=d['A90'],d['A92']
    need(isinstance(a90,list) and isinstance(a92,list) and len(a90)==len(a92)==384,
         'Private audits must contain 384 records')
    map90={key90(r):r for r in a90}
    map92={key92(r):r for r in a92}
    need(len(map90)==len(map92)==384 and set(map90)==set(map92),
         'P90/P92 case alignment mismatch')
    counts=Counter(k[0] for k in map90)
    need(counts==Counter({i:16 for i in range(24)}),
         'Expected 24 base scenes x 16 conditions')
    for k in map90:
        r90,r92=map90[k],map92[k]
        for cfg in CONFIGS:
            d90=r90['decisions'][cfg]['MID']
            d92=r92['decisions']['FROZEN'][cfg]['MID']
            need(d90['winner']==d92['winner'] and
                 bool(d90['available'])==bool(d92['available']),
                 f'Frozen Mid parity mismatch: {k}/{cfg}')
        for dom in ('wide','outer_box'):
            for law in LAWS:
                old=r90['fit_status_PRIVATE'][dom][law]
                original=r92['fit_audit_PRIVATE'][dom][law]['original']
                need(bool(old['converged'])==bool(original['converged']) and
                     bool(old['valid'])==bool(original['valid']),
                     f'Fit flag mismatch: {k}/{dom}/{law}')
                x,y=old['train_mse'],original['train_mse']
                if x is not None and y is not None:
                    need(math.isclose(float(x),float(y),rel_tol=1e-8,abs_tol=1e-8),
                         f'Train MSE mismatch: {k}/{dom}/{law}')
    return d, map90, map92, {k:sha(p) for k,p in PATHS.items()}


def candidate_fits(r90,r92,arm,config):
    result={}
    for i,law in enumerate(LAWS):
        dom=domain(config,i)
        if arm=='FROZEN':
            x=r90['fit_status_PRIVATE'][dom][law]
            valid=bool(x['available'])
            mse=x['train_mse']
            reason=x.get('reasons',[])
            origin='frozen_two_starts'
        else:
            x=r92['fit_audit_PRIVATE'][dom][law]['selected']
            valid=bool(x and x.get('converged') is True)
            mse=x['mse'] if x else None
            reason=[] if valid else ['unavailable_matched5_fit']
            origin=x.get('origin') if x else None
        if mse is None:
            valid=False
        else:
            try:
                mse=float(mse)
                if not math.isfinite(mse) or mse < 0:
                    valid=False
            except (ValueError,TypeError):
                valid=False
        result[law]={'available':valid,'mse':mse if valid else None,
                     'domain':dom,'origin':origin,'unavailable_reasons':reason}
    return result


def score(mse,law,selector):
    k=K[law]
    if selector=='TRAIN_PEM':
        return mse
    if selector=='KNOWN_SIGMA_AIC':
        return N_TRAIN*mse + 2*k
    if selector=='KNOWN_SIGMA_BIC':
        return N_TRAIN*mse + math.log(N_TRAIN)*k
    # Profiled-variance scores: for equal N, common additive terms cancel.
    # Log only when residual MSE > 0; undefined on a perfect exact fit.
    need(mse>0, f'Profile likelihood undefined on zero MSE: {law}')
    if selector=='PROFILE_SIGMA_AIC':
        return N_TRAIN*math.log(mse) + 2*k
    if selector=='PROFILE_SIGMA_BIC':
        return N_TRAIN*math.log(mse) + math.log(N_TRAIN)*k
    raise ValueError(selector)


def decide(fits,selector):
    if not all(fits[law]['available'] for law in LAWS):
        return {'available':False,'winner':None,'scores_PRIVATE':None,
                'selection_margin':None}
    scores={law:score(float(fits[law]['mse']),law,selector) for law in LAWS}
    need(all(math.isfinite(v) for v in scores.values()),
         'Nonfinite information criterion')
    order=sorted(LAWS,key=lambda l:(scores[l],LAWS.index(l)))
    return {'available':True,'winner':order[0],
            'scores_PRIVATE':scores,
            'selection_margin':float(scores[order[1]]-scores[order[0]])}


def build_case(k,r90,r92):
    tr=truth(r90)
    row={'key_PRIVATE':list(k),'scene_id':k[0],
         'mode_PRIVATE':k[2],'true_law_PRIVATE':tr,
         'arms':{},
         'historical_mid':{cfg:{'winner':r90['decisions'][cfg]['MID']['winner'],
                                'available':r90['decisions'][cfg]['MID']['available']}
                           for cfg in CONFIGS}}
    for arm in ARMS:
        row['arms'][arm]={}
        for cfg in CONFIGS:
            fits=candidate_fits(r90,r92,arm,cfg)
            row['arms'][arm][cfg]={
                'fits_PRIVATE':fits,
                'selectors':{selector:decide(fits,selector) for selector in SELECTORS}
            }
    return row


def evaluate(rows):
    n=len(rows)
    need(n>0,'Empty cohort')
    out={}
    for arm in ARMS:
        out[arm]={}
        for cfg in CONFIGS:
            out[arm][cfg]={}
            for selector in SELECTORS:
                ds=[r['arms'][arm][cfg]['selectors'][selector] for r in rows]
                avail=sum(d['available'] for d in ds)
                c=sum(d['winner']==r['true_law_PRIVATE'] for d,r in zip(ds,rows))
                by_mode={}
                for mode in MODE_TO_LAW:
                    subset=[(d,r) for d,r in zip(ds,rows) if r['mode_PRIVATE']==mode]
                    by_mode[mode]={'n':len(subset),
                                   'correct':sum(d['winner']==r['true_law_PRIVATE'] for d,r in subset),
                                   'available':sum(d['available'] for d,r in subset)}
                out[arm][cfg][selector]={
                    'correct':c,'available':avail,'total':n,
                    'coverage':avail/n,'accuracy_all':c/n,
                    'accuracy_available':c/avail if avail else None,
                    'by_mode_PRIVATE':by_mode,
                    'winner_counts_including_abstain':dict(Counter(d['winner'] or 'abstain' for d in ds))
                }
    return out


def scene_bootstrap(rows):
    # Retrospective sensitivity CIs: not a new confirmatory hypothesis test.
    grp=defaultdict(list)
    for r in rows:
        grp[r['scene_id']].append(r)
    need(len(grp)==24 and all(len(v)==16 for v in grp.values()),
         'Bootstrap expects 24 x 16 full cases')
    clusters=[grp[i] for i in range(24)]
    out={}
    rng=np.random.default_rng(BOOT_SEED)
    pick=rng.integers(0,24,size=(N_BOOT,24))
    for arm in ARMS:
        out[arm]={}
        for cfg in CONFIGS:
            diffs={}
            for selector in SELECTORS[1:]:
                per_scene=[]
                for scene in clusters:
                    v=[]
                    for r in scene:
                        cur=r['arms'][arm][cfg]['selectors'][selector]['winner']==r['true_law_PRIVATE']
                        pem=r['arms'][arm][cfg]['selectors']['TRAIN_PEM']['winner']==r['true_law_PRIVATE']
                        v.append(int(cur)-int(pem))
                    per_scene.append(sum(v)/len(v))
                arr=np.asarray(per_scene)
                sample=np.mean(arr[pick],axis=1)
                diffs[selector+'_minus_TRAIN_PEM']={
                    'difference_pp':float(100*np.mean(arr)),
                    'CI95_pp':[float(100*np.quantile(sample,.025)),float(100*np.quantile(sample,.975))]
                }
            out[arm][cfg]=diffs
    return {'unit':'base_scene_id','clusters':24,'resamples':N_BOOT,
            'seed':BOOT_SEED,'retrospective_not_predeclared':True,
            'contrasts':out}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    print('='*76)
    print('P4-B1A — FROZEN PEM / AIC / BIC BASELINE PILOT (READ-ONLY)')
    print('='*76)
    print('Smoke:',args.smoke,'| E4 sealed | No fits or simulations')
    data,a90,a92,shas=check_provenance()
    print('[PASS] Hash lineage, complete aligned 384 P90/P92 cases, E4 sealed')
    keys=sorted(a90)
    if args.smoke:
        keys=[k for k in keys if k[0]==0]
    rows=[build_case(k,a90[k],a92[k]) for k in keys]
    need(len(rows)==(16 if args.smoke else 384),'Wrong output matrix')
    stats=evaluate(rows)
    if not args.smoke:
        ci=scene_bootstrap(rows)
        print('[PASS] 384 cases, 30 baseline cells, 24-scene retrospective CIs')
    else:
        ci=None
        print('[PASS] 16-case interface smoke; no scientific inference')
    report={
        'protocol':'P4-B1A training-only classical model selection baseline pilot',
        'smoke':args.smoke,'n_cases':len(rows),'n_scenes':1 if args.smoke else 24,
        'n_training_scalar_residuals':N_TRAIN,
        'training_actions':list(TRAIN_ACTIONS),
        'candidate_parameters':K,
        'fit_arms':{'FROZEN':'original two start fits; abstentions preserved',
                    'MATCHED5':'best converged among original2 and fixed extra3 starts'},
        'score_definitions':{
            'TRAIN_PEM':'training normalized MSE (already the original optimizer objective)',
            'KNOWN_SIGMA_AIC':'n * normalized_MSE + 2*k',
            'KNOWN_SIGMA_BIC':'n * normalized_MSE + ln(n)*k',
            'PROFILE_SIGMA_AIC':'n * ln(normalized_MSE) + 2*k',
            'PROFILE_SIGMA_BIC':'n * ln(normalized_MSE) + ln(n)*k',
        },
        'metrics':stats,'bootstrap_exploratory':ci,
        'original_mid_reference':{
            arm:{cfg:data['R92']['metrics'][arm][cfg]['MID']
                 for cfg in CONFIGS} for arm in ARMS
        } if not args.smoke else None,
        'historical_sensitivity_reference':{
            'P90_Nonlinear_M2_recall_full_minus_half':data['R90']['predeclared_effects']['PRIMARY_WOO_nonlinear_M2_recall_full_minus_half'],
            'P95_posthoc_relative_margin_mean':data['R95']['P90_nonlinear_M2']['delta_relative_margin']['mean'],
        },
        'source_sha256':shas,
        'E4_accessed':False,'new_fits':0,'new_simulations':0,
        'new_actions':0,'truth_used_for_fit_or_selection':False,
        'heldout_mid_half_full_used_for_selection':False,
        'model_selection_threshold_tuning':False,
        'cohort_previously_examined':True,
        'limitations':[
            'Post-hoc P90 cohort, baseline feasibility only; requires pre-frozen fresh-scene B1B confirmation',
            'Training-only AIC/BIC uses four training actions; MID validation-based selector sees extra observations',
            'AIC/BIC are Gaussian pseudo-information criteria unless temporal covariance and likelihood validated',
            'Known-sigma scoring assumes residual variance 0.005^2 and independence; effective residual variance may differ',
            'Profile-variance AIC/BIC uses a free Gaussian residual scale; imperfect under correlated time samples',
            'k=2 or 3 counts nominal candidate parameters only; active-boundary effective degrees of freedom unaccounted',
            'AIC/BIC are scoring rules applied to existing PEM fits; not separate optimizers or new physics models',
            'Equal start counts are not equal function evaluation counts',
            'Truth labels enter only after each model choice, to evaluate accuracy and confusion',
            'No structural identifiability proof, no hardware transfer claim, no E4',
        ],
    }
    audit={'rows_PRIVATE':rows}
    tag='smoke' if args.smoke else 'pilot'
    rp=OUT/f'p4b1a_{tag}_report.json'
    ap=OUT/f'p4b1a_{tag}_audit_PRIVATE.json'
    write(rp,report);write(ap,audit)
    if not args.smoke:
        for arm in ARMS:
            for cfg in CONFIGS:
                labels=[]
                for selector in SELECTORS:
                    m=stats[arm][cfg][selector]
                    labels.append(f'{selector}:{m["correct"]}/{m["total"]}')
                print(f'{arm}/{cfg}: '+' | '.join(labels))
        print('[CAUTION] Pseudo-AIC/BIC exploratory scores, not confirmatory; B1B fresh-scene required')
    print('Saved:',rp)
    print('Saved:',ap)
    print('P4-B1A COMPLETE')

if __name__=='__main__':
    main()
