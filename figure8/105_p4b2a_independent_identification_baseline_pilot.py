"""#105 P4-B2A: historical-cohort feasibility pilot for independent ID baselines.

NOT a fresh-scene or confirmatory comparison. Replays the immutable P103 ODE
SHIFTED observations and its original SLSQP results without modifying them.
Two truth-blind estimators see EXACTLY the same four (204 B-position sample)
training actions as SLSQP:
  EQERR_SG: independent derivative/equation-error regression (Savitzky-Golay,
            bounded linear regression; fixed gap grid for deadzone).
  TRF_OUT:  independent trust-region-reflective trajectory-output-error least
            squares, SAME 2+3 start points as historical SLSQP.
Both are scored only on the original P103 heldout MID+HALF / MID+FULL, via the
same A-conditioned B predictor and WOO domain configuration. All models use
Elastic/Deadzone/Cubic candidates and identical bounds. E4 is never read.

Prior P103 is an ALREADY OBSERVED cohort, and this is only an interface and
compute-budget pilot. No accuracy or statistics from this pilot can be sold as
independent evidence; the next step must freeze a fresh-scene B2B protocol.

Giant usage:
 .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\105_p4b2a_independent_identification_baseline_pilot.py --smoke
 .\\Giant\\Scripts\\python.exe .\\experiments\\p2d1\\105_p4b2a_independent_identification_baseline_pilot.py
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from runpy import run_path
import sys
import time

import numpy as np
import scipy
from scipy.optimize import least_squares, lsq_linear
from scipy.signal import savgol_filter

SELF = Path(__file__).resolve()
HERE = SELF.parent
BASE = HERE / 'results'
OUT = BASE / 'p4b2a'
P98 = HERE / '98_p4g1a_independent_physics_generator_feasibility.py'
P99 = HERE / '99_p4g1b_independent_generator_identification.py'
P103 = HERE / '103_p4g2c_fresh_frozen_vs_refit_confirmation.py'
P104 = HERE / '104_p4g2d_fresh_confirmation_closure_audit.py'
R103 = BASE / 'p4g2c' / 'p4g2c_fresh_report.json'
A103 = BASE / 'p4g2c' / 'p4g2c_fresh_audit_PRIVATE.json'
M103 = BASE / 'p4g2c' / 'p4g2c_source_manifest.json'
R104 = BASE / 'p4g2d' / 'p4g2d_closure_report.json'

LAWS = ('elastic', 'deadzone', 'cubic')
DOMAINS = ('wide', 'outer_box')
METHODS = ('MID_PLUS_HALF', 'MID_PLUS_FULL')
TRAIN = ('e1', 'e2', 'weak', 'strong')
ARMS = ('EQERR_SG', 'TRF_OUT', 'SLSQP_MATCHED5_REFERENCE')
SIGMA = 0.005
DT = 0.02
N_SAMPLES = 204
SG_WINDOW, SG_POLY, SG_EDGE = 11, 3, 5
DEADZONE_GRID = 21
TRF_MAX_NFEV = 150
TOL = 1e-8
SMOKE_CASES = 1
PILOT_SCENES = 1  # retrospective feasibility only; not a population sample
PILOT_CASES = 3  # default: one M0 case for EACH true mechanism (original band)
SEED = 20300203  # reserved only for possible deterministic tie handling


def need(cond, msg):
    if not cond: raise RuntimeError(msg)


def read(path):
    need(path.is_file(), f'Missing immutable evidence: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    need(path.is_file(), f'Missing source: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def objhash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                   encoding='utf-8')
    os.replace(tmp, path)


def preflight():
    files = {'P98':P98, 'P99':P99, 'P103':P103, 'P104':P104,
             'R103':R103, 'A103':A103, 'M103':M103, 'R104':R104, 'SELF':SELF}
    manifest = {k:sha(v) for k,v in files.items()}
    rep, rows, man, closure = read(R103),read(A103),read(M103),read(R104)
    need(rep['n_cases']==len(rows)==384 and rep['n_base_scenes']==24,
         'Frozen 103 cohort size differs')
    need(rep['E4_accessed'] is False and closure['E4_accessed'] is False and
         closure['status']=='PASS_WITH_PRIVILEGED_SOURCE_AND_NUMERICAL_LIMITATIONS',
         '103/104 closure or E4 status differs')
    need(rep['sources']['SELF']==manifest['P103'] and
         man['source_sha256']['SELF']==manifest['P103'] and
         closure['input_sha256']=={'R103':manifest['R103'],
             'A103':manifest['A103'],'M103':manifest['M103']},
         'Historical source/hash mismatch')
    need(manifest['P103']==closure.get('source_sha256'),
         'P104 verified P103 source differs')
    p103 = run_path(str(P103))
    p99, gen, p77, p92, p22, commands, starts, lineage, historical = p103['preflight']()
    need(p103['TRAIN']==TRAIN and p103['N_TRAIN_RESIDUALS']==N_SAMPLES
         and p99['SIGMA']==SIGMA and p77['MAXITER']==150,
         'Training protocol differs')
    need(rep['sources']['P98']==manifest['P98'] and
         rep['sources']['P99']==manifest['P99'], 'ODE provenance differs')
    need(len(man['scene_fingerprints_PRIVATE'])==48 and
         set(int(x['scene_id']) for x in rows)==set(range(24)), 'Scene structure differs')
    rmap={tuple(r['key_PRIVATE']):r for r in rows}
    need(len(rmap)==384,'Duplicate case key in historical evidence')
    return p103,p99,gen,p77,p92,p22,commands,starts,rmap,rep,man,manifest


def training_features(probes):
    """Use only the 4 training probes. Differentiate noisy observed B positions.

    SG(11,3) at 50 Hz, excluding five samples at each edge => 41/action;
    164 acceleration residuals. These are DERIVED from the same 204 B samples
    accessible to all methods, not independent additional observations.
    """
    X=[]; Y=[]
    for name in TRAIN:
        p=probes[name]
        b=np.asarray(p.y_b,dtype=float)
        a=np.asarray(p.a_half[::2],dtype=float)
        va=np.asarray(p.va_half[::2],dtype=float)
        need(b.shape==a.shape==va.shape==(51,) and np.isfinite(b).all(),
             'Training public position/velocity interface mismatch')
        vb=savgol_filter(b,SG_WINDOW,SG_POLY,deriv=1,delta=DT,mode='interp')
        ab=savgol_filter(b,SG_WINDOW,SG_POLY,deriv=2,delta=DT,mode='interp')
        sel=slice(SG_EDGE,-SG_EDGE)
        r=((b-p.b0)-(a-p.a0))[sel]
        vr=(vb-va)[sel]
        target=ab[sel]
        need(r.shape==vr.shape==target.shape==(41,), 'SG residual size changed')
        X.append((r,vr));Y.append(target)
    return np.concatenate([a for a,b in X]),np.concatenate([b for a,b in X]),np.concatenate(Y)


def equation_fit(probes,p77,law,domain):
    """Bounded equation-error estimator; grid is pre-fixed, truth-blind."""
    r,vr,acc=training_features(probes)
    lo,hi=p77['specifications'](law,domain)
    need(len(r)==164 and len(lo)==(2 if law=='elastic' else 3),
         'Equation regression shape differs')
    starts=time.perf_counter()
    evaluated=0
    candidates=[]
    if law=='elastic':
        mats=[np.column_stack((-r,-vr))]; gaps=[None]
    elif law=='cubic':
        mats=[np.column_stack((-r,-vr,-r**3))]; gaps=[None]
    else:
        gaps=np.linspace(float(lo[2]),float(hi[2]),DEADZONE_GRID)
        mats=[]
        for gap in gaps:
            mask=(np.abs(r)>gap)
            disp=np.sign(r)*np.maximum(np.abs(r)-gap,0.0)
            mats.append(np.column_stack((-disp,-vr*mask)))
    for idx,(mat,gap) in enumerate(zip(mats,gaps)):
        evaluated+=1
        lower=lo[:2] if gap is not None else lo
        upper=hi[:2] if gap is not None else hi
        try:
            opt=lsq_linear(mat,acc,bounds=(lower,upper),method='trf',
                           max_iter=100,lsmr_tol='auto',tol=1e-9)
            pars=list(np.asarray(opt.x,dtype=float))
            if gap is not None: pars.append(float(gap))
            mse=float(np.mean((mat @ opt.x-acc)**2))
            if opt.success and np.isfinite(mse) and p77['is_feasible'](law,pars,domain):
                candidates.append({'mse':mse,'params_PRIVATE':pars,'origin':f'eqerr_grid_{idx}',
                                   'converged':True})
        except (ValueError,FloatingPointError,RuntimeError):
            pass
    fit=min(candidates,key=lambda x:x['mse']) if candidates else None
    return fit,{'attempts':evaluated,'training_equation_samples':164,
                'nonlinear_grid_size':DEADZONE_GRID if law=='deadzone' else 0,
                'elapsed_s':float(time.perf_counter()-starts),'available':fit is not None,
                'metric':'acceleration_mse_m2_per_s4; not output-error MSE'}


def output_error_fit(probes,p77,law,domain,starts):
    """Independent TRF solver of the SAME original output-error objective.

    Counter records ALL residual function calls (including finite-difference
    Jacobian evaluations) rather than relying only on scipy's nfev field.
    """
    training=[probes[x] for x in TRAIN]
    obs=[np.asarray(p.y_b,dtype=float) for p in training]
    lo,hi=p77['specifications'](law,domain)
    x0s=[*p77['initial_points'](law),*starts[law]]
    need(len(x0s)==5,'Original 2+3 starts changed')
    attempts=[]; candidates=[]
    t0=time.perf_counter()
    for idx,x0 in enumerate(x0s):
        calls=[0]
        def residual(q):
            calls[0]+=1
            try:
                rr=[]
                for p,y in zip(training,obs):
                    prediction=np.asarray(p.predict(law,q.tolist()),dtype=float)
                    if prediction.shape!=(51,) or not np.isfinite(prediction).all():
                        raise FloatingPointError('Nonfinite output prediction')
                    rr.append((prediction-y)/SIGMA)
                return np.concatenate(rr)
            except (ValueError,OverflowError,FloatingPointError):
                return np.full(N_SAMPLES,1e6,dtype=float)
        try:
            res=least_squares(residual,np.asarray(x0,dtype=float),
                              bounds=(lo,hi),method='trf',
                              max_nfev=TRF_MAX_NFEV,ftol=1e-9,
                              xtol=1e-9,gtol=1e-9)
            q=res.x.tolist();mse=float(np.mean(res.fun**2))
            ok=bool(res.success and p77['is_feasible'](law,q,domain)
                    and np.isfinite(mse) and mse<1e10)
            attempts.append({'start':idx,'converged':bool(res.success),
                             'valid':ok,'mse':mse if np.isfinite(mse) else None,
                             'nfev_scipy':int(res.nfev),'njev':int(res.njev or 0),
                             'residual_calls_ALL':int(calls[0]),
                             'status':int(res.status)})
            if ok:
                candidates.append({'params_PRIVATE':q,'mse':mse,'origin':f'trf_start_{idx}',
                                   'converged':True})
        except (RuntimeError,ValueError,FloatingPointError,OverflowError) as exc:
            attempts.append({'start':idx,'converged':False,'valid':False,
                             'mse':None,'residual_calls_ALL':int(calls[0]),
                             'error':f'{type(exc).__name__}: {exc}'[:150]})
    chosen=min(candidates,key=lambda x:x['mse']) if candidates else None
    return chosen,{'attempts':attempts,'residual_calls_ALL':sum(a['residual_calls_ALL'] for a in attempts),
                   'elapsed_s':float(time.perf_counter()-t0),'available':chosen is not None,
                   'objective':'204-sample normalized B-position output-error MSE',
                   'start_count':5,'max_nfev_scipy_per_start':TRF_MAX_NFEV}


def score_fit(fits,probes,p103,config='WOO'):
    record={'MATCHED5':fits}
    return p103['scored_decision'](record,probes,config)


def fit_methods(probes,p77,p103,starts):
    fitted={x:{d:{} for d in DOMAINS} for x in ARMS[:2]}
    logs={x:{d:{} for d in DOMAINS} for x in ARMS[:2]}
    for law in LAWS:
        for domain in DOMAINS:
            a,la=equation_fit(probes,p77,law,domain)
            b,lb=output_error_fit(probes,p77,law,domain,starts)
            fitted['EQERR_SG'][domain][law]=a
            fitted['TRF_OUT'][domain][law]=b
            logs['EQERR_SG'][domain][law]=la
            logs['TRF_OUT'][domain][law]=lb
    return fitted,logs


def check_case(key,rmap,gen,base,mode,axis,level,probes,fps,p103):
    old=rmap[key]
    need(old['physical_scene_sha256_PRIVATE']==gen['digest_object'](
             gen['apply_condition'](base,mode,axis,level)),
         'P103 physical scene digest mismatch')
    need(fps==old['clean_fingerprints_PRIVATE'],
         'P103 clean observation digest mismatch')
    return old


def one_case(sid,band,mode,axis,level,base,p99,p103,gen,p77,Probe,commands,starts,rmap):
    key=(sid,band,mode,axis,level)
    scene=gen['apply_condition'](base,mode,axis,level)
    obs,clean,fps=p99['sample_noisy_probes'](gen,Probe,scene,mode,commands,
                    sid,band,axis,level,p103['NOISE_SEED'])
    reference=check_case(key,rmap,gen,base,mode,axis,level,obs,fps,p103)
    target=obs['SHIFTED_ODE']
    # Verify full noisy-observation replay, not only clean ODE fingerprints.
    # Existing SLSQP target fits were frozen in P103, so this check creates
    # no new SLSQP fits and reads no heldout outcome for optimization.
    replay=p103['scored_decision'](reference['candidate_fits_PRIVATE']['SHIFTED_ODE'],
                                   target,'WOO')
    for method in METHODS:
        ref=reference['decisions']['SHIFTED_REFIT'][method]
        got=replay[method]
        need(ref['available']==got['available'] and ref['winner']==got['winner'],
             f'Frozen P103 reference decision differs after noisy replay: {key}')
        if ref['available']:
            need(all(math.isclose(ref['scores_mm_PRIVATE'][law],
                       got['scores_mm_PRIVATE'][law],abs_tol=1e-7,rel_tol=1e-9)
                     for law in LAWS),
                 f'Frozen P103 reference scores differ after noisy replay: {key}')
    fits,logs=fit_methods(target,p77,p103,starts)
    choices={}
    for alg in ARMS[:2]:
        choices[alg]={cfg:score_fit(fits[alg],target,p103,cfg)
                      for cfg in ('WWW','OOO','WOO')}
    # Reference decisions reused, NOT a newly optimized SLSQP treatment.
    choices['SLSQP_MATCHED5_REFERENCE']={'WOO':reference['decisions']['SHIFTED_REFIT']}
    truth=reference['true_law_PRIVATE']  # only post-selection audit
    return {'key_PRIVATE':list(key),'scene_id':sid,'truth_PRIVATE':truth,
            'decisions':choices,'fits_PRIVATE':fits,'fit_logs_PRIVATE':logs,
            'P103_target_reference_sha256_PRIVATE':objhash(reference['decisions']['SHIFTED_REFIT'])}


def summarize(rows):
    out={}
    for alg in ARMS:
        out[alg]={}
        cfgs=('WOO',) if alg=='SLSQP_MATCHED5_REFERENCE' else ('WWW','OOO','WOO')
        for cfg in cfgs:
            out[alg][cfg]={}
            for m in METHODS:
                ds=[r['decisions'][alg][cfg][m] for r in rows]
                n=len(rows);correct=sum(int(d['available'] and d['winner']==r['truth_PRIVATE'])
                                        for d,r in zip(ds,rows))
                avail=sum(int(d['available']) for d in ds)
                out[alg][cfg][m]={'n':n,'correct':correct,'available':avail,
                                  'accuracy_all':correct/n if n else None}
    metrics={}
    for a in ARMS[:2]:
        calls=sum(r['fit_logs_PRIVATE'][a][d][l].get('residual_calls_ALL',0)
                  for r in rows for d in DOMAINS for l in LAWS)
        secs=sum(r['fit_logs_PRIVATE'][a][d][l]['elapsed_s']
                  for r in rows for d in DOMAINS for l in LAWS)
        metrics[a]={'total_residual_calls_ALL':calls,'cumulative_fit_wall_seconds':secs,
                    'n_fit_law_domain_units':len(rows)*6,
                    'note':'Not equalized in compute; report actual cost before formal B2B'}
    return out,metrics


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--smoke',action='store_true')
    arg.add_argument('--pilot-scenes',type=int,default=PILOT_SCENES,
                     help='Retrospective interface pilot only; 1 or 2 old scenes')
    arg.add_argument('--pilot-cases',type=int,default=PILOT_CASES,choices=(3,16),
                     help='3 mechanism-representative M0 rows/scene (default), or all 16 rows/scene')
    a=arg.parse_args()
    need(a.pilot_scenes in (1,2),'Pilot scenes must be 1 or 2')
    print('='*76)
    print('P4-B2A — INDEPENDENT ID ESTIMATOR / OPTIMIZER FEASIBILITY PILOT')
    print('='*76)
    print(f'Smoke: {a.smoke} | E4 sealed | Historical P103 cohort only')
    (p103,p99,gen,p77,p92,p22,commands,starts,rmap,report,man,manifest)=preflight()
    print('[PASS] Historical P103/P104 source hashes, 384-case frozen cohort')
    scene_rng=np.random.default_rng(p103['SCENE_SEED'])
    nscenes=1 if a.smoke else a.pilot_scenes
    cases=[]
    for sid in range(nscenes):
        bases={band:gen['sample_scene'](scene_rng,band) for band in gen['BANDS']}
        for band in gen['BANDS']:
            need(gen['digest_object'](bases[band])==man['scene_fingerprints_PRIVATE'][2*sid+gen['BANDS'].index(band)],
                 'Historical base scene fingerprint changed')
        for band in gen['BANDS']:
            for mode in p99['MODES']:
                for axis,level in p99['CONDITIONS'][mode]:
                    if a.smoke and cases: break
                    if not a.smoke and a.pilot_cases==3 and not (band=='original' and level=='M0'):
                        continue
                    cases.append(one_case(sid,band,mode,axis,level,bases[band],p99,p103,
                                         gen,p77,p22['RevisionProbe'],commands,starts,rmap))
                if a.smoke and cases: break
            if a.smoke and cases: break
        print(f'[DONE] historical scene {sid+1}/{nscenes}: '+str(sum(r['scene_id']==sid for r in cases))+' cases')
    expected=1 if a.smoke else a.pilot_cases*nscenes
    need(len(cases)==expected,'Incomplete pilot matrix')
    stats,cost=summarize(cases)
    result={'protocol':'P4-B2A historical P103 estimator feasibility; retrospective ONLY',
            'smoke':bool(a.smoke),'n_scenes':nscenes,'n_cases':len(cases),
            'pilot_case_selection':'original-band M0 each law' if a.pilot_cases==3 else 'full historical 16-case grid',
            'methods':list(ARMS),'same_204_train_B_positions':True,
            'train_actions':list(TRAIN),'heldout_actions':['mid','HALF','FULL'],
            'ODE_generator':'SHIFTED_ODE','candidate_domain_configs':['WWW','OOO','WOO'],
            'TRF':{'starts':'original 2 + fixed common 3','max_nfev_scipy_per_start':TRF_MAX_NFEV,
                   'note':'total residual calls including numerical Jacobian separately audited'},
            'EQERR':{'SG_window':SG_WINDOW,'SG_poly':SG_POLY,'SG_trim_each_end':SG_EDGE,
                      'eqerr_derived_samples_per_case':164,'deadzone_grid':DEADZONE_GRID,
                      'note':'Independent equation-error estimator; not equal CPU budget'},
            'classification':stats,'fit_compute_cost':cost,
            'input_sha256':{k:manifest[k] for k in manifest if k!='SELF'},
            'SELF_sha256':manifest['SELF'],
            'P103_reference_read_only':True,'truth_used_during_fit':False,
            'heldout_used_for_fit':False,'new_fresh_scenes':False,
            'scientific_effect_estimate':False,'E4_accessed':False,
            'environment':{'python':sys.version,'scipy':scipy.__version__,
                           'numpy':np.__version__,'platform':platform.platform()}}
    stem='p4b2a_smoke' if a.smoke else 'p4b2a_pilot'
    save(OUT/(stem+'_report.json'),result)
    save(OUT/(stem+'_audit_PRIVATE.json'),{'rows_PRIVATE':cases})
    print('[PASS] EQERR and TRF use 4 training actions; both scored on unchanged SHIFTED heldouts')
    print('[WARNING] P103 retrospective pilot; optimizer costs NOT equalized; NO confirmatory inference')
    print('[SUMMARY WOO FULL]',{alg:stats[alg]['WOO']['MID_PLUS_FULL'] for alg in ARMS})
    print('Saved:',OUT/(stem+'_report.json'))
    print('Saved:',OUT/(stem+'_audit_PRIVATE.json'))
    print('P4-B2A PILOT COMPLETE')

if __name__=='__main__':
    main()
