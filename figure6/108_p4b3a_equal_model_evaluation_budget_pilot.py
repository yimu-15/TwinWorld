
"""#108 P4-B3A: historical-only HARD model-evaluation-cap feasibility pilot.

Same P103 SHIFTED noisy observations; SLSQP vs TRF on identical 204-position
training residual objective, identical 2+3 starts, bounds, and budget caps.
Every call to the 204-residual evaluator (including finite-difference Jacobian
calls) is counted; per-start hard caps are 20 and 40, total maximum 100/200.
Early optimizer convergence may use fewer calls: equal CAP != equal realized work.

Historical P103 original-band M0 cases only, one per true mechanism, three cases.
Not confirmatory, no tuning, no E4, no new scene. No accuracy claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from runpy import run_path
import time

import numpy as np
from scipy.optimize import Bounds, least_squares, minimize

HERE = Path(__file__).resolve().parent
BASE = HERE / 'results'
OUT = BASE / 'p4b3a'

P105 = HERE / '105_p4b2a_independent_identification_baseline_pilot.py'
R106 = BASE / 'p4b2b' / 'p4b2b_fresh_report.json'
M106 = BASE / 'p4b2b' / 'p4b2b_source_manifest.json'
R107 = BASE / 'p4b2c' / 'p4b2c_closure_report.json'

LAWS = ('elastic', 'deadzone', 'cubic')
TRAIN = ('e1', 'e2', 'weak', 'strong')
METHODS = ('MID_PLUS_HALF', 'MID_PLUS_FULL')
SOLVERS = ('TRF_CAP', 'SLSQP_CAP')

DOMAIN_BY_LAW = {
    'elastic': 'wide',
    'deadzone': 'outer_box',
    'cubic': 'outer_box',
}

BUDGETS = (100, 200)
SMOKE_BUDGET = 50
SIGMA = 0.005
N = 204
MAXITER = 150


def need(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def read(path):
    need(path.is_file(), f'Missing frozen input: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    need(path.is_file(), f'Missing source: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(
        json.dumps(
            obj,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        ).encode('utf-8')
    ).hexdigest()


def save_once(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        doc, ensure_ascii=False, indent=2, allow_nan=False
    ) + '\n'

    if path.exists():
        if 'smoke' not in path.name:
            need(
                path.read_text(encoding='utf-8') == content,
                f'Existing immutable result differs, refuse overwrite: {path}',
            )
            return

        old = json.loads(path.read_text(encoding='utf-8'))
        need(
            old['source_sha256'] == doc['source_sha256'],
            'Smoke source/protocol changed; archive existing smoke before rerun',
        )

    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(content, encoding='utf-8')
    os.replace(tmp, path)


def preflight():
    r106, m106, r107 = read(R106), read(M106), read(R107)

    need(
        r106['source_sha256'] == m106['source_sha256']
        and r106['n_cases'] == 384
        and not r106['new_e4_accessed']
        and r107['status'] == 'PASS_WITH_COMPUTE_AND_MODEL_LIMITATIONS'
        and r107['n_cases'] == 384
        and r107['checkpoint_rows_verified'] == 384
        and r107['score_values_verified'] == 20736
        and r107['source_provenance_warnings'] == []
        and not r107['E4_accessed']
        and r107['classification'] == r106['classification'],
        'P106/P107 baseline closure is not intact',
    )

    need(
        sha(P105) == r106['source_sha256']['P105'],
        'P105 source differs from frozen P106 baseline',
    )

    p105 = run_path(str(P105))

    (
        p103, p99, gen, p77, p92, p22,
        commands, starts, rmap, r103, m103, lineage
    ) = p105['preflight']()

    need(
        p105['N_SAMPLES'] == N
        and p105['TRAIN'] == TRAIN
        and tuple(p105['LAWS']) == LAWS
        and r103['n_cases'] == 384
        and r103['E4_accessed'] is False,
        'Historical P103 interface changed',
    )

    need(
        all(
            len(p77['initial_points'](law)) == 2
            and len(starts[law]) == 3
            for law in LAWS
        ),
        'P103 2+3 starts changed',
    )

    need(
        r106['source_sha256']['R103']
        == sha(BASE / 'p4g2c' / 'p4g2c_fresh_report.json'),
        'P106 source P103 hash changed',
    )

    return p103, p99, gen, p77, p22, commands, starts, rmap, {
        'P105': sha(P105),
        'R106': sha(R106),
        'M106': sha(M106),
        'R107': sha(R107),
        'SELF': sha(Path(__file__)),
        'P103': sha(
            HERE / '103_p4g2c_fresh_frozen_vs_refit_confirmation.py'
        ),
    }


class BudgetExhausted(Exception):
    pass


class CountedResidual:
    """One call = a full 204-vector training trajectory residual."""

    def __init__(self, probes, law, p77, domain, cap):
        self.probes = [probes[k] for k in TRAIN]
        self.observed = [
            np.asarray(p.y_b, dtype=float)
            for p in self.probes
        ]

        need(
            all(x.shape == (51,) for x in self.observed),
            'Unexpected 51-position train action',
        )

        self.law = law
        self.p77 = p77
        self.domain = domain
        self.lo, self.hi = p77['specifications'](law, domain)
        self.cap = int(cap)

        self.calls = 0
        self.best = None
        self.trace = []
        self.failures = 0

    def __call__(self, q):
        if self.calls >= self.cap:
            raise BudgetExhausted(
                'hard 204-vector evaluation cap reached'
            )

        self.calls += 1
        x = np.asarray(q, dtype=float)
        valid = True

        try:
            parts = []

            for probe, y in zip(self.probes, self.observed):
                pred = np.asarray(
                    probe.predict(self.law, x.tolist()),
                    dtype=float,
                )

                if pred.shape != (51,) or not np.isfinite(pred).all():
                    raise FloatingPointError(
                        'invalid model prediction'
                    )

                parts.append((pred - y) / SIGMA)

            residual = np.concatenate(parts)

            if residual.shape != (N,) or not np.isfinite(residual).all():
                raise FloatingPointError(
                    'invalid train residual'
                )

        except (
            FloatingPointError,
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ):
            valid = False
            self.failures += 1
            residual = np.full(N, 1e6, dtype=float)

        mse = float(np.mean(residual ** 2))

        admissible = (
            valid
            and np.all(x >= self.lo - 1e-9)
            and np.all(x <= self.hi + 1e-9)
            and self.p77['is_feasible'](
                self.law, x, self.domain
            )
        )

        # Training-only best evaluated candidate.
        # No truth label or heldout score is used here.
        if admissible and (
            self.best is None or mse < self.best['mse']
        ):
            self.best = {
                'mse': mse,
                'params_PRIVATE': x.tolist(),
                'at_call': self.calls,
            }

        self.trace.append({
            'q': x.tolist(),
            'mse': mse,
            'admissible': bool(admissible),
        })

        return residual


def fit_capped(probes, p77, law, domain, starts, solver, budget):
    need(
        budget % 5 == 0,
        'Budget must split evenly into five starts',
    )

    lo, hi = p77['specifications'](law, domain)
    x0s = [
        *p77['initial_points'](law),
        *starts[law],
    ]

    need(len(x0s) == 5, 'Not 2+3 starts')

    per = budget // 5
    all_attempts = []
    best = None
    total = 0
    wall = time.perf_counter()

    for idx, x0 in enumerate(x0s):
        x0 = np.asarray(x0, dtype=float)

        need(
            p77['is_feasible'](law, x0, domain),
            f'Frozen optimizer start {idx} infeasible for {law}/{domain}',
        )

        evaluator = CountedResidual(
            probes, law, p77, domain, per
        )

        success = False
        stopped = 'unknown'
        status = None

        try:
            if solver == 'TRF_CAP':
                result = least_squares(
                    evaluator,
                    x0,
                    bounds=(lo, hi),
                    method='trf',
                    max_nfev=MAXITER,
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-9,
                )

            elif solver == 'SLSQP_CAP':
                result = minimize(
                    lambda q: float(
                        np.mean(evaluator(q) ** 2)
                    ),
                    x0,
                    method='SLSQP',
                    bounds=Bounds(lo, hi),
                    options={
                        'maxiter': MAXITER,
                        'ftol': 1e-9,
                        'disp': False,
                    },
                )

            else:
                raise ValueError('Unknown solver')

            success = bool(result.success)
            status = int(result.status)
            stopped = (
                'optimizer_success'
                if success
                else 'optimizer_non_success'
            )

        except BudgetExhausted:
            stopped = 'budget_cap'

        need(
            evaluator.calls <= per
            and len(evaluator.trace) == evaluator.calls,
            'HARD BUDGET VIOLATION',
        )

        total += evaluator.calls

        if evaluator.best is not None:
            record = {
                **evaluator.best,
                'origin': f'{solver}_start_{idx}',
                'converged': success,
            }

            if best is None or record['mse'] < best['mse']:
                best = record

        all_attempts.append({
            'start': idx,
            'cap': per,
            'calls': evaluator.calls,
            'status': stopped,
            'scipy_status': status,
            'solver_success': success,
            'invalid_residual_calls': evaluator.failures,
            'best_mse': (
                None
                if evaluator.best is None
                else evaluator.best['mse']
            ),
            'best_params_PRIVATE': (
                None
                if evaluator.best is None
                else evaluator.best['params_PRIVATE']
            ),
            'trace_PRIVATE': evaluator.trace,
        })

    need(
        total <= budget and len(all_attempts) == 5,
        'TOTAL HARD BUDGET VIOLATION',
    )

    return best, {
        'budget': budget,
        'per_start_cap': per,
        'calls': total,
        'unused_calls': budget - total,
        'wall_seconds': float(
            time.perf_counter() - wall
        ),
        'attempts': all_attempts,
        'best_available': best is not None,
    }


def prefix_audit(a, b, label):
    """A larger budget must preserve the smaller run's call prefix."""

    for x, y in zip(a['attempts'], b['attempts']):
        tx = x['trace_PRIVATE']
        ty = y['trace_PRIVATE']
        common = min(len(tx), len(ty))

        for k in range(common):
            xx, yy = tx[k], ty[k]

            need(
                xx['admissible'] == yy['admissible']
                and np.allclose(
                    xx['q'], yy['q'],
                    rtol=1e-11,
                    atol=1e-11,
                )
                and math.isclose(
                    xx['mse'], yy['mse'],
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ),
                f'Budget prefix mismatch '
                f'{label}/start{x["start"]}/call{k}',
            )

        need(common > 0, f'No model calls {label}')


def replay_case(
    sid, band, mode, axis, level, base, context
):
    p103, p99, gen, p77, p22, commands, starts, rmap, _ = context

    scene = gen['apply_condition'](
        base, mode, axis, level
    )
    key = (sid, band, mode, axis, level)
    historical = rmap[key]

    need(
        gen['digest_object'](scene)
        == historical['physical_scene_sha256_PRIVATE'],
        'P103 physical scene fingerprint mismatch',
    )

    obs, clean, fps = p99['sample_noisy_probes'](
        gen,
        p22['RevisionProbe'],
        scene,
        mode,
        commands,
        sid,
        band,
        axis,
        level,
        p103['NOISE_SEED'],
    )

    need(
        fps == historical['clean_fingerprints_PRIVATE'],
        'P103 clean action fingerprints mismatch',
    )

    target = obs['SHIFTED_ODE']

    historical_fit = historical[
        'candidate_fits_PRIVATE'
    ]['SHIFTED_ODE']

    check = p103['scored_decision'](
        historical_fit, target, 'WOO'
    )

    for meth in METHODS:
        old = historical['decisions']['SHIFTED_REFIT'][meth]

        need(
            check[meth]['winner'] == old['winner']
            and check[meth]['available'] == old['available']
            and all(
                math.isclose(
                    check[meth]['scores_mm_PRIVATE'][l],
                    old['scores_mm_PRIVATE'][l],
                    rel_tol=1e-9,
                    abs_tol=1e-7,
                )
                for l in LAWS
            ),
            'P103 noisy heldout replay differs',
        )

    return target, historical


def test_case(
    sid, band, mode, axis, level, base, context, smoke=False
):
    p103, p99, gen, p77, p22, commands, starts, rmap, _ = context

    probes, old = replay_case(
        sid, band, mode, axis, level, base, context
    )

    budgets = (
        (SMOKE_BUDGET,)
        if smoke
        else BUDGETS
    )

    laws = (
        ('elastic',)
        if smoke
        else LAWS
    )

    domains = (
        ('wide',)
        if smoke
        else tuple(dict.fromkeys(DOMAIN_BY_LAW.values()))
    )

    fits = {}
    logs = {}

    for cap in budgets:
        fits[str(cap)] = {}
        logs[str(cap)] = {}

        for solver in SOLVERS:
            fits[str(cap)][solver] = {
                d: {} for d in domains
            }
            logs[str(cap)][solver] = {
                d: {} for d in domains
            }

            for law in laws:
                d = DOMAIN_BY_LAW[law]

                chosen, info = fit_capped(
                    probes,
                    p77,
                    law,
                    d,
                    starts,
                    solver,
                    cap,
                )

                fits[str(cap)][solver][d][law] = chosen
                logs[str(cap)][solver][d][law] = info

    if not smoke:
        for solver in SOLVERS:
            for law in LAWS:
                domain = DOMAIN_BY_LAW[law]

                prefix_audit(
                    logs[str(BUDGETS[0])][solver][domain][law],
                    logs[str(BUDGETS[1])][solver][domain][law],
                    f'{solver}/{domain}/{law}/{mode}',
                )

    decisions = {}

    if not smoke:
        for budget in BUDGETS:
            decisions[str(budget)] = {}

            for solver in SOLVERS:
                scores = p103['scored_decision'](
                    {
                        'MATCHED5':
                        fits[str(budget)][solver]
                    },
                    probes,
                    'WOO',
                )

                decisions[str(budget)][solver] = scores

    return {
        'key_PRIVATE': [
            sid, band, mode, axis, level
        ],
        'observations_replayed_sha256_PRIVATE': digest(
            old['clean_fingerprints_PRIVATE']
        ),
        'fits_PRIVATE': fits,
        'fit_logs_PRIVATE': logs,
        'decisions_PRIVATE': decisions,
        'label_for_retrospective_audit_ONLY': (
            old['true_law_PRIVATE']
            if not smoke
            else None
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    print('=' * 74)
    print(
        'P4-B3A — HARD TRAINING MODEL-EVALUATION CAP / HISTORICAL PILOT'
    )
    print('=' * 74)
    print(
        'Smoke:', args.smoke,
        '| no new scenes | E4 sealed | no confirmation',
    )

    context = preflight()

    (
        p103, p99, gen, p77, p22,
        commands, starts, rmap, lineage
    ) = context

    rng = np.random.default_rng(p103['SCENE_SEED'])
    base = {
        b: gen['sample_scene'](rng, b)
        for b in gen['BANDS']
    }

    need(
        gen['digest_object'](base['original'])
        == read(
            BASE / 'p4g2c' / 'p4g2c_source_manifest.json'
        )['scene_fingerprints_PRIVATE'][0],
        'Historical scene ID=0 mismatch',
    )

    conditions = (
        [('elastic', 'matched', 'M0')]
        if args.smoke
        else [
            ('elastic', 'matched', 'M0'),
            ('deadzone', 'matched', 'M0'),
            ('nonlinear', 'matched', 'M0'),
        ]
    )

    source = {
        **lineage,
        'protocol_budget_caps': list(BUDGETS),
        'smoke_budget': SMOKE_BUDGET,
        'same_five_starts': True,
        'domain_by_law_WOO': DOMAIN_BY_LAW,
    }

    rows = []

    for mode, axis, level in conditions:
        cp = (
            OUT / 'checkpoints'
            / f'original_{mode}_M0.json'
        )

        if not args.smoke and cp.is_file():
            pack = read(cp)

            need(
                pack['source_sha256'] == source
                and pack['row_sha256'] == digest(pack['row']),
                'Frozen checkpoint does not match source',
            )

            row = pack['row']
            print('[RESUME]', mode, flush=True)

        else:
            row = test_case(
                0,
                'original',
                mode,
                axis,
                level,
                base['original'],
                context,
                args.smoke,
            )

            if not args.smoke:
                save_once(
                    cp,
                    {
                        'source_sha256': source,
                        'row_sha256': digest(row),
                        'row': row,
                    },
                )

            print(
                '[PASS] hard budget and prefix:',
                mode,
                flush=True,
            )

        rows.append(row)

    if args.smoke:
        for sol in SOLVERS:
            log = rows[0]['fit_logs_PRIVATE'][
                str(SMOKE_BUDGET)
            ][sol]['wide']['elastic']

            need(
                len(log['attempts']) == 5
                and log['calls'] <= SMOKE_BUDGET,
                'Smoke cap violated',
            )

        stem = 'p4b3a_smoke'

    else:
        for row in rows:
            for cap in BUDGETS:
                for sol in SOLVERS:
                    for law in LAWS:
                        d = DOMAIN_BY_LAW[law]
                        info = row['fit_logs_PRIVATE'][
                            str(cap)
                        ][sol][d][law]

                        need(
                            info['calls'] <= cap
                            and len(info['attempts']) == 5,
                            'Formal cap violated',
                        )

        stem = 'p4b3a_pilot'

    aggregate = {}

    for cap in (
        (SMOKE_BUDGET,)
        if args.smoke
        else BUDGETS
    ):
        aggregate[str(cap)] = {}

        for sol in SOLVERS:
            logs = [
                r['fit_logs_PRIVATE'][str(cap)][sol][
                    DOMAIN_BY_LAW[law]
                ][law]
                for r in rows
                for law in (
                    ('elastic',)
                    if args.smoke
                    else LAWS
                )
            ]

            aggregate[str(cap)][sol] = {
                'fit_units': len(logs),
                'max_calls_total': sum(
                    x['budget'] for x in logs
                ),
                'actual_204vector_calls': sum(
                    x['calls'] for x in logs
                ),
                'unused_calls': sum(
                    x['unused_calls'] for x in logs
                ),
                'start_stop_status_counts': dict(
                    Counter(
                        a['status']
                        for x in logs
                        for a in x['attempts']
                    )
                ),
                'unavailable_fits': sum(
                    not x['best_available']
                    for x in logs
                ),
                'wall_seconds_descriptive_only': sum(
                    x['wall_seconds'] for x in logs
                ),
            }

    doc = {
        'protocol':
            'P4-B3A historical WOO-only maximum model-evaluations feasibility',
        'smoke': bool(args.smoke),
        'n_base_scenes': 1,
        'n_cases': len(rows),
        'source_sha256': source,
        'budgets': list(
            (SMOKE_BUDGET,)
            if args.smoke
            else BUDGETS
        ),
        'per_start_caps': [
            int(b / 5)
            for b in (
                (SMOKE_BUDGET,)
                if args.smoke
                else BUDGETS
            )
        ],
        'training_actions': list(TRAIN),
        'training_residual_count': N,
        'domain_by_law_WOO': DOMAIN_BY_LAW,
        'methods': list(SOLVERS),
        'aggregate': aggregate,
        'scoring_not_used_to_select_budget': True,
        'truth_not_used_for_fitting': True,
        'heldout_not_used_for_fitting': True,
        'no_confirmatory_inference': True,
        'new_scenes': False,
        'E4_accessed': False,
        'limits': [
            'Equal maximum calls, NOT equal actual calls or wall time',
            'Retain best feasible evaluated parameter even without successful optimizer termination',
            'The 3 old cases are a preflight diagnostic only, not a representative accuracy sample',
            'Holdout action scores are for end-of-pilot diagnostic only; never used in fitting',
            'One RK4 physics implementation, not hardware verification',
        ],
    }

    report = OUT / (stem + '_report.json')
    audit = OUT / (stem + '_audit_PRIVATE.json')

    save_once(report, doc)
    save_once(
        audit,
        {
            'rows_PRIVATE': rows,
            'source_sha256': source,
        },
    )

    print('[AGGREGATE]', aggregate)
    print(
        '[SCIENTIFIC STATUS] FEASIBILITY ONLY; no performance estimate'
    )
    print('Saved:', report)
    print('Saved:', audit)
    print('P4-B3A COMPLETE')


if __name__ == '__main__':
    main()
