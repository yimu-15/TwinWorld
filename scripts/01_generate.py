"""Generate three splits of paired twins, with shuffled labels held separately.

Public arrays contain no hidden mechanism labels. P0 evaluator can read the
private files to check dataset validity, NOT as training inputs.
"""
from __future__ import annotations
from pathlib import Path
import sys
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twincore import sample_params, simulate, DT, STEPS, PULSE_STEPS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data'


def make_split(split: str, pairs: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    passive = np.empty((pairs * 2, STEPS, 2), dtype=np.float64)
    active = np.empty_like(passive)
    action = np.empty((pairs * 2, STEPS), dtype=np.float64)
    labels = np.empty((pairs * 2,), dtype=np.int64)
    ids = np.empty((pairs * 2,), dtype=np.int64)
    params_all = []
    for i in range(pairs):
        params = sample_params(rng)
        order = rng.permutation([0, 1])  # index is NOT mechanism label
        params_all.append(vars(params))
        for j, mechanism in enumerate(order):
            idx = 2 * i + j
            p0, _ = simulate(params, coupled=bool(mechanism), force=False)
            p1, a = simulate(params, coupled=bool(mechanism), force=True)
            passive[idx] = p0
            active[idx] = p1
            action[idx] = a
            labels[idx] = int(mechanism)
            ids[idx] = i
    assert np.all(np.isfinite(passive)) and np.all(np.isfinite(active))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / f'{split}_public.npz', passive=passive,
                        active=active, action=action, pair_id=ids)
    # This file is FOR EVALUATION ONLY. Models must not read it at inference.
    np.savez_compressed(OUTPUT / f'{split}_private_labels.npz', label=labels)
    (OUTPUT / f'{split}_private_params.json').write_text(
        json.dumps(params_all, indent=2), encoding='utf-8')
    print(f'{split}: {pairs} paired worlds, {pairs * 2} episodes, seed={seed}')


if __name__ == '__main__':
    for args in [('train', 150, 11), ('val', 50, 22), ('test', 100, 33)]:
        make_split(*args)
    (OUTPUT / 'generation_config.json').write_text(json.dumps({
        'time_step': DT, 'steps': STEPS, 'pulse_steps': PULSE_STEPS,
        'splits': {'train': 150, 'val': 50, 'test': 100},
        'pair_definition': 'two different mechanisms, exactly matched passive initial conditions',
        'public_inputs': ['passive', 'active', 'action', 'pair_id'],
        'private_labels': 'for blind final evaluator only',
        'warning': 'P0 only: matched passive counterfactual available; not a full embodied benchmark'
    }, indent=2), encoding='utf-8')
