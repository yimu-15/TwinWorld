"""Sanity-check baselines only. No trained AI, no conceptual revision.

Passive baseline predicts one label for all worlds. Balance => exactly 50%.
Active diagnostic makes a high-information matched-counterfactual comparison
against a no-action rollout. The matched rollout is an unrealistic P0 privilege;
it must NOT be reported later as a fair real-world active agent baseline.
"""
from pathlib import Path
import json
import numpy as np

DATA = Path(__file__).resolve().parents[1] / 'data'
OUT = Path(__file__).resolve().parents[1] / 'outputs'


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(pred == true))

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    results = {}
    for split in ('val', 'test'):
        d = np.load(DATA / f'{split}_public.npz')
        y = np.load(DATA / f'{split}_private_labels.npz')['label']
        assert np.array_equal(np.bincount(y), [len(y)//2, len(y)//2])
        blind = np.zeros_like(y)
        # B changes only if an unseen coupling transmits the A-only impulse.
        # This is a P0 oracle diagnostic with paired no-action rollout.
        b_effect = np.max(np.abs(d['active'][:,:,1] - d['passive'][:,:,1]), axis=1)
        active_rule = (b_effect > 1.e-6).astype(np.int64)
        results[split] = {
            'episodes': int(len(y)),
            'passive_majority_accuracy': accuracy(blind, y),
            'active_oracle_diagnostic_accuracy': accuracy(active_rule, y),
            'minimum_coupled_effect': float(np.min(b_effect[y == 1])),
            'maximum_independent_effect': float(np.max(b_effect[y == 0])),
            'important': 'Active diagnostic has privileged matched passive rollout; not a learned agent.'
        }
        print(f'{split}: passive chance={results[split]["passive_majority_accuracy"]:.3f}, '
              f'active oracle={results[split]["active_oracle_diagnostic_accuracy"]:.3f}')
    (OUT / 'p0_baselines.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
