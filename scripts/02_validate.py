from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'data'

if __name__ == '__main__':
    print('TwinWorld P0 - dataset protocol validation')
    for split in ('train', 'val', 'test'):
        d = np.load(ROOT / f'{split}_public.npz')
        labels = np.load(ROOT / f'{split}_private_labels.npz')['label']
        p, a, u, pair_ids = (d[k] for k in ('passive','active','action','pair_id'))
        # For each pair, exact same history by construction, not merely similar.
        max_passive_delta = max(np.max(np.abs(p[2*i]-p[2*i+1])) for i in range(len(p)//2))
        # Identical action applied to each world of the pair.
        max_action_delta = max(np.max(np.abs(u[2*i]-u[2*i+1])) for i in range(len(p)//2))
        separations = [float(np.max(np.abs(a[2*i]-a[2*i+1]))) for i in range(len(p)//2)]
        assert np.all(labels.reshape(-1,2).sum(axis=1)==1), 'Label pair is not balanced'
        assert np.array_equal(pair_ids.reshape(-1,2)[:,0], pair_ids.reshape(-1,2)[:,1])
        assert max_passive_delta == 0.0, 'Passive observational equivalence broken'
        assert max_action_delta == 0.0, 'Intervention not matched across worlds'
        assert np.min(separations) > 1e-7, 'An intervention was uninformative'
        print(f'{split}: passive_max_abs_diff={max_passive_delta:.3g}; '
              f'matched_action_diff={max_action_delta:.3g}; '
              f'min_active_divergence={min(separations):.6g}; '
              f'median_active_divergence={np.median(separations):.6g} -- PASS')
