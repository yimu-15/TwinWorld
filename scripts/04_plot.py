from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    d = np.load(ROOT/'data'/'test_public.npz')
    y = np.load(ROOT/'data'/'test_private_labels.npz')['label']
    labels = ['independent' if y[k]==0 else 'coupled' for k in (0,1)]
    tt=np.arange(d['passive'].shape[1])*.02
    fig, axes=plt.subplots(1, 2, figsize=(11,4), constrained_layout=True)
    for j in (0,1):
        axes[0].plot(tt, d['passive'][j,:,1],label=labels[j],linestyle='-' if j==0 else '--')
        axes[1].plot(tt, d['active'][j,:,1],label=labels[j],linestyle='-' if j==0 else '--')
    axes[0].set_title('Passive observations: indistinguishable')
    axes[1].set_title('A-only intervention: B responds differently')
    for ax in axes:
        ax.set(xlabel='time (s)', ylabel='object B position (m)')
        ax.grid(alpha=.2)
        ax.legend()
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    fig.savefig(out/'p0_twinworld_comparison.png',dpi=180)
    plt.close(fig)
    print('Saved:',out/'p0_twinworld_comparison.png')
