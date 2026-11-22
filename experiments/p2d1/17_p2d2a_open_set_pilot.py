
from pathlib import Path
import json
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT / "experiments/p2d1/results/p2d1b"
    / "p2d1b_audit_only.npz"
)
OUT = ROOT / "experiments/p2d1/results/p2d2a"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261113
SIGMA = 0.005
PULSE = 1

KNOWN = (0, 1, 2)
UNKNOWN = (3, 4)

rng = np.random.default_rng(SEED)

with np.load(SOURCE, allow_pickle=False) as f:
    trajectories = f["active"]

# scene, action, mechanism, frame, position
assert trajectories.shape == (24, 5, 5, 51, 2)
assert np.isfinite(trajectories).all()

order = rng.permutation(24)

train_scenes = order[:12]
val_scenes = order[12:18]
test_scenes = order[18:]


def build(scenes, modes, view, augment=1):
    xs, ys = [], []

    for scene in scenes:
        for mode in modes:
            clean = trajectories[
                scene, PULSE, mode
            ]

            for _ in range(augment):
                observed = clean + rng.normal(
                    0, SIGMA, clean.shape
                )

                delta = observed - observed[:1]

                if view == "A_only":
                    feature = delta[:, 0]
                elif view == "B_only":
                    feature = delta[:, 1]
                else:
                    feature = delta.reshape(-1)

                xs.append(feature)
                ys.append(mode)

    return np.asarray(xs), np.asarray(ys)


results = {}

for view in ("A_only", "B_only", "A_plus_B"):

    # Training contains known mechanisms only.
    xtr, ytr = build(
        train_scenes, KNOWN, view, augment=5
    )

    # Calibrate rejection on known validation
    # samples. Unknown labels are never used.
    xv, _ = build(val_scenes, KNOWN, view)

    # Unknown mechanisms only enter evaluation.
    xt, yt = build(
        test_scenes, KNOWN + UNKNOWN, view
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=4000,
            random_state=SEED
        )
    )

    clf.fit(xtr, ytr)

    val_novelty = (
        1.0 - clf.predict_proba(xv).max(axis=1)
    )
    threshold = float(
        np.quantile(val_novelty, 0.95)
    )

    probabilities = clf.predict_proba(xt)
    novelty = 1.0 - probabilities.max(axis=1)

    predictions = clf.classes_[
        probabilities.argmax(axis=1)
    ]

    known_mask = np.isin(yt, KNOWN)
    unknown_mask = ~known_mask

    accepted = novelty <= threshold

    results[view] = {
        "known_accuracy": float(
            accuracy_score(
                yt[known_mask],
                predictions[known_mask]
            )
        ),
        "known_acceptance_rate": float(
            np.mean(accepted[known_mask])
        ),
        "unknown_rejection_rate": float(
            np.mean(~accepted[unknown_mask])
        ),
        "unknown_detection_AUROC": float(
            roc_auc_score(
                unknown_mask.astype(int),
                novelty
            )
        ),
        "threshold": threshold,
        "n_known": int(known_mask.sum()),
        "n_unknown": int(unknown_mask.sum())
    }

report = {
    "protocol": "P2-D2A open-set pilot",
    "development_only": True,
    "seed": SEED,
    "action": "pulse",
    "noise_std_m": SIGMA,
    "known_classes": list(KNOWN),
    "unknown_classes": list(UNKNOWN),
    "scene_splits": {
        "train": train_scenes.tolist(),
        "val": val_scenes.tolist(),
        "test": test_scenes.tolist()
    },
    "results": results,
    "note": (
        "Unknown labels used only for evaluation. "
        "No concept revision is implemented. "
        "This is not a confirmatory experiment."
    )
}

path = OUT / "p2d2a_open_set_pilot.json"
path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print(json.dumps(results, indent=2))
print("\nP2-D2A exploratory run finished")
print("Saved:", path)
