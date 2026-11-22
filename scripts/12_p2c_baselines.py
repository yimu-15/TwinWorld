"""TwinWorld-v2 P2-C0: fixed supervised diagnostic baselines, CPU only.

Prerequisites: numpy, scikit-learn. Train/validation labels are allowed;
test labels are loaded only after all candidate selection and predictions.
Each test prediction uses a SINGLE episode (no test-triplet pairing).
This is a state-trajectory baseline, not visual concept revision.

Run from project root:
  .\\Giant\\Scripts\\python.exe scripts\\12_p2c_baselines.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODES = ("independent", "rigid", "elastic")
SEED = 20261113


def load_observations(directory: Path, split: str):
    with np.load(directory / f"{split}_observations.npz", allow_pickle=False) as z:
        p = np.asarray(z["passive_xy"], dtype=np.float64)
        a = np.asarray(z["active_xy"], dtype=np.float64)
        cmd = np.asarray(z["commanded_accel"], dtype=np.float64)
    n = p.shape[0]
    assert p.shape == a.shape == (n, 51, 2), (split, p.shape, a.shape)
    assert cmd.shape == (n, 51), (split, cmd.shape)
    assert np.isfinite(p).all() and np.isfinite(a).all() and np.isfinite(cmd).all()
    return p, a, cmd


def load_labels(directory: Path, split: str):
    with np.load(directory / f"{split}_private.npz", allow_pickle=False) as z:
        labels = np.asarray(z["mechanism"], dtype=np.int64)
        groups = np.asarray(z["pair_id"], dtype=np.int64)
    assert set(np.unique(labels)) == {0, 1, 2}
    return labels, groups


def features(observations: tuple[np.ndarray, np.ndarray, np.ndarray]):
    p, a, cmd = observations
    n = p.shape[0]
    t = np.arange(p.shape[1], dtype=np.float64) / 50.0
    # Estimate passive velocity using the observed endpoints only. No simulator state.
    v_est = (p[:, -1, :] - p[:, 0, :]) / 1.0
    expected = p[:, -1:, :] + v_est[:, None, :] * t[None, :, None]
    residual = a - expected
    return {
        "passive_only": p.reshape(n, -1),
        "A_only": np.column_stack((residual[:, :, 0], cmd)),
        "B_only": np.column_stack((residual[:, :, 1], cmd)),
        "A_plus_B": np.column_stack((residual.reshape(n, -1), cmd)),
        "B_final_residual": residual[:, -1, 1:2],
    }


def candidates():
    return [
        ("LogReg_C0.1", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))),
        ("LogReg_C1", make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))),
        ("RandomForest", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
    ]


def group_bootstrap_ci(y_true, y_pred, group_id, n_boot=2000):
    correct = (y_true == y_pred).astype(float)
    ids = np.unique(group_id)
    group_rates = np.array([correct[group_id == g].mean() for g in ids])
    rng = np.random.default_rng(SEED)
    draw = rng.choice(group_rates, size=(n_boot, len(ids)), replace=True)
    return np.percentile(draw.mean(axis=1), [2.5, 97.5]).tolist()


def audit_pairs(p, y, group_id, split):
    # Used ONLY for auditing paired observational equivalence, never for inference.
    group_count = len(np.unique(group_id))
    assert len(y) == group_count * 3
    max_gap = 0.0
    for g in np.unique(group_id):
        idx = np.where(group_id == g)[0]
        assert len(idx) == 3 and set(y[idx]) == {0, 1, 2}, (split, g)
        max_gap = max(max_gap, float(np.max(np.abs(p[idx] - p[idx[0]]))))
    return {"pairs": group_count, "max_public_passive_within_pair_diff": max_gap}


def run(directory: Path, out_dir: Path):
    train_obs = load_observations(directory, "train")
    val_obs = load_observations(directory, "val")
    test_obs = load_observations(directory, "test")
    x_train = features(train_obs)
    x_val = features(val_obs)
    x_test = features(test_obs)

    y_train, group_train = load_labels(directory, "train")
    y_val, group_val = load_labels(directory, "val")

    # No test labels loaded above. Validation selects method for each view.
    predictions = {}
    selection = {}
    for view in x_train:
        leaderboard = []
        best = None
        for name, model in candidates():
            model.fit(x_train[view], y_train)
            val_pred = model.predict(x_val[view])
            val_acc = float(accuracy_score(y_val, val_pred))
            leaderboard.append({"model": name, "val_accuracy": val_acc})
            if best is None or val_acc > best[0]:
                best = (val_acc, name, model)
        assert best is not None
        predictions[view] = best[2].predict(x_test[view])
        selection[view] = {"selected_model": best[1], "val_accuracy": best[0], "candidates": leaderboard}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train["passive_only"], y_train)
    predictions["majority_control"] = dummy.predict(x_test["passive_only"])

    # Evaluation boundary: load test labels only AFTER all model selection/predictions.
    y_test, group_test = load_labels(directory, "test")
    checks = {
        "train": audit_pairs(train_obs[0], y_train, group_train, "train"),
        "val": audit_pairs(val_obs[0], y_val, group_val, "val"),
        "test": audit_pairs(test_obs[0], y_test, group_test, "test"),
    }
    assert all(d["max_public_passive_within_pair_diff"] == 0.0 for d in checks.values())
    assert len(y_test) == 90

    metrics = {}
    for view, pred in predictions.items():
        acc = float(accuracy_score(y_test, pred))
        metrics[view] = {
            "accuracy": acc,
            "macro_f1": float(f1_score(y_test, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
            "per_class_recall": recall_score(y_test, pred, average=None, labels=[0, 1, 2], zero_division=0).tolist(),
            "confusion_matrix_true_rows": confusion_matrix(y_test, pred, labels=[0, 1, 2]).tolist(),
            "pair_cluster_bootstrap_ci95": group_bootstrap_ci(y_test, pred, group_test),
        }

    result = {
        "protocol": "P2-C0 CPU supervised fixed baselines; NOT concept discovery",
        "random_seed": SEED,
        "labels": list(MODES),
        "input_policy": "Predict each test episode independently; no test-time paired-triplet comparison, no force or constraint-state inputs",
        "train_validation_policy": "Train labels for fit; validation labels for model selection; test labels for final scoring only",
        "notes": [
            "Test includes mass and spring/damper distribution shift (not fully unseen mechanism types).",
            "95% CI bootstraps 30 independent test scene groups, rather than 90 independent episodes.",
            "Passive sensor noise is identical within each mechanism triplet by P2-B design.",
            "Do not select a new model or alter feature extraction based on the test results.",
        ],
        "pair_audit": checks,
        "selection": selection,
        "test": metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "p2c0_baselines.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("P2-C0 fixed baseline audit and evaluation")
    print("train/val/test pairs:", [checks[s]["pairs"] for s in ("train", "val", "test")])
    print("view                  selected          val_acc test_acc pair_95%_CI")
    for view in predictions:
        m = metrics[view]
        s = selection.get(view, {})
        print(f"{view:21s} {s.get('selected_model', 'dummy'):16s} "
              f"{s.get('val_accuracy', float('nan')):7.3f} {m['accuracy']:8.3f} "
              f"[{m['pair_cluster_bootstrap_ci95'][0]:.3f}, {m['pair_cluster_bootstrap_ci95'][1]:.3f}]")
    print("Saved:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=default_root / "outputs" / "p2b")
    parser.add_argument("--out", type=Path, default=default_root / "outputs" / "p2c0")
    args = parser.parse_args()
    run(args.data, args.out)
