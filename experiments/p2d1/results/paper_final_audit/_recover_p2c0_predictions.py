"""Recover P2-C0 per-scene test predictions + B-only-minus-Passive-only paired bootstrap.

The original 12_p2c_baselines.py only persisted aggregate metrics; per-episode
predictions and scene grouping were discarded. This re-runs the SAME deterministic
pipeline (SEED=20261113, fixed train/val/test from outputs/p2b) and:
  1) saves per-scene predictions (scene_id, true mechanism, per-condition preds)
  2) computes the paired B-only minus Passive-only scene-cluster bootstrap

Verification: the reconstructed aggregate accuracy / marginal CI must match
p2c0_baselines.json exactly. No original file is modified.
"""
import csv
import json
import runpy
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0")
DATA = ROOT / "outputs" / "p2b"
OUT = ROOT / "outputs" / "p2c0"
REPORT = json.loads((OUT / "p2c0_baselines.json").read_text(encoding="utf-8"))
LABELS = REPORT["labels"]  # ["independent", "rigid", "elastic"]

mod = runpy.run_path(str(ROOT / "scripts" / "12_p2c_baselines.py"))
load_observations = mod["load_observations"]
load_labels = mod["load_labels"]
features = mod["features"]
candidates = mod["candidates"]
SEED = mod["SEED"]

train_obs = load_observations(DATA, "train")
val_obs = load_observations(DATA, "val")
test_obs = load_observations(DATA, "test")
x_train = features(train_obs)
x_val = features(val_obs)
x_test = features(test_obs)
y_train, _ = load_labels(DATA, "train")
y_val, _ = load_labels(DATA, "val")

# Replicate the exact selection/prediction loop from 12_p2c_baselines.py
predictions = {}
selection = {}
for view in x_train:
    leaderboard = []
    best = None
    for name, model in candidates():
        model.fit(x_train[view], y_train)
        val_pred = model.predict(x_val[view])
        val_acc = float((val_pred == y_val).mean())
        leaderboard.append((name, val_acc))
        if best is None or val_acc > best[0]:
            best = (val_acc, name, model)
    predictions[view] = best[2].predict(x_test[view])
    selection[view] = best[1]

# Dummy majority control
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(x_train["passive_only"], y_train)
predictions["majority_control"] = dummy.predict(x_test["passive_only"])

# Load test labels/groups AFTER predictions (as original does)
y_test, group_test = load_labels(DATA, "test")
assert len(y_test) == 90

# Verify reconstruction matches saved report exactly (accuracy + marginal CI)
def group_bootstrap_ci(y_true, y_pred, group_id, n_boot=2000):
    correct = (y_true == y_pred).astype(float)
    ids = np.unique(group_id)
    rates = np.array([correct[group_id == g].mean() for g in ids])
    rng = np.random.default_rng(SEED)
    draw = rng.choice(rates, size=(n_boot, len(ids)), replace=True)
    return np.percentile(draw.mean(axis=1), [2.5, 97.5]).tolist()

ok = True
for view, pred in predictions.items():
    exp = REPORT["test"][view]
    acc = float((y_test == pred).mean())
    ci = group_bootstrap_ci(y_test, pred, group_test)
    a = abs(acc - exp["accuracy"]) < 1e-12
    c = all(abs(x - y) < 1e-9 for x, y in zip(ci, exp["pair_cluster_bootstrap_ci95"]))
    if not (a and c):
        ok = False
        print("MISMATCH", view, acc, exp["accuracy"], ci, exp["pair_cluster_bootstrap_ci95"])
print("VERIFY_AGGREGATE_MATCH:", ok)
assert ok

# --- 1) Save per-scene predictions CSV ---
# scene_id = pair_id (group). Each scene has 3 episodes (one per mechanism).
cond_names = ["passive_only", "A_only", "B_only", "A_plus_B", "B_final_residual", "majority_control"]
rows = []
for i in range(len(y_test)):
    rows.append({
        "scene_id": int(group_test[i]),
        "true_class_id": int(y_test[i]),
        "true_class": LABELS[int(y_test[i])],
        **{c: LABELS[int(predictions[c][i])] for c in cond_names},
    })
with open(OUT / "p2c0_per_scene_predictions.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["scene_id", "true_class_id", "true_class"] + cond_names)
    w.writeheader()
    w.writerows(rows)

# --- 2) Paired B-only minus Passive-only scene-cluster bootstrap ---
b_pred = predictions["B_only"]
p_pred = predictions["passive_only"]
ids = np.unique(group_test)
b_correct = (y_test == b_pred).astype(float)
p_correct = (y_test == p_pred).astype(float)
b_rate = np.array([b_correct[group_test == g].mean() for g in ids])
p_rate = np.array([p_correct[group_test == g].mean() for g in ids])
diff = b_rate - p_rate  # per-scene paired difference
rng = np.random.default_rng(SEED)
draws = rng.choice(len(ids), size=(2000, len(ids)), replace=True)
boot_diffs = diff[draws].mean(axis=1)
paired = {
    "contrast": "B_only_minus_Passive_only",
    "unit": "test scene group (30 clusters)",
    "n_clusters": int(len(ids)),
    "b_only_accuracy": float(b_correct.mean()),
    "passive_only_accuracy": float(p_correct.mean()),
    "point_estimate": float(diff.mean()),
    "paired_bootstrap_ci95": [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))],
    "bootstrap_repetitions": 2000,
    "bootstrap_seed": SEED,
}
with open(OUT / "p2c0_B_only_minus_Passive_only_paired_bootstrap.json", "w") as fh:
    json.dump(paired, fh, indent=2)

print("SAVED p2c0_per_scene_predictions.csv + p2c0_B_only_minus_Passive_only_paired_bootstrap.json")
print(json.dumps(paired, indent=2))
print("selected models:", selection)
