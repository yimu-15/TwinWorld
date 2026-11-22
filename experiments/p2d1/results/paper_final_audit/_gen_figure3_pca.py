"""Generate Figure 3 PCA 2D coordinates + feature matrix (deterministic).

Loads P2-C0 test observations/labels, computes the B_only feature matrix via the
frozen feature() from scripts/12_p2c_baselines.py, then StandardScaler + PCA(2).
Writes feature matrix CSV and PCA 2D coordinate CSV into figure3/.
"""
import csv, runpy
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"D:\OnShouldersGiants\TwinWorld_P0")
OUT = ROOT / "figure3"
LABELS = ["independent", "rigid", "elastic"]

mod = runpy.run_path(str(ROOT / "scripts" / "12_p2c_baselines.py"))
load_observations = mod["load_observations"]
load_labels = mod["load_labels"]
features = mod["features"]

DATA = ROOT / "outputs" / "p2b"
test_obs = load_observations(DATA, "test")
y_test, group_test = load_labels(DATA, "test")
X = features(test_obs)["B_only"]  # (90, 102) B residual + commanded accel

# standardize + PCA(2) — deterministic SVD
scaler = StandardScaler()
Z = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=20261113)
coord = pca.fit_transform(Z)

# load B_only predictions (winner of B_only, already computed deterministically)
pred_csv = list(csv.DictReader(open(OUT / "p2c0_per_scene_predictions.csv", newline="")))
b_pred = {int(r["scene_id"]): r for r in pred_csv}  # placeholder

# feature matrix CSV (90 x 102)
with open(OUT / "p2c0_B_only_feature_matrix.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["episode", "scene_id", "true_class_id"] + [f"f{i}" for i in range(X.shape[1])])
    for i in range(X.shape[0]):
        w.writerow([i, int(group_test[i]), int(y_test[i])] + [round(float(v), 6) for v in X[i]])

# PCA coords CSV
with open(OUT / "p2c0_PCA_2d_coordinates.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["episode", "scene_id", "true_class", "PC1", "PC2", "explained_var_ratio_PC1", "explained_var_ratio_PC2"])
    for i in range(X.shape[0]):
        w.writerow([i, int(group_test[i]), LABELS[int(y_test[i])],
                    round(float(coord[i, 0]), 6), round(float(coord[i, 1]), 6),
                    round(float(pca.explained_variance_ratio_[0]), 6),
                    round(float(pca.explained_variance_ratio_[1]), 6)])

# method note
with open(OUT / "p2c0_PCA_method.md", "w") as fh:
    fh.write("# Figure 3 PCA method\n\n")
    fh.write("- Feature: B_only = column_stack(B residual[:, :, 1], commanded_accel) -> 102 dims (90 test episodes).\n")
    fh.write("- Reducer: sklearn StandardScaler + PCA(n_components=2, random_state=20261113).\n")
    fh.write("- Deterministic (no stochastic step beyond fixed-seed standardization/PCA).\n")
    fh.write(f"- Explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}\n")
    fh.write("- Generator: scripts/12_p2c_baselines.py features() + this script.\n")

print("B_only feature matrix:", X.shape)
print("explained variance ratio:", pca.explained_variance_ratio_.round(4))
print("saved p2c0_B_only_feature_matrix.csv, p2c0_PCA_2d_coordinates.csv, p2c0_PCA_method.md ->", OUT)
