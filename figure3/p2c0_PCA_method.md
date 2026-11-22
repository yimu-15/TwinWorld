# Figure 3 PCA method

- Feature: B_only = column_stack(B residual[:, :, 1], commanded_accel) -> 102 dims (90 test episodes).
- Reducer: sklearn StandardScaler + PCA(n_components=2, random_state=20261113).
- Deterministic (no stochastic step beyond fixed-seed standardization/PCA).
- Explained variance ratio: PC1=0.3949, PC2=0.2383
- Generator: scripts/12_p2c_baselines.py features() + this script.
