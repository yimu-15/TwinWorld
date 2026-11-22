# PAPER_FIGURE_DATA_INDEX — Figure 1–6 data source map (English)

> Read-only. Each figure lists axes, units, statistic, and original file location.

## Figure 1 — Problem setup & passive equivalence
- Content: matched passive trajectories (two worlds overlap), A-only action, divergent B response.
- X: time step (DT=0.02, 120 steps). Y: position (m). Unit: meters.
- Source: `outputs/p0_twinworld_comparison.png` (script `scripts/04_plot.py`); `outputs/p1b/p1b_comparison.png`; `outputs/p1c/p1c_matched_motion.png`.
- Underlying data: `data/*.npz`, `outputs/p1b/p1b_report.json` (active_B_final_abs_diff=0.2505).

## Figure 2 — Passive at chance vs active oracle (P0/P1/P2)
- X: diagnostic (passive / A-only / B-only / A+B). Y: accuracy. Unit: fraction.
- Statistic: test accuracy + pair/scene-cluster bootstrap 95% CI.
- Source: `outputs/p0_baselines.json`; `outputs/p2c0/p2c0_baselines.json` (B_only 0.878 CI[0.822,0.933]).

## Figure 3 — Physical domain ablation (outer_box vs wide)
- X: domain config (WIDE / OUTER_BOX / SHARED_MASS or WWW/OOO/WOO). Y: identification accuracy.
- Statistic: accuracy + scene-cluster CI; delta outer−wide.
- Source: `results/p4m2b2b/p4m2b2b_fresh_report.json` (outer−wide +0.0833 CI[0.051,0.116]); `results/p4m2c2e/p4m2c2e_fresh_report.json` (OOO−WWW +0.1797 CI[0.128,0.232]).

## Figure 4 — Amplitude dose-response
- X: amplitude (0.5x / 1.0x). Y: nonlinear-M2 recall (and separation in mm).
- Statistic: recall full−half +0.198 CI[0.094,0.313]; pairwise separation +10.2mm CI[8.34,12.15].
- Source: `results/p4m2d2b/p4m2d2b_v2_fresh_report.json`.

## Figure 5 — Classical identification comparison
- X: method (EQERR_SG / TRF_OUT / SLSQP). Y: correct/n (WOO·FULL, n=384).
- Statistic: EQERR 313, TRF 328, SLSQP 328; EQERR−SLSQP CI[-0.070,-0.005].
- Source: `results/p4b2b/p4b2b_fresh_report.json`.

## Figure 6 — Equal-budget optimizer comparison
- X: budget (100 / 200). Y: correct/n (WOO·FULL). Two lines: TRF_CAP, SLSQP_CAP.
- Statistic: cap100 SLSQP 338 vs TRF 320 (CI[-0.070,-0.021]); cap200 324 vs 321 (CI[-0.026,0.010]).
- Source: `results/p4b3c/p4b3c_fresh_report.json`.
