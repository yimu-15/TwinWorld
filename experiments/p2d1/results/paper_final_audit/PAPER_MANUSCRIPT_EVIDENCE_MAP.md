# PAPER_MANUSCRIPT_EVIDENCE_MAP — Direct-to-manuscript evidence mapping (English)

> Read-only audit. Every paper claim below is traceable to a source file + JSON field. No number is invented.

## 1. Central scientific question (1)

**How should an embodied agent choose physical interventions and parameter priors to identify the hidden dynamical mechanism of a passively indistinguishable system, and what are the hard limits of that identifiability under weak excitation, open-world mechanisms, and imperfect solvers?**

Source: `docs/PROTOCOL_V1.md`; motivated by P0 (01–03), P1 (06–09), P2 (10–19); limits from P3-A7F0(61), P4-M1A(72), P3-A6L(52).

## 2. Three main contributions (candidates)

- **C1 (fair, strongest):** A physically-feasible outer-box parameter prior yields reproducible identification-accuracy gains over a legacy wide domain across independent scenes.
  - Evidence: P4-M2B2B(79) outer−wide +0.0833, CI[0.051,0.116]; P4-M2C2E(86) OOO−WWW +0.1797, CI[0.128,0.232].
  - Source fields: `results/p4m2b2b/p4m2b2b_fresh_report.json` primary; `results/p4m2c2e/p4m2c2e_fresh_report.json`.

- **C2 (fair):** Intervention amplitude/dose has a monotone, reproducible effect on mechanism discrimination.
  - Evidence: P4-M2D2B(90 v2) recall full−half +0.198, CI[0.094,0.313], separation +10.2mm; P4-M2D1(88) HIGH−MID +0.120, CI[0.081,0.156].
  - Source fields: `results/p4m2d2b/p4m2d2b_v2_fresh_report.json` primary/secondary; `results/p4m2d1/p4m2d1_fresh_report.json`.

- **C3 (secondary, careful):** Evidence-gated, mechanism-aware stop/strong control saves intervention work with near-zero identification loss.
  - Evidence: P3-A4C(31) prospective E4 diff −0.0023mm CI[-0.0234,0.0239], work saved −0.291J; P3-A4D(32) zero mis-stop vs old policy 15.
  - Source fields: `results/p3a4c/p3a4c_prospective_report.json`.

> Explicitly NOT a contribution: adaptive action selection beating the best fixed action (P3-A8C2B fails); frozen-IDEAL transfer (P4-G2C is privileged).

## 3. Five most prominent empirical results

1. Passive observations are at chance; B-observation + active pulse recovers mechanism (P2-C0 B_only 0.878 CI[0.822,0.933]).
2. Physical outer-box prior beats wide domain reproducibly (C1 above).
3. Amplitude dose-response: 0.5x→1.0x raises nonlinear-M2 recall +0.198 (C2 above).
4. Output-error least squares (TRF/SLSQP) beat equation-error (EQERR) at WOO/FULL: EQERR 313 vs 328, net −15 CI[-0.070,-0.005] (P4-B2B 106).
5. At equal hard budget 100, SLSQP beats TRF (−18/384, CI[-0.070,-0.021]); at 200 they tie (P4-B3C 110).

## 4. Three anti-intuitive / negative findings

1. **Lower fit error ⇒ worse mechanism selection** (P4-G2B 102): refit RMS 9.79<11.02mm but fixed identification effect +0.0859; elastic sub-group fixed +0.4583.
2. **Adaptive action selection does not beat the strongest fixed action** (P3-A8C2B 71): 174<180; disagreement margin cannot rank actions (P3-A8C3A0 78, Spearman CI crosses 0).
3. **Low-excitation structural ambiguity is physical, not algorithmic** (P3-A6K/A6L 50-52): saturating vs same-parameter elastic differ by only ~9.88mm (near 5mm noise).

## 5. Strongest fair baseline comparison

Physical outer-box vs legacy wide domain (C1) — fair, non-privileged, replicated across independent scenes. This is the headline baseline comparison for the abstract/contributions.

## 6. Method-section equations / algorithm definitions (required)

- **P0 core physics** (`scripts/twincore.py`): 1D two-body spring-damper with `coupling = k·Δx + d·Δv` for coupled world, `0` for independent; `Δx = x_B − x_A − distance`; acceleration `a_A=(F+coupling)/m_A − γ·v_A`, `a_B=−coupling/m_B − γ·v_B`; Euler step with DT=0.02.
- **P3/P4 candidate laws**: elastic (`F = k·x`), deadzone (`F = k·(x−g)·[|x|>g]`), cubic (`F = k·x + c·x^3`) — recover via `scripts`/experiment docstrings; parameter domains wide / outer_box.
- **Model-selection baselines** (P4-B1): train-set PEM / Gaussian AIC / BIC; residual n=204 B-position residual vector.
- **Classical identification** (P4-B2/B3): equation-error `EQERR_SG` (grid), output-error least squares via `TRF` / `SLSQP` (scipy), 5 starts, per-start hard caps.
- **Scoring/CI**: paired scene-cluster percentile bootstrap, 24 clusters, 2000 resamples.

## 7. Main Results tables → see PAPER_MAIN_TABLES.csv

## 8. Ablation tables → see PAPER_ABLATION_RESULTS.csv

## 9. Discussion failure cases

- Low-excitation saturating/elastic ambiguity (P3-A6K/L).
- Nested-domain solver violations persist (P4-S1B/C, 33–36).
- Privileged frozen-IDEAL advantage not deployable (P4-G2C).
- Fit-error vs selection counterexample (P4-G2B).

## 10. Figure 1–6 data index → see PAPER_FIGURE_DATA_INDEX.md
