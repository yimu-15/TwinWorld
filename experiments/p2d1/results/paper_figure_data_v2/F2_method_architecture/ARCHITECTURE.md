# F2 — Method Architecture (read-only documentation)

> This documents the ACTUAL experimental workflow implemented in the source. No adaptive algorithm is invented.

## 1. Experimental workflow (as implemented)

```
P0 (scripts/twincore.py + 01–04): CPU 1D two-body spring-damper
  generate matched-passive twin pairs -> validate passive equivalence -> baselines
P1 (05–09): MuJoCo rigid bodies + hidden coupling; match A motion with PD; noise + unknown-param diagnostics
P2 (10–19): 3/5 mechanisms, open-set rejection, probe/action catalog (B-observation + active pulse)
P3 (20–71): active mechanism identification
  A0–A2  known-law parameter refit + restricted structural revision (deadzone/cubic)
  A3–A4  evidence-gated stop/strong intervention
  A5–A6  identifiability signal, open-world, calibrated revision, excitation-shift failure attribution
  A7–A8  action families (high/reverse/biphasic), affine gain predictor, bidirectional selection
P4 (72–110): four closing lines
  M  physical prior / parameter domain / amplitude dose
  S  solver budget + nested-domain violations
  T  predictive margin consistency
  B/G classical system-ID baselines + independent ODE generator frozen-vs-refit
```

## 2. Candidate mechanism (restoring law) definitions

- **elastic** — linear spring: `F(x) = k * x`  (parameters: k)
- **deadzone** — gap with linear stiffness beyond threshold: `F(x) = k * (x - g) * 1[|x| > g]`  (parameters: k, g)
- **cubic** — linear + cubic nonlinearity: `F(x) = k * x + c * x^3`  (parameters: k, c)
- Diagnostic non-candidate modes (not fitted): **independent**, **rigid**, **nonlinear** (used as truth modes).
- Source of truth: experiment scripts 74–90 and report `candidate_laws` fields (e.g. `results/p4m2b2b/p4m2b2b_fresh_report.json`).

## 3. Parameter-domain definitions

- **wide** — legacy broad box (unconstrained physical feasibility).
- **outer_box** — physically-feasible box (explicit physical constraints).
- **shared_mass** — outer_box plus shared-mass constraint variant.
- Config shorthands **WWW / OOO / WOO** encode per-candidate domain choices (W=wide, O=outer_box) across the three candidates.
- Source: `results/p4m2b2b/p4m2b2b_fresh_report.json` (`domains`, `paired_comparisons`), `results/p4m2c2e/p4m2c2e_fresh_report.json`.

## 4. Fitting algorithms

- **EQERR_SG** — equation-error method: Savitzky-Golay (window 11, poly 3) differentiates noisy positions to acceleration, then constrained linear regression over a 21-point deadzone-gap grid.
- **TRF / TRF_OUT / TRF_CAP** — scipy `least_squares` Trust-Region-Reflective; 5 starts; per-start max_nfev=150 (or hard caps 20/40 in P4-B3); objective = 204 B-position residual vector (incl. numerical Jacobian calls).
- **SLSQP / SLSQP_MATCHED5_REFERENCE / SLSQP_CAP** — scipy `SLSQP`; 5 matched starts; maxiter=150; same 204-residual objective.
- Source: `results/p4b2b/p4b2b_fresh_report.json` (`EQERR`, `TRF`, `SLSQP`), `results/p4b3c/p4b3c_fresh_report.json` (`compute_cost`, `solver_interface`).

## 5. Scoring and selection functions

- **Identification score** — candidate with minimum heldout B-position RMS (finite feasible) is the winner; strict convergence variants require optimizer convergence.
- **Model-selection baselines (P4-B1)** — train-set PEM, Gaussian AIC, Gaussian BIC over the n=204 B-position residual vector.
- **Calibrated revision (P3-A6I)** — BIC-anchored switch rule (`tau_absolute`, `tau_switch`) for known-vs-unknown.
- Source: `results/p4b1b_v2/p4b1b_v2_fresh_report.json`, `results/p3a6i/p3a6i_fresh_report.json`.

## 6. Relevant source-code paths

| Component | Path |
|---|---|
| P0 core physics | `scripts/twincore.py` |
| P0/P1/P2 scripts | `scripts/01_generate.py` … `scripts/14_p2d0_probe_catalog.py` |
| P3/P4 scripts | `experiments/p2d1/20_p3a0_parameter_refit.py` … `110_p4b3c_fresh_equal_budget_confirmation.py` |
| Frozen protocol | `docs/PROTOCOL_V1.md` |
| Dataset config | `data/generation_config.json` |

## 7. Explicit non-claims

- No adaptive action-selection algorithm is claimed (P3-A8C2B did NOT beat the best fixed action).
- No global identifiability theorem; no optimal active planner; no RGB/real-robot pipeline.
