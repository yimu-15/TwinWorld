# TwinWorld — Publication-Ready Tables 1–5

All values are copied verbatim from the original report JSONs / checkpoints. No value is pooled across physical implementations. "—" = not reported in source (no CI / not applicable), not a missing value to fabricate.

---

## Table 1. Passive versus active mechanism distinguishability across physical implementations

| Experiment | Condition | Metric | Value | Unit | 95% CI | Evidence Status |
|---|---|---|---|---|---|---|
| P0 | Passive trajectory | Maximum absolute difference (coupled vs independent) | 0.0 | m | — | Formal sanity-check |
| P0 | Active oracle diagnostic (val) | Identification accuracy | 1.0 | fraction | — | **Privileged (matched passive rollout) — non-deployable** |
| P0 | Active oracle diagnostic (test) | Identification accuracy | 1.0 | fraction | — | **Privileged (matched passive rollout) — non-deployable** |
| P1-B | Active intervention on A | B final displacement difference (coupled vs independent) | 0.250478 | m | — | Formal physical audit |
| P1-C | Matched A motion | A trajectory maximum gap | 0.001632 | m | — | Formal physical audit |
| P1-C | Divergent B response | B final gap | 0.196828 | m | — | Formal physical audit |
| P2-C0 | Passive-only | Accuracy (n = 90 episodes) | 0.3333 | fraction | [0.3333, 0.3333] | Formal (fixed supervised baseline) |
| P2-C0 | A-only | Accuracy (n = 90 episodes) | 0.4556 | fraction | [0.3556, 0.5556] | Formal |
| P2-C0 | B-only | Accuracy (n = 90 episodes) | 0.8778 | fraction | [0.8222, 0.9333] | Formal |
| P2-C0 | A+B | Accuracy (n = 90 episodes) | 0.8556 | fraction | [0.8000, 0.9111] | Formal |

**Footnotes.** (1) P0, P1, and P2 use different physical generators (CPU twincore vs MuJoCo) and evaluation protocols, so their results are NOT pooled. (2) P0 active oracle uses a privileged matched passive counterfactual and is not a deployable agent; P1-C PD controller uses privileged A state for control only. (3) P2-C0 95% CIs are pair-cluster bootstraps over 30 independent test scene groups (not 90 independent episodes); seed = 20261113. Sources: `outputs/p0_baselines.json`, `outputs/p1b/p1b_report.json`, `outputs/p1c/p1c_report.json`, `outputs/p2c0/p2c0_baselines.json`.

---

## Table 2. Physical prior and parameter-domain ablation (cohorts kept separate; no pooled effect)

| Experiment | Configuration (A vs B) | Indep. scenes | Accuracy A | Accuracy B | Paired effect | Scene-clustered 95% CI | Evidence |
|---|---|---|---|---|---|---|---|
| #79 (pooled) | OUTER_BOX vs WIDE | 72 clusters | 0.673611 | 0.590278 | +0.083333 | [0.050926, 0.115741] | Formal — SUPPORTED_PRACTICAL_GAIN |
| #79 (above_original_range) | OUTER_BOX vs WIDE | 72 clusters | 0.685185 | 0.574074 | +0.111111 | [0.069444, 0.157407] | Formal — SUPPORTED_PRACTICAL_GAIN |
| #79 (pooled) | SHARED_MASS vs OUTER_BOX | 72 clusters | 0.668981 | 0.673611 | −0.004630 | [−0.016204, 0.006944] | Formal — NOT_CONFIRMED |
| #79 (pooled) | SHARED_MASS vs WIDE | 72 clusters | 0.668981 | 0.590278 | +0.078704 | [0.048611, 0.111111] | Formal — SUPPORTED_PRACTICAL_GAIN |
| #86 (fresh) | OOO vs WWW | 24 clusters | 0.760417 (292/384) | 0.580729 (223/384) | +0.179688 | [0.127604, 0.231771] | Formal (fresh-scene replication) |

**Footnotes.** (1) #79 and #86 are independent cohorts; no pooled overall effect is reported. (2) #79 uses 72 base-scene clusters with 2 starts/150 iters SLSQP and the "mid" validation action; #86 uses 24 fresh base-scene clusters (M0 + fixed M2 mismatch), 384 cases. (3) SHARED_MASS configurations are included because they appear in the #79 formal report. (4) The physical (outer-box) prior does NOT universally improve identification under extreme prior mismatch (see `p4m2c0_dev_report.json`: deadzone alpha_high M3, outer_box 0.5 vs wide 0.917). Sources: `results/p4m2b2b/p4m2b2b_fresh_report.json`, `results/p4m2c2e/p4m2c2e_fresh_report.json`.

---

## Table 3. Intervention amplitude and mechanism discrimination (recall and separation NOT merged)

| Experiment | Mechanism | Amplitude Contrast | Primary Metric | Effect | 95% CI | Sample Unit |
|---|---|---|---|---|---|---|
| P4-M2D2B v2 | nonlinear-M2 | FULL (1.0×) vs HALF (0.5×) | Recall difference | +0.1979 | [0.0938, 0.3125] | 96 nonlinear-M2 cases |
| P4-M2D2B v2 | nonlinear-M2 | FULL (1.0×) vs HALF (0.5×) | Pairwise separation difference | +10.156 mm | [8.336, 12.146] mm | physical separation (mm) |
| P4-M2D1 | overall (elastic/deadzone/nonlinear) | HIGH vs MID | Identification accuracy | +0.1198 | [0.0807, 0.1563] | 384 cases (24 scenes) |

**Footnotes.** (1) Recall difference and physical separation are separate metrics and are not combined. (2) Only two amplitude levels are tested (0.5× and 1.0×); no continuous dose-response curve is inferred. (3) Original statistical denominators are preserved (96 nonlinear-M2 cases for M2D2B; 384 cases for M2D1). Sources: `results/p4m2d2b/p4m2d2b_v2_fresh_report.json`, `results/p4m2d1/p4m2d1_fresh_report.json`.

---

## Table 4. Classical mechanism-identification baselines (P4-B2B, experiment #106)

| Method | Correct/Total | Accuracy | Paired Difference | Effect | 95% CI | Evaluation Unit |
|---|---|---|---|---|---|---|
| EQERR_SG | 313/384 | 0.8151 | EQERR − SLSQP = −15/384 | −0.0391 | [−0.0704, −0.0052] | 384 cases, 24 scene clusters |
| TRF_OUT | 328/384 | 0.8542 | TRF − SLSQP = 0/384 | 0.0000 | [0.0000, 0.0000] | 384 cases, 24 scene clusters |
| SLSQP_MATCHED5_REFERENCE | 328/384 | 0.8542 | reference | — | — | 384 cases, 24 scene clusters |

**Footnotes.** (1) All methods use the same WOO domain and MID_PLUS_FULL heldout action; n = 384 with 24 base-scene bootstrap clusters (paired percentile CI, 2000 resamples, seed = 2030022302). (2) **Fairness disclosure:** the three methods use incomparable computational evaluation units — EQERR_SG = 17,664 grid/regression evaluations (~6.6 s); TRF_OUT = 463,732 residual calls (~1155.6 s); SLSQP = 918,685 objective nfev (~2294.9 s). This is NOT an equal-compute comparison. (3) No claim is made that the proposed framework outperforms all classical system-identification methods. Source: `results/p4b2b/p4b2b_fresh_report.json`.

---

## Table 5. Equal hard-cap optimizer comparison (P4-B3C, experiment #110)

| Budget Cap | Optimizer | Correct/Total | Accuracy | Actual Calls | Paired Difference | Effect | 95% CI |
|---|---|---|---|---|---|---|---|
| 100 | TRF_CAP | 320/384 | 0.8333 | 114350 | TRF − SLSQP = −18 | −0.0469 | [−0.0703, −0.0208] |
| 100 | SLSQP_CAP | 338/384 | 0.8802 | 111199 | — | — | — |
| 200 | TRF_CAP | 321/384 | 0.8359 | 189743 | TRF − SLSQP = −3 | −0.0078 | [−0.0260, 0.0104] |
| 200 | SLSQP_CAP | 324/384 | 0.8438 | 193082 | — | — | — |

**Footnotes.** (1) "Hard cap" = maximum residual-vector calls; the 204-position residual vector is the evaluation unit. Equal hard maximum call caps do NOT imply equal realized calls or equal wall-clock time (cap-200 realized calls: TRF 189,743 vs SLSQP 193,082; per-start caps 20/40, 5 starts). (2) Optimizer success and mechanism-identification correctness are kept as separate metrics (e.g. cap-200 TRF: optimizer_success = 4073, budget_cap = 1687 start records). (3) 24 base-scene clusters are the independent bootstrap unit; 384 rows are not IID. Source: `results/p4b3c/p4b3c_fresh_report.json`.
