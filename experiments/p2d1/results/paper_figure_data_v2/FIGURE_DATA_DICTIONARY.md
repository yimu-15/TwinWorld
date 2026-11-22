# FIGURE_DATA_DICTIONARY — TwinWorld Figure Data Package V2

> Read-only export. Every file is traceable to an original source. No data was fabricated; missing per-scene/per-case detail is marked explicitly.

## Data types and semantics

### F1 — Trajectories (raw npz → CSV)
- Directory: `F1_trajectories/`
- Each original `.npz` was dumped key-by-key to `*__<key>.csv` (flattened, one value per row with an `index` column) plus a `*__meta.json` recording keys/shapes/dtypes and per-CSV SHA256.
- P0 (`data/*_public.npz`): `passive`/`active` have shape `(N, 120, 2)` = `(sample, timestep, {A,B})`; `action` shape `(N,120)`; `pair_id` shape `(N,)`. Time step DT=0.02 (from `data/generation_config.json`), so `time = timestep * 0.02 s`. World identity (coupled=1 / independent=0) is in `data/*_private_labels.npz` (`label`).
- P1/P2 audit (`outputs/p1b`, `p2a`, `p2d0`, `p2d1a`, `p2d1b`): raw arrays exported; column semantics follow generator (see each `*__meta.json` + report). `p2b` observations are `passive_xy`/`active_xy` `(N,51,2)` and `commanded_accel` `(N,51)`.
- NOTE: the flattened CSV preserves values exactly; reconstruct shape from the `shape` field in `*__meta.json`.

### F2 — Method architecture
- Directory: `F2_method_architecture/ARCHITECTURE.md` (single authoritative doc): workflow, candidate laws (elastic/deadzone/cubic), parameter domains (wide/outer_box/shared_mass), fitting algorithms (EQERR_SG/TRF/SLSQP), scoring/selection, source-code paths, and explicit non-claims.

### F3 — Passive/active identification
- Directory: `F3_passive_active/`
- `summary.csv` columns: `condition, accuracy, correct, total, ci95_low, ci95_high, privilege, source_report, exp_id`.
- Verbatim copies of P0/P1/P2 report JSONs are also present (e.g. `P0_baselines.json`, `P2C0_baselines.json`).

### F4 — Physical prior (#79, #86)
- Directory: `F4_physical_prior/`
- `summary.csv`: paired comparisons with `accuracy_a/accuracy_b/delta_accuracy/ci95/decision/n_pairs/n_clusters`.
- Verbatim `79_p4m2b2b_report.json` (full per-mode×domain×band `summary` + `paired_comparisons`) and `86_p4m2c2e_report.json`.
- Scene identifiers: `79_p4m2b2b_report.json` `scene_independence_audit.new_scene_fingerprints` (144 fingerprints). Individual-scene correct flags are aggregated into winner counts (n=72 per cell), not per-scene rows in the report.

### F5 — Intervention amplitude (#88, #90)
- Directory: `F5_amplitude/`
- `summary.csv`: interaction/HIGH−MID/recall/separation/score-gap with CI.
- Verbatim `88_p4m2d1_report.json`, `90_p4m2d2b_v2_report.json`.

### F6 — Classical identification & budget (#106, #110)
- Directory: `F6_classical_baselines/`
- `summary.csv`: per-comparison correct counts + effect + CI.
- Verbatim `106_p4b2b_report.json` (methods EQERR_SG/TRF_OUT/SLSQP, `classification`, `compute_cost`, `nested_violations`, `coverage`) and `110_p4b3c_report.json` (primary/secondary contrasts, `aggregate_classification`, `compute_cost` with `start_stop_status_counts`/`actual_204vector_calls`/`wall_seconds`, `full_denominator_coverage`, `bootstrap`).
- Per-case decisions (chosen law per case): available in original `checkpoints/scene_*.json` (`rows` array), NOT extracted here (8.8MB/scene). The report JSON provides per-method/per-truth-law aggregate correct counts and optimizer status counts.

### S1 — Fit vs selection (#102)
- Directory: `S1_fit_vs_selection/`; `summary.csv` + verbatim `102_p4g2b_report.json`.

### S2 — Open-world rejection (#47, #48, #50)
- Directory: `S2_open_world/`; `summary.csv` + verbatim reports.

### S3 — Solver audit (#91, #92, #93)
- Directory: `S3_solver_audit/`; `summary.csv` + verbatim reports.

### S4 — Adaptive action (#71)
- Directory: `S4_adaptive_action/`; `summary.csv` (per-band per-strategy correct + work_J) + verbatim report.

### S5 — Prior mismatch (#80, #81)
- Directory: `S5_prior_mismatch/`; `summary.csv` + verbatim reports.

## source_manifest
- `source_manifest/sha256_manifest.json`: SHA256 of every original source file referenced (npz, report JSON, twincore.py, PROTOCOL_V1.md). `MISSING` marks a source that did not exist at export time.
- `source_manifest/provenance.json`: per-exported-file record of `source`, `dst`, `exp_id`, `evidence` class (formal/dev/audit/diagnostic/retrospective), `type`, and `sha256`.

## Evidence classification legend
- **formal** — prospective/fresh/confirm experiment with protocol frozen before outcome.
- **dev** — development/pilot experiment (not confirmatory).
- **audit** — read-only/closure/retrospective audit (reuses historical queues).
- **diagnostic** — post-hoc / mechanism attribution.
- **smoke** — small-scale smoke test.
- **retrospective** — historical data reused after observation (not independent).
