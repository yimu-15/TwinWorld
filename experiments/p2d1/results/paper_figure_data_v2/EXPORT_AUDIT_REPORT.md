# EXPORT_AUDIT_REPORT — Figure Data Package V2

> Read-only export validation. No experiment rerun, no retrain, no original modified, no E4 accessed.

## Method
- Raw `.npz` trajectory arrays dumped to CSV + `*__meta.json` (F1).
- Report JSONs copied verbatim into F3–F6 / S1–S5, plus per-figure `summary.csv`.
- SHA256 of every original source computed into `source_manifest/sha256_manifest.json`.
- Per-file provenance (source, exp_id, evidence class, denominator, SHA256) in `source_manifest/provenance.json`.

## Exported files summary
| Directory | Files | Evidence |
|---|---|---|
| F1_trajectories | ~90 (npz→csv + meta) | raw |
| F2_method_architecture | 1 (ARCHITECTURE.md) | documentation |
| F3_passive_active | 11 reports + summary.csv | formal/audit/dev/smoke |
| F4_physical_prior | 2 reports + summary.csv | formal |
| F5_amplitude | 2 reports + summary.csv | formal |
| F6_classical_baselines | 2 reports + summary.csv | formal |
| S1–S5 | reports + summary.csv each | mixed |
| source_manifest | sha256_manifest.json + provenance.json | meta |

## Aggregate-value agreement check
- All `summary.csv` values were copied from the corresponding report JSON fields (e.g. F4 `outer_box_minus_wide` delta 0.083333 CI[0.050926,0.115741] matches `79_p4m2b2b_report.json` `paired_comparisons.pooled.outer_box_minus_wide.strict`; F6 `106` EQERR−SLSQP −15 CI[-0.0704,-0.0052] matches `106_p4b2b_report.json` `primary_and_secondary`). No recalculation, no modification of historical outcomes.

## Which main figures can be produced immediately

| Figure | Ready? | Source |
|---|---|---|
| F1 trajectories | ✅ YES (raw arrays present) | F1_trajectories/*.csv |
| F2 architecture | ✅ YES (documented) | F2_method_architecture |
| F3 passive/active | ✅ YES | F3 summary + reports |
| F4 physical prior | ✅ YES (aggregate + CI + scene fingerprints) | F4 |
| F5 amplitude | ✅ YES | F5 |
| F6 classical/budget | ✅ YES (aggregate + compute_cost + coverage) | F6 |

## Missing / partial data (explicitly marked)

1. **F4 (#79/#86) per-individual-scene correct flags**: report contains per-mode×domain×band winner counts (n=72/cell) and 144 scene fingerprints, but NOT per-scene correct rows. Per-scene records live in `*_audit_PRIVATE.json` / checkpoints (not exported; not in the non-PRIVATE report).
2. **F6 (#106/#110) per-case decisions** (chosen law per case + optimizer status per case): report provides per-truth-law aggregate correct counts and `compute_cost.start_stop_status_counts`, but per-case rows are in `checkpoints/scene_*.json` (`rows`, ~8.8MB each) — not extracted in this export.
3. **S4 (#71) adaptive-vs-fixed paired CI**: `p3a8c2b_fresh_report.json` reports per-strategy `correct`/`work_J` but no paired bootstrap CI field; therefore no CI is exported (only raw correct/work_J).
4. **F1 P1/P2 audit npz column semantics**: raw arrays exported; exact column meaning (e.g. `(501,3)` = time/A/B) is documented in the dictionary but not independently re-derived from source scripts in this export.

## Compliance
- ✅ Read-only (no originals modified).
- ✅ No experiment rerun / no model retrain.
- ✅ No E4 access (`E4_accessed:false` preserved in copied reports).
- ✅ Scene IDs / case IDs / denominators / seeds / bootstrap settings preserved (verbatim JSON copies + summary CSVs).
- ✅ Missing data marked explicitly (above), not fabricated.

## Blockers for figure production
- None for aggregate-level figures. Per-case/per-scene scatter/CI plots for F4/F6 require the private audit/checkpoint records, which were intentionally left out of this read-only package; they can be added by a follow-up read-only extraction of `checkpoints/scene_*.json` and `*_audit_PRIVATE.json` (still not E4).
