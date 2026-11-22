# PAPER_MANUSCRIPT_READINESS — Readiness assessment (English)

> Read-only audit of whether the evidence supports a high-quality SCI submission.

## Verdict: CONDITIONALLY READY (blocked by E4 gate + SHA recomputation)

## What is ready
- Complete experiment registry (1–110) with type/phase/status classified.
- Structured scientific findings across 12 categories, each with source + CI + counterexample.
- Baseline/ablation comparisons with fairness audit and explicit privileged/not-fair flags.
- Negative results and protocol deviations documented honestly.
- Manuscript evidence map, main tables, figure data index, contribution candidates.

## What blocks final submission
1. **E4 final independent evaluation not executed** (frozen protocol requires it pre-submission). → BLOCKED.
2. **Full source SHA256 not independently recomputed** (P0–P3 lack per-experiment manifests).
3. **Nested-domain solver violations not eliminated** (25–36 deadzone cases; global optimum uncertified) — must be disclosed, not fixed in this audit.
4. **Several early P3 experiments have no formal report** (23,28,33–39,41,43,44,46,49,53) — cannot be cited as formal evidence.

## Evidence quality grades
- P4 fresh/confirm experiments (79,86,88,90,92,97v2,99,103,106,110): **HIGH** (paired scene-cluster CI, protocol-frozen, explicit limitations).
- P3 prospective/confirm experiments (31,32,45,47,48,50,69): **MEDIUM-HIGH**.
- P0/P1/P2 and P3 dev/pilot: **MEDIUM** (motivating/developmental).
- Read-only/PRIVATE-only experiments: **NOT FORMAL** (excluded from claims).

## Required pre-submission actions (not performed here)
1. Execute the frozen final independent E4 evaluation.
2. Recompute SHA256 for all 110 scripts and freeze a full source manifest.
3. Optionally add a solver-exhaustive/global check to resolve or bound the nested-domain violations.

## Bottom line
The audit evidence is sufficient to draft a defensible manuscript whose main claims are: (C1) physical outer-box prior improves identification; (C2) intervention amplitude has a reproducible dose-response; plus honest negative/limit findings. Final submission is **BLOCKED_FOR_FINAL_SUBMISSION** until the E4 gate and SHA recomputation are completed.
