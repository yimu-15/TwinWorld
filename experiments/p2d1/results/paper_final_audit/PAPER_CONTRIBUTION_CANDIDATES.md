# PAPER_CONTRIBUTION_CANDIDATES — Contribution statements & 3 title candidates (English)

> Read-only. Contribution statements are evidence-bounded; no unproven algorithm claims.

## Contribution candidates (ranked by fairness + reproducibility)

1. **Physical prior for mechanism identification** — We show that constraining parameter fitting to a physically-feasible outer-box domain yields reproducible identification-accuracy gains over an unconstrained wide domain across independent scenes (P4-M2B2B +0.0833 CI[0.051,0.116]; P4-M2C2E OOO−WWW +0.1797 CI[0.128,0.232]). *(fair, non-privileged)*

2. **Intervention amplitude as an identification knob** — We show a monotone, reproducible dose-response: increasing excitation amplitude raises recall of a nonlinear mechanism (P4-M2D2B +0.198 CI[0.094,0.313]) and high-vs-mid action improves identification (P4-M2D1 +0.120 CI[0.081,0.156]). *(fair)*

3. **Evidence-gated, mechanism-aware intervention economy** — We show a mechanism-aware stop/strong policy saves intervention work with near-zero identification loss (P3-A4C E4 diff CI[-0.0234,0.0239], work −0.291J; P3-A4D zero mis-stop). *(secondary, careful: not "beats best fixed action")*

## Anti-contributions (must NOT be claimed)
- Adaptive action selection beats best fixed action (P3-A8C2B fails).
- Frozen-IDEAL transfer as deployable advantage (P4-G2C privileged).
- Global structural identifiability theorem.
- Optimal active planner.
- Real-robot / RGB concept revision.

## 3 title candidates (avoiding unproven algorithm claims)

1. "Physical Priors and Intervention Amplitude for Identifying Passively Indistinguishable Dynamical Mechanisms"
2. "Intervention Design for Mechanism Identification in Passively Equivalent Physical Systems: Physical Priors, Amplitude Effects, and Identifiability Limits"
3. "How Much Excitation and What Prior? A Controlled Study of Active Mechanism Identification in Matched-Passive Physical Worlds"

## Evidence strengths (for journal fit)
- Strengths: rigorous paired scene-cluster bootstrap CI; explicit fairness disclosures; frozen-protocol prospective confirmations; honest negative results; sealed E4 gate.
- Weaknesses: synthetic (MuJoCo/RK4) only; no hardware; single-engine two-arm generator; nested-domain solver violations; several early P3 experiments lack formal reports; E4 final gate unexecuted.

## Fit assessment
- Suitable for: applied/computational venues focused on active learning, system identification, or embodied AI (empirical methodology + careful baselines + negative results).
- Not yet suitable for: top venues requiring real-robot validation or formal identifiability theorems, until E4 gate + full SHA recomputation are completed.
