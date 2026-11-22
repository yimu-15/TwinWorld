# TwinWorld Research Protocol V1 — frozen before model selection

## Aim
Investigate whether an agent can use interventions to discriminate mechanisms that are observationally indistinguishable under an explicitly specified passive-action distribution, then make evidence-grounded revisions to its structured concept/relationship model.

## Scientific cautions
- Passive equivalence is within the constructed experimental protocol, NOT universal metaphysical indistinguishability.
- Paired no-action and action rollouts in P0 are diagnostic privileges, NOT available by default to future autonomous agents.
- At P0, labels reveal simple coupled/independent alternatives preselected by the researcher; this is NOT open-ended concept invention.
- Exact paired passive equality is constructed. Do not report chance performance on these pairs as an empirical discovery.
- A sufficiently expressive latent recurrent network may solve the task without explicit relation symbols. Capacity-matched baselines are mandatory.

## P0 verification (must all pass)
1. For every pair, full passive trajectory equality under label=0 and label=1 (strict, before adding noise).
2. Same action sequence applied to both worlds of each pair.
3. Active trajectories reliably diverge with an observation signal above chosen sensor noise.
4. Class labels balanced, world order shuffled within each pair; avoid labels, params, filenames, simulator configurations in model input.
5. Passive majority-class baseline at 50%; intervention-based diagnostic can identify world type.
6. No claims of learning, innovation, or superior scientific discovery from this phase.

## P1 and above: controls
- Explicit intervention budget, observation modality, environmental parameters, reset capability, force bounds, sensor noise, camera and backgrounds.
- Model and investigator cannot inspect hidden mechanism or simulation internals during test evaluation. Private generator/evaluator only.
- Split across scene parameters and *held-out mechanism compositions*, avoid random-frame split leakage.
- 5 or more random seeds for train/evaluation stochasticity; multiple environment seeds; paired bootstrap intervals over scene groups.
- Match parameter counts, training data, compute, action budget and labels/privileged state availability when claiming fairness.
- Primary metrics: active structural recovery accuracy or equivalence-class recovery, cost-adjusted identification rate and intervention budget; secondaries: false revision rate, calibration, OOD prediction MSE, downstream control success, inference expense.
- Report uncertainty/abstention when interventions cannot distinguish models, and performance when correct answer lies outside the allowed edit grammar.
- Pre-specify P1/P2 thresholds and do not modify them after looking at test labels.

## Promotion gates (planning criteria, not existing results)
P0: all six verification checks pass.
P1: paired mechanisms remain sensor-indistinguishable without action; show divergence after viable actions with a noise-aware SNR analysis.
P2: run matched baselines, including a non-symbolic high-capacity recurrent predictor and active system identification.
P3: compare revision algorithm to strongest matched baseline; use paired confidence intervals and independently held-out mechanism compositions. No claim of positive gain if intervals overlap null.
P4: show visual rather than ground-truth state performance, viewpoint robustness, latency and control transfer; quantify perception errors.
