# PAPER_FIGURE7_MATERIALS — P4-G 生成器与冻结/重拟合证据

> 只读汇总。所有数值取自报告 JSON 与生成器源码。核心结论：#103 的 Fixed 方法依赖**同场景特权 IDEAL 参数**，必须标注为 privileged same-scene source，不可宣称可部署零样本迁移。

## 1. 物理参数表（单位已标注）

| 臂 | viscous_rate_s_inv (s⁻¹) | dry_acc_ms2 (m/s²) | velocity_smoothing_ms (m/s) | actuator_tau_s (s) |
|---|---|---|---|---|
| IDEAL_ODE | 0.0 | 0.0 | 0.05 | 0.0 |
| VISCOUS_ONLY | 0.08 | 0.0 | 0.05 | 0.0 |
| DRY_ONLY | 0.0 | 0.02 | 0.05 | 0.0 |
| LAG_ONLY | 0.0 | 0.0 | 0.05 | 0.025 |
| SHIFTED_ODE | 0.08 | 0.02 | 0.05 | 0.025 |

含义：viscous_rate = 线性粘滞阻尼率；dry_acc = 干摩擦（库仑）加速度幅值；velocity_smoothing = tanh 干摩擦平滑速度尺度；actuator_tau = 一阶执行器滞后时间常数。来源：`98_p4g1a...py` 的 `ARMS`、`p4g1a_feasibility_report.json`、`p4g2a_diagnostic_report.json`。

## 2. 生成器物理方程（`98_p4g1a_independent_physics_generator_feasibility.py`）

- 状态 `y = (x_a, x_b, v_a, v_b, u_a)`；`Δx = x_b − x_a`，`Δv = v_b − v_a`。
- 恢复力（B 上，`force_b`）：
  - elastic：`F = −k·Δx − c·Δv`
  - deadzone：`F = 0 if |Δx| ≤ g else −k·sign(Δx)·(|Δx|−g) − c·Δv`
  - nonlinear(cubic)：`F = −k·Δx − k3·Δx³ − c·Δv`
- 地面阻力：`drag(v) = v_rate·v + f_rate·tanh(v / v_smooth)`
- 执行器滞后：`du_a/dt = (u_desired − u_a)/τ`（τ=0 时 u_a = u_desired）
- 动力学：`a_a = (u_a − F)/m_a − drag(v_a)`；`a_b = F/m_b − drag(v_b)`；RK4 积分（DT=0.002, 500 步, 每 10 步采样）。

## 3. #98 P4-G1A 可行性（生成器真实性）

- 报告：`p4g1a_feasibility_report.json`；源码：`98_p4g1a_...py`（`source_sha256` 内嵌）。
- `ideal_vs_shifted_B_rms_mm_median = 23.667`；`shifted_half_vs_full_B_rms_mm_median = 198.797`。
- `max_passive_within_case_difference_m = 0.0`、`max_passive_cross_law_difference_m = 0.0`（被动无机制泄漏）。
- 声明：**单一 RK4 引擎两臂**，非两台独立引擎（`cross_simulator_claim: false`）。

## 4. #99 P4-G1B（IDEAL vs SHIFTED）主对比

报告 `p4g1b_fresh_report.json`。主终点（n=96 nonlinear-M2，FULL−HALF 剂量）：

| 臂 | half | full | net | effect | 95% CI |
|---|---|---|---|---|---|
| IDEAL_ODE | 79 | 83 | +4 | +0.0417 | [−0.0521, 0.1354] |
| SHIFTED_ODE | 80 | 85 | +5 | +0.0521 | [−0.0521, 0.1563] |
| SHIFT−IDEAL DD | — | — | +1 | +0.0104 | [−0.0417, 0.0729] |

机制级（WOO MATCHED5 MID_PLUS_FULL，elastic 正确数）：IDEAL=62/96，SHIFTED=42/96（SHIFTED 主要损害 elastic）。bootstrap：24 簇、2000 次、seed 2029122702。SHIFTED 剂量效应 CI 含 0，**未确立**。

## 5. #100 P4-G1C（物理漂移下的类别变化）

报告 `p4g1c_closure_report.json` → `cross_arm_shift_exploratory`（SHIFTED−IDEAL，WOO MATCHED5）：

| 机制 | net | pp | 95% CI (posthoc) |
|---|---|---|---|
| elastic | −20 | −20.83 | [−28.125, −13.542] |
| deadzone | +2 | +1.39 | [0.0, 3.472] |
| nonlinear | 0 | 0.0 | [−2.778, 2.778] |
| all | −18 | −4.69 | [−6.510, −3.125] |

结论：物理漂移（地面阻力+执行器滞后）几乎**只**损害 elastic 的识别（−20/96）。

## 6. #101 P4-G2A（五物理因素对照，冻结 IDEAL 参数）

报告 `p4g2a_diagnostic_report.json`。MID_PLUS_FULL 正确数（/384）与 elastic（/96）：

| 因素 | 总 correct | elastic correct |
|---|---|---|
| IDEAL_ODE | 331 | 62 |
| VISCOUS_ONLY | 344 | 80 |
| DRY_ONLY | 343 | 73 |
| LAG_ONLY | 332 | 63 |
| SHIFTED_ODE | 346 | 86 |

`IDEAL_to_factor_classification_transitions`（elastic，MID_PLUS_FULL）：VISCOUS gained +19、DRY gained +11、SHIFTED gained +25（粘滞/组合使 elastic 识别反而上升）。注意：`is_posthoc_exploratory: true`（事后探索，非预注册确认）。

## 7. #102 P4-G2B（固定 vs 重拟合，事后对照）

报告 `p4g2b_closure_report.json` → `paired_fixed_vs_refit`（MID_PLUS_FULL）：

| 组 | net(fixed−refit) | effect | fixed 真律 RMS(mm) | refit 真律 RMS(mm) |
|---|---|---|---|---|
| ALL (n=384) | +33 | +0.0859 | 11.015 | 9.787 |
| elastic (n=96) | +44 | +0.4583 | 11.476 | 10.027 |
| deadzone (n=144) | −4 | −0.0278 | 11.473 | 9.767 |
| cubic (n=144) | −7 | −0.0486 | 10.251 | 9.646 |

**关键反例（拟合改善 vs 选择反转）**：refit 真律 RMS 全面更低（拟合更好），但 elastic 上 fixed 正确率反而领先 +0.4583（44/96）；`critical_limits` 明示 "Lower true-model prediction RMS does not imply a larger model-selection margin"。CI（posthoc，seed 2029122901）：elastic [0.3542, 0.5625]。

## 8. #103 P4-G2C（新场景确认 + 逐场景配对 + bootstrap）

报告 `p4g2c_fresh_report.json`。主/次终点：

| 对比 | a(fixed) | b(refit) | net | effect | 95% CI |
|---|---|---|---|---|---|
| PRIMARY ALL FULL (n=384) | 347 | 327 | +20 | +0.0521 | [0.0130, 0.0885] |
| SECONDARY elastic FULL (n=96) | 79 | 47 | +32 | +0.3333 | [0.2188, 0.4479] |

bootstrap：`base_scene`，24 簇，2000 次，seed 2030011702，paired percentile 双侧。**逐案例 CSV**（384 行）：`p4g2c_384case_results.csv`（含 scene_id、true_law、FIXED/REFIT winner、correct、true_law_rms、REFIT_better_fit、REFIT_worse_selection）。elastic 反转：64/96 refit 拟合更优、37/96 refit 选择更差、**19/96 两者同时发生**。

## 9. #104 P4-G2D（最终只读审计）

报告 `p4g2d_closure_report.json`：`status = PASS_WITH_PRIVILEGED_SOURCE_AND_NUMERICAL_LIMITATIONS`。`interpretation`：
- `same_scene_ideal_source_privilege: YES`
- `deployable_zero_shot_superiority: NOT_ESTABLISHED`
- `two_independent_engines: NO`（单 RK4 实现两臂）
- `global_SLSQP_optimum: NOT_ESTABLISHED`；`E4_sealed: true`

## 10. bootstrap 记录汇总

| 实验 | 单位 | 簇数 | 次数 | seed | 状态 |
|---|---|---|---|---|---|
| #99 | base_scene | 24 | 2000 | 2029122702 | pre-frozen |
| #101 | — | 24 | — | — | posthoc exploratory |
| #102 | scene_clusters | 24 | 2000 | 2029122901 | post-hoc descriptive |
| #103 | base_scene | 24 | 2000 | 2030011702 | paired percentile 双侧（预注册） |

## 11. 逐案例数据定位

- **#103（固定 vs 重拟合，384 行）**：`results/p4g2c/p4g2c_384case_results.csv`（已提取，含可用性 correct 与 RMS）。
- **#99（IDEAL vs SHIFTED 逐案例）**：`results/p4g1b/checkpoints/scene_00..23.json` 的 `rows`（未单独导出；报告 `stats` 已给配置级/机制级聚合）。
- **#101（五因素逐案例）**：`results/p4g2a/checkpoints/scene_00.json` 仅 1 场景（部分）；聚合在报告 `summary`。
