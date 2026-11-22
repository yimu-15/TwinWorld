# PAPER_COMPUTE_COST_AUDIT — compute_cost 原始记录

> 只读汇总。所有数值逐字段取自各报告 JSON 的 `compute_cost` / `function_evaluations` / `solver_nfev_*` 字段。关键结论：不同方法的"调用/评估"量纲不可直接比较；"等硬上限" ≠ "等实际调用" ≠ "等墙钟时间"。

## 1. P4-B2B (#106) 经典辨识方法 — 算力量纲不可比

| Method | 成本量 | 数值 | wall (s) | cost_unit |
|---|---|---|---|---|
| EQERR_SG | total_regression_grid_evaluations | 17,664 | 6.617 | 约束线性回归/网格点（**非**残差 nfev） |
| TRF_OUT | total_optimizer_objective_calls | 463,732 | 1,155.592 | 全部 TRF 残差调用（含数值导数） |
| SLSQP_MATCHED5_REFERENCE | total_optimizer_objective_calls | 918,685 | 2,294.920 | SLSQP 目标 nfev（2 原始 + 3 额外起点；墙钟含内部 heldout 评分） |

nested_violations (deadzone)：EQERR=66、TRF=21、SLSQP=26。

## 2. P4-B3C (#110) 等硬预算优化器 — 等上限 ≠ 等实际调用

| budget | solver | fit_units | per_start_cap | max_calls | actual_calls | unused | optimizer_success | budget_cap | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| 100 | TRF_CAP | 1152 | 20 | 115200 | 114350 | 850 | 338 | 5422 | 225.474 |
| 100 | SLSQP_CAP | 1152 | 20 | 115200 | 111199 | 4001 | 632 | 5128 | 217.115 |
| 200 | TRF_CAP | 1152 | 40 | 230400 | 189743 | 40657 | 4073 | 1687 | 378.065 |
| 200 | SLSQP_CAP | 1152 | 40 | 230400 | 193082 | 37318 | 2830 | 2930 | 377.369 |

（每 solver `optimizer_start_records=5760`；`unavailable_fits=0`、`invalid_residual_calls=0`；优化器成功/预算截断与机制识别正确是**分离**指标。）

## 3. P4-G2C (#103) 冻结 vs 重拟合 — 函数评估量

| 生成器臂 | 域 | original | extra |
|---|---|---|---|
| IDEAL_ODE | wide | 245,084 | 365,865 |
| IDEAL_ODE | outer_box | 132,906 | 205,686 |
| SHIFTED_ODE | wide | 223,390 | 342,280 |
| SHIFTED_ODE | outer_box | 135,750 | 205,361 |

（`inference = P22 A-conditioned B, P77/P92 five-start SLSQP`；MATCHED5 匹配起点/maxiter，**不**匹配实际函数评估。）

## 4. P4-B3A (#108) 等预算 pilot（1 scene, 3 cases，仅可行性）

| budget | TRF_CAP actual | SLSQP_CAP actual |
|---|---|---|
| 100 | 896 | 899 |
| 200 | 1,535 | 1,591 |

（报告声明 `no_confirmatory_inference: true`；3 个历史 case 仅为预检，非代表性准确率样本。）

## 5. P4-S1B (#92) / P4-S1A (#91) 求解器预算

| 实验 | wide nfev | outer_box nfev | 备注 |
|---|---|---|---|
| #92 (full, 384 cases) | 611,022（extra 370,296） | 333,855（extra 203,029） | matched5 起点，nested 违例 36→33 |
| #91 (diagnostic, 37 targets) | 41,728 | 36,384 | +3 共享起点后仍剩 25 处 nested 违例 |

## 关键公平性结论（论文必须声明）

1. **算力量纲不可比**：EQERR（网格点）≠ TRF（残差调用）≠ SLSQP（nfev）≠ 函数评估——不得用这些数字对方法做"计算效率排序"。
2. **等硬上限 ≠ 等实际调用**（#110 cap200：TRF 189,743 vs SLSQP 193,082；cap100：114,350 vs 111,199）。
3. **等硬上限 ≠ 等墙钟**（#110 墙钟仅作描述性记录）。
4. **优化器成功 ≠ 机制识别正确**（#110 optimizer_success 与 correct 分别统计）。
5. **MATCHED5 匹配起点/迭代上限，不匹配实际 nfev**（#103、#92）。
