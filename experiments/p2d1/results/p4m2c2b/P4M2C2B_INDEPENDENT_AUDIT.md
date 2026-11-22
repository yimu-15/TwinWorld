# P4-M2C2B 独立结果验收报告（Independent Audit）

- 审计对象：P4-M2C2B（Warm-Start Transfer Audit）正式实验结果
- 审计性质：**只读、独立、可复现** 验收
- 审计日期：2026-11-15
- 结论：**PASS**（全部核验项通过；无 FAIL、无 BLOCKED）

---

## 0. 审计范围与方法

本次审计未修改 P4-M2C2A / P4-M2C2B 的任何结果，未重跑优化器，未修改历史参数，未访问 Mid validation、E4 或测试集，未进行训练，未提交 Git。

核验方法：

1. **静态核验**：读取冻结的 JSON 结果（`p4m2c2a_dev_*`、`p4m2c2b_dev_*`、`p4m2c2b_smoke_*`）做字段级比对。
2. **独立复核**：使用冻结实现 `P74/P75/P77/P79`（经 `P80` 的 `check_provenance()` 与 `initialize()` 加载）重建原始 12 个开发场景与噪声探针，对 21 个 `final_params` **只重新计算目标函数**（全程不调用 `scipy.optimize.minimize`）。
3. **源哈希核验**：对冻结源文件与输入报告重新计算 SHA256，与记录值比对。

冻结源文件 SHA256（均已与报告记录值一致）：

| 文件 | SHA256（前 16 位） |
| --- | --- |
| `80_p4m2c0_prior_mismatch_stress.py` | `08fae735…` |
| `81_p4m2c1_prior_conflict_audit.py` | `77cac2ed…` |
| `77_p4m2b2a_matched_domain_fit.py` | `641be70d…` |
| `p4m2c2a_dev_report.json` | `a20438b8…` |
| `p4m2c2a_dev_audit_PRIVATE.json` | `e06f9710…` |

---

## 1. 完整性检查（A）— PASS

| 检查项 | 结果 |
| --- | --- |
| 正式报告 `n_cases == 21` | PASS |
| 正式审计 JSON 长度 == 21 | PASS |
| case 编号 1..21 连续 | PASS（`[1..21]`） |
| 无缺失 / 重复 / 异常记录 | PASS（21 条记录，21 个唯一 `(scene,band,mode,axis,level)`） |
| Smoke 与正式 Case 1 完全一致 | PASS（JSON 全等） |
| 报告计数器（`warm_success` 等）全部 == 21 | PASS |
| `law == "deadzone"` / `smoke == false` | PASS |
| `E4_accessed == false` / `uses_truth_in_fitting == false` / `uses_mid_validation == false` | PASS |

21 个案例仅覆盖 **8 个 distinct `scene_id`**：`{2, 3, 4, 5, 6, 7, 10, 11}`。

---

## 2. 参数来源核验（B）— PASS

逐案例核对结果：

| 检查项 | 结果 |
| --- | --- |
| `warm_start.initial_params == P4-M2C2A outer_box.best_valid.final_params` | PASS（21/21 精确相等） |
| `warm_start.initial_fun == historical_outer_fun == outer_box.best_valid.result_fun` | PASS（21/21） |
| `historical_wide_fun == P4-M2C2A wide.best_valid.result_fun` | PASS（21/21） |
| 21 案例 `(scene,band,mode,axis,level)` 与 P4-M2C2A 记录一致 | PASS |
| 21 案例 `historical_outer < historical_wide`（嵌套违规成立） | PASS |

历史 Outer 相对历史 Wide 的差距（`wide - outer`）：

- min = `2.62e-05`，median = `1.88e-03`，max = `3.15e-02`（Case 10）

**无测试/验证数据用于拟合**：目标函数仅使用 `TRAIN = (e1, e2, weak, strong)` 探针；`mode`（真值生成模式）仅用于生成动力学，不进入拟合；`VALID = "mid"` 探针未进入目标函数。报告字段 `uses_truth_in_fitting=false`、`uses_mid_validation=false` 与代码路径一致。

---

## 3. 目标函数独立复核（C）— PASS

使用冻结 P74/P75/P77/P79 与原始开发数据，对 21 个 `final_params` 重新计算目标函数：

| 检查项 | 结果 |
| --- | --- |
| 21 个重算目标函数 == `final_fun` | PASS（`max abs diff = 0.0`，精确一致） |
| 21 个 `final_params` 均为 Wide 可行 | PASS |
| 21 个 `final_params` 均在 Wide 边界内 | PASS |
| 21 个 `final_fun <= initial_fun`（不劣于初始解） | PASS |
| 21 个 `final_fun < historical_wide`（优于历史 Wide） | PASS |

Wide 边界（deadzone 候选律）：`alpha ∈ [0.01, 70]`、`beta ∈ [0.001, 15]`、`gap ∈ [0.001, 0.2]`。

Warm-start 优化相对初始解的额外改善（`improvement_from_seed`）：

- min = `7.53e-14`，median = `4.50e-09`，max = `3.30e-04`（Case 19）

---

## 4. 可行性核验 — PASS

21 个 `final_params` 均通过 `is_feasible("deadzone", params, "wide")`，且均位于 Wide 上/下界内；无边界或可行性违规。

---

## 5. 元数据问题（D）

- **全局 `law == "deadzone"`**：指被拟合的**候选模型律**（candidate law），全 21 案例统一为 deadzone。
- **Case 10 `mode == "nonlinear"`**：指该案例的**真值生成模式**（ground-truth mode，对应 cubic 动力学），其余案例 `mode == "deadzone"`。

**语义关系**：`law` 与 `mode` 是两个正交字段——`law` 是“假设/被拟合模型”，`mode` 是“真值系统模式”。二者不矛盾：Case 10 是一个“真值为非线性、但用 deadzone 候选律拟合”的先验失配案例，其 deadzone 拟合目标函数值（≈2.92/2.95）明显高于纯 deadzone 案例（≈1.0~1.3），与预期一致。

**结论**：报告字段**无需修正**；字段本身正确且区分清晰。建议（可选）在报告/文档中加一句注记，说明 `mode` 为真值模式、`law` 为候选拟合律，以消除读者歧义。

---

## 6. 科研证据边界（E）

1. **Warm-start 初始解已闭合历史差距**：是。21/21 案例中 `historical_outer < historical_wide`，即 Outer Box 最优解在 Wide 域内已闭合 P4-M2C1 报告的嵌套违规。
2. **Warm-start 优化带来的额外改善**：极小。median ≈ `4.5e-09`，max ≈ `3.3e-04`。Outer 解已接近 Wide 域内的局部极小，进一步优化收益可忽略。
3. **局部优化 vs 全局最优**：仅 SLSQP 局部优化，**无全局最优性证明**；`final_fun` 是最优性受限于局部搜索的结论，不能解读为全局最优。
4. **21 案例 vs 8 个 scene_id**：21 个案例**并非 21 个独立场景**，而是 21 个 `(scene, band, mode, axis, level)` 组合，聚簇在 8 个基础场景上（scene 5 有 5 例、scene 10 有 5 例、scene 3 与 7 各 3 例、scene 11 有 2 例、scene 2/4/6 各 1 例）。因此统计上存在场景级相关，不能按 21 个独立样本处理。

---

## 7. PASS / FAIL / BLOCKED 清单

| 编号 | 检查项 | 结论 |
| --- | --- | --- |
| A1 | 正式报告 21 案例 | PASS |
| A2 | case 编号连续 1..21 | PASS |
| A3 | 无缺失/重复/异常记录 | PASS |
| A4 | Smoke 与正式 Case 1 完全一致 | PASS |
| B1 | `initial_params` 对应 P4-M2C2A outer best | PASS |
| B2 | `initial_fun` 与 `historical_outer_fun` 一致 | PASS |
| B3 | 历史 Wide / 历史 Outer 目标函数比较（outer < wide） | PASS |
| B4 | 无测试/验证数据用于拟合 | PASS |
| C1 | 21 个 `final_params` 重算目标函数 == `final_fun` | PASS |
| C2 | 参数边界与可行性核验 | PASS |
| C3 | 全部不劣于初始解 | PASS |
| C4 | 全部优于历史 Wide | PASS |
| D1 | 全局 `law=deadzone` 语义 | PASS（无需修正） |
| D2 | Case 10 `mode=nonlinear` 语义 | PASS（无需修正） |
| E1 | 科研证据边界明确 | PASS |

**FAIL：0 项；BLOCKED：0 项。**

---

## 8. 结论与限制

**结论**：P4-M2C2B 正式实验结果**可信、可复现、可用于后续科研分析**。21 个案例的输出完整、参数来源可追溯、目标函数可被独立精确复算，元数据语义正确，无需要修正的报告字段。

**限制（不可外推的边界）**：

- 仅开发数据（development-only），无独立评估。
- 仅 21 个历史违规案例，且聚簇于 8 个基础场景（非独立样本）。
- SLSQP 局部优化，无全局最优性证明。
- Warm-start 的额外改善幅度极小；主要结论“闭合历史差距”本质上来自 Outer Box 初始解本身，而非 warm-start 优化。
- 未访问 E4、未使用 Mid validation，故不构成独立泛化结论。
