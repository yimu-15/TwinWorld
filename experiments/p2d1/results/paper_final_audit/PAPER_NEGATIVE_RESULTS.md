# PAPER_NEGATIVE_RESULTS — 未达确认、方向反转与失败发现

> 只读审计。诚实记录不利结果，不修饰、不剔除。每条注明实验编号、真实数字、是否独立确认、原因。

## 1. 自适应动作选择未优于最强固定动作（方向性失败）
- **实验**：P3-A8C2B(71)、P3-A8C0(67)、P3-A8B0(64)。
- **事实**：original 带 affine_disagreement correct=174 < fixed_reverse 180，且 work_J 更高（2405.94 vs 2238.56）；above 带 198 < 201。空集校正后 affine 相对 fair_high 仅省 0.76%。
- **结论**：基于动作前分歧的自适应动作选择**不**构成优于最强固定动作的贡献。
- **原因**：分歧信号本身（见下条）不足以可靠排序动作。

## 2. 动作排序预测不可靠
- **实验**：P3-A8C3A0(78)。
- **事实**：分歧 margin 与实际 High/Reverse 偏好的 concordance≈0.5（old 0.533 / affine 0.517），Spearman ρ CI 跨 0（old[-0.190,0.313]、affine[-0.164,0.331]）。
- **结论**：结构分歧 margin 不能作为可靠的动作排序信号。

## 3. 低激励下的结构歧义（激励漂移脆弱）
- **实验**：P3-A6K(50)、P3-A6L(51/52)。
- **事实**：low regime 下 saturating 17/48 被误接受（全判为 elastic）；真实 saturating vs 同参数 elastic 轨迹差仅 9.88mm（拟合残差 7.60mm，噪声~5mm）。
- **结论**：低激励下柔顺律结构歧义**物理上不可区分**，是识别极限，非算法缺陷。

## 4. 冻结参数优势依赖特权信息（不可部署）
- **实验**：P4-G2C(103)、P4-G2D(104)。
- **事实**：FIXED−REFIT +20/384（CI 不含 0），但 report 自声明 `paired_source_privileged: true`、`same_scene_ideal_source_privilege: YES`、`deployable_zero_shot_superiority: NOT_ESTABLISHED`。
- **结论**：不能作为"可部署迁移优势"。

## 5. 独立生成器迁移的剂量效应未确立
- **实验**：P4-G1B(99)、P4-G1C(100)。
- **事实**：SHIFTED FULL−HALF +5/96（0.0521），CI[-0.052,0.156] 含 0；SHIFTED 主要损害 elastic（−20/96）。

## 6. 更好的拟合 ≠ 更好的选择（反例）
- **实验**：P4-G2B(102)。
- **事实**：refit 把真模型 RMS 从 11.02→9.79mm，但识别 effect fixed 领先 +0.0859；elastic 亚组 RMS 更低但正确率落后 +0.4583。

## 7. 方程误差法显著差于输出误差法
- **实验**：P4-B2B(106)。
- **事实**：EQERR_SG−SLSQP net −15/384，CI[-0.070,-0.005]（EQERR 更差）；且算力量纲不可比。

## 8. 分辨率感知规则无增量
- **实验**：P3-A8A(63)。
- **事实**：resolution 与 baseline 决策完全相同，未产生额外确认/弃权。

## 9. Biphasic 动作不优于 High
- **实验**：P4-M2D0(87)。
- **事实**：OOO elastic-M2 biphasic−high = −0.125，CI[-0.292,0.021]（负方向）。

## 10. 反向动作统计学不排除等效
- **实验**：P3-A8C1B(69)。
- **事实**：reverse_high 正确 170 vs high 160（复现），但 correct_delta 95% CI[-2,22] 含 0。

## 11. 自适应探针余量为零
- **实验**：P2-D1B(16)。
- **事实**：fixed_best_action=pulse 在全部 24 场景胜出，`privileged_adaptivity_headroom=0.0`。

## 12. 无正式报告的只读/PRIVATE 实验
- **编号**：33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 49, 53（以及 23, 28）。
- **事实**：这些编号只有脚本 + 只读/PRIVATE 审计，无确认性 report.json，不能作为论文正式证据。

## 13. 经典选择器彼此差异不显著
- **实验**：P4-B1B_v2(97v2)。
- **事实**：BIC−PEM +5/384，CI[-0.003,0.031] 含 0。

## 14. 嵌套域违例未随起点匹配消除
- **实验**：P4-S1B(92)、P4-S1C(93)。
- **事实**：三臂匹配五起点后 nested 违例 36→33 仍存活（判定为数值缺口，非物理）。
