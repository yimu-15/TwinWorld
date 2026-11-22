# PAPER_FINAL_STATISTICAL_AUDIT — 统计、数值与科研诚信核验

> 只读审计。逐项核验任务 D 提出的全部问题。

## 1. 来源 SHA 与原始报告一致性
- **状态：PARTIAL / WARNING**。仅部分 P4 实验携带 `source_manifest.json`（p4b3c、p4b2b、p4g1b、p4g2a/b/c、p4b1b_v2、p4s1b 等）；P4 report JSON 内嵌 `source_sha256` 映射（P98/P99/P103/R98/... /SELF）。P0/P1/P2/P3 实验**普遍无逐实验 SHA manifest**；顶层 `source_hashes.txt` 仅覆盖 2 个脚本。
- **本次审计未独立重算全部 110 个脚本的 SHA256**（只读约束）。投稿前需补齐全量源码 SHA256 复核。

## 2. 场景指纹与 train/cal/test 重叠
- 各 P4 fresh 实验报告含 `scene_fingerprints_PRIVATE`（如 p4b3c 有 48 个 fingerprint）与 `scene_seed`，用于保证独立场景。
- P4 fresh 实验（97v2/99/103/106/110 等）使用**独立新场景 seed**（scene_seed 各不相同），训练/校准/测试场景分离。
- **未发现跨实验 train/test 场景重叠的迹象**，但 P96 复用 P90 历史队列（回溯，非独立）。

## 3. 真实独立样本单位
- 正式 P4 实验：**24 base scene 聚类**为 bootstrap 单位，384 行（16 case/场景）非 IID。
- P0/P1/P2：pair 或 scene 聚类。
- **禁止**把 384 行当 384 个 IID 样本做逐行二项 CI。

## 4. Bootstrap 种子、次数、聚类方法、CI
- P4 报告 `bootstrap` 字段明确：unit=base_scene，n_clusters=24，resamples=2000，method="paired scene-cluster percentile 95% CI"，seed 各异（如 p4b3c seed=2030032302）。
- P3 部分实验用 20000 次 bootstrap（p3a6j 计划）或 Wilson（p2d2b）。
- **口径一致**，均保留在原始报告内。

## 5. 主要配对统计能否复算
- P4 report JSON 直接给出配对四格（a_correct / b_correct / both_correct / a_only / b_only / both_wrong / net / effect / scene_cluster_CI95），可复算（如 p4b3c 主终点 net=−3，effect=−0.0078125）。
- **本次审计未重新运行 bootstrap 复算**（只读），但报告结构允许第三方复算。

## 6. 未收敛/不可用候选/拒识是否保留在分母
- P4-B3C 明确：`full_denominator_coverage` 全部 rate=1.0、passes_90pct_guard=true；`unavailable_fits=0`；"Unavailable decisions remain errors in the full 384-case denominator"。
- **合规**：无剔除分母行为。

## 7. 嵌套参数域数值违例
- 跨实验存在 deadzone nested-domain 违例：P4-M2C1 21 处、P4-M2C2E 25、P4-M2D1 27、P4-M2D2B 36、P4-S1B 33。P4-S1A/B/C 判定为**有限起点 SLSQP 的优化搜索缺口**，非物理违例；修复后总体决策变化 CI 含 0（P4-M2C2C）。
- **需在论文 Limitations 声明**：残余 33–36 处 nested 违例未消除，全局最优无证书。

## 8. 有无测试结果驱动的调参/方法选择/协议修订
- 存在**事后修正**：P96（n 从 408 改为 204）为"已观察 P90 队列后"的回溯修正，P97 原版冒烟失败后改 v2。这些已在报告内标记非独立设计。
- P4 正式实验声明 `protocol_fixed_before_formal_outcomes: true`（如 p4b3c）。
- **WARNING**：P96/P97 的历史修正构成"测试结果驱动的协议修订"风险，必须在论文中明确披露，不得与真正预注册的前瞻确认混为一谈。

## 9. 是否重复报告同一数据为独立确认
- 存在一条数据被多个只读审计复用（如 P90 队列被 P96、P102、P105、P108 回溯复用；P4-A4D 队列被 p3a5 系列只读复用）。这些均为**只读/回溯审计**，报告中已标注 `development_only / post_hoc / retrospective`。
- **合规**：回溯复用与独立前瞻确认在报告中区分清楚；但论文中需避免把"回溯复核"写成"独立验证"。

## 10. 是否把相同 RK4 两个物理分支当独立仿真引擎
- **已识别风险**：IDEAL 与 SHIFTED 是**同一 RK4 引擎的两个假设臂**，非两台独立仿真器。报告已声明（"one RK4 implementation, not hardware validation"）。
- **论文必须明确**：不得宣称"跨两个独立仿真引擎"验证。

## 11. 是否把优化器成功终止误写成机制识别正确
- **已识别风险**：P4-B3C limitations 明确 "Optimizer success is not equivalent to correct mechanism identification"；compute_cost 记录 optimizer_success vs budget_cap 计数。
- **论文必须区分**：优化器收敛状态 ≠ 机制识别正确性。

## 12. 是否把相同最大预算误写成相同实际计算成本
- **已识别风险**：P4-B3C "Equal hard maximum residual-vector calls, NOT equal realized calls or wall time"；实际调用数 TRF/SLSQP 不同。
- **论文措辞**：必须用"等硬模型评估上限"，不得用"等计算成本"。

## 13. 论文不得宣称的结果（逐项确认）
- 真实机器人验证：**无**（全部 MuJoCo/RK4 合成仿真）。不得宣称。
- RGB 视觉概念修正：**无**（无 RGB 感知实验）。不得宣称。
- 全局结构可辨识性定理：**无**（P3-A7F0/P4-M1A 明确"非全局证明"）。不得宣称。
- 最优主动规划器：**无**（P3-A8C2B 自适应未优于固定）。不得宣称。
- 跨独立仿真引擎验证：**无**（单 RK4 引擎）。不得宣称。

## 14. E4 是否封存 + 投稿前是否需一次最终独立 E4 评估
- E4 全程 `E4_accessed: false`（所有 P4 报告一致）。**封存状态成立**。
- 现有冻结协议（PROTOCOL_V1 P4 gate）要求投稿前开展一次最终独立 E4 评估；本次审计**未执行 E4**。
- 详见 `PAPER_E4_FINAL_GATE.md`。

## 重要缺口标记
- **BLOCKED_FOR_FINAL_SUBMISSION**（见下）：E4 最终独立评估未执行。
- **WARNING**：全量源码 SHA256 未独立重算；部分 P3 早期实验无正式 report（只读/PRIVATE）；nested-domain 违例未消除。
