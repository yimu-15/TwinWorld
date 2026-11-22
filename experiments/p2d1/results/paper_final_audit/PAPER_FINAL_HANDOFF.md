# PAPER_FINAL_HANDOFF — TwinWorld 论文证据最终交接文档

> 本文件面向一个完全不了解 TwinWorld 的新对话。读完即可理解项目目标、110 号实验脉络、最终方法、全部发现、最强 Baseline 优势、统计证据、负面结果、投稿边界与章节计划。
> 本文件为只读审计产物，位于 `experiments/p2d1/results/paper_final_audit/`，不覆盖任何历史文件。

---

## 1. 项目最初目标

TwinWorld-P0 研究一个问题：**当一个智能体面对"被动观测完全不可区分、但隐藏物理机制不同"的多个世界时，能否通过主动干预来识别机制，并据此修正它的结构化物理模型？**

初始 P0 是一个"站在巨人肩膀上"的最小 sanity-check：用 CPU 玩具模型验证——严格匹配的被动轨迹下，coupled（隐藏弹簧阻尼耦合）与 independent 两个世界无法区分；而对物体 A 施加外力后，物体 B 的响应不同。README 明确声明：这是物理可辨识性与数据生成的 sanity-check，**不是**新算法/科学定理/论文最终实验。

## 2. 研究问题的演变

1. **P0（能否构造匹配被动反事实）** → 验证被动等价 + 主动可分离（sanity check）。
2. **P1（迁移到真实物理）** → MuJoCo 刚体 + 隐藏耦合，隔离 A 干预对 B 的影响（匹配 A 运动）。
3. **P2（多机制 + 开集）** → 三/五机制、未知机制拒绝、探针动作；发现"B 观测 + pulse 探针"最有效。
4. **P3（主动机制辨识算法线）** → 从"已知定律重拟合"到"受限结构修订"到"证据门控主动干预"，研究主动实验能否区分被动不可分机制，并探索门控省功与自适应动作。
5. **P4（收官四条线）** → (M) 物理先验/参数域/动作剂量；(S) 求解器预算与嵌套违例；(T) 预测边际一致性；(B/G) 与经典系统辨识方法（PEM/AIC/BIC、EQERR、TRF/SLSQP）的公平横向比较 + 独立 ODE 生成器的冻结 vs 重拟合。

## 3. 第1–110号实验完整研究脉络

（详见 `PAPER_EXPERIMENT_TIMELINE.md` 与 `PAPER_EXPERIMENT_REGISTRY.csv`）

- **P0（1–4）**：CPU twincore 生成/验证/基线/绘图。被动多数类=50%，特权主动 oracle=100%（非可推广）。
- **P1（5–9）**：MuJoCo 迁移。B 观测测试集 100% 可分（P1-C3）。
- **P2（10–19）**：三/五机制 + 开集 + 探针。B_only 0.878；pulse 恒最优（自适应余量=0）。
- **P3（20–71, 78）**：三子段——A0–A2 重拟合与结构修订；A3–A4 门控停/强；A5–A6 可辨识性/开集/激励漂移；A7–A8 动作族/affine 预测器/双向选择。
- **P4（72–110）**：M（先验/域/剂量）、S（求解器）、T（边际一致性）、B/G（经典基线 + 生成器）。

**关键诚实声明**：110 个编号**不是** 110 组统计独立确认实验，而是开发→诊断→确认→审计链条。编号 23,28,33–39,41,43,44,46,49,53 等**无正式 report**（只读/PRIVATE），不能作正式证据。

## 4. 最终物理系统和方法

- **P0 核心物理**（`scripts/twincore.py`）：1D 两体弹簧阻尼；`coupling = k·Δx + d·Δv`（coupled）或 0（independent）；`Δx = x_B − x_A − distance`；`a_A=(F+coupling)/m_A − γv_A`，`a_B=−coupling/m_B − γv_B`；欧拉步 DT=0.02。
- **P3/P4 候选恢复律**：elastic（F=kx）、deadzone（F=k(x−g)·[|x|>g]）、cubic（F=kx+cx³）；参数域 wide / outer_box（物理外箱）。
- **辨识方法**：方程误差 EQERR_SG、输出误差最小二乘（scipy TRF / SLSQP，5 起点，per-start 硬上限）；训练集经典模型选择 PEM/AIC/BIC（n=204 B 位置残差）。
- **统计口径**：24 base scene 聚类，384 行（16/场景）非 IID；配对 scene-cluster 百分位 bootstrap，2000 次。

## 5. 全部主要科学发现（12 类，详见 `PAPER_SCIENTIFIC_DISCOVERIES.md`）

A. 被动观测不携带机制信息（机会水平），主动 A-only 干预使 B 分化。
B. 低耦合极限下多个柔顺律（elastic/deadzone/cubic）被动经验等价（<0.08mm）。
C. 更强正向幅值单调放大机制分离与识别召回。
D. 自适应动作选择**未**优于最强固定动作（省功与识别存在权衡）。
E. 参考激励下库外机制可高可靠拒识；低激励下结构歧义物理上不可消除。
F. 物理外箱先验带来跨场景复现的识别增益。
G. 物理先验中等失配稳健、极端失配反噬；嵌套域违例为求解器缺口。
H. **更低拟合误差 ≠ 更好机制选择**（系统反例）。
I. 冻结 IDEAL 参数优于重拟合，但依赖特权同场景信息。
J. 输出误差法（TRF/SLSQP）优于方程误差法（EQERR），两者决策一致。
K. 等硬预算下 cap100 SLSQP 优于 TRF，cap200 相当。
L. 多项未达确认/方向反转/失败结果（见负面结果）。

## 6. 最高价值的 Baseline 优势（详见 `PAPER_STRONGEST_BASELINE_FINDINGS.md`）

1. **物理外箱先验 vs legacy 宽域**（公平，跨场景复现）：outer−wide +0.0833 CI[0.051,0.116]；OOO−WWW +0.1797 CI[0.128,0.232]。
2. **激励幅值剂量反应**（公平）：0.5x→1.0x recall +0.198 CI[0.094,0.313]。
3. 其余（EQERR 差、SLSQP/TRF 选择、门控省功）作为结果/消融/讨论支撑，**不进入摘要/贡献**。

## 7. 原始统计证据（详见 `PAPER_FINAL_STATISTICAL_AUDIT.md`）

- 所有 CI 为场景聚类配对 bootstrap（24 簇、2000 次），seed 各报告明确。
- 不可用候选/拒识保留在 384 分母（rate=1.0）。
- 嵌套域违例 25–36 处未消除（数值缺口，非物理）。
- 等预算≠等实际调用；优化器成功≠机制识别正确；单 RK4 引擎≠双引擎。

## 8. 负面结果及局限（详见 `PAPER_NEGATIVE_RESULTS.md` / `PAPER_LIMITATIONS_AND_THREATS.md`）

- 自适应动作未优于最强固定动作；分歧 margin 不能可靠排序动作。
- 低激励下 saturating/elastic 物理上不可分（~9.88mm 接近噪声）。
- 冻结参数优势依赖特权信息（不可部署）。
- 更低拟合误差反而选择更差（反例）。
- 全合成仿真（MuJoCo/RK4），无硬件、无 RGB、无全局可辨识性定理、无最优规划器。

## 9. 投稿方向与贡献边界（详见 `PAPER_CONTRIBUTION_CANDIDATES.md`）

- **可主打的 2 个公平贡献**：物理外箱先验增益 + 激励幅值剂量反应。
- **次要贡献**：机制感知门控省功。
- **红线（不得宣称）**：真实机器人验证、RGB 视觉概念修正、全局结构可辨识性定理、最优主动规划器、跨独立仿真引擎验证、"等计算成本"的公平比较。
- **适合投稿**：主动学习 / 系统辨识 / 具身智能方向的实证方法学期刊。

## 10. 英文论文完整章节计划

1. **Introduction** — 被动不可分 vs 主动可辨识动机（A/B 类发现）。
2. **Related Work** — 主动系统辨识、实验设计、物理先验、模型选择。
3. **Method** — 物理生成器方程、候选恢复律、参数域（wide/outer_box）、辨识器（EQERR/TRF/SLSQP）、模型选择器（PEM/AIC/BIC）、场景聚类 bootstrap。
4. **Experiments** — 24 场景协议、训练/校准/测试分离、seed、预算口径。
5. **Main Results** — 被动等价、外箱先验增益、幅值剂量反应、经典辨识比较（T1–T7）。
6. **Ablation** — 域消融、动作剂量、求解器起点、预算（AB 表）。
7. **Discussion** — 失败案例（低激励歧义、嵌套违例、特权冻结）、反直觉（拟合 vs 选择）、局限。
8. **Conclusion** — 收敛到 2 个公平贡献 + 识别极限。
9. **Appendix** — 完整实验注册表、SHA manifest、负面结果全清单。

## 11. 待解决的最终问题（投稿前必须完成）

1. **执行冻结协议的最终独立 E4 评估**（当前 BLOCKED）。
2. **全量重算 110 脚本 SHA256 并冻结完整 source manifest**（P0–P3 缺失）。
3. 嵌套域违例的全局最优确认或明确边界（当前 25–36 处未消除）。

> 最终判定：证据体系足以支撑一篇以"物理外箱先验 + 激励幅值剂量"为核心的、诚实披露负面结果与局限的论文，但**最终投稿被 E4 门禁与 SHA 复核所阻塞（BLOCKED_FOR_FINAL_SUBMISSION）**。

## 附：本审计生成的产物清单（位于 `experiments/p2d1/results/paper_final_audit/`）

- PAPER_EXPERIMENT_REGISTRY.csv / PAPER_EXPERIMENT_TIMELINE.md / PAPER_SOURCE_MANIFEST.json
- PAPER_SCIENTIFIC_DISCOVERIES.md / PAPER_CLAIM_EVIDENCE_MATRIX.csv / PAPER_NEGATIVE_RESULTS.md
- PAPER_BASELINE_COMPARISON.csv / PAPER_BASELINE_FAIRNESS_AUDIT.md / PAPER_ABLATION_RESULTS.csv / PAPER_STRONGEST_BASELINE_FINDINGS.md
- PAPER_FINAL_STATISTICAL_AUDIT.md / PAPER_PROTOCOL_DEVIATIONS.md / PAPER_LIMITATIONS_AND_THREATS.md / PAPER_E4_FINAL_GATE.md
- PAPER_MANUSCRIPT_EVIDENCE_MAP.md / PAPER_MAIN_TABLES.csv / PAPER_FIGURE_DATA_INDEX.md / PAPER_CONTRIBUTION_CANDIDATES.md / PAPER_MANUSCRIPT_READINESS.md
- PAPER_FINAL_HANDOFF.md（本文件）
