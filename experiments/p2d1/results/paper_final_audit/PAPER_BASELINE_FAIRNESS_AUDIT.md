# PAPER_BASELINE_FAIRNESS_AUDIT — 横向比较公平性核验

> 只读审计。逐项回答"这是否是真正的公平算法优势"，并特别核验任务 C 指定的五个问题。

## 逐项特别核验

### 1. P3-A8C2B 自适应动作是否真正优于最强固定动作？—— **否**
- 事实：original 带 affine_disagreement correct=174 < fixed_reverse 180（work_J 2405.94 > 2238.56）；above 带 198 < 201。
- 报告未提供自适应 vs 固定的成对 CI（p3a8c2b_fresh_report.json 无配对 CI 字段）。
- 结论：自适应动作选择**不占优**。仅相对 fixed_high（161/192）正确数更高，但成本仍高于 fixed_reverse。

### 2. P4-M2D1/P4-M2D2B 激励幅值的改善 —— 真实且可部署
- P4-M2D1：动作×先验交互 0.0833（CI[0.044,0.122]），HIGH−MID 0.1198（CI[0.081,0.156]）—— fresh 独立新场景确认，**非特权**。
- P4-M2D2B v2：幅值 0.5x→1.0x recall +0.198（CI[0.094,0.313]），分离 +10.2mm—— fresh 确认，**非特权**。
- 公平性判定：**公平优势**（冻结模型、相同观测预算、无特权信息）。

### 3. P4-G2C 参数来源优势是否含特权同场景 IDEAL 信息？—— **是（特权）**
- 事实：report `limitations` 首句即 "Fixed-IDEAL uses privileged same-scene IDEAL observations"；manifest `paired_source_privileged: true`；P104 结论 `same_scene_ideal_source_privilege: YES`、`deployable_zero_shot_superiority: NOT_ESTABLISHED`。
- 公平性判定：**非公平优势**。FIXED−REFIT +20/384 的"冻结优于重拟合"不能作为可部署算法贡献，只能作为"当同场景 IDEAL 参数可得时"的条件结论。

### 4. P4-B2B 的 EQERR / TRF / SLSQP 比较 —— 结果可信但算力不可比
- 结果真实：EQERR=313/384 < SLSQP=328 < TRF=328，EQERR−SLSQP net −15（CI 不含 0）。
- 公平性隐患：三法**模型评估/计算成本量纲不同**——EQERR 17664 网格评估（6.6s）；TRF 463732 残差调用（1155.6s）；SLSQP 918685 目标 nfev（2294.9s）。"同观测权"≠"同算力"。
- 公平性判定：结论方向可信，但**不能**宣称"在等计算成本下 EQERR 更差"。

### 5. P4-B3C 的 100/200 等预算比较 —— 等上限≠等实际调用
- 结果真实：cap200 TRF≈SLSQP（net −3，CI 含 0）；cap100 SLSQP 显著占优（net −18，CI 不含 0）。
- 公平性隐患：报告明确 "Equal hard maximum residual-vector calls, NOT equal realized calls or wall time"。cap200 下 SLSQP 实际 193082 vs TRF 189743 残差调用；per-start cap 20/40。
- 公平性判定：这是"等硬上限"的公平比较（预注册口径），但**不得**把它表述成"等实际计算成本"。

## 通用公平性核验结论

| 维度 | 结论 |
|---|---|
| 是否使用相同数据 | P4-B/G 系列均为 24 base scene 独立队列，同一实验内两方法共用相同观测；**但 P96 与 P97v2 的 n 口径不同（408 vs 204），P96 为事后修正且非盲设计** |
| 参数域 | WOO/OWO/WWW 嵌套域在公平比较中一致指定；但 deadzone 存在 nested-domain 违例（各实验 20–66 次不等） |
| 优化起点 | P4-B3C 统一 5 起点 + per-start cap；TRF/SLSQP 起点相同 |
| 预算 | P4-B3C 统一硬上限；但"上限"≠"实际调用" |
| 特权信息 | P4-G2C 冻结 IDEAL 含**同场景特权**信息（不可部署）；P0 active oracle 含特权反事实 |
| 统计单位 | 24 base scene 为 bootstrap 聚类，384 行非 IID；各报告 seed 明确 |
| 事后选组 | P96 为"已观察 P90 队列后回溯"，明确非独立设计；P97v2 为独立新场景确认 |

## 最重要的公平性风险清单（论文必须声明）

1. **P4-G2C 冻结优势 = 特权**，不得列为"跨域迁移贡献"。
2. **等预算 ≠ 等实际计算成本**（P4-B3C），措辞须精确为"等硬模型评估上限"。
3. **EQERR/TRF/SLSQP 算力量纲不可比**（P4-B2B），只能报告"同观测权"下的准确率。
4. **P96 为事后回溯**，P97v2 才是独立确认。
5. **nested-domain 违例未消除**（25–36 处），全局最优未确立。
6. **同一 RK4 引擎的两个物理分支（IDEAL/SHIFTED）不是两台独立仿真器**。
7. **384 行非 IID**，所有 CI 必须用 24 场景聚类口径，不得用逐行二项 CI。
