# 订单流/盘口预测 · 研究工艺标准（SOP）

> 用途：规定本项目"数据—特征—模型—评测—状态"五条工艺线的标准口径、命令、验收与防错清单。
> 照此执行即可复现；任何偏离必须在实验台账登记原因。
> 标的：BTCUSDT 为主（ETH 需另行下载，当前缺失）。

---

## 0. 总览：三条产线，验收口径不同，禁止混用

| 产线 | 数据 | 回答的问题 | 验收指标 |
|---|---|---|---|
| 方向分类线 | L2 逐档（细、12 周） | 往哪做 | Macro-F1 对 Logistic 的配对 CI 下界 > 0 |
| 交易变现线 | L2 投注 + mid | 做了能不能赚钱 | 净收益（扣成本）配对 CI 下界 > 0 |
| 状态表征线 | 桶（粗、8.7 个月） | 该不该做 / 做多大 | 预注册 GATE（见 §7），**不用方向 F1 判生死** |

核心分工：**桶管状态，L2 管方向。** 桶不回答"往哪做"。

---

## 1. 标准口径（数据字典，所有实验共用）

- 帧长：**30 秒**
- 输入窗口：**120 帧 = 60 分钟**
- 方向标签：看**未来 10 帧 = 5 分钟**均价收益
- 投注点间隔：**STRIDE = 10 帧 = 5 分钟**，见 [run_l2_level_experiment.py L38](file:///Users/bytedance/Downloads/order%20book/experiments/runners/run_l2_level_experiment.py#L38)
- 三分类：down=0 / neutral=1 / up=2；**阈值 = 该折 train |forward| 的 70% 分位**
- 逐档张量 **L `(B,120,10,5)`**：10 真实档 × [bid_dist, bid_qty(log), ask_dist, ask_qty(log), imbalance]
- 全局张量 **G `(B,120,4)`**：`["spread_bps","imb_top5","ofi_1","return_1"]`，见 [L41](file:///Users/bytedance/Downloads/order%20book/experiments/runners/run_l2_level_experiment.py#L41)
- 每折切分：train = 更早各周全量 + 本周前 56%；dev = 56%~70%；test = 后 30%
- 泄漏隔离：阈值阈值化器、Robust scaler 只在该折 train 拟合；训测交界留 **purge/embargo = horizon**（5 分钟硬空窗）
- Logistic 每折用窗口末帧重训

---

## 2. 数据制备工艺

### 2.1 桶数据（Binance 官方百分比深度桶）
- 来源：Binance Vision，bookDepth + aggTrades，30s 聚合帧
- 产物：`artifacts/historical/binance_1y/event_features.parquet`（735,317 行、49 列，2026-01-15 ~ 10-07）
- **schema 演进处理**：±0.2 桶 2026-01-15 才发布，早期缺失。pivot 后必须 reindex 到完整 schema，见 [binance_archive.py L128-L136](file:///Users/bytedance/Downloads/order%20book/lob_mvp/binance_archive.py#L128-L136)
- 因果状态列（可直接用）：`rv_1h/rv_1d, liquidity_1h, depth_imbalance_mean, trade_imbalance_mean, ofi_proxy`

### 2.2 L2 逐档数据
- 来源：CryptoHFTData，10 档挂单
- 12 周清单（双周间隔周四，04-23~09-24）：见 [run_l2_multiweek.py WEEKS](file:///Users/bytedance/Downloads/order%20book/experiments/runners/run_l2_multiweek.py)
- 每周产物：`artifacts/historical/l2_weeks_features/{周}/l2_30s_features.parquet`

### 2.3 账本重放必须 stateful（关键工艺）
- 增量流会丢失"挂单被成交后的删除更新"，非 stateful 重放会留下**僵尸价**，导致买卖价交叉（曾现 mid 38k–627k、spread_bps 均值 1568）
- 标准处理：stateful replay + 撮合自愈 [`heal_crossed`](file:///Users/bytedance/Downloads/order%20book/experiments/runners/replay_l2_stateful.py#L25-L48)：清除所有 bid ≥ best_ask，再清反向 ask ≤ best_bid
- 验收：best_bid < best_ask 全样本成立；mid、spread_bps 量级正常

### 2.4 大数据量分位数工艺
- expanding 大折行数超限（`quantile() input tensor is too large`）：对 >20 万行的 flatten 数据均匀抽稀后再算分位；mean/std/clip 仍用全量，见 [fit_robust_scaler L71-L87](file:///Users/bytedance/Downloads/order%20book/experiments/runners/run_l2_multiweek.py#L71-L87)

---

## 3. 特征工艺

### 3.1 快变量
盘口即时微结构：spread、档位失衡、OFI、单帧收益。直接进方向模型。

### 3.2 慢变量——只喂置信门，绝不碰方向（硬纪律）
- 时间尺度用**可配置表**，不写死，见 [medium_long_features.py L32-L44](file:///Users/bytedance/Downloads/order%20book/lob_mvp/medium_long_features.py#L32-L44)：
  - 反转动量 mom：1h / 4h / 12h
  - 区间位置 relpos：2h / 8h
  - 成交压力 pressure：1d / 7d
- 每个尺度各自因果 z-score（[`_rolling_z` L73-L76](file:///Users/bytedance/Downloads/order%20book/lob_mvp/medium_long_features.py#L73-L76)）；合成用尺度数量无关的平均（[`_mean_z` L79-L81](file:///Users/bytedance/Downloads/order%20book/lob_mvp/medium_long_features.py#L79-L81)），加减尺度不改量级
- 合成：reversion / continuation，见 [L125-L126](file:///Users/bytedance/Downloads/order%20book/lob_mvp/medium_long_features.py#L125-L126)
- 实证依据：1h–7d 盘口以短期均值回归为主，朴素动量 carry 负 IC；延续只出现在订单流加速度与长期成交压力

### 3.3 学习型慢状态隐变量 z（替代手工公式）
- 输入 13 个因果慢变量的过去 120 帧 → 输出 **12 维 z**
- **多任务、无方向头**：只预测未来 rv / 流动性 / 活跃度，防止 latent 被方向标签锁死，见 [run_slow_state_encoder.py L53-L75](file:///Users/bytedance/Downloads/order%20book/experiments/runners/run_slow_state_encoder.py#L53-L75)
- 产物：`artifacts/historical/slow_state/slow_z.parquet`（分段 train / valid / oos，统计检验只用 **oos**）

---

## 4. 模型工艺

### 4.1 标准积木（[two_axis_model.py](file:///Users/bytedance/Downloads/order%20book/lob_mvp/two_axis_model.py)）
| 积木 | 行号 | 作用 |
|---|---|---|
| DilatedCausalConv | [L26-L43](file:///Users/bytedance/Downloads/order%20book/lob_mvp/two_axis_model.py#L26-L43) | 深度可分离空洞因果卷积，左 pad 右截断，不看未来 |
| AttentionPool | [L46-L56](file:///Users/bytedance/Downloads/order%20book/lob_mvp/two_axis_model.py#L46-L56) | 可学习注意力把 120 帧压成 1 向量 |
| FeatureAxisAttention | [L59-L85](file:///Users/bytedance/Downloads/order%20book/lob_mvp/two_axis_model.py#L59-L85) | 第二轴轻量门控混合 |
| TwoAxisLOB（桶 / TLOB-lite） | [L88-L154](file:///Users/bytedance/Downloads/order%20book/lob_mvp/two_axis_model.py#L88-L154) | 空洞卷积 + 因果时间 Transformer + 特征轴 |
| LevelStructuredLOB（L2） | [L157-L221](file:///Users/bytedance/Downloads/order%20book/lob_mvp/two_axis_model.py#L157-L221) | token l 对应真实第 l 档；每档过共享时间轴再跨档自注意 |
| ConfidenceFilter | [L224-L240](file:///Users/bytedance/Downloads/order%20book/lob_mvp/two_axis_model.py#L224-L240) | 吃慢变量出可交易度，方向无关 |

### 4.2 两个必备结构补丁
- **因果掩码**：TransformerEncoder 默认双向有泄漏，必须补三角掩码，见 [L121-L128](file:///Users/bytedance/Downloads/order%20book/lob_mvp/two_axis_model.py#L121-L128)
- **双路读出**：拼接注意力全局向量与末帧瞬时向量（借鉴 MBOFormer），见 [L130-L131](file:///Users/bytedance/Downloads/order%20book/lob_mvp/two_axis_model.py#L130-L131)

### 4.3 选型顺序（先简单后复杂，禁止倒置）
majority → Logistic（单帧 / 序列）→ 卷积 → 双轴 Transformer。每步必须显著优于前者才允许加复杂度。

### 4.4 类权重口径（公平性红线）
- 神经网络：逆频次 **pow 0.7**，`CLASS_WEIGHT_POWER=0.7`，见 [L47](file:///Users/bytedance/Downloads/order%20book/experiments/runners/run_l2_level_experiment.py#L47) 与 [L219-L220](file:///Users/bytedance/Downloads/order%20book/experiments/runners/run_l2_level_experiment.py#L219-L220)
- Logistic 默认 `class_weight="balanced"`（= pow 1.0）
- **同一次比较必须对齐 pow，否则人为制造 gap**

---

## 5. 评测工艺

### 5.1 对称基线矩阵（先做，消除偏心）
2×2：（单帧 / 序列）×（线性 / 非线性）；类权重全对齐（pow 1 / 0.7 / 0.5 / 0 全扫）。
用于区分胜负来自"模型非线性"还是"历史长度"。

### 5.2 统计检验
- **配对 moving-block bootstrap**：同索引重采两个模型，块长 = 收益跨度换算的投注点数，见 [paired_bootstrap L154-L168](file:///Users/bytedance/Downloads/order%20book/experiments/runners/run_bucket_walkforward.py#L154-L168)
- 嵌套 **regime（周）层级重采样**，避免少量折给出伪精度
- 显著性裁决：**95% CI 下界 > 0**；重复 ≥ 5,000 次

### 5.3 交易口径
- 成本：主口径 **7bp round-trip**；maker 敏感性 **2bp**
- 方向押注：pred=2 → +1，pred=0 → −1，neutral 不开仓
- 通过标准（事先定死）：净均值 > 0 且 CI 下界 > 0；≥ 9/12 周为正；毛 edge 中位数 > 成本

### 5.4 多 horizon / 时间口径扫描
- 固定时钟：h = 10/60/240/480/960 = 5min~8h，见 [run_horizon_scan.py L41](file:///Users/bytedance/Downloads/order%20book/experiments/runners/run_horizon_scan.py#L41)
- 事件口径：用 12 个训练日标定中位事件率，把时钟时长折算成 N 笔 aggTrade，见 [calibrate_event_n L181](file:///Users/bytedance/Downloads/order%20book/experiments/runners/run_horizon_scan.py#L181)

---

## 6. 工程工艺

- 训练默认走 **MPS**（实测比 CPU 快约 28×、数值一致）；bootstrap 无 GPU 实现走 CPU
- 长任务一律 `caffeinate -is`（tmux 不防睡眠，本机未装）
- 每次实验用 run_io 自动归档代码 + 配置 + 产物，保证可追溯
- 训练集内部窗口重叠可作数据增广（不影响）；独立性只属检验环节，训测必须硬隔离
- 每次只改一个变量；省时间的抽样必须先单折对照无损再推广
- **时间预估先实测再承诺，禁止先喊"成了"**

---

## 7. 市场状态线工艺（4 阶段 + GATE，顺序不可倒置）

1. **规则状态**：高/低波动、厚/薄流动性、买/卖压力（仅用预测时点之前信息）→ 条件胜率/收益分析
   - **GATE**：若不同状态下方向信号的条件收益/胜率差异 CI 全跨 0 → "状态无交易价值"，停在此处，禁止进第 2 步
2. 状态有效后才训练多任务 latent z（方向 + 波动 + 流动性 + 活跃度，不被方向锁死）
3. z 聚类成 regime，验收稳定性 / 可解释 / 可复现
4. 状态过滤 L2 信号用于交易；乘法门控 `P = 方向 × 置信 × 可交易度 g × 仓位尺度 s`，**状态只调仓位、不翻转方向**

### 7.1 聚类三 GATE（阈值事先注册，仅在 oos 段做）
| GATE | 标准 |
|---|---|
| G1 非方向马甲 | Cramér's V < 0.40 且 CI 上界 < 0.50；簇内多空偏斜 ≤ 0.15 |
| G2 优于原始慢变量聚类 | silhouette 与段间复现 ARI 的 (z−raw) 配对 CI 下界均 > 0 |
| G3 状态持久 | min-dwell 滤波后平均驻留 > 360 帧、CI 下界 > 240 帧 |

脚本：[run_slow_state_clustering.py](file:///Users/bytedance/Downloads/order%20book/experiments/runners/run_slow_state_clustering.py)。

---

## 8. 标准命令清单

```bash
# 桶一年 walk-forward
python -m experiments.runners.run_bucket_walkforward

# L2 多周：下载/特征 + 12 折训练
python -m experiments.runners.run_l2_multiweek

# 多 horizon / 时间口径扫描（A 时钟 / B 事件）
python -m experiments.runners.run_horizon_scan --mode A
python -m experiments.runners.run_horizon_scan --mode B

# 慢状态：多任务 encoder → 聚类 GATE → 状态条件化 edge
python -m experiments.runners.run_slow_state_encoder
python -m experiments.runners.run_slow_state_clustering
python -m experiments.runners.run_state_conditioned_edge
```

---

## 9. 交付前检查清单

- [ ] 阈值 / scaler 只在 train 拟合，训测间有 embargo
- [ ] Transformer 已补因果掩码
- [ ] 对比双方类权重 pow 已对齐
- [ ] 已出对称基线矩阵，而非只比一个模型
- [ ] 独立折数足够；CI 用 regime 层级 + moving-block，非单次抽样
- [ ] 收益结论同时给出 7bp 与 2bp
- [ ] GATE 阈值先于结果注册，结论不做事后硬圆
- [ ] 代码 / 配置 / 产物已归档；台账已更新

---

## 10. 防错清单（本项目踩过的坑）

1. 类权重不对称制造假差距
2. NN 喂序列、LR 喂单帧 → 输入维度不对等
3. train-stride 过大导致 dev 与 train 重叠、早停失效
4. 折数过少 → bootstrap 伪精度
5. 抽样训练削弱模型（点估计由正转负）
6. 桶预训练迁 L2 产生负迁移（数据语言不同）
7. 非 stateful 重放产生僵尸价 / 交叉账本
8. 误把逐帧掩码用到窗口级数组
9. "正确率差"可能是 neutral 类别构成假象，须剔除 neutral + 收益口径双验证
10. 长 horizon 相邻投注收益重叠，block 长度必须 ≥ h/STRIDE
11. 普通 `&` 后台遭 SIGHUP；大文件需放宽看门狗超时；Mac 睡眠挂起用 caffeinate

---

## 11. 当前研究结论（截至 2026-10-09）

1. 方向分类成功：L2 12 折、7,176 独立样本，对 Logistic gap **+0.077，CI(+0.061,+0.092)**；类权重对齐最强基线后仍 **+0.042，下界 +0.026**
2. 交易变现失败：5min–8h、时钟/事件、7bp/2bp 共 20 组合，净收益 CI 无一可靠为正；毛 edge 太薄
3. 状态 GATE 阶段 1 不通过（5min 毛 edge ≈ 0、状态间无差异），已按纪律停
4. 状态表征：多任务 z 有效（oos 三任务相关 0.69/0.69/0.74），K=5 聚类三 GATE 全过；但状态条件化 edge 不集中、不可变现
5. 唯一剩余现实方向：maker 2bp 实盘降费，并需补独立月份验证（当前不做）
