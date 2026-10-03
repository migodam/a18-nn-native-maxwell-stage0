# A18 — Lightweight Experiment Specification

状态：PROPOSED / NOT RUN。本轮研究只做了文献、固定源码审查与独立随机代数验证。以下数值门槛、模型大小和资源限额是下一步预注册合同，不是 A18 已达到的结果。

## 0. 固定对象与范围

保留 A17 四对象：smooth Gaussian、near-contact Gaussian、layered voxel、asymmetric voxel；各 early/middle/late；k=4/8/16。共十二个 physical states、三十六个共同 receiver queries。所有物理 material gauge、residual、prior/LM、illumination layout、dictionary 原始 ID、basis hash 沿用固定 commit。

第一轮只读现有数据，零新增 full-wave tangent RHS、零 nonlinear trajectories、零训练 epoch。不重建 A17 所依赖的四十二个外部内核，不借缺数据之名重跑全部项目。[A17-start]

物理小矩阵使用现有 complex128/float64 合同，feasibility 和 rank/stability threshold 不改动。No-op 保留。

## 1. Stage G0：数据和能力盘点

输入来源至少对应：`public_workspace.npz`、`runtime_receipt.json`、cell receiver/teacher/moves 文件、accepted endpoint NPZ、`offline_information_reference.npz`、A17 labels index、setup/cost receipts。路径通过已有索引解析；下列数组名来自源码的保存行为，不构成对每个 archive 文件已存在的声明。[A17-run]

输出 `A18_DATA_INVENTORY.csv`：

```text
source_path, sha256, object_id, phase, k, array_name, shape, dtype,
coordinate_scope, role(public/teacher/evaluator), source_commit,
can_reconstruct_endpoint, can_form_JFd, can_form_full_injection,
can_form_unresolved_current_label, status, missing_reason
```

必须区别：Gaussian fullJ；voxel V/fullJV；projected L/S/PB；真实 full L/B；Js 端点输出；仅 scalar Q 的标签。存在 fullJ 不等于存在 full-current snapshot。

去重 key：object/state hash、U original atom IDs/hash、current step hash、anchor hash、drop/add ID、direction hash、prior/LM 和 receiver layout。相同物理查询的多个 policy 副本不当成独立样本；三十六个共同 receiver 起点只计一次。

输出 source rows、feasible rows、unique queries、unique moves、unique incoming labels 分别的计数。47,712/43,108 是 A17 报告的来源行数，不是本轮提前填写去重计数。[A17-NN]

## 2. Stage G0：复现已知 bottleneck

从共同起点完整一换一标签计算：full-dictionary best gain、anchor top-one capture、old balanced-12 coverage。按对象先 sum gains 再 ratio，不平均极小分母的逐点比值。

参考值用于 sanity check，而不是调参目标：shortpool smooth≈0.969、near≈0.353、layered≈0.920、asymmetric≈0.659；full-dictionary anchor capture≈0.981–0.996。[A17-report]

若不匹配，输出范围差异（共同起点 vs path、no-op vs forced top-one、重复计数、feasibility、floor），在解开原因前不训练。

## 3. Stage G0-A：强解析 proposal 与 feature audit

固定 shortlist=12。运行 CANDIDATE_GENERATOR 中 B0–B5，尤其 O/P/M union 与 cheap signed-transfer+diversity。所有非 teacher proposal 只能访问 F0/F1。

输出 `proposal_common_queries.csv`：

```text
query_id, object_id, method, N_dictionary, N_legal, shortlist_size,
full_best_positive_gain, shortlist_best_positive_gain,
top1_recall, top4_recall, anchor_selected_Qtrue,
accepted_Qtrue, no_op, feature_wall_s, proposal_wall_s,
endpoint_count, endpoint_wall_s, physical_action_count,
measurement_scope(replay/measured), seed
```

Qtrue 查表 replay 用于评估，不算在线 NN 所需输入。可以用现有 full teacher table 计算单步反事实，但不能把 lookup wall time 当成真实 full Jd solver 时间。

一次 feature profiling 逐项统计：LC/SC、retained cross blocks、N 草图、稀疏边建立、host/device copy、小矩阵求解。缓存构建费用另列。原 setup 内嵌的费用不重复相加。[A17-cost]

A 的非学习 gate：若 B2/B3 已达到 below A target，并且不存在实测费用优势空间，不训练 neural proposer；输出 `A_NO_NEURAL_HEADROOM`。

## 4. Stage G0-B：仅已有向量的 defect oracle

按实际可恢复数据生成同一个 d 的 z_U、z_A、z_F 及两个 defect。Gaussian 用保存 fullJ；voxel 的 fullJV 仅在 d 的 V-projection residual 达到原精度门槛时使用。其他 voxel 方向只能来自真实已存 Js differences 等可核对记录；没有就标 missing。

输出 `defect_direction_manifest.csv`：identity、zF provenance、direction support check、各 norm、full gain、predicted mean gain、margin、ε_crit、source feature scope。每条 label 都绑定完整 d 与对应 reference，不混用不同 U/x 的 finite gain。

仅使用 training objects 拟合 q=8/16/32 的输出 POD。单列：measurement POD、可构造的 physical-current decoder basis、解析 anchor/residual baseline、POD+ridge。先比较 oracle projection 能否达到 ε_crit，而不是先训练网络。

仅 Gaussian 向量齐备时，可做两个对象的 leave-one-object-out、超参固定的表示 oracle；不能伪称四对象 B split 完整，更不能用同一 test object 同时挑 q、调不确定性门槛和报结果。B 若缺独立 validation，保持结构诊断而不校准安全概率。

## 5. object-wise split 合同

A 的主检验采用固定循环四折。顺序为 smooth、near-contact、layered、asymmetric。第 i 折 test=i，validation=(i+1) mod 4，其余两个训练。这样一个对象的所有 phase/k/历史路径/重复记录永不跨 train、validation、test。

每折主查询数：训练十八、validation 九、test 九。历史 query 增强只加入所属对象的对应 partition。不要为了避免 representation shift 而事后调整循环划分；分别报告 Gaussian↔voxel 外推带来的困难。

scaler、POD、learned sketch、图结构超参和 loss weight 只能在 training/validation 定义。测试对象允许使用纯物理的 per-query normalization，不能使用测试标签做 fitting。

A17 的四对象结果已经被看过，所以这是 held-out-object retrospective pilot，不是 pristine prospective blind test。随机 phase augmentation、atom phase augmentation 和 permutation augmentation 只增加不变性检查，不增加独立物理对象数。

## 6. Stage A1：后续 tiny training，默认不在第一轮执行

只有 G0 的 `A_GO_TINY_PILOT` 后，另以显式 stage 参数运行。模型：shared MLP、Set Transformer、普通 GAT、physics-edge GNN。统一 listwise+固定 top-m auxiliary loss，固定 m=4。

资源上限：四模型×四折×两个固定 seeds=最多三十二 fits；每 fit≤200k参数、≤60 epochs；最多两 GPU 小时（占用上限，不是预计运行时间）；device allocated≤3 GiB；无 architecture search，无多次挑最好 seed。

width 32 或 64 的选择在看 test 之前固定，不两者都扫。batch 单位是 query graph/set，而不是从不同 query 混抽 atom 后丢失上下文。记录每个 epoch 的 train/validation；按统一 validation objective 选 epoch，所有 seed 结果都保留。

必要 ablation 在通过主 gate 后才运行，最多额外八 fits：移除 M；删除 directed P edges；打乱物理边但保留 topology；只保留 norms 丢 signed phase。优先以 inference-time perturbation 做零训练诊断，再决定是否值得 retrain。

## 7. Stage A1 的预注册成功门槛

以下是实用 effect-size gate，不是来自 A17 的定理：

|指标|目标|
|---|---|
|near-contact Cov12|≥0.70|
|asymmetric Cov12|≥0.85|
|四对象等权 mean coverage|≥0.85|
|其他对象相对最佳 cheap analytic baseline|每对象下降不超过0.03|
|NN 相对最佳廉价解析方法|平均覆盖提升≥0.05；或差≤0.01且同质量的 feature+proposal+required-endpoint 费用至少减半|
|后接同一 full-gain/no-op 验收|不新增超过 tolerance 的错误接受|
|physics-native claim|物理边/材料注入消融有跨对象可重复效应，预设实用差异0.03|

不把在三十六点上的标准差包装成大量独立样本的置信区间。展示四个 object 的完整结果与所有 seed，而不是只展示 mean。

## 8. Stage B1：补偿网络单独 gated

B training 不因 A 通过而自动启动。必须先通过数据可用性、source/latent oracle 和 margin headroom 三项。优先 state-conditioned linear low-rank map；比较 direct measurement output、unresolved current coefficients（仅有合法 teacher 时）、physical latent，以及无学习 anchor/POD+ridge。

B 初版规模≤200k参数、rank≤32、≤60epochs、最多两种子。不与 A 一起全组合搜索；无完整 voxel labels 时，不开始四对象 “B training”。第一阶段全部 shadow：真实向量标签用于验证每个 hypothetical skip，实际策略仍不跳过 full check。

B 的成功门槛：matched finalist 下可跳过比例足够达到至少两倍全方向 action 减少；初始化、cache refresh 和 audit 都计入；保留≥95% matched 正收益；每个 held-out object 不新增实质负 gain 接受，并报告 tolerance 内的未定接受；优于 anchor/POD+ridge 的误差—费用曲线。

半径来自 empirical uncertainty 时结果列写 `EMPIRICAL_GATE`，不得写 `CERTIFIED`。four-object pilot 不为未来任意对象提供 error-rate guarantee。

## 9. Stage C：最终四格消融，不是当前第一轮

保持同样 frozen object/state/current budget，比较：

```text
C0 analytical shortlist + anchor + full checks
C1 learned shortlist    + anchor + full checks
C2 analytical shortlist + anchor + learned defect / exact fallback
C3 learned shortlist    + anchor + learned defect / exact fallback
```

先固定 common checkpoint/finalist 的诊断，再做路径级比较。新路径状态无真实 full label 时不得从别的 path 借用。后续真实计时与新 label acquisition 都需要独立授权/预算，不能伪装成当前已有 frozen-data replay。

C 中必须记录 `cache_error_radius`、`cache_error_type`、`cache_refresh_rhs`。改变 proposal 后重新审查 B 的 selected-finalist uncertainty calibration。

## 10. 统一的 cost ledger

```text
attempt_id, object_id, state_hash, method, stage, status,
setup_inclusive_wall_s, feature_wall_s, proposal_wall_s,
endpoint_wall_s, anchor_wall_s, neural_wall_s, transfer_wall_s,
full_initialization_rhs, full_finalist_rhs, full_refresh_rhs,
full_audit_rhs, offline_label_rhs, training_wall_s,
peak_device_bytes, total_attempt_wall_s, reused_cache_scope,
measured_or_counterfactual, failure_reason
```

总 wall 与内部阶段是嵌套关系，不能全部相加。所有失败、prewarm、拒绝、no-op、GPU transfer、补充 exact checks 都收费。以 CPU teacher lookup 对 GPU full physics 的比较不能叫 measured speedup。

RHS 以六照明计数，不把一个 shared material direction 写成一个物理 RHS。若同时报告方向数，另有 `direction_count` 列。[A17-report]

## 11. 第一轮硬上限和停止状态

第一轮：只读四对象已有文件；最多 47,712 来源 priority rows；三个既有 k 与三个 phases；单 worker；最多一 CPU 小时的计算资源预算；最多4 GiB进程内存，超大数据分块；新 full-wave RHS=0；训练 epochs=0；GPU 训练=0。

资源上限到达记 `BUDGET_STOP_PARTIAL`，保留已完成行，不把未完成表填零。输入缺失记 `DATA_INCOMPLETE`，不绕过权限访问 evaluator-only 字段来“补齐”输入。禁止自动扩大对象、字典、频率、噪声或 nonlinear trajectories。

## 12. 必须输出的首轮结果

`A18_DATA_INVENTORY.csv`、`A18_QUERY_MANIFEST.jsonl`、`A18_FEATURE_COSTS.csv`、`A18_ANALYTIC_SHORTLIST_BASELINES.csv`、`A18_DEFECT_LABEL_AVAILABILITY.csv`、`A18_DEFECT_ORACLE.csv`（没有合法向量时写状态，不编数据）、`A18_SPLITS.json`、`A18_COST_LEDGER.jsonl`、`A18_STAGE0_DECISION.md`。

最终 Stage0 verdict 精确取值：

- `A_GO_TINY_PILOT` / `A_NO_NEURAL_HEADROOM` / `A_DATA_INCOMPLETE`。
- `B_GO_ORACLE_TO_PILOT` / `B_NO_MARGIN_OR_COST_HEADROOM` / `B_DATA_INCOMPLETE`。

两个分支分别裁决。A 失败不自动否定 B；两者都被合法、匹配的非学习基线或 oracle 否定时，停止 NN branch，不启动 Flow。