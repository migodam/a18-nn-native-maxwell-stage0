# A18 — Codex 第一轮执行指令

## 角色与目标

你负责 A18 NN-native Maxwell current design 的 **Stage 0：无训练、无新增 full-wave actions 的可行性审计**。本轮不是复跑 A17、不是训练 NN、不是寻找更快 PCG、不是做 U-Net/Flow。

固定上游仓库：

`https://github.com/migodam/a17-maxwell-certified-exchange`

固定 commit：

`56097653e6a814399581c3759fc3e3e8f0c2284c`

入口：

`https://raw.githubusercontent.com/migodam/a17-maxwell-certified-exchange/56097653e6a814399581c3759fc3e3e8f0c2284c/START_HERE.md`

在当前工作目录新建 `A18_NN_NATIVE/`，全部新代码、派生数据和报告置于其中。上游 A17 输入、结果与代码只读。不覆盖历史数据，不自动 git push，不启动远端训练。

首先读随本指令交付的七份 A18 研究文档和 REFERENCES，随后核对固定 A17 source。没有随附研究文档时，本指令以下定义仍是必须遵守的自包含合同。

## 1. 为什么选择 Candidate Generator first

A17 full-dictionary analytical anchor 的 top-one 捕获已经很高；near-contact/asymmetric 的在线十二候选池覆盖只有约35.3%/65.9%。`balanced_ids` 实际是按 current family 轮流取前缀，不是强物理筛选。已有六次 learned priority pilot 未超过 analytical anchor。

我们要先判断：合法廉价 O/P/M 特征能否改进 incoming shortlist，而不是重新预测 Q。第一道对手是无学习的 O/P/M analytic shortlist；如果它已经消除缺口，不为了 NN 项目继续训练。

独立第二分支是完整 directional defect compensation，但向量 teacher 的可用性不能由 scalar Q labels 推断。

## 2. 数学与数据合同

共享实材料向量 d；每个方向包含全部六个 illumination RHS。保留 A17 的 material gauge、实化顺序和 Λ、ell。

\[
R_U=U(U^*LU)^{-1}U^*,\quad J_U=SR_UB,
\quad I_U=(I-LR_U)B,\quad O_U=S(I-R_UL),
\]

\[
J_F-J_U=O_UL^{-1}I_U,
\quad J_{V+c}-J_{V+a}=Y_c\Gamma_c^{-1}N_c-Y_a\Gamma_a^{-1}N_a.
\]

incoming proposal 的 teacher：

\[
t_i(q)=\max(0,\max_{a\in U,\;feasible}Q_{a,i}(q)).
\]

实际动作使用原始 dictionary atom ID。QR/SVD 只表示相同 span，不改变删除动作身份。No-op 不删除。L/Γ 不假定 Hermitian/SPD，稳定性沿用原代码，禁止任意 jitter。

完整 gain 对固定物理 d：

\[
Q=-\operatorname{Re}\langle u,z\rangle-(\ell+\Lambda x)^Td
-\tfrac12(\|z\|^2+d^T\Lambda d),
\quad u=r+J_Fx,\ z=J_Fd.
\]

## 3. 第一轮资源硬限制

只使用原四对象、early/middle/late、k=4/8/16。共同 receiver 检查点为三十六个。已有 priority 源记录报告上限47,712，feasible43,108；实际去重计数重新输出。

- 新 full-wave tangent/adjoint RHS：0。
- 新 nonlinear trajectories：0。
- NN training epochs：0；GPU training：0。
- 单 worker，进程内存≤4 GiB，累计 CPU 计算资源预算≤1小时。
- 原始文件只读，派生结果可分块/缓存。

如果匹配完整 query 需要比指定 scope 更多记录，输出 `SCOPE_MISMATCH`；禁止截断一个 teacher dictionary 再把它当 full-dictionary oracle。预算到达保存 partial results，状态 `BUDGET_STOP_PARTIAL`。

不要为了补缺文件重建四十二个 external kernels。依赖或数组缺失输出明确 missing inventory，并继续所有不依赖缺失项的审计。

## 4. 第一步：数据盘点与泄漏防护

通过已有项目索引/manifest 定位本地 A17 数据；不要猜历史机器路径。读取固定 source：

- `A17_R1/code/run_exchange.py`：`balanced_ids`、`moves`、`evaluate_neighborhood`、`policy_run`、`run`。
- `A17_R1/code/maxwell_state.py`：`public_workspace`、`offline_gaussian_J`、Teacher/Probe capabilities。
- `A17_R1/code/exchange_core.py`：realification、endpoint、Schur、gain 和错误界。

为每个 source 文件记录 path、sha256、object/state/k、shape、dtype、gauge、coordinate scope、public/teacher/evaluator role。

重点核验：

`public_workspace.npz` 的 L/S/PB 是公共 current embedding 的投影映射，不是完整物理 L/S/B。它可用于现有字典内的 reduced endpoint 计算，不能凭其 inverse 生成 full truth。

Gaussian 的 `offline_information_reference.npz/fullJ` 可用于离线生成 measurement response labels。Voxel 的 `V/fullJV` 只有十六方向；任意 d 不在其 span 时，禁止用 fullJV 作为真实 J_Fd。

`moves.jsonl` 常保存 Q_true、Qhat、hash、norm，不等于保存了所有 d/Jd。核对 accepted endpoint NPZ 的 step/Js 是否足以恢复个别合法差分。current coefficient labels 需要独立 full-current 证据。

Teacher/evaluator 数组不能进入 proposal/correction 的 public input。建立禁止字段 canary：随机改动 fullJ、Qtrue、truth、reference step 等，student feature/prediction 路径必须不变。A17 Qhat 可在 shortlist 之后做已付费 replay，不能拿它给 shortlist 模型当廉价输入。

## 5. 第二步：共同 query 与 teacher 表

以 object/state hash、current U/selected atom IDs、x hash、anchor hash、Λ/ell、receiver layout 定义 query。多个 policy 共用的同一 receiver 起点只保留一次。

以 query + drop/add physical IDs 定义 move。去除重复副本，保留 feasibility mask、actual rank、core stability、numerical floor、Q_true 及 exact source path。

每个 incoming i 的 teacher 是所有合法 drop 中最大的正 gain，另保存最优 drop 和 tolerance 内并列集合。无正收益 query 保留 no-op 标签。不要默认每个 query 恰好有一个真正正标签。

输出来源行数、unique query/move/incoming 数量，禁止将43,108可行行称为43,108独立场景。

## 6. 第三步：只做廉价物理特征

F0：cached receiver signatures、norm、selected mask、current-family、rank/cost。

F1：批量 LC/SC、双向 U*LC 与 C*LU、受控 N 草图、dressed Y、少量有向候选耦合和 cheap signed task alignment。

F2：actual child endpoint step、完整 anchor score。只允许 shortlist 之后使用。

F3：full Jd/J/reference/truth。只做 teacher/evaluation。

对 F1 单独计费，不为了一个输入特征悄悄对全字典做 child GN。scalar γ 的通常 condition number=1，应改用 normalized |γ| 与核心 reciprocal condition；块 Γ 才记录 condition。

无对象ID、数据路径、seed、几何类别名等网络捷径。候选 current-family 可作为合法物理来源元数据。跨 Gaussian/voxel 的材料草图必须尊重各自保存的物理 gauge，不把坐标编号当共同语义。

## 7. 第四步：无学习 shortlist 比较

固定十二 incoming，并由原 anchor 对全部合法 drop×incoming 组合排序。

B0：原 family-prefix balanced-12。

B1：family-matched random-12，固定三个 seeds，全数报告。

B2：O/P/M 排序的固定轮转 union-12，去重并补齐。

B3：cheap signed-transfer+overlap diversity-12，只用 F0/F1。

B4：已有 full-dictionary endpoint+anchor 的昂贵参照。

B5：已有 full-objective one-swap teacher，包含 no-op，不叫全局 optimum。

算每 object 的九 query 汇总：

\[
Cov_{12}=\frac{\sum_q\max(0,\max_{a,i\in P_q}Q_{q,a,i})}
{\sum_q\max(0,\max_{a,i\in D_q}Q_{q,a,i})}.
\]

零分母记 NA。另报每 query gain regret、top1/top4 recall、no-positive-query 数和四对象等权均值。

参考 A17 短池覆盖约0.969/0.353/0.920/0.659；anchor capture约0.981–0.996。不同则先找范围/重复/no-op/floor 原因，不改阈值去凑报告值。

只有已有相同 q,d 的高保真 label 时，才能回放单步 final acceptance。不要从历史 teacher 的另一条路径借用标签来模拟 learner 的新三步路径。

## 8. 第五步：B 的 vector-label / latent / margin oracle

不要训练 B。对合法已存 zF 计算 zU、zA、ΔzU、ΔzA。

先取 richer anchor tail：

\[
\Delta z_U=(J_A-J_U)d+(J_F-J_A)d.
\]

已知第一项不学习。标明 A 和 U 不一定嵌套。公共 Q-space 的 full solve 是更丰富的 reduced model，不是 F。

训练对象拟合 q=8/16/32 的 measurement POD；held-out object 只投影评价。合法 current snapshot 不存在时不构造虚假的 unresolved-current teacher。可运行固定 ridge baseline，不运行 NN。

每个固定 d、精确 u 下：

\[
\epsilon_{crit}=\frac{2m}{\sqrt{v^2+2m}+v},\quad
m=\hat Q-\tau>0,\quad v=\|u+\hat z\|.
\]

先判断最理想 latent projection 能否跨过所需精度；记 `oracle_can_skip` 只是反事实诊断。source sketch 或 output basis 的残差不达要求时，指出缺信息/低秩失败，不自动增大网络。

若只具备两个 Gaussian 对象的数据，就只报告两对象的表示 oracle；不报告不存在的四对象 B result，不做95%安全校准。

## 9. Split 与后续训练计划，本轮只生成 manifest

固定对象顺序 smooth/near-contact/layered/asymmetric。第 i 折 test=i、validation=(i+1) mod4、其余训练；所有同对象 phases/k/moves/augmentations 在同一 partition。

主36 queries每折18train/9validation/9test。fit scaler、POD、sketch、图/损失超参均不看 test labels。不要以 phase/k 随机分割制造泄漏。

后续若 A gate 通过，建议比较 shared MLP、Set Transformer、普通 GAT、phase-consistent physics-edge GNN；相同输入预算、listwise 主损失、top-m辅助（m=4），≤200k参数、≤60epochs、4fold×2seed、总训练≤2GPU小时。**本轮只生成配置，不执行训练。**

B 是独立 gated task，不能因 A 通过而自动训练。它的 NN 应保持 frozen-state 对 d 的线性，输出 Δz 而不是 Q。

## 10. 必要结构测试

直接运行提供的 `verification/verify_a18.py`，或等价复写其代数测试但保留原版结果。

将来网络实现必须通过：atom permutation、随机复相位、original-ID action consistency；补偿器还须通过 zero/source scaling/additivity。N、Γ、Y 的变换规则是 N'=E*N、Γ'=E*ΓE、Y'=YE，其中星表示共轭转置，不是元素乘法。

没有这些约束的普通 feature concatenation 只能标成 generic baseline，不写“代数强制的 Maxwell-native architecture”。

## 11. 不确定性与费用的禁止误述

A17 当前只提供 high-fidelity numerical checks，不提供确定性 output error bounds。

低 ensemble σ 不构成 certificate。统计校准必须说明独立单元；不能把四对象的相关动作当几千独立样本。

连续 learned acceptance 使 cached Jx error 累积；下一次 exact Jd 不会修复旧缓存。未来 C 必须计 cache radius/refresh，并对 selected-finalist 分布重新校准。

每方向六 RHS。若按A17的1164 finalist RHS与36×6初始化计算，总 verification RHS 减半的理想 skip 门槛约59.28%，尚未计 refresh/audit。仅减少 finalist 的一半不能叫全部流程两倍。

本轮 fullJ 查表及历史 Q 查表仅是 offline reuse；不据此声称 measured runtime saving。GPU transfer、feature construction、所有 failed attempts 和初始化都入账。

## 12. Stage0 裁决与退出

输出：

`A18_DATA_INVENTORY.csv`
`A18_QUERY_MANIFEST.jsonl`
`A18_SPLITS.json`
`A18_FEATURE_COSTS.csv`
`A18_ANALYTIC_SHORTLIST_BASELINES.csv`
`A18_DEFECT_LABEL_AVAILABILITY.csv`
`A18_DEFECT_ORACLE.csv`
`A18_COST_LEDGER.jsonl`
`A18_STAGE0_DECISION.md`

A 的后续实用目标：near-contact Cov≥0.70、asymmetric≥0.85、四对象mean≥0.85，并胜最佳廉价解析基线至少0.05或同质量费用减半。这些是预设 gate，不是已实现数值。如果廉价解析已经达到且没有 NN 费用空间，判 `A_NO_NEURAL_HEADROOM`，不启动 NN。

B 要先有合法 labels 和 rank/source oracle，能在 matched finalists 达到约60%以上理想跳过机会，才判 `B_GO_ORACLE_TO_PILOT`；数据缺失不是科学 NO-GO，应单列。

最后一页明确：Candidate Generator first 是否仍成立；A/B 各自 GO/NO-GO/INCOMPLETE；下一阶段所需的最少新证据；所有实际运行和未运行项目。不要只列“未来工作”。

A 失败但 B 独立 oracle 通过，不停止整个 NN 分支；A 被强解析方法消除、B 也没有精度—费用余量时，明确停止 NN branch，保留 A17 current-design 结论。禁止以 full-material Flow、U-Net 或新 solver 变体规避失败。

完成这些首轮输出后停止；不自动进入任何训练、补充 full-wave label acquisition 或 nonlinear reconstruction。