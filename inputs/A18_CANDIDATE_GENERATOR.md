# A18 — Learned Current Candidate Generator

## 1. 学习问题不是重新预测 Q

输入一个当前 receiver-centered 电流空间 U、同一 frozen material state、residual、richer anchor、合法原始 dictionary，输出十二个 distinct incoming atom IDs。之后保留 A17 原来的：实际 child GN step → analytical anchor 排序 → 完整 Jd 验收。

物理决定动作的定义、可行性、反馈特征和最终接受；学习只决定**先把哪些 incoming directions 送进昂贵的端点求解**。初版不生成任意线性组合，不改变 current budget，不直接回归材料图像。

A17 的 measured bottleneck 是池外遗漏，不是已构建端点的 anchor 无法排序。[A17-report] 将网络训练成重算同一个 anchor 分数，即使推理很快，也可能已经为获取它的输入支付了几乎全部原始成本。

## 2. 准确的条件 teacher

对一个 query q=(state,U,x,r,Λ,anchor,dictionary)：

\[
Q_{a,i}=\Phi_F(s_U)-\Phi_F(s_{U-a+c_i}),
\]

仅在原始 atom 删除/加入之后实际秩满足预算、current block 合法且同一材料坐标下定义。对 incoming atom 的主标签是

\[
t_i=\max\{0,\max_{a\in U,\;feasible}Q_{a,i}\}.
\]

这是在删除动作上边缘化的 utility label，不是单纯 addition gain。移出 a 与移入 c 的组合是实际语义；ID 不得被 QR/SVD 后的列编号替代。[A17-core, A17-run]

存储：best incoming（允许 tolerance 内并列）、top-m membership（m=4 的预注册主设置）、每个 i 的最佳 drop、完整可行性 mask、Q 的数值 floor、query 和 source hash。m=4 是 pilot 参数，不是理论最优值。

### 损失

主损失采用按 query 归一化的 gain-weighted listwise cross-entropy：

\[
p_i^*=\frac{(t_i-\tau_q)_+}{\sum_j(t_j-\tau_q)_+},\quad
L_{list}=-\sum_i p_i^*\log p_{\theta,i}.
\]

如果分母为零，标记 no-op/no-positive-opportunity query；不强制赋给某个 atom 正标签。可加入 top-m binary auxiliary loss。Pairwise loss 仅对 teacher gain 间隔超过数值 floor 的配对启用，避免学习无意义的微小差别。禁止以全字典 Q regression MSE 作为主结果。

所有 query 等权，再按 object 等权；不能让大字典或记录较多的某对象主导训练。教师 Q_true、完整 J、完整最优步和真值只存在于 teacher/evaluator 命名空间。

## 3. 特征必须有取得成本

定义 F0/F1/F2/F3 四级能力边界。

|级别|内容|何时取得|成本边界|
|---|---|---|---|
|F0|原始 receiver 坐标、atom norm、candidate family、当前 selected mask、rank 与合法成本元数据|公共缓存|仍需计入首次公共缓存构建|
|F1|LC、SC、U*LC、C*LU、注入草图、dressed observation、少量稀疏耦合|全字典批量或缓存增量|这是 proposal 的最高默认能力；禁止每 atom child GN|
|F2|实际 child step、完整 N 或高维材料收缩、anchor.j(d)、精确 Γ block 配对评分|仅十二个 incoming 与可行 drop 组合|NN 前拿到这些可能已支付大部分目标成本|
|F3|完整 J_Fd、J_F、真值、完整最优 GN 步|teacher/evaluator 或最终物理检验|不作为 proposal 输入|

若已有 A17 workspace 缓存使某项 F2 非常便宜，则以实测成本重新分级，不通过命名宣称节省。首轮同时报告 cached/warm 和含必要 setup 的 cold-reconstructed 口径；后者未经独立测量不能叫实测 cold speedup。[A17-cost]

## 4. O/P/M 节点特征

### O：观测

使用归一化 ||Sc_i||、receiver singular coordinate 的能量摘要、||Y_i||、dressed/direct ratio。与接收任务相关的复量不能直接取模后全部丢弃相位。

### P：内部反馈

使用 ||(I-L)c_i||/||c_i||、投影到 retained space 的反馈、两向 U*Lc_i 与 c_i*LU、归一化 |γ_i| 及核心 reciprocal condition。γ_i=c_i*L(I-R_UL)c_i 为 scalar 时，不使用其恒为 1 的通常条件数；pair/block 情况才使用真正的 Γ condition。

L 不应被未经证明地对称化。所有稳定性门槛沿用 A17；不加 undocumented jitter 把本来不合法的 endpoint 改成新问题。

### M：材料切线与 signed task

使用 ||N_i|| 或成本受控的 N_iT 草图、与当前材料方向/残差 pullback 的对齐、第一阶 signed features。T 采用固定物理 probe，经各对象保存的材料 gauge 投影；不能把 Gaussian 第 7 参数与 voxel 第 7 参数默认为相同物理自由度。

建议把每个 probe 的解析响应

\[
w_{i,t}=Y_i\,\gamma_i^{-1}(N_i t)
\]

及 Re⟨u_A,w_{i,t}⟩、||w_{i,t}|| 作为 gauge-invariant 特征。它是便宜的响应摘要，不冒充真实交换 d 的 full finite gain。

材料 norm 与草图的缩放采用固定物理度量或训练对象拟合的 scaler。测试对象可以做公式明确的单 query 归一化，但不能使用该对象的标签分布调 scaler。

### 当前状态与成本

包含 k、当前 receiver rank、离线固定的 LM/prior 数值、合法 atom mask、current-family 标识、rank-dependence 指标以及估计的 feature/endpoint cost。禁止 object ID、文件名、随机 seed、smooth/near-contact/asymmetric 人工类别、teacher hash 等捷径特征。

## 5. 边与 NN-native 的最小实现

候选不是像素。先采用 candidate-to-retained bipartite edges，其物理内容是 U*LC 和 C*LU；再按廉价耦合幅度给每个候选最多八条 candidate-candidate directed edges。八条是资源上限，不是物理近邻假设，不能按欧氏距离直接截掉强远程反馈而不做比较。

可用边字段：c_i*Lc_j、Γ_ij、c_i*c_j、归一化 feedback product。精确 pair synergy 需要两个 child endpoint 或额外求解，不属于免费边特征；只能用固定的低成本近似，并单独计费。1-swap 的主任务不因观察到 addition pair synergy 就改成未经授权的 2-swap 搜索。

### 相位一致路径

以 η_i=N_iT 初始化复隐藏量 h_i；它随 atom 相位作 e^{-iφ_i} 变换。传播使用 Γ_ij h_j 或同等协变的有向边操作，attention 权重只依赖不变量。读出以 Y_i h_i、h_i*Γ_ij h_j 及其与 residual 的实内积构成 invariant logits。

不要对复隐藏量逐分量套 ReLU 后声称严格相位等变。可使用依赖模长的实 gate、复线性通道混合或先构造完备程度明确的 invariant 特征。每个架构都必须通过重排/相位测试。

对这里的 h，message passing 是提案特征构造，不是在取代小 Γ 精确求解。网络不因写出了 Γ^{-1} 就自动获得“学习 Maxwell solver”的贡献。

## 6. 四种架构公平比较

|架构|输入与输出|应检验的假说|风险|
|---|---|---|---|
|共享 MLP|每候选 invariant O/P/M + pooled core context → logits|单候选特征是否足够|无法显式建模候选间条件依赖|
|Set Transformer|candidate tokens 与 selected-core tokens；每元素 logits|集合上下文/多样性是否足够|dense attention 成本；本身不保证复相位一致|
|普通 GAT|相同节点、固定合法图；不使用复 feedback edge|拓扑与 learned attention 的作用|只换网络名字并不 native|
|physics-edge GNN|相位一致的 Γ/retained coupling message + Y/N 读出|真正的反馈边能否提升 held-out 覆盖|复杂特征成本与小样本过拟合|

相同 O/P/M 特征预算、参数上限、训练 split 和 loss。普通 GAT 与 physics-edge GNN 的区别应只在明确的物理边路径，不同时增加更多 teacher 信息。若 MLP 胜出就采用 MLP，不为了文章标题强行保留 GNN。[R12,R13]

## 7. 必须先跑的非学习基线

B0：原 family-prefix balanced-12，忠实复现 A17。

B1：family-matched random-12，固定三个种子，揭示原顺序偏差；它不贡献新独立物理样本。

B2：O/P/M analytic union-12。分别按 observable energy、feedback strength、material injection/cheap signed alignment 排序，按冻结轮转规则合并去重；只用 F0/F1。

B3：同特征的 normalized transfer/signed surrogate + diversity-12；保留复 signed alignment，并以固定 overlap penalty 去重。不使用 actual child d。

B4：full-dictionary exact endpoint + analytical anchor。昂贵的 opportunity upper benchmark，不是廉价部署基线。

B5：full-objective best one-swap teacher，包含 no-op；不是全局 current-space optimum。

B2/B3 的权重在 train/validation 选择一次，不能在四个已知测试对象上调出最佳组合再当无学习基线。也保留完全固定无调参版本。

## 8. 覆盖指标与 downstream 指标

对 object o，其九个共同 receiver checkpoint 的 gain-weighted opportunity coverage：

\[
\mathrm{Cov}_{12}(o)=
\frac{\sum_{q\in o}\max(0,\max_{a,\;i\in P_q}Q_{q,a,i})}
{\sum_{q\in o}\max(0,\max_{a,\;i\in D_q}Q_{q,a,i})}.
\]

零分母记为 NA，并另外报告 no-positive-query 数。另报 top-1/top-4 recall、per-query gain regret、对象等权均值、四个对象各自数值。

下游评价必须固定 analytical anchor、finalist 上限、数值 floor 和完整 Jd 检验。第一轮使用已付费标签做单次决策 replay；它可以计算反事实单步接受结果，但不具有真实 online wall time，也不能凭离线查表宣称省掉了 full solve。

多步 learner 路径若走到历史中未保存的 U/x，已有 teacher 其他路径标签不能拿来顶替。该项标成 NEED_NEW_LABEL/NOT_RUN，不要伪造三步轨迹收益。

## 9. 数据量和预注册 gate

已有 47,712/43,108 是 source rows/feasible labels，去重后的 query 与 incoming label 数必须实际输出。四对象按 outer leave-one-object-out 检验；每折另留一个完整对象做 validation，其余两个训练。主结果只有三十六个共同 receiver queries，后续历史路径可作为同对象训练增强或单列测试。[A17-NN]

拟定的工程 gate，不是已经达到的结论：

- 在固定十二候选下，near-contact Cov≥0.70、asymmetric Cov≥0.85，四对象等权均值≥0.85；其余对象不比强解析基线低超过 0.03。
- 相对最佳廉价解析 B2/B3，平均覆盖至少提升 0.05；或覆盖差不超过 0.01，且 proposal+feature+所需 endpoint 总费用至少减半。两者满足其一。
- 后接固定 anchor/高保真验收，保持 no-op 语义，不新增高于数值 tolerance 的错误接受；只改善覆盖、不降低 Jd 次数，也是 Route A 的有效结果，但不是 Route B 的加速结论。
- 对 physics-native 论文主张，物理边打乱/移除后的性能应有可重复、跨对象的实际下降；3 个百分点可作为 pilot 的最小实用效应阈值，不是显著性定理。

第一轮不训练：先检查强解析 B2/B3 是否已达到目标。如果是，A 分支停止以 NN 为中心继续投入，保留解析提案改进。

通过非学习 gate 后才允许后续 tiny pilot：四种架构、每种最多 200k 参数、两层/宽度 32 或 64 中预先固定一个设置、四折×两种子、最多 60 epochs，无搜索架构/超参网格，最多两 GPU 小时。固定物理状态不扩展，未付费 full RHS 为零。这个上限是执行合同，不是能学会的样本复杂度保证。