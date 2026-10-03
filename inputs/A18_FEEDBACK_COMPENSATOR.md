# A18 — Feedback-Defect Compensation Network

## 1. 本分支的独立问题

给定物理由真实 Galerkin endpoints 产生的固定材料位移 d，减少评估其完整 frozen gain 所需的 J_Fd 次数。网络不改变 d，不更新材料真值，不取代整个 Maxwell solver，不重新输出 scalar Q。

基础分解是

\[
z_F=z_U+\Delta z_U,\qquad
\Delta z_U=O_UL^{-1}I_Ud.
\]

但是 A17 已有 richer anchor A；应同时比较

\[
\Delta z_U=(J_A-J_U)d+\Delta z_A,\quad
\boxed{\Delta z_A=(J_F-J_A)d}.
\]

**优先检验 anchor-tail compensation。** 已知的 J_A−J_U 不交给网络重新学习。anchor 评分准确不等于 Δz_A 很小或低秩；这恰好是需要用向量标签检查的 oracle。[A17-run]

最接近的成熟范式是 far-field paper 的物理近似输入加修正；更直接的数学近邻是 learned operator correction、ROM closure 与 residual-based neural-operator correction。[R1,R7,R11,R16]

## 2. 固定状态下要学习一个线性响应算子

用 A 表示具体选择的 baseline basis，定义 b=I_Ad、t=L^{-1}b。满足

\[
A^*b=0,\qquad A^*Lt=0,\qquad\Delta z_A=O_At.
\]

建议网络由 χ、A、已有物理反馈摘要生成低秩算子参数，随后线性地作用于 b 的廉价草图：

\[
\widehat{\Delta z}_A
=O_AP_A^LW_q\,K_\theta(\mathrm{state},A)\,E_A b.
\]

这里 E_A 是明确的线性 injection encoder；W_q 是合法的 current decoder basis；K_θ 是小矩阵。所有矩阵可以随 frozen material state 与电流空间变化，但不能随当前 d 非线性变化。输出满足 Δz(0)=0、Δz(d_1+d_2)=Δz(d_1)+Δz(d_2)、实数缩放一致性。

这个表示是一个候选低秩模型，不是普适表达定理。如果 E_Ab 丢失必要激励，或 O_AP_A^LW_q 无法表示真实输出，再大的 K 网络也补不回来。必须分别做 source-compression 与 output-basis oracle。

禁止把完整 projected workspace 的 inverse 标成 full physical inverse。`public_workspace.npz/PB` 不必包含完整 I_Ad 所需的全部分量；使用其草图时清楚标注 source-projected 模型。[A17-state]

## 3. 三种输出空间公平比较

|方案|输出与 decoder|优点|必须承担的限制|
|---|---|---|---|
|Measurement correction|直接输出同一照明/receiver 坐标的 Δz|标签取得最容易，直接对应 gain|可能不由合法 current 产生；固定 receiver layout；须显式保持方向线性|
|Unresolved-current coefficients|输出 W_q 中的系数，再经 P_A^L 与 S|保留 current/measurement 关系，可计算物理残差|需要 current snapshot 或可信系数 teacher；测量 fullJ 本身不够反推唯一系数|
|Low-rank physical latent|线性 injection encoder → state-conditioned feedback matrix → physical output decoder|可变维度与相位规则清楚，可压缩计算|latent 截断必须达到 gain margin；物理基构建也要计费|

另设 measurement-POD + linear ridge 的极强廉价基线。它是统计输出 latent，不把它叫 current-native。若其与神经模型相当，应停止复杂网络，而不是改名包装。

当 q×q 的 Γ 已准确已知且可便宜求解时，先精确算 Γ^{-1}。学习重点只能是未表示的 tail feedback/注入/返回或跨状态压缩，不是为了使用 NN 而逼近小矩阵求逆。

## 4. Tail feedback 不等于一个 self-energy 参数

完整 complement 分块公式见 THEORY 文件。被消去的 tail E 同时改变：effective Γ、effective injection、dressed observation，以及 direct tail output。只预测 Σ=Γ_CEΓ_EE^{-1}Γ_EC 不保证学到 Δz。

如果用输出 Δz 监督这些隐藏块，它们还可能不可辨识：不同隐藏参数给出同一 measurement response。因此第一轮不拟合四个大型自由矩阵。先用最小低秩端到端线性响应表示，并以 exact small-block physics 作已知部分；仅在 oracle 显示 tail 拟合有稳定结构时讨论可解释 self-energy 参数化。

反馈网络不能将 Γ 强制为 SPD：current propagation 与材料 GN Hessian 的性质不同。任何数值稳定化都作为显式算法改变记录，不能悄悄加 jitter。

## 5. Teacher labels 与现有数据的实际边界

每条 response label 必须包含 identity=(object,state,k,U hash,step hash,d hash,receiver layout,material gauge)。训练标签是

\[
\Delta z_U=J_Fd-J_Ud,\quad\Delta z_A=J_Fd-J_Ad.
\]

主比较固定同一组实际 d，不拿一个模型的准确动作与另一个模型的随机方向混比。

### 已有可用性

Gaussian：源码保存已付费 `offline_information_reference.npz/fullJ`。只要相应 baseline basis、公共 PB、d 或可复建 endpoint 可读取，就能离线得到任意合法 d 的 measurement label。不得再次支付 full tangent RHS 获取已存在标签。[A17-run]

Voxel：源码保存的是十六方向 `V/fullJV`。只有 d 确实在 V 的张成空间内时，才可以由 fullJV 重建完整 J_Fd；必须先做投影残差检查。A17 的任意交换 d 一般不能假定满足该条件。接受路径保存的 step/Js 可能恢复部分 directional labels；须检查基准和相邻端点是否齐备。其余缺失项记 MISSING，不以 projected Jd 充当 full label。

两种表示都没有由上述测量数组自动得到 full unresolved currents。Current-coefficient teacher 的存在需要另行盘点。

所有已有 fullJ/fullJV 都是 evaluator-only。训练只取允许的目标列，测试预测不得访问它们；生成 training POD、scaler 和 latent basis 时不得包含 held-out full outputs。

## 6. 不训练之前的三个 oracle

**Oracle B1：能表示吗？** 对真实 Δz_U、Δz_A 分别做训练对象拟合的输出 POD，固定 q=8/16/32，在 held-out 对象投影。记录 field error 的绝对值以及是否小于各动作的 ε_crit。全对象联合 SVD 只能标成 transductive upper bound，不当泛化结果。

**Oracle B2：输入是否够？** 用相同合法 feature/source sketches 做线性 ridge 或固定解析 correction。若许多几乎相同的 cheap input 对应相反的大输出修正，说明缺信息或跨状态歧义，不能靠增大网络自动消除。采用 learned encoder 前也要计其需要的物理 actions。

**Oracle B3：有 gate headroom 吗？** 直接以输出基的 oracle projection 代入真实 gain，测哪些 finalist 达到 margin。即使假设系数完美，若可跳过比例仍低于成本门槛，本分支不值得训练。

先做 Gaussian，并把 voxel 可支持方向单列。不能将仅两个 Gaussian 对象的正结果宣称跨四类对象。若 voxel vector labels 不足，B 在当前“仅已有数据”合同内保持 UNRESOLVED，不启动补标签大实验。

## 7. Loss：输出保真为主，动作敏感性为辅

基础是 scaled measurement MSE，但不能让 ||Δz|| 很小的样本因相对误差分母过小主导。使用固定物理尺度和训练集决定的 floor；报告原始绝对误差。

增加 goal-weighted loss：

\[
L_{goal}=|\operatorname{Re}\langle u+z_A,\widehat{\Delta z}_A-\Delta z_A\rangle|^2.
\]

它强调当前 gain 最敏感的输出分量，但不替代向量监督。训练权重可以使用 teacher 信息；网络输入不读取 teacher。NN 不输出 Q，只由固定解析公式评估 Q。

物理解码器可加 ||b-L\hat t|| 的 residual loss；这是 a posteriori diagnostic，不在缺少稳定性/输出映射界时叫证书。对一致的 \hat t=P_A^LW_q a，有

\[
\rho=b-L\hat t,\qquad\Delta z-\widehat{\Delta z}=SL^{-1}\rho.
\]

这也解释了为什么“残差小”与“measurement error 可靠小”之间仍需要额外控制。

## 8. Uncertainty 必须与 gain margin 对齐

精确 u 的条件下：

\[
Q-\hat Q=-\operatorname{Re}\langle u+\hat z,e\rangle-\tfrac12\|e\|^2.
\]

以误差半径 ε 得到 κ=||u+\hat z||ε+ε²/2。仅当

\[
\hat Q-\kappa>\tau
\]

才有条件接受。ε 是已证明的界时是确定性结论；是校准的统计半径时，只能在对应抽样/校准假设下解释；是 ensemble σ 时只是一项经验 gate。

单纯 low-uncertainty -> accept 是错误的。例如 u=-0.009、预测 z=0.01、真实 z=0.02 时，预测 gain=4×10^-5，真实 gain=-2×10^-5。绝对输出误差 0.01 是否“小”并不重要，关键是它超过该动作 margin 允许的误差。

若预测分布均值为 \bar z、协方差 C_z，在固定精确 u 下：

\[
\mathbb E Q(z)=Q(\bar z)-\tfrac12\operatorname{tr}C_z.
\]

因此只把均值代入二次式也会遗漏曲率上的不确定性惩罚；这个期望修正仍不等于高概率安全。

小 ensemble 可以作为 pilot 的 epistemic indicator。必须在**完成 proposal+anchor 选择后的 finalist 分布**上评价门槛，而不是在容易的全字典平均误差上校准。

## 9. 连续接受会污染缓存

A17 的 u=r+J_Fx 是真实输出缓存。[A17-run] 若接受一个 learned-only action 并更新 \hat u←\hat u+\hat z_d，则其误差满足

\[
\epsilon_{u,new}\le\epsilon_u+\epsilon_{z,d}.
\]

下一步 κ 必须加入 ε_u||\hat z||、ε_uε_z 项。精确计算下一次 J_Fd 不修复旧的 u。可选策略：累计半径；每若干次 paid refresh J_Fx；或把当前所有历史方向批量重算。所有 refresh/audit RHS 都要入账。

第一阶段采用 shadow mode：每个 finalist 都实际有完整标签，模型只记录假如跳过会发生什么，不真正用近似缓存去驱动下一条未验证路径。它给出 counterfactual skip fraction，不能记成实际节省时间。

## 10. 四对象不能提供强分布外安全认证

四个固定对象及其重复 k/state/move 不等于几万独立校准样本。标准 split-conformal 上界的 rank 为 ceil((n+1)(1-α))；即使对象可交换、模型和 gate 已固定，要获得非无穷的 95%/99% order-statistic 上界，至少需要 19/99 个独立校准单元。这只是分辨率条件，不是充足的统计样本保证。[R14]

本 pilot 每折只有一个完整 validation object，更不能据几千条相关 move 宣称 95% unseen-object acceptance safety。CRC 对 expected monotone loss 的结论也不能直接换成选择后的 conditional wrong-acceptance rate。[R15]

同时报告：实质负 gain 接受（Q_true<-τ）；未达到明确正 gain 的接受（Q_true≤τ）；收益损失总量；对象最大风险；拒绝好动作比例。零观测错误只能写“本样本未观察到”，不能写理论保证。

## 11. 两倍 RHS 减少的正确成本式

设 T 是原 finalist actions，T0 是初始化缓存，Tr 是刷新，Ta 是随机审计，f 为跳过 fraction：

\[
\mathrm{ratio}=\frac{T_0+(1-f)T+T_r+T_a}{T_0+T}.
\]

要把总 verification RHS 减半，需

\[
f\ge\tfrac12+\frac{T_0}{2T}+\frac{T_r+T_a}{T}.
\]

用 A17 报告的 1164 finalist RHS，并假设 36 cells 各一个六照明初始化（216 RHS），在完全无刷新/审计的理想情况下就需要跳过约 59.28%，不只是 50%。这是条件成本推算，不是已完成的 A18 加速实验。[A17-report]

如果只报告 finalist RHS 减半，必须明确排除初始化。NN feature、CPU/GPU transfer、latent basis 构造、校准、拒绝和训练摊销另列；少一半 tangent actions 不意味着整体 wall time 两倍。

已有 dense Gaussian fullJ 在离线标签制作中可免费查表，但若部署也缓存完整 J，则 baseline 本身就是 cheap matrix multiplication，不能仍给 baseline 虚构每列一个 full solve 的成本。

## 12. GO / NO-GO

PROPOSED gate：训练对象构建的 rank≤32 oracle 表示，在 matched finalist 分布上已经能达到至少 60% 的理想 skip headroom；再要求实际小模型保持≥95% 可得正收益，所有 held-out 对象均未观察到超过 tolerance 的新增错误接受，且初始化/refresh/audit 计入后完整方向 RHS 至少减少两倍。

所有门槛都要同时和 J_A、较丰富解析 ROM、measurement-POD+ridge 比较。若只赢非常粗的 J_U，却输已经存在的 J_A，则不能保留为 A18 补偿贡献。

这是一项是否值得扩展数据的 pilot gate，不是论文终局安全证据。不能达到时优先停止 B，而不是立即增加网络深度、换 Flow 或改成全材料生成。