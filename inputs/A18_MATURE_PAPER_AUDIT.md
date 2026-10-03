# A18 — Mature Paper Audit

文献身份、链接和逐篇访问层级见 [REFERENCES.md](REFERENCES.md)。本审查把“论文实际支持的内容”和“A18 的比较/推论”分开。关键三篇尚未取得全文：far-field learning、2025 PEC EGAT、2019 physics-inspired current CNN。它们的 abstract 足以识别范式重合，但不支持断言其所有层、训练协议或理论边界；这些项目明确保留 OPEN。

## 1. Yin, Tan, Chen：Far-Field Approximation Learning，TAP 2023 [R1]

### 原文支持

其公开 abstract 描述：用 far-field approximation 构造可快速 inverse-Fourier 计算的输入，CNN 修正近似与 limited-aperture 所致的失真，目标输出是三维 reflectivity，包含 synthetic 与 experimental 验证。全文未获得，不补写 CNN 层数、loss 或样本数。

### A18 对照

最相近的不是“NN 给 current 评分”，而是

\[
\underbrace{J_A d}_{\text{cheap known physical response}}
\; + \;
\underbrace{(J_F-J_A)d}_{\text{specific approximation defect}}
\longrightarrow J_Fd.
\]

A17 的 exact algebraic defect 是 O_A L^{-1}I_A d；D_C=Y_CΓ_C^{-1}N_C 只表示指定 bank 的模型改变，不能一般性地冒充完整近似误差。

差别在于：far-field paper 的输出是成像表示，abstract 同时涉及 limited aperture，因此不能把其所有改善都解释成确定性 forward approximation repair。A18-B 在冻结状态下修正一个线性方向响应，不生成弱材料成分，也不利用图像先验填补观测缺失。

**对本项目的启发：先做 cheap physics，再只学剩余缺陷；不是先选一个 CNN。** 原范式已经存在，A18 不能宣称首次 physical approximation + neural compensation。

## 2. SOM-Net：current/material 交替与 unrolling [R2]

### 原文支持

全文 Algorithm 1 及 equations (6)–(11) 给出 deterministic current 和 BP permittivity 输入；学习 current update，再显式由 current 更新 material。损失同时包含 current、scattered field 和 material terms，实验设计使用四个 unrolled blocks。此处依据正文，不依赖未成功渲染的图表。

### A18 对照

SOM-Net 已经把 current 放在学习中心，并用物理关系连接 current、data 与 material。因此“网络先估 current 再得到材料”“把散射方程写进 loss”均不足为 A18 新意。

A18-A 的输出是 dictionary incoming IDs，不是 full current 或 material image；其 teacher 是条件实际材料步收益。A18-B 输出固定 d 的遗漏 tangent response，其用途是验收一个已经定义好的物理动作，而非学习整条重建迭代。

严格比较应保持这三点：动作自由度不同；学习标签不同；评价对象不同。SOM-Net 的图像指标不能直接与 A18 的 full-GN gap ratio 数值比较。论文级重建比较需另立共享噪声、数据和轨迹合同。

## 3. Guo et al.：Physics Embedded Deep Neural Network，TAP 2022 [R3]

### 原文支持

作者上传全文把 VIE/FWD SOLVER 嵌入 ISP SOLVER。网络从模拟与观测的 residual 预测 material update；公式 (11)–(16) 展示迭代结构，FWD 网络在 inversion training 中固定。期刊卷期为 2022，DOI/early-online 为 2021，不与同组 VIE-forward-only 论文混淆。

### A18 对照

“forward neural surrogate + inversion feedback”已经很成熟。A18 不应把同样的结构换成 Maxwell/current 命名后声称独立理论。

可区分之处：A18 保留现有 endpoint material step，不学习 residual→material 的大映射；学习仅用于提案顺序或某个 full directional check 的 defect。这样任务更小、验收接口清楚，也避免当前三十六冻结单元无法支撑完整图像生成训练的错配。

但是更小任务不自动代表更强论文；需要证明在同等 current budget 下取得更多合法收益，或者在相同收益/错误接受水平下确实省掉大量完整方向 actions。

## 4. Cao et al.：Physics-Data Hybrid Driven EGAT for 3D PEC，TAP 2025 [R4]

### 原文支持及访问边界

作者出版列表与论文记录确认 3D PEC surface-current learning、physics-data hybrid、edge-featured graph attention 的定位。公开摘要层面的描述包括用物理 Green/导数类信息构造边以及 mesh 信息构造节点。本轮没有取得全文，确切边公式、CFIE 离散细节、网络层数和消融协议仍 OPEN；不转述未经同口径复核的速度百分比。

### A18 对照

这是对“physics-edge GNN 就是创新”的直接压力。物理耦合进入图边、NN 输出电流这两件事本身不是新贡献。

A18 的节点是**dictionary modes**，而不是表面 mesh elements；边应依赖当前 retained core 消元后的 Γ 或双向 retained coupling，而不只是固定几何 Green 距离。材料 injection N 和当前 signed task 必须参与决策；输出是 mode shortlist 或 tangent-defect correction，而不是 incident-field→完整 PEC surface current 的 end-to-end forward solve。

这是研究对象与接口的实质区分，但在其全文未读齐前，只能说“已检索来源尚未给出同构的条件 current-budget task”，不能断言该文完全没有任何关联机制。

## 5. Wei & Chen：Physics-Inspired CNN，TAP 2019 [R5]

### 原文支持

摘要明确提出 induced-current learning、以多标签损失处理学习任务、通过 skip connections 旁路主要 current 部分并学习较小部分。全文未取得，因此不猜测“minor current”与 A18 residual-current projector 的逐项对应。

### A18 对照

这是比泛泛 ResNet 更直接的重合：**不学全部电流、保留已知主要部分、只学修正**，东哥的工作已经做过。A18-B 必须说明自己的“剩余”由 O_A L^{-1}I_A d 精确定义，随 current model 和材料状态变化，并且输出精度直接受 finite-gain margin 控制。

没有这些内容，只说“learn receiver-dark current completion”，会很接近既有 induced-current learning 的叙述。更换 CNN 为 GNN 不解决这个重合。

## 6. DeepNIS：物理粗图到精细材料图 [R6]

### 原文支持

正文将 BP image 作为第一个模块输入，多个 complex CNN 模块级联，后一个接收前一个输出，形成从粗图到细化 inverse-scattering image 的映射。

### A18 对照

Cheap physics + learned refinement 这一高层套路并不稀缺。A18 不做 BP image enhancement，更不把全部材料先验隐式塞进 network。只有 genuine full-material weak directions 与 prior headroom 被单独证明后，才应考虑该类图像先验。Gaussian 已有表示 weak dimension=0、voxel full space 未知的 A17 结果不能被改写成“应该立即上 Flow”。[A17-report]

## 7. Lunz et al.：On Learned Operator Correction，SIIMS 2021 [R7]

### 原文支持

全文 sections 3–4 分析只在 data space 修正 approximate forward map 的不足，并讨论 forward-adjoint correction；approximate adjoint 的作用范围和迭代中的 gradient consistency 是重要问题。

### A18 对照

A18-B 是一个更窄的接口：d 已经由现有物理 endpoints 给定，只需标量 finite gain，故完整梯度/adjoint 不必为了这一检查重新计算。这里的便利来自任务定义，不代表 A18 解决了所有 learned-operator adjoint 问题。

若后续把修正网络用于构造 GN Jacobian、Hessian 或优化 d，而非只验证固定 d，就必须重新研究一致导数/伴随。不能将“验证固定方向无需额外 adjoint”扩展成“学习 operator 永远不用处理 adjoint”。

## 8. DINO 与 MR-DINO：导数相关 surrogate [R8,R9]

### 原文支持

DINO 通过压缩 derivative information 改善高维算子及其导数的学习；MR-DINO 将 reduced architecture 与 optimization-variable derivatives 用于 PDE-constrained optimization under uncertainty。这些判断来自 primary abstract/method descriptions，非本轮逐公式复现。

### A18 对照

“NN 必须保持优化相关的导数”“用低秩导数而非完整 image MSE”已经是明确路线。A18 的差异应是 current-realizable 的 fixed-budget 动作，以及特定 omitted-feedback defect 的 source/propagation/measurement 分解，而不是一般性声称 task-aware surrogate 比 field-only 更好。

DINO 是必要概念基线：一个一般 derivative-informed low-rank surrogate 能否同样减少 Jd？如果能，而 Γ/Y/N 没有额外实用作用，A18 不宜把网络命名为新的 Maxwell-native 机制。

## 9. Certified RB–ML–ROM hierarchy [R10]

### 原文支持

作者提出 FOM/ROM/ML 分层、按需适应并以 a posteriori error estimates 控制模型切换；这比简单地用 ensemble variance 做 heuristic fallback 更强。

### A18 对照

“低误差走 NN，高误差退回物理”不是新体系。A18 可以贡献的是一个直接由 **当前交换 gain margin** 决定的输出精度预算，以及 current feedback 特征。但没有合法输出界时，不从该文借用 certified 一词；A17 自身也明确没有取得这样的界。[A17-decision]

## 10. Closure learning 与 residual correction [R11,R16]

### 原文支持

ROM closure learning 已研究以网络表示 resolved/unresolved mode 的作用。Cao et al. 的 residual-based operator correction 则用 PDE residual 定义一个线性变分修正，并在相应条件下分析误差改善。

### A18 对照

把 Γ_CC−Γ_CEΓ_EE^{-1}Γ_EC 解释为有效反馈并不是首创。A18 需要明确：遗漏 tail 不只改变 self-energy，也改变注入和观测返回；而 linear Maxwell tangent 的完整 residual correction 本身可能就是原本昂贵的线性 solve，不能直接移植 nonlinear-PDE 一次 Newton 校正的收益结论。

因此要比较最小非学习 correction。若已有小 ROM/投影修正同样跨过 ε_crit，就不需要为这一步训练 NN。

## 11. Set/GAT 与 uncertainty 基础 [R12–R15]

Set Transformer 提供集合处理，GAT 提供图注意力；它们不自行保证电流相位规范或 frozen tangent 线性。Conformal 方法需要其抽样/校准合同，expected risk control 也不等价于任意选中 finalist 的条件安全。

A18 的不可省略检查：同一 atom 的随机复相位；字典重排；冻结状态的线性叠加；selected-finalist 校准分布；cached Jx error 的传播；所有 fallback/refresh 费用。

## 12. 三个直接问题的回答

**哪个 exact defect 最像 far-field approximation defect？** J_F−J_A 或 J_F−J_U 的完整模型缺陷，而不是单独某个候选的 YΓ^{-1}N。更实用的是先减去已有 anchor 可以解释的部分，再学剩余 O_A L^{-1}I_A。

**哪些层由代数决定？** 线性材料注入或其带标注的草图、保留空间消元、方向/相位一致的 feedback 作用、已知 measurement readout、fixed gain evaluator。这些接口是结构要求；Transformer/GNN/MLP 的具体层数不是定理强制的。

**去掉 feedback algebra 会怎样？** 若只剩候选向量列表、普通 attention 和经验 loss，它就是 generic set ranking；若只剩 coarse output + unconstrained residual MLP，就是 generic operator correction。二者可以是好的工程基线，但不能同时声称发现独立 Maxwell-native neural mechanism。

## 13. Paper-level novelty ledger

|拟议说法|裁决|需要的额外证据|
|---|---|---|
|首次物理近似加 NN 补偿|否|R1/R5/R7 已有直接范式|
|首次 current-centered 或 physics-edge network|否|R2/R4/R5 已有直接重合|
|Γ/Y/N 接口支持条件 current proposal|可能有独立价值|同预算 strong analytic/MLP/GAT 比较与边消融|
|学习剩余反馈并以 gain margin 选择精度|可能有独立价值|vector oracle、实际少算 full Jd、缓存误差控制|
|低 ensemble σ 因此 certified|否|合法输出界或明确统计保证合同|
|三十六冻结单元证明最终重建改进|否|独立 nonlinear trajectory 与噪声/约束实验|
|一次检索已排除所有同构先例|否|至少关闭 R1/R4/R5 full-text OPEN 项后再收紧措辞|

**研究路线可执行；NN 的成功、两倍方向动作减少和论文级原创性仍是待检验结论，不是本轮文献审查已经证明的事实。**