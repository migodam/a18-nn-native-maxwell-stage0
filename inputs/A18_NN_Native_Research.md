# A18 — NN-Native Maxwell Current Design

## 裁决

**Candidate Generator first。** 网络首先学习“哪些 incoming current directions 应进入十二候选 shortlist”，不重算已准确的 analytical anchor，也不直接学习材料反演。先用强解析 O/P/M 预筛选判断是否真的还需要学习。

Feedback-Defect Compensator 作为独立次线保留。其首要学习目标建议从 `(J_F-J_U)d` 收缩到扣除了已知 inner correction 的 **`(J_F-J_anchor)d`**，并用 actual finite-gain margin 而非全局 RMSE 决定精度是否够。

## 包含的七份交付

|文件|内容|
|---|---|
|[A18_NN_NATIVE_THEORY.md](#a18-1)|双侧 defect、Schur tail、相位/线性约束、margin 精度公式、路线裁决|
|[A18_MATURE_PAPER_AUDIT.md](#a18-2)|逐篇比较必读论文及 operator/ROM/uncertainty 近邻，列明全文访问边界|
|[A18_CANDIDATE_GENERATOR.md](#a18-3)|teacher、O/P/M node/edge、四架构、非学习对手、coverage 指标与门槛|
|[A18_FEEDBACK_COMPENSATOR.md](#a18-4)|三种输出空间、标签可用性、线性物理 latent、误差门、缓存与费用|
|[A18_ARCHITECTURE_COMPARISON.md](#a18-5)|Candidate first 原因、组合架构、能力隔离、分支失败后的停止规则|
|[A18_LIGHTWEIGHT_EXPERIMENT_SPEC.md](#a18-6)|object split、零训练 Stage0、后续 tiny pilot、计费与完整数据合同|
|[A18_CODEX_PROMPT.md](#a18-7)|可直接交给 Codex 的自包含 audit-only 第一轮指令|

另有 [REFERENCES.md](#reference-registry)、`stage0_config.json`、`verification/verify_a18.py`、`verification/results.json`、文件校验与 SHA-256 manifest。ZIP 还包含合并阅读版 `A18_COMBINED.md`。

## 本轮已经做了什么

读取了指定 commit 的 START_HERE、决策/完整中文报告/NN pilot/成本审计及 current exchange、runtime/data-saving 源码；检索并对照必读论文和相关原始研究；给出可执行的数学结构、标签定义、架构比较、资源与 GO/NO-GO 合同。

运行了128组随机复数非 Hermitian 有限维代数测试。验证项包括双侧残差恒等式、完整补空间表达、tail Schur closure、相位不变性、anchor 分解、线性与 gain-error bound；最大相对误差约5.4×10^-14，无误差界违反。脚本使用固定seed，结果可复现。

## 本轮没有做什么

没有训练神经网络，没有新增 Maxwell 求解，没有启动远端 GPU，没有重新逐数组分析完整 A17 大归档，也没有验证新的 nonlinear reconstruction。A17 实验数值均标记为固定报告来源；A18 的 coverage、精度与加速门槛是拟议目标，不是新实验结果。

SOM-Net、Physics Embedded full-wave 与 learned-operator-correction 等取得正文文本并核对方法；far-field、2025 PEC EGAT 和2019 current CNN 本轮主要取得摘要/出版记录，具体全文缺口列在文献审查中。PDF screenshot 服务请求失败，没有宣称图表已经视觉复核。正式投稿前应关闭这些明确的 full-text OPEN 项；此限制不妨碍当前的结构比较与小型可行性设计。

## 最重要的新增判断

原十二候选池是 family-prefix round-robin，因此不是强解析预筛选；NN 必须先赢得使用它的必要性。Gaussian 源码保存 fullJ，而 voxel 仅有十六方向投影，二者不能共用“完整向量 teacher 已齐”的假设。

B 的误差必须小于当前动作所需的 ε_crit；连续 learned acceptance 的 Jx 缓存误差必须传播或刷新。以A17报告的finalist RHS及一次/单元初始化作条件推算，总验证RHS减半所需理想跳过比例约59.28%，尚未计额外刷新和审计。

Candidate first 是首轮优先次序，不是架构预判。若强解析 shortlist 已消除缺口，停止 A 的 neural 投入；B 若有独立 label/latent/margin headroom，仍可单独继续。两者都无额外价值时停止 NN branch，不改做未经 full-material oracle 支持的 Flow。

---

<a id="a18-1"></a>

# A18 — NN-Native Maxwell Current Design：理论裁决

版本：2026-10-02。基准：A17 commit `56097653e6a814399581c3759fc3e3e8f0c2284c`。本文件中的 A17 数字来自固定版本报告与源码审查，不是本轮重新运行 Maxwell 的结果。外部文献编号见 [REFERENCES.md](#reference-registry)。

## 1. 首选路线与真正要学的东西

**Candidate Generator first。** 第一目标是从同一个合法 current dictionary 中提出十二个 incoming directions，提高进入昂贵端点计算之前的机会覆盖率，而不是改写已经很准确的 analytical anchor。学习目标是条件提案分布，不是完整材料反演，也不是精确标量 Q 回归。

理由不是 GNN 比 CNN 更先进，而是 A17 已提供了可测的瓶颈：full-dictionary anchor top-one 的对象汇总捕获率为 98.1–99.6%，短池覆盖却在 near-contact/asymmetric 为 35.3%/65.9%；六次 learned priority pilot 没有胜过 anchor。[A17-report, A17-NN]

源码还给出一个更严格的基线要求：`balanced_ids` 按 family 顺序轮流取未选条目的第一个、第二个……，它不是 O/P/M 物理预筛选。A18 必须先排除“廉价解析预筛选已经修好短池”的可能；只击败原十二候选规则不足以建立 NN 贡献。[A17-run]

**Feedback-Defect Compensator 是独立的第二机会，不由 A 是否成功决定。** 它更接近物理近似补偿范式，但要先确认可学习的向量标签、廉价可用输入、任务相关低秩结构及可用的误差门槛。两条分支不能共用一个模糊的“预测更准”指标。

## 2. 空间、坐标与符号合同

材料增量 d 属于固定物理度量下的实空间 R^p；所有照明共享同一个 d。电流与测量可为复数。P 次照明分别有 B_l、S_l，电流算子 L 及共享复电流基 U 相同。最终 GN 在实化测量空间求解，不得把复数数据实化误当成材料自由度翻倍。

对一个照明先写：

\[
J_F=SL^{-1}B,\qquad A_U=U^*LU,\qquad R_U=UA_U^{-1}U^*,\qquad J_U=SR_UB.
\]

U 满列秩，可取正交表示；A_U 可逆。L 通常不是 Hermitian/SPD，以下推导不以此为前提。所有多照明结论按 A17 的 `[Re(data_l); Im(data_l)]` 次序堆叠。[A17-core]

定义双侧残差映射：

\[
P_U^L=I-R_UL,\quad P_U^R=I-LR_U,
\quad O_U=SP_U^L,\quad I_U=P_U^RB.
\]

R_ULR_U=R_U，所以

\[
\boxed{L^{-1}-R_U=P_U^L L^{-1}P_U^R},
\qquad
\boxed{J_F-J_U=O_UL^{-1}I_U}.
\]

证明是直接展开右端：L^{-1}-R_U-R_U+R_ULR_U=L^{-1}-R_U。它是标准投影/残差代数，不作为 A18 首创。A18 的问题是：用什么廉价可观测量预测这个特定缺陷，并把精度转换成当前动作的接受可靠性。

## 3. current change 与 full defect 的区别

给定新增 bank C，相对当前核心 U：

\[
\bar C=P_U^L C,\quad Y_C=S\bar C,\quad
\Gamma_C=C^*L\bar C,\quad N_C=C^*I_U.
\]

则指定 Galerkin 增广造成的改变是

\[
J_{[U,C]}-J_U=Y_C\Gamma_C^{-1}N_C.
\]

右端不是一般情况下的 J_F-J_U。只有 C 完成全部遗漏电流空间、并且相应块可逆时，二者才相等。学习小 bank 内的 Γ^{-1}，也不能自动补偿 bank 外的反馈。

对 fixed-budget swap，必须使用共同核心 V=U\setminus\{a\}，不是把 incoming c 当作在完整 U 上的单纯 addition：

\[
J_{V+c}-J_{V+a}=Y_c\Gamma_c^{-1}N_c-Y_a\Gamma_a^{-1}N_a.
\]

候选器可以先按整个 U 构建廉价特征，但 teacher 和最终 anchor 必须按实际 drop/add 端点计算。A17 源码枚举所有 retained atom 的移除与 incoming 的组合。[A17-run]

## 4. 三个 physics axes 各自强制什么

**O：输出必须经过观测。** 原始 Sc 只表示直接辐射；Y_c 包含保留电流反馈。两者都可作特征，不可相互替代。

**P：传播是有向复耦合。** Γ 的非对角块包含反馈，原始 c_i^*Lc_j 与 c_j^*Lc_i 不必共轭相等。以无向、非负相似图替代全部复耦合，会丢失需要验证的物理信息。

**M：激励来自材料切线。** N_c 或 I_Ud 说明该方向是否被当前材料任务激发。大的 ||Y_c||、大的内部响应，不足以说明当前 residual 下的 signed utility 为正。

这三个轴决定信息接口与不变性，而不唯一决定 MLP、attention 还是 GNN。网络结构是一项待比较的建模选择。

特别注意：单个非零标量 Γ_cc 的通常条件数恒为 1。应记录归一化 |Γ_cc|、核心稳定性、块 Γ 的最小奇异值/条件数，而不是用“scalar condition number”伪装共振诊断。

## 5. 必须先扣除已知的廉价物理

设 A 是 A17 已有 independent richer anchor 的基，而不是要求 A 包含 U。恒等分解为

\[
\boxed{(J_F-J_U)d=(J_A-J_U)d+(J_F-J_A)d}.
\]

第一项已经能由两个廉价模型计算。优先学习第二项：

\[
\widehat z_F=J_A d+\mathcal T_\theta(\chi,A)[I_A d].
\]

这仍是用户要求的 reduced physics + defect compensation，只是不浪费网络容量重学已知的 inner correction。基于 U 的 direct defect 是必要消融，而不是默认最好目标。A 与 U 不嵌套时，不得把第一项直接写成单次 addition 的 Schur 公式。

若使用公共 workspace 基 Q，L_Q=Q^*LQ、S_Q=SQ、B_Q=Q^*B 只描述其张成空间。解 L_Q 不能当成 full Maxwell teacher。它可以提供更丰富的解析基线；是否值得支付该求解费用，要实测。[A17-state, A17-run]

## 6. unresolved feedback 的完整闭包包含不止 self-energy

对完整遗漏基 W，定义 Γ=W^*LP_U^LW、η=W^*I_Ud、Y=SP_U^LW。分块 W=[C,E]：

\[
\Gamma=\begin{bmatrix}\Gamma_{CC}&\Gamma_{CE}\\\Gamma_{EC}&\Gamma_{EE}\end{bmatrix},
\quad \eta=\binom{\eta_C}{\eta_E},\quad Y=[Y_C,Y_E].
\]

消去 E 得到

\[
a_C=(\Gamma_{CC}-\Sigma)^{-1}(\eta_C-b_E),
\quad \Sigma=\Gamma_{CE}\Gamma_{EE}^{-1}\Gamma_{EC},
\quad b_E=\Gamma_{CE}\Gamma_{EE}^{-1}\eta_E,
\]

\[
\boxed{\Delta z=(Y_C-Y_E\Gamma_{EE}^{-1}\Gamma_{EC})a_C
+Y_E\Gamma_{EE}^{-1}\eta_E}.
\]

因此只训练一个 Γ 的修正矩阵，而不处理 tail injection、dressed return 和 direct tail output，并不是完整 feedback-defect compensation。小 Γ 已知可精确求逆时，以 NN 替代该小求逆本身没有默认价值。

## 7. NN-native 的两个硬约束

### 7.1 复相位规范与集合顺序

对 C'=CE，其中 E 是对角单位复相位矩阵：

\[
Y'=YE,\quad\Gamma'=E^*\Gamma E,\quad N'=E^*N.
\]

Γ^{-1}Nd 是协变量，YΓ^{-1}Nd 是不变量。若节点复隐藏量 h_i 随 e^{-iφ_i} 变换，Γ_ij h_j 就具有正确的接收节点相位；以 Y_i h_i 做观测读出可以消除该任意相位。attention gate 只能依赖相位不变量，或用经过证明的 equivariant 操作。

必须测试随机列相位与字典重排后，proposal 对应物理 atom ID 不变。对固定 atom 删除菜单，任意混合整个字典会改变可行动作，因此不要求对这种改变仍保持同一个选点结果。保留空间的纯坐标换基与真实动作集变化也要分开。

### 7.2 对方向的线性

冻结 χ、L、U 后，d↦(J_F-J_U)d 是实线性映射。补偿器最好由 state network 产生一个低秩线性算子，再作用于 I_Ud；算子系数不能随当前 d 改变。必须满足零输入、加法和实数缩放测试。

proposal 网络不需要对 residual 或 d 线性，因为“哪个动作值得提出”本来是非线性条件决策。不要把这两个网络的约束混在一起。

## 8. 从输出误差到实际 gain margin

固定真实步差 d，u=r+J_Fx，z=J_Fd。定义

\[
Q(z)=-\operatorname{Re}\langle u,z\rangle-(\ell+\Lambda x)^Td
-\tfrac12(\|z\|^2+d^T\Lambda d).
\]

若 z=\hat z+e 且 u 精确，直接得到

\[
Q(z)-Q(\hat z)=-\operatorname{Re}\langle u+\hat z,e\rangle-\tfrac12\|e\|^2.
\]

若 ||e||≤ε，则

\[
|Q-\hat Q|\le\|u+\hat z\|\epsilon+\tfrac12\epsilon^2.
\]

令 m=\hat Q-τ>0，v=||u+\hat z||，可接受的输出半径为

\[
\boxed{\epsilon_{crit}=\frac{2m}{\sqrt{v^2+2m}+v}}.
\]

这给出明确的学习目标：不是所有方向都必须同样准确，而是选中的正收益方向必须达到其 margin 所要求的绝对精度。低整体 RMSE 或低 ensemble variance 不能代替这个条件。

u 也有误差时，若 ||u-\hat u||≤ε_u、||z-\hat z||≤ε_z，则

\[
|Q-\hat Q|\le\epsilon_u\|\hat z\|+\|\hat u+\hat z\|\epsilon_z
+\epsilon_u\epsilon_z+\tfrac12\epsilon_z^2.
\]

连续学习接受后，缓存 J_Fx 的误差会累积。下一次仅计算精确 J_Fd 不会清除已有缓存误差；需要累计界或付费刷新 J_Fx。这个问题是 Route C 的必要状态变量，而非事后校准细节。

统计预测半径、经验 ensemble σ 与合法确定性输出半径不同。A17 当前没有最后一种。[A17-decision] 在只有四个相关对象的 pilot 中，只报告经验选择风险，不宣称 95% OOD 安全。

## 9. 数据与可实现性边界

A17 有 47,712 条 priority 记录、43,108 条 feasible labels，但它们不是同数量的独立对象。主研究单元仍是四对象、十二个物理状态、三十六个共同 receiver 检查点。[A17-NN]

`run_exchange.py` 对 Gaussian 保存 `offline_information_reference.npz/fullJ`，可从已付费完整 J 生成任意合法材料方向的测量标签；voxel 保存的是 `V/fullJV` 十六方向投影，不能生成任意 3456 维 d 的完整标签。接受路径保存 step/Js；某些方向还可能由相邻已存端点恢复，必须逐条核实来源，不能按理论存在就宣布数据齐备。[A17-run]

`public_workspace.npz` 的 L/S/PB 是公共电流坐标映射。完整 I_Ud 不一定可由这些数组恢复。压缩 injection signature 可以合法做输入，但必须标为 projected；不能把它当 full physical injection residual。真实 current labels 更不能由 measurement-only fullJ 唯一恢复。

## 10. 可主张的新贡献与不能主张的内容

可能形成独立工作的贡献组合是：材料任务驱动的合法 current proposal、相位一致的有向 feedback 信息接口、预算内机会覆盖、由 finite-gain margin 定义的补偿精度及完整成本验证。每一项都需要对应消融或实验。

不是新贡献的内容包括：一般残差学习、普通 GNN、已知小矩阵求逆、泛化的 learned operator correction、FOM/ROM/ML fallback、把 NN 嵌入迭代反演。这些都有明确近邻。[R1–R11]

本轮做了 128 组随机复数非 Hermitian 有限维代数验证，验证双侧恒等式、完整补空间 Schur 表达、tail closure、相位不变性、anchor 分解、线性与 gain 误差界。最大相对误差约 5.4×10^-14；不是 Maxwell 场景、训练或性能实验。脚本及原始结果见 `verification/`。

**结论：先验证 A 的 proposal headroom，再运行小网络比较；B 独立保留，但必须先做 vector-label 与 margin oracle。不能为了保留 NN 论文，跳过强解析基线或把未知 voxel weak space 改称生成先验空间。**### 信息度量补充

A17 的数据尺度来自原 measured-data normalization，不是本轮重新估计的 calibrated noise covariance。[A17-state] 本文所有 output-error norm 沿用该既有尺度。Gaussian 的 weak dimension=0 也仅指 A17 保存材料表示、信息度量与指定阈值下的结果；它不证明连续 full-material problem 在任意噪声水平下全局可辨识。

---

<a id="a18-2"></a>

# A18 — Mature Paper Audit

文献身份、链接和逐篇访问层级见 [REFERENCES.md](#reference-registry)。本审查把“论文实际支持的内容”和“A18 的比较/推论”分开。关键三篇尚未取得全文：far-field learning、2025 PEC EGAT、2019 physics-inspired current CNN。它们的 abstract 足以识别范式重合，但不支持断言其所有层、训练协议或理论边界；这些项目明确保留 OPEN。

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

---

<a id="a18-3"></a>

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

---

<a id="a18-4"></a>

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

---

<a id="a18-5"></a>

# A18 — Architecture Comparison and Final Route Decision

## 1. 最终二选一

# Candidate Generator first

选择依据是现有证据与第一轮可辨别性，不是先认定 GNN 最强。

|维度|Candidate Generator|Feedback-Defect Compensator|
|---|---|---|
|A17 已证实的问题|near-contact/asymmetric 短池机会遗漏|完整 Jd 检验有额外费用|
|明确的可学对象|给定 U/state/task 的 incoming shortlist|给定 frozen physics 的线性 response defect|
|已有监督|全字典 feasible scalar labels 已存在|Gaussian fullJ 已存；voxel 向量覆盖不完整|
|第一轮安全性|最终继续用原 full Jd 验收|若真跳过 Jd，需要新 gate 与缓存误差管理|
|首要非学习对手|O/P/M analytic shortlist，不是旧 prefix pool|richer anchor、解析小 ROM、POD+ridge|
|本轮最便宜可做的否证|解析 shortlist 是否已消除瓶颈|latent oracle 是否能达到动作 margin|
|与 far-field compensation 的类比|间接|最直接|
|建议|先核验并做小型 proposal pilot|独立做 label/latent/margin oracle，暂不抢先训练|

Candidate first 不等于 Candidate guaranteed to work。旧十二池只是 family-prefix round-robin；若无学习的 O/P/M 预筛选已经覆盖足够多机会，就应停止 A 的 neural 主线。[A17-run]

## 2. 先做的模型不是一张巨大的网络图

第一比较是 shared MLP、Set Transformer、普通 GAT、physics-edge GNN。输入取得成本相同，输出均为原始 current atom 的 proposal logits，十二 shortlist 大小相同。

MLP 是不可省略基线。Set Transformer 检验是否只要集合上下文；GAT 检验是否只要图结构；physics-edge GNN 检验有向复反馈和 material-source/receiver readout 是否确实额外有效。

若 physic-edge 模型没有超过同特征 MLP，就没有证据要求 message passing。若只在增加 expensive endpoint 特征后超过 MLP，就没有证明 NN 原生结构节省了原来的工作。

## 3. Proposed phase-consistent proposal block

```text
Cached legal current dictionary + selected atom IDs
       |
       +-- N / injection sketches ------ source channels
       +-- directed retained feedback -- Gamma-like edge channels
       +-- S, Y ------------------------ observation channels
       +-- residual, prior/LM, costs ---- invariant task context
       |
Invariant MLP / set model / phase-equivariant graph block
       |
12 original incoming IDs + validity mask + optional confidence
       |
ALL legal removal cores x proposed incoming directions
       |
Exact reduced endpoint steps d + unchanged A17 anchor
       |
At most 2 finalists / round -> original high-fidelity Jd acceptance
```

该结构保留物理生成动作、网络只做低成本提案的能力分离。loss 不赋予网络更改 Q、Λ、full-reference 或真实 data residual 的能力。confidence 第一阶段仅用于安排备选候选，不作为省略物理检查的授权。

## 4. Feedback block 的更合适的共享位置

```text
Frozen state + fixed richer anchor A
       |
State network creates a small, source-linear defect operator
       |
For each actual material d:
   analytic J_A d + learned outer-tail response
       |
Analytical finite-gain evaluator + output-error indicator
       |
shadow diagnostic first; later calibrated gate / exact fallback
```

A17 在一次 frozen case 内固定 richer anchor；因此 T_A=J_F−J_A 与当前 U、drop/add ID、残差值无关。它只作用于不同的 d。对同一 material state，跨 k 和多次候选使用同一个 state-conditioned operator 可以摊销构建费用，并避免网络把无关 candidate ID 记成 correction 规律。

对于用户定义的 z_U，最后再加已知的 `(J_A-J_U)d`。这不是改变目标，而是把可计算的 inner correction 从学习目标里拿掉。

operator 本身不应依赖 residual 来改变物理映射；residual 可以用于决定精度门槛、样本权重、是否 fallback。若采用 task-specific nonlinear correction，应明确它只是局部预测器，不再声称是可复用的线性 tangent operator。

## 5. Route C 如何真正耦合，而不只是并排两个 NN

1. Proposal 改变进入 endpoint evaluator 的候选集合，因而改变实际 d 与 B 网络遇到的 query 分布。
2. Anchor 仍决定 finalist，防止 proposal confidence 直接变成 physical utility。
3. Compensator 只修正选中 d 的 full directional response；它不允许反向挑选一个靠模型误差获得高 Q 的新 d。
4. Uncertainty gate 使用实际 gain margin 与当前 cached-output error，决定接受、拒绝、exact check 或刷新。
5. 接受以后 U/x/cache 更新；下一次 proposal 使用更新的 retained core，B 则复用固定 anchor-tail operator，重新编码新的 d。

因此 C 的耦合发生在 **物理动作分布、完整目标验收及状态更新**，不是任意两个 latent 拼接。

不得端到端联合训练一个最大化 surrogate Q 的 A+B，再用同一 surrogate 当验收者；proposal 可能学会利用 compensator 的偏差。先分支独立训练、冻结 A 与 anchor，再按 selected-finalist 分布校准 B。改变 A、pool size 或 B 后，原有校准结果不自动适用。

## 6. 最小软件能力接口

```python
# Specification; not an implementation or permission to train.
ProposalContext = {
    "state_public": ..., "current_atom_ids": ...,
    "cheap_nodes": ..., "cheap_edges": ..., "feasibility_mask": ...
}
ProposalOutput = {"incoming_ids": ..., "proposal_logits": ...}

DefectContext = {
    "frozen_state_public": ..., "fixed_anchor": ...,
    "linear_source_sketch": ..., "output_decoder": ...
}
DefectOutput = {
    "delta_z": ..., "error_indicator": ...,
    "indicator_type": "EMPIRICAL_NOT_CERTIFIED"
}
```

完整 J、full optimal step、Q_true、truth、full-field teacher vectors 不应出现在上述 context。Evaluator 是不同对象；canary 测试故意替换 forbidden fields，要求预测不变。Source hash 用于审计，不作为网络输入。

## 7. 两个分支如何公平评价

A 主要问：同十二候选预算下，held-out object 的 opportunity coverage 是否更高，且 features+proposal+required endpoints 是否值得支付。

B 主要问：固定同一 candidate/finalist/d 集，能否在相同收益保留和错误接受标准下显著少算完整方向 actions。

C 最后比较四格：无学习、仅 A、仅 B、A+B。相同 current dimension、max three exchanges、max two finalists、full objective、tolerance、outer state 和 data。不能用 A+B 的新候选去比较 B-only 的旧候选后，将所有收益归因 B。

Full nonlinear reconstruction 不在本轮合同内。单步 full-GN gap 好转不是最终材料图像改进；不增加 truth labels 到 student 后再称为 task-only learning。[A17-decision]

## 8. 第一轮数据量与计算上限

**第一轮 Codex：audit-only。** 四个既有对象、十二个保存 material states、三十六个共同 receiver cells、已有 labels，零新增 full-wave RHS、零训练 epoch、零新 trajectory。可以重算公共小矩阵、单步标签 replay、CPU baseline、已有向量的 SVD/ridge oracle。

这一步不是形式：它同时决定 A 的强解析基线、B 的标签可用性与 latent margin headroom。没有这些结果，直接训练四种网络只是在重复 A17 的旧 priority pilot。

之后允许的 tiny training 上限在 EXPERIMENT_SPEC 中预注册：最多四类 A 架构×四 object folds×两 seeds，单模型≤200k参数、≤60 epochs、总训练≤2 GPU hours、峰值设备占用≤3 GiB。B training 需要独立 gate，不默认与 A 一起启动。

现有 43,108 feasible rows 可以支持一次受限学习可行性诊断，但只有四个 object domains。不存在由这些行数推出的普适“足够训练”保证；论文级跨几何/对比度/噪声结论需要未来独立样本，不能拿 phase/k/随机增广当新对象。

## 9. 失败后如何裁决

- **A 输强解析 shortlist**：停止 A 的 NN 主线，保留解析改进；不认为 current design 机制失败。
- **A 有 teacher gap、MLP 能学、physics-edge 无增益**：可保留轻量工程模型，但停止强 NN-native architecture claim。
- **B 的 rank/source oracle 不达 margin**：停止 B，先不追加网络规模；说明是表示/信息精度瓶颈。
- **B 精度好但全部成本不省**：停止加速主张；不把低 inference latency 充当全费用胜利。
- **A 失败而 B 独立 oracle 通过**：可转向 B，不强行杀死整个 NN branch。
- **A 已无非学习以外的机会，B 也无精度—费用优势**：停止本轮整个 NN branch。保留 A17 mechanism/design paper，不改投 full-material Flow 作为逃生路线。

**最终立场：A 先做，是因为证据链最短、监督最齐、不会先降低物理验收可靠性；B 的机制更像真正的 approximation-defect learning，但目前必须先通过更严格的数据与 margin gate。**

---

<a id="a18-6"></a>

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

---

<a id="a18-7"></a>

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

---

<a id="reference-registry"></a>

# A18 sources and access registry

Access date: 2026-10-02. A17 repository reads are pinned to commit `56097653e6a814399581c3759fc3e3e8f0c2284c`. Reports and source code were inspected through the GitHub connector. This package does not claim a raw-array reanalysis of the full A17 archive. No copyrighted paper PDFs are redistributed.

## Project sources

Root: https://github.com/migodam/a17-maxwell-certified-exchange/tree/56097653e6a814399581c3759fc3e3e8f0c2284c

- **[A17-start]** `START_HERE.md`: measured scope, limitations, read order, external kernels.
- **[A17-decision]** `A17_DECISION.md`: local design GO, score/coverage separation, learned-priority NO-GO, certificates unavailable.
- **[A17-report]** `A17_FINAL_RESEARCH_REPORT_ZH.md`: object-wise ratios, short-pool coverage, illumination count, material dimensions, 1164 finalist RHS, weak-space scope.
- **[A17-NN]** `A17_NN_PILOT.md`: 47712 rows/43108 feasible labels, object split, six runs, cost exclusions.
- **[A17-cost]** `A17_COST_AUDIT.md`: matched costs, initialization/setup distinctions, no demonstrated net acceleration.
- **[A17-core]** `A17_R1/code/exchange_core.py`: realification, Galerkin/Schur formulas, legal stability checks, finite directional gain.
- **[A17-run]** `A17_R1/code/run_exchange.py`: `balanced_ids`, `moves`, `evaluate_neighborhood`, `policy_run`, and `run`; scalar label storage, accepted endpoint arrays, fullJ versus projected fullJV storage.
- **[A17-state]** `A17_R1/code/maxwell_state.py`: `public_workspace`, `EvaluatorContext.offline_gaussian_J`, `ProbeContext`, `TeacherContext`; separation of projected public maps and full actions.

To form an exact file URL, append the listed path to `https://github.com/migodam/a17-maxwell-certified-exchange/blob/56097653e6a814399581c3759fc3e3e8f0c2284c/`.

## Primary research and method sources

**[R1]** Tiantian Yin, Kai Tan, Xudong Chen. *Far-Field Approximation Learning Method for Millimeter-Wave Short-Range Imaging*. IEEE TAP, 2023. DOI: https://doi.org/10.1109/TAP.2023.3242424
Publication abstract: https://www.researchgate.net/publication/368409180_Far-Field_Approximation_Learning_Method_for_Millimeter-Wave_Short-Range_Imaging
Access: paper abstract retrieved; full text not obtained. Claims restricted to Fourier-based approximate input, correction of approximation/limited-aperture distortions, and reflectivity output. Exact layer/loss/training-size audit remains OPEN.

**[R2]** Yu Liu et al. *SOM-Net: Unrolling the Subspace-based Optimization for Solving Full-wave Inverse Scattering Problems*. IEEE TGRS, DOI 10.1109/TGRS.2022.3217162. Author preprint https://arxiv.org/abs/2209.03567 ; full PDF https://arxiv.org/pdf/2209.03567 . 2022.
Access: full PDF text, especially Algorithm 1 and equations (6)–(11), inspected. The method alternates learned current updates and explicit material updates. PDF screenshot service failed, so no claim of visual figure/table QA is made. The audit does not quote experimental tables.

**[R3]** Rui Guo, Zhichao Lin, Tao Shan, Xiaoqian Song, Maokun Li, Fan Yang, Shenheng Xu, Aria Abubakar. *Physics Embedded Deep Neural Network for Solving Full-Wave Inverse Scattering Problems*. IEEE TAP 70(8), 6148–6159, 2022; early online 2021. DOI: https://doi.org/10.1109/TAP.2021.3102135
Author-uploaded full text: https://www.researchgate.net/publication/353782924_Physics_Embedded_Deep_Neural_Network_for_Solving_Full-wave_Inverse_Scattering_Problems
Access: author-uploaded full-text extraction, including VIE/FWD solver definitions and equations (11)–(16), inspected. Direct PDF download failed; no visual architecture-figure QA claimed. Do not confuse it with the related VIE-forward-only paper, DOI 10.1109/TAP.2021.3070152.

**[R4]** Huilin Cao, Haoran Sun, Jingwei Liang, Tao Shan, Yuxiang Ren, Maokun Li. *Physics-Data Hybrid Driven Edge-Featured Graph Attention Networks for Surface Currents Learning of 3D PEC Targets*. IEEE TAP, 2025. DOI: https://doi.org/10.1109/TAP.2025.3602467
Author publication list: https://yuxiangren.github.io/publications/
IEEE record: https://ieeexplore.ieee.org/document/11146424/
Publication abstract record: https://www.researchgate.net/publication/395174268_Physics-Data_Hybrid_Driven_Edge-Featured_Graph_Attention_Networks_for_Surface_Currents_Learning_of_3D_PEC_Targets
Access: title/authors/year and abstract-level method scope; full text unavailable in this session. Layer counts, discretization details, exact edge formula, split and speedup denominator remain OPEN. Do not reproduce reported speed percentages without that audit.

**[R5]** Zhun Wei, Xudong Chen. *Physics-Inspired Convolutional Neural Network for Solving Full-Wave Inverse Scattering Problems*. IEEE TAP 67(9), 6138–6148, 2019. DOI: https://doi.org/10.1109/TAP.2019.2922779
Publication abstract: https://www.researchgate.net/publication/333889082_Physics-Inspired_Convolutional_Neural_Network_for_Solving_Full-Wave_Inverse_Scattering_Problems
Access: primary paper abstract reproduced at publication record; full text not obtained. Abstract supports induced-current learning, bypass of major current and learning the smaller correction, and multi-label loss. No deeper layer/formula assertions.

**[R6]** Lianlin Li et al. *DeepNIS: Deep Neural Network for Nonlinear Electromagnetic Inverse Scattering*. IEEE TAP, 2019; author preprint 2018. https://arxiv.org/abs/1810.03990 ; https://arxiv.org/pdf/1810.03990
Access: full PDF text, BP-image input and cascaded complex CNN method paragraphs inspected. Screenshot service failed; numerical tables not used.

**[R7]** Sebastian Lunz, Andreas Hauptmann, Tanja Tarvainen, Carola-Bibiane Schonlieb, Simon Arridge. *On Learned Operator Correction in Inverse Problems*. SIAM J. Imaging Sciences 14(1), 92–127, 2021. DOI: https://doi.org/10.1137/20M1338460 ; author preprint https://arxiv.org/abs/2005.07069
Access: full-text sections 3–4 on correction and adjoint consistency inspected. Screenshot service failed. No empirical performance figures used.

**[R8]** Thomas O'Leary-Roseberry, Peng Chen, Umberto Villa, Omar Ghattas. *Derivative-Informed Neural Operator: An Efficient Framework for High-Dimensional Parametric Derivative Learning*. JCP 496, 112555, 2024. DOI: https://doi.org/10.1016/j.jcp.2023.112555 ; https://arxiv.org/abs/2206.10745 ; https://tomoleary.github.io/dino/
Access: primary abstract/project method description, not an equation-by-equation full-text audit.

**[R9]** Dingcheng Luo, Thomas O'Leary-Roseberry, Peng Chen, Omar Ghattas. *Efficient PDE-Constrained Optimization Under High-Dimensional Uncertainty Using Derivative-Informed Neural Operators*. SIAM J. Scientific Computing 47(4), C899–C931, 2025. DOI: https://doi.org/10.1137/23M157956X
Access: publisher abstract/method scope; no detailed empirical reproduction.

**[R10]** Bernard Haasdonk, Hendrik Kleikamp, Mario Ohlberger, Felix Schindler, Tizian Wenzel. *A new certified hierarchical and adaptive RB-ML-ROM surrogate model for parametrized PDEs*. Author preprint 2022. https://arxiv.org/abs/2204.13454
Access: primary abstract and authors' conference description. Its rigorous a posteriori hierarchy is a prior-art boundary, not evidence that A18 inherits a certificate.

**[R11]** *Closure Learning for Nonlinear Model Reduction Using Deep Residual Neural Network*. Fluids 5(1), 39, 2020. DOI: https://doi.org/10.3390/fluids5010039
Access: primary journal abstract/method summary. Resolved/unresolved closure is established prior art.

**[R12]** Juho Lee et al. *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks*. ICML 2019. https://proceedings.mlr.press/v97/lee19d.html
Access: primary paper landing page. A candidate proposal requires equivariant per-element outputs, not only an invariant pooled output.

**[R13]** Petar Velickovic et al. *Graph Attention Networks*. ICLR 2018. https://arxiv.org/abs/1710.10903
Access: primary abstract. Generic attention alone does not encode Maxwell phase or Schur coupling.

**[R14]** Anastasios N. Angelopoulos, Stephen Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. https://arxiv.org/abs/2107.07511
Access: primary method reference. The finite-order-statistic calculation in A18 is a direct mathematical application, not a new theorem or a demonstrated A18 guarantee.

**[R15]** Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, Tal Schuster. *Conformal Risk Control*. ICLR 2024. https://research.google/pubs/conformal-risk-control/
Access: authors' primary abstract. Expected monotone-loss risk control is not automatically conditional wrong-acceptance control after an adaptive shortlist.

## Evidence classification

- PROJECT_REPORTED: metrics directly stated by pinned A17 reports.
- SOURCE_VERIFIED: implementation/data-saving behavior directly visible in pinned source.
- DERIVED: algebra or cost calculation in this package.
- PROPOSED: pilot sizes, thresholds and architecture options; not achieved results.
- OPEN: inaccessible full-text detail or uninspected archive arrays.

A source access failure is not evidence that a paper lacks a feature. Novelty judgments below are provisional where full text is OPEN. No first-ever claim is made.
**[R16]** Lianghao Cao, Thomas O'Leary-Roseberry, Prashant K. Jha, J. Tinsley Oden, Omar Ghattas. *Residual-based error correction for neural operator accelerated infinite-dimensional Bayesian inverse problems*. JCP 486, 112104, 2023. https://arxiv.org/abs/2210.03008 ; https://doi.org/10.1016/j.jcp.2023.112104 . Access: primary abstract and journal method overview. A residual-based linear variational correction and its accuracy conditions are established prior art; no A18 performance inference is made.
