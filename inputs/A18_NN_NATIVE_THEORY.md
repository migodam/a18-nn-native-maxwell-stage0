# A18 — NN-Native Maxwell Current Design：理论裁决

版本：2026-10-02。基准：A17 commit `56097653e6a814399581c3759fc3e3e8f0c2284c`。本文件中的 A17 数字来自固定版本报告与源码审查，不是本轮重新运行 Maxwell 的结果。外部文献编号见 [REFERENCES.md](REFERENCES.md)。

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