# A18 — NN-Native Maxwell Current Design

## 裁决

**Candidate Generator first。** 网络首先学习“哪些 incoming current directions 应进入十二候选 shortlist”，不重算已准确的 analytical anchor，也不直接学习材料反演。先用强解析 O/P/M 预筛选判断是否真的还需要学习。

Feedback-Defect Compensator 作为独立次线保留。其首要学习目标建议从 `(J_F-J_U)d` 收缩到扣除了已知 inner correction 的 **`(J_F-J_anchor)d`**，并用 actual finite-gain margin 而非全局 RMSE 决定精度是否够。

## 包含的七份交付

|文件|内容|
|---|---|
|[A18_NN_NATIVE_THEORY.md](A18_NN_NATIVE_THEORY.md)|双侧 defect、Schur tail、相位/线性约束、margin 精度公式、路线裁决|
|[A18_MATURE_PAPER_AUDIT.md](A18_MATURE_PAPER_AUDIT.md)|逐篇比较必读论文及 operator/ROM/uncertainty 近邻，列明全文访问边界|
|[A18_CANDIDATE_GENERATOR.md](A18_CANDIDATE_GENERATOR.md)|teacher、O/P/M node/edge、四架构、非学习对手、coverage 指标与门槛|
|[A18_FEEDBACK_COMPENSATOR.md](A18_FEEDBACK_COMPENSATOR.md)|三种输出空间、标签可用性、线性物理 latent、误差门、缓存与费用|
|[A18_ARCHITECTURE_COMPARISON.md](A18_ARCHITECTURE_COMPARISON.md)|Candidate first 原因、组合架构、能力隔离、分支失败后的停止规则|
|[A18_LIGHTWEIGHT_EXPERIMENT_SPEC.md](A18_LIGHTWEIGHT_EXPERIMENT_SPEC.md)|object split、零训练 Stage0、后续 tiny pilot、计费与完整数据合同|
|[A18_CODEX_PROMPT.md](A18_CODEX_PROMPT.md)|可直接交给 Codex 的自包含 audit-only 第一轮指令|

另有 [REFERENCES.md](REFERENCES.md)、`stage0_config.json`、`verification/verify_a18.py`、`verification/results.json`、文件校验与 SHA-256 manifest。ZIP 还包含合并阅读版 `A18_COMBINED.md`。

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