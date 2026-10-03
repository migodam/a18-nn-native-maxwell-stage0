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