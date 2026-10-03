# A18 NN Native Stage 0：完成记录与独立分支裁决

Stage 0 已完成；A=`A_GO_TINY_PILOT`（仅建议下一阶段），B 全范围=`B_DATA_INCOMPLETE`。有完整标签的 Gaussian primary test 对当前 rank8/16/32 POD 表示判 `B_NO_MARGIN_OR_COST_HEADROOM_FOR_TESTED_POD_BASES`。Candidate Generator first 仍是较合理的下一阶段顺序。没有进入训练或补标签。

## 来源与本轮边界

固定 commit `56097653e6a814399581c3759fc3e3e8f0c2284c`。36 个 canonical receiver queries，四对象×三 frozen states×三 k，全部 dictionary 覆盖；47,712 个唯一 move、43,108 个可行 move、5,208 个 incoming utility 标签，duplicate=0。公共 manifest 的输出 hash 与原始 manifest/source-epoch 的 hash 分别核验；36 original teacher hashes 另对 NN dataset manifest 核验。公开 JSON 元数据整理可能改变字节哈希，不能要求它等于 raw teacher 字节哈希。没有 source/scope mismatch。

新增 full-wave RHS=0，NN epochs=0，新增 nonlinear trajectories=0，GPU training=0。所有计算均为单 CPU worker；实际进程/失败费用见 PROCESS_RESOURCE_RECEIPTS。DeepSeek profiles 未配置，实际 routing 为主线程直接实现，不更改 provider/model。

公共接口只使用投影 L/S/PB、dictionary、公共 receiver GN step 与 r/ell/lambda；完整 Jacobian、Q_true、truth、reference step 均不进入 proposal。原始 atom IDs、共享实材料 gauge、六 illumination、data packing 和1e-10稳定性门槛保留。公共 Q-space inverse 不是真实 full L inverse。

## A：实际结果

| Method | smooth | near-contact | layered | asymmetric | Equal mean |
|---|---:|---:|---:|---:|---:|
| B0 | 0.9689 | 0.3528 | 0.9198 | 0.6589 | 0.7251 |
| B1 seed20261002 | 0.9317 | 0.5809 | 0.9010 | 0.7647 | 0.7946 |
| B1 seed20261003 | 0.9658 | 0.8326 | 0.3139 | 0.8541 | 0.7416 |
| B1 seed20261004 | 0.9760 | 0.5426 | 0.9379 | 0.9287 | 0.8463 |
| B2 | 0.9718 | 0.5178 | 0.9852 | 0.6751 | 0.7875 |
| B3 | 0.7283 | 0.4142 | 0.7644 | 0.5969 | 0.6259 |
| B4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| B5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |


B0 恢复 A17 的已报瓶颈。B4 的 anchor top1 capture 分别为0.9960/0.9810/0.9937/0.9883，确认困难主要在 incoming pool。B2 提高 near-contact 的覆盖，却仍低于0.70；asymmetric0.6751仍低于0.85；平均0.7875低于0.85。B3 固定 receiver-step addition proxy 的符号信息并未带来有效 one-swap material utility，平均0.6259的失败保留。三随机 seeds 全报，不能逐对象或逐 query 事后选 seed。

最强单个已冻结廉价比较对象的描述性 mean 为0.8463，离 one-swap teacher ceiling 尚有0.1537；这是机会空间，不是 learned success。下一阶段须同时满足 near-contact>=0.70、asymmetric>=0.85、mean>=0.85，并超过最佳廉价基线0.05或在质量差<=0.01时成本减半。没有本轮 NN 实测值。所有 Cov12、top1/top4 recall、gain regret 和零正收益统计在 object summary 和 query CSV 中；36 queries 的 no-positive 数为0，no-op作为合法选项保留。

F1 joint components 的 warm median=4.60ms/query，max=5.67ms；每项单独计费，child GN/Qhat 只在 shortlist 后。此时间仅是已缓存投影数据上的计算，不是 cold full-wave workflow 或加速实测。

## B：标签、表示与实际 margin

合法 measurement vectors：smooth101、near-contact103、layered0、asymmetric1。194 条 voxel matched vectors 缺失；V/fullJV 的非支撑方向投影残差为0.9948-0.9992；全 saved-step/Js span 审计也只有 asymmetric late k8 的 receiver 可合法恢复。该孤立向量是 inventory 发现，不能构成 voxel object-level oracle。full-current / unresolved-current labels=0。

扣除已知 inner correction 后，tail/direct norm 的中位数为0.3729/0.4356。Gaussian source sketch 在 rank8/16/32（每 illumination complex rows）均保留保存的54维 real material gauge，state-specific nullspace error最大4.47e-15 relative；source 信息诊断通过，但 shared state-blind ridge 不等于可泛化的物理补偿器。

严格遵守四折：Fold1 以 smooth 的101方向拟合，对 near-contact 的103方向做 test；Fold3 以 near-contact 拟合，smooth 仅为 validation。Fold0 的指定 train objects 没有完整 Gaussian 标签，不把 validation 改作 train。所有 POD/RMS/ridge 参数仅来自合法 train 对象，固定 ridge alpha，不用 test 选参数。

在 primary test 的 B4 matched18个 finalists 中：direct-defect POD rank8/16/32 的 skip 为0/18；anchor-tail 为10/18=55.6%；uncorrected anchor也是10/18。rank8 ridge10/18，rank16/32 ridge9/18。B2 的 rank32 tail projection 为7/18=38.9%。当前输出 basis 对未见对象的 tail orientation 泛化不足；不能通过放大 NN 自动闭合这一瓶颈。这是所测 projection/ridge/bases 的负结果，未证明任何 state-conditioned compensation 都不可能。

## 成本、误差与判定范围

对保存 reference 的精确评价侧 u，计算 margin m、v=||u+zhat|| 和 epsilon_crit=2m/(sqrt(v²+2m)+v)。oracle_can_skip 使用实际 full-response residual，仅是反事实可行性；在线没有这种精确误差上界，A17 high-fidelity label 也不是物理确定性 certificate。

历史 A17 1164 finalist RHS、216 initialization RHS 对应理想 skip floor59.28%。本次 single-decision test 的54初始化RHS和108finalistRHS，对应75%；55.6%只能给理想总 RHS ratio0.6296，而非减半。refresh、audit、失败、source构造、模型与 cold setup 成本只会提高所需 skip。连续接受时的 Jx cache error 需要累积 radius/refresh，下一次 exact Jd 并不自动修复旧缓存。

## 验证与保留的失败

128 个新写非 Hermitian complex algebra cases PASS，identity残差最大9.69e-16；双侧defect、Schur tail、anchor decomposition、phase covariance、physical-ID permutation和gain-error bound全部检查。四个对象的公共输入 canary、phase/permutation action consistency以及线性接口检查 PASS。ridge预测路径与线性检查单独输出。没有 NN implementation，因此不宣称架构已通过网络测试。

原始 A18 verify_a18.py、config、results 和独立 REFERENCES 原件缺失：REFERENCES 从合并文稿恢复，新写 equivalent tests，不冒称复现原版128PASS/5.4e-14。错误 tester 的 SVD gauge、两处 parser、adapter/JSON serialization 失败全部保留。最终16个 upstream Woodbury units 的残差失败，经同一 normal equation 的一次固定 residual correction 降到原1e-10门槛以下；vendor未改、无jitter、无新物理算子，失败和修复费用入账。

## 下一阶段所需最少证据与停止点

A：一个按固定四折和两seed进行的 bounded candidate pilot，公共输入预算相同，与全部随机seed和B2比较；首先兑现 utility coverage，再谈 native architecture 额外价值。配置已生成，但 enabled=false。

B：若继续，先补足 matched voxel vector/exact receiver-output provenance，并提出能在 train-only basis 下改变当前 margin ceiling 的 state-conditioned linear target representation。当前 Gaussian oracle 没有足够额外 margin/cost headroom，不建议直接训练 B。缺失数据与这个 scoped负结果独立记录。

实际已运行：source/scope/hash audit、完整 dictionary reuse、B0-B5、selected-direction reduced endpoints、合法向量标签盘点、POD、固定ridge、full-gauge source oracle、margin/cost诊断、结构测试、图表和三份文稿。未运行：NN training、new full-wave labels、nonlinear reconstruction、95% calibration、真实在线跳过和整流程加速、远端任务或发布。Stage0交付后停止。

## 最终交付封存

三份正式TeX/PDF为8/6/14页，原生compiler成功、28页逐页查看；独立目录复现的全部科学数值一致。最终source/PDF/render、来源与交付文件hash见qa、verification和DELIVERY_MANIFEST。

Lambda的文字定义已统一：原lambda_total包含prior与LM damping，数值实现始终使用保存值。全public manifest5556项、42个raw/pinned module hashes、36份raw teacher核验通过，597个consumed inputs保持原hash。

费用主账按进程envelope加总，nested component/replay只作为视图；累计数值CPU49.113秒，所有已计时进程CPU80.294秒，失败进程CPU19.095秒，峰值RSS884965376bytes。Native UI compiler CPU未在shell仪表中。新增物理actions与NN训练均为0；没有把离线warm计时当作实测加速。

Status为COMPLETE_STAGE0；本轮到此停止，后续配置enabled=false。
