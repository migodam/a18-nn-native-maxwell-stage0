# A18 Stage 0 — web reading bundle

Read [START_HERE.md](START_HERE.md) for scope and navigation. This combines decision, common ledgers and extracted text from all three final PDFs. PDF/TeX equations and tables are authoritative when text extraction loses formatting. No new scientific result is introduced here.

## Branch decision

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


## Claim ledger

```csv
claim_id,status,claim,evidence,scope_limit
C01,VERIFIED_DATA,"36 complete queries, 47712 unique moves, 43108 feasible moves, 5208 incoming labels",results/COUNTS.json,"Four objects, twelve frozen states; moves are correlated"
C02,REANALYZED,B0 Cov12=0.9689/0.3528/0.9198/0.6589; original short-pool bottleneck reproduced,results/A18_BASELINE_OBJECT_SUMMARY.csv,One-swap dictionary scope
C03,REANALYZED,B4 anchor top1 capture=0.9960/0.9810/0.9937/0.9883,results/A18_BASELINE_OBJECT_SUMMARY.csv,Ranking quality distinct from pool coverage
C04,REANALYZED,"B2 Cov12=0.9718/0.5178/0.9852/0.6751, mean0.7875",results/A18_BASELINE_OBJECT_SUMMARY.csv,Fails simultaneous opportunity-elimination gates
C05,NEGATIVE_RESULT,"B3 mean Cov12=0.6259, below B0 and B2",results/A18_BASELINE_OBJECT_SUMMARY.csv,Fixed receiver-direction addition proxy; no retuning
C06,OPPORTUNITY_NOT_SUCCESS,Best single frozen cheap baseline mean0.8463; teacher-relative gap0.1537,results/STAGE0_SUMMARY.json,Best random seed is descriptive; no object-wise or query-wise seed oracle
C07,VERIFIED_LABEL_SCOPE,204 Gaussian measurement vectors plus 1 voxel vector; zero full-current teachers,A18_DEFECT_LABEL_AVAILABILITY.csv,194 voxel matched vectors missing; scalar gains are not vector labels
C08,REANALYZED,Median tail/direct norm0.3729 smooth and0.4356 near-contact,figures/04_tail_norms.csv,A and U need not be nested; not a monotonic theorem
C09,EVALUATION_ORACLE,Source sketch rank54 on saved54-dimensional Gaussian gauge; null-tail error<=4.47e-15 relative,results/A18_SOURCE_ORACLE.csv,"State-specific fullJ oracle, not a learned transfer result or full-current label"
C10,SPLIT_AUDIT,"Fold1 train smooth, test near-contact; reversed smooth results are Fold3 validation",verification/FIT_RECEIPTS.json,Fold0 lacks complete Gaussian training object; no four-object B test
C11,NEGATIVE_RESULT,B4 held-out POD anchor-tail skip10/18 for ranks8/16/32; uncorrected anchor also10/18; direct defect0/18,results/A18_B_OBJECT_SUMMARY.csv,Projection oracle under tested bases; not a proof excluding all compensation architectures
C12,DERIVED_COST,Matched single-decision RHS-halving threshold75%; historical multi-round floor59.28%,results/A18_B_OBJECT_SUMMARY.csv,"Excludes refresh, audit and model overhead; no measured acceleration"
C13,DECISION,A_GO_TINY_PILOT; B_DATA_INCOMPLETE globally; Gaussian primary tested bases have no margin/cost headroom,A18_STAGE0_DECISION.md,Recommendations only; training remains disabled
C14,IMPLEMENTATION_VERIFIED,"128 fresh non-Hermitian algebra cases PASS, largest identity residual9.69e-16",verification/algebra_results.json,Equivalent newly written tests; original A18 verification source unavailable
C15,NUMERICAL_REPAIR,16 upstream Woodbury failed units recovered with one residual correction and unchanged1e-10 gate,FAILURES.json,"No jitter, new physical operator, PCG or speed comparison"
C16,EXECUTION_VERIFIED,"New full-wave RHS0, NN epochs0, new nonlinear trajectories0",results/PROCESS_RESOURCE_RECEIPTS.jsonl,Single computation child; failed attempts retained
C17,NOVELTY_OPEN,"Minor-current learning, operator correction and derivative compression have prior art",literature/REFERENCE_LEDGER.csv,Search rate limited; no first-ever or global novelty claim
C18,SOURCE_VERIFIED,Step and frozen gain use saved lambda_total including prior precision plus LM damping; smooth early example0.01001,INPUT_HASHES.json; pinned maxwell_state.py and setup_cost.json,Preserved damped frozen quadratic; not a new undamped or nonlinear objective
```

## Failure ledger

```csv
failure_id,classification,failure,resolution,evidence,status
IMPLEMENTATION_1,TEST_GAUGE_ERROR_NOT_OPERATOR_COUNTEREXAMPLE,phase covariance tester applied E to input columns instead of the SVD basis returned by bank_increment,apply E to returned inc[Z]; compare the same gauge,verification/algebra_results_attempt01_FAIL.json,RECORDED_AND_RESOLVED
IMPLEMENTATION_2,IMPLEMENTATION_ERROR,IndentationError before all calculations,corrected for-loop indentation and AST-checked all project code,results/process_03_run_stage0.log,RECORDED_AND_RESOLVED
IMPLEMENTATION_3,NUMERICAL_PATH_FAILURE_NOT_MISSING_PHYSICS_OR_NN_FAILURE,Woodbury normal residual exceeds unchanged 1e-10 gate for late asymmetric receiver and 2 selected children,"keep failed attempts; evaluate only failed reduced endpoints with upstream direct branch, same gate and no jitter; this records an attempted path; final unchanged-gate resolution is IMPLEMENTATION_5 residual correction",results/attempt04_partial/FAILURES.json,RECORDED_AND_RESOLVED
IMPLEMENTATION_4,IMPLEMENTATION_ERROR,adapter accessed a nonexistent PB attribute; direct branch also restricted by upstream to small material space,use saved PB layout to invoke existing torch_cpu branch only on failures; restore original contiguous SVD basis convention; this records an attempted path; final unchanged-gate resolution is IMPLEMENTATION_5 residual correction,results/attempt05_partial/FAILURES.json,RECORDED_AND_RESOLVED
IMPLEMENTATION_5,FLOATING_POINT_CANCELLATION_IN_EQUIVALENT_NORMAL_SOLVE,"NumPy, existing torch_cpu and fixed Schur basis fail residual gate in late asymmetric state, residual about 2e-10","equivalent Woodbury normal equation with one fixed residual correction; preserves operator, lambda, and 1e-10 gate; upstream vendor untouched",results/attempt06_partial/FAILURES.json,RECORDED_AND_RESOLVED
IMPLEMENTATION_6,IMPLEMENTATION_ERROR,unmatched closing parenthesis before any numerical work,removed unmatched parenthesis and AST validated,results/process_10_analyze_stage0.log,RECORDED_AND_RESOLVED
IMPLEMENTATION_7,OUTPUT_SERIALIZATION_ERROR,NumPy boolean not JSON-native; numerical files already saved,convert NumPy scalars explicitly without altering results,results/process_11_analyze_stage0.log,RECORDED_AND_RESOLVED
A_NEGATIVE_B3,SCIENTIFIC_NEGATIVE,B3 coverage inferior to both prefix and O/P/M union,Keep negative result; do not retune on held-out objects,results/A18_BASELINE_OBJECT_SUMMARY.csv,RETAINED
B_VOXEL_LABEL_GAP,DATA_INCOMPLETE,194 matched voxel directions have no legal JFd label,Do not synthesize from scalar Q or unsupported fullJV,A18_DEFECT_LABEL_AVAILABILITY.csv,OPEN_DATA_GAP
B_PRIMARY_MARGIN,SCIENTIFIC_NEGATIVE_SCOPED,"Rank8/16/32 projection skips55.6%, adds no skips over anchor and misses60%/75% gates",Do not recommend B training from this representation,results/A18_B_OBJECT_SUMMARY.csv,RETAINED
ORIGINAL_VERIFY_ABSENT,SOURCE_INCOMPLETE,Original A18 verification script/config/results not supplied,"Write equivalent new tests, preserve missing inventory, never claim original reproduction",A18_DATA_INVENTORY.csv,EQUIVALENT_AUDIT_COMPLETE_ORIGINAL_OPEN
LITERATURE_429,ACCESS_INCOMPLETE,Five collect operations HTTP429; R4 full text unavailable,"Keep partial search, verify citations, inspect identified primary sources",literature/scholarqa_collect.json,NOVELTY_SCOPE_OPEN
DOCUMENT_DEFINITION,AUTHORING_CORRECTION_NOT_NUMERICAL_CHANGE,Draft guide called total Lambda only prior precision; source lambda_total also includes LM damping,Define archived total regularization in all three current manuscripts; numerical code always used saved lambda_total,manuscript/COMPLETE_EXPLANATION_ZH.tex; pinned maxwell_state.py; INPUT_HASHES.json,RECORDED_AND_RESOLVED
```

## Reference ledger

```csv
reference_id,verified_identifier,title,authors,metadata_year,year_scope,tier,primary_url,evidence,limitation,metadata_resolved
R5,DOI:10.1109/TAP.2019.2922779,Physics-Inspired Convolutional Neural Network for Solving Full-Wave Inverse Scattering Problems,"[""Zhun Wei"", ""Xu-Dong Chen""]",2019,PREPRINT_OR_DATABASE_YEAR; see DOI for journal year,B,https://doi.org/10.1109/TAP.2019.2922779,collect paper_search abstract,Induced-current learning with bypassed major component; detailed layers OPEN,True
R7,DOI:10.1137/20M1338460,On Learned Operator Correction in Inverse Problems,"[""Sebastian Lunz"", ""A. Hauptmann"", ""T. Tarvainen"", ""C. Schönlieb"", ""S. Arridge""]",2020,PREPRINT_OR_DATABASE_YEAR; see DOI for journal year,A,https://arxiv.org/html/2005.07069v2,Sections3-4; primary full text,Forward correction may fail gradient consistency; this study only tests fixed-direction acceptance,True
R2,ARXIV:2209.03567,SOM-Net: Unrolling the Subspace-Based Optimization for Solving Full-Wave Inverse Scattering Problems,"[""Yu Liu"", ""Hao Zhao"", ""Ren-Cheng Song"", ""Xu-Dong Chen"", ""Chang Li"", ""Xun Chen""]",2022,PREPRINT_OR_DATABASE_YEAR; see DOI for journal year,A,https://arxiv.org/html/2209.03567v2,"Method, Algorithm1; primary full text",Alternating learned current and analytical material update; no claim of direct Stage0 comparator performance,True
R8,ARXIV:2206.10745,Derivative-Informed Neural Operator: An Efficient Framework for High-Dimensional Parametric Derivative Learning,"[""Thomas O'Leary-Roseberry"", ""Peng Chen"", ""U. Villa"", ""O. Ghattas""]",2022,PREPRINT_OR_DATABASE_YEAR; see DOI for journal year,B,https://arxiv.org/abs/2206.10745,primary abstract,Derivative compression is established; no empirical extrapolation to this archive,True
R4,DOI:10.1109/TAP.2025.3602467,Physics-Data of Hybrid-Driven Edge-Featured Graph Attention Networks for Surface Currents Learning 3-D PEC Targets,"[""Huilin Cao"", ""Haoran Sun"", ""Jing-Wei Liang"", ""Shan Tao"", ""Yuxiang Ren"", ""Mao-Kun Li""]",2025,PREPRINT_OR_DATABASE_YEAR; see DOI for journal year,C,https://yuxiangren.github.io/publications/,author publication list + metadata verification,"PEC surface-current graph prior art; full text, edge formula and speed denominator OPEN; S2 title malformed",True
R16,ARXIV:2210.03008,Residual-based error correction for neural operator accelerated infinite-dimensional Bayesian inverse problems,"[""Liang-Hao Cao"", ""Thomas O'Leary-Roseberry"", ""P. Jha"", ""J. Oden"", ""O. Ghattas""]",2022,PREPRINT_OR_DATABASE_YEAR; see DOI for journal year,B,https://arxiv.org/abs/2210.03008,primary abstract,Residual-based operator error correction prior art; no imported certificate,True
R10,ARXIV:2204.13454,A new certified hierarchical and adaptive RB-ML-ROM surrogate model for parametrized PDEs,"[""B. Haasdonk"", ""H. Kleikamp"", ""Mario Ohlberger"", ""F. Schindler"", ""T. Wenzel""]",2022,PREPRINT_OR_DATABASE_YEAR; see DOI for journal year,B,https://arxiv.org/abs/2204.13454,primary abstract,Certified adaptive ROM uses error control; this study lacks physical deterministic envelopes,True
```

## English study — extracted PDF text

      Can Neural Current Proposals Outperform Cheap Physics?
       An Offline Feasibility Study for Maxwell Current Design

                                        Stage 0 research record

                                            3 October 2026


                                                Abstract

         We study two decision-oriented uses of learning in reduced Maxwell models: proposing dictio-
     nary currents before costly endpoint scoring, and correcting the directional observation defect of
     a reduced model. The study reuses a pinned numerical archive and runs no neural training, new
     full-wave right-hand sides (RHS), or nonlinear reconstruction. Across four objects and 36 com-
     plete receiver queries, a fixed observation/coupling/injection shortlist improves gain-weighted
     coverage from 0.7251 to 0.7875 but leaves the diﬀicult near-contact case at 0.5178. A signed-
     transfer heuristic performs worse, at 0.6259. The remaining coverage gap supports a bounded
     proposal pilot. Compensation has a different outcome: only two Gaussian objects have complete
     matched vector labels. With a train-only proper orthogonal decomposition (POD) basis, the
     primary near-contact test yields 10/18 counterfactual skipped checks for anchor-tail ranks 8, 16
     and 32, identical to the uncorrected anchor. The direct-defect oracle yields 0/18. Initialization
     raises the matched total-RHS-halving requirement to 75%. We therefore recommend the candi-
     date pilot and withhold a compensation training recommendation. Missing voxel labels and the
     negative Gaussian representation diagnostic are recorded separately.


1 Research question and decision unit

A reduced current model replaces a large Maxwell solve by a Galerkin solve in a selected current
subspace. Its value for inversion depends on the material step it produces, rather than solely on
how much observation energy its currents carry. We ask whether a learned candidate generator
has useful opportunity after cheap physical screening, and whether an existing directional-defect
representation is accurate enough to preserve actual gain margins.
A query is one frozen physical state, receiver current selection, shared material step and anchor. A
move replaces one original dictionary atom by one incoming atom. The anchor is a richer, already
constructed reduced current model used to score shortlisted moves. These terms define a local
decision problem; the one-swap teacher measures the best feasible move in this dictionary, with
no-op available. It does not solve unrestricted current or material optimization.
Learning small current corrections has prior art in induced-current learning [1]; SOM-Net com-
bines learned current updates with analytical material updates [2]. Explicit operator correction [3],
derivative compression [4], and residual correction [5] also predate this study. Our contribution
here is an evidence-backed feasibility decision with physical input boundaries, complete local teach-
ers and margin-aware costs. A search with three formulations encountered five HTTP 429 op-
eration failures; seven selected bibliographic records resolved. This is a scoped prior-art audit,
without a global novelty claim. The recent PEC surface-current graph paper associated with DOI
10.1109/TAP.2025.3602467 remains a full-text audit gap.




                                                     1
NN-native current design: Stage 0                                                               3 October 2026


2 Physical and information contract

Let L denote the full complex Maxwell current operator, S the observation map, and Bp the material-
to-current tangent map for illumination p. The identities assume an invertible full operator and
admitted nonsingular compressed blocks. All six illuminations share one real material vector. For
a current basis U ,
                  RU = U (U ∗ LU )−1 U ∗ ,                        JU = realify{SRU Bp }6p=1 ,              (1)
                 IU,p = (I − LRU )Bp ,                           OU = S(I − RU L).                         (2)
The complex L and Schur blocks are not assumed Hermitian or positive definite. The archived
stability tolerance is 10−10 ; no jitter is used. Atom deletion refers to the original physical dictionary
ID, even when SVD or QR changes the coordinates used to represent its span.
The saved public arrays are L  e = Q∗ LQ, Se = SQ, and P Bp = Q∗ Bp , with a 128-dimensional current
embedding Q. Their inverse cannot produce full-physics truth. We use them for reduced endpoint
calculations and public features. The saved orthonormal physical material tangent, its fingerprint,
ℓ and Λ = λtotal I remain coupled. The archived total regularization includes prior precision and
Levenberg–Marquardt (LM) damping: for smooth early, λtotal = 10−5 + 10−2 = 0.01001. It is
retained in both the step and frozen gain. Gaussian material dimension is 54 real coordinates, voxel
dimension 3456. Complex material coordinates are packed as [ℜc; ℑc]; observations are packed
illumination-major as [ℜyp ; ℑyp ], giving 1536 real data entries. Illumination stacking does not
create six independent material steps.
The shared receiver step x minimizes the frozen reduced quadratic. With material-to-observation
Jacobian JU , its Gauss–Newton (GN) equation is
                                      x = −(JUT JU + Λ)−1 (JUT r + ℓ).
The proposal interface receives F0 cached public signatures and F1 inexpensive projected physical
features. Teacher utility and vector labels are separate interfaces. Actual child steps and complete
anchor gains (F2) are used only after a shortlist; full Jacobians, reference steps, truth and full
gains (F3) remain on the evaluator side. Object identity, geometry category, path and seed are
partition/provenance metadata, not learner features.


3 Complete archive and frozen comparisons

We use commit 56097653e6a814399581c3759fc3e3e8f0c2284c of the pinned archive. Public out-
put hashes and original source hashes are checked against their respective manifests. The raw
teacher files are separately checked against the original dataset manifest. Public metadata rewrites
can change JSON bytes, so raw and public hashes are not conflated.
The four objects are smooth, near-contact, layered and asymmetric. Each has early, middle and late
frozen states with k = 4, 8, 16: 36 queries in total. Query identity contains object/state/pool hashes,
original selected IDs, the public step hash, anchor, prior, gauge and receiver/operator layout. It
avoids collapsing distinct zero-contrast states with equal contrast hashes. Every query contains its
full k(154 − k) one-swap dictionary: 47,712 unique moves, 43,108 feasible moves and 5208 incoming
utility labels. No duplicate canonical move remains. These are correlated local records from four
physical objects, not tens of thousands of independent scenes.
For incoming atom i, the teacher is
                                                                            
                                    ti (q) = max 0,        max          Qq,a,i .                           (3)
                                                      a∈Uq , feasible

All feasible maximizing drops within the numerical tolerance are retained, and zero-positive cases
retain no-op. All 36 archived queries have a gain above their floor; no-op is available but never
optimal in this cohort.

                                                       2
NN-native current design: Stage 0                                                                    3 October 2026


3.1   The six opponents

B0 is the original family-prefix round-robin selection of twelve incoming atoms. B1 matches those
family counts with random draws, using frozen seeds 20261002/03/04. B2 cyclically takes the
next unused candidate from descending observation, coupling and material-injection lists, filling
exactly twelve. B3 greedily maximizes query-normalized signed-transfer rank minus a fixed 0.25
observation-overlap penalty. B4 uses the already paid full-dictionary endpoint/anchor scores. B5 is
the high-fidelity one-swap teacher with no-op.
To define B2/B3, use the projected public operators and receiver U . Set

                c̄i = ci − U (U ∗ LU )−1 U ∗ Lci ,                         Yi = Sc̄i ,                          (4)
                γi = c∗i Lc̄i ,                                        np,i = c∗i (I − LRU )Bp x.               (5)

Here and below this feature computation uses the saved projected maps, while the full-operator
equations explain the mechanism. The three nonnegative axes are

                                    Oi = ∥Yi ∥2 / ∥ci ∥2 ,                                                      (6)
                                    Pi = (∥U ∗ Lci ∥2 + ∥c∗i LU ∥2 )/ ∥ci ∥2 ,                                  (7)
                                         X
                                    Mi =    |np,i |2 /(∥x∥2 ∥ci ∥2 ).                                           (8)
                                             p


B3 forms wi = realify{Yi γi−1 np,i }p and

                                      si = −(r + JA x)T wi − 21 ∥wi ∥2 .                                        (9)

This is an addition response at the public receiver step, not the finite material gain of a swap.
Percentile ranks are computed within each query. The diversity penalty uses squared normalized
Yi∗ Yj overlap. Numerical ties are fixed at relative 10−12 score resolution and resolved by original
atom ID. A scalar γ has usual condition number one; normalized |γ| and retained-core reciprocal
condition are tracked instead.


4 Proposal results and mechanism

Gain-weighted coverage is                        P
                                                     q max(0, maxa,i∈Pq Qq,a,i )
                                  Cov12 = P                                        .                          (10)
                                                   q max(0, maxa,i∈Dq Qq,a,i )

An absent denominator would be NA. Object means give the four objects equal weight. Top-one
recall asks whether the pool contains a maximizing incoming atom; top-four recall averages the
fraction of tied top-four incoming labels captured. Gain regret is the full-dictionary best positive
gain minus the pool’s best positive gain, in the saved objective units.

                  Method          smooth         near-contact    layered     asymmetric     Mean
                  B0                0.9689              0.3528   0.9198            0.6589   0.7251
                  B1 (s = 02)       0.9317              0.5809   0.9010            0.7647   0.7946
                  B1 (s = 03)       0.9658              0.8326   0.3139            0.8541   0.7416
                  B1 (s = 04)       0.9760              0.5426   0.9379            0.9287   0.8463
                  B2                0.9718              0.5178   0.9852            0.6751   0.7875
                  B3                0.7283              0.4142   0.7644            0.5969   0.6259
                  B4                1.0000              1.0000   1.0000            1.0000   1.0000
                  B5                1.0000              1.0000   1.0000            1.0000   1.0000




                                                            3
NN-native current design: Stage 0                                                                         3 October 2026


                                                B0 prefix   B1 random mean   B2 O/P/M    B3 signed




    Gain-weighted coverage
                              1




                             0.5




                              0
                                   smooth               near-contact           layered               asymmetric

Figure 1: Cheap screening leaves opportunity in the diﬀicult objects. Random bars show the three
fixed-seed mean; all individual seeds appear in the table. These are descriptive values, not confidence
intervals.

B1 seed suﬀixes 02/03/04 mean 20261002/03/04. B4 and B5 have coverage one because their pools
contain the full dictionary; their decision roles and costs differ.
B0 reproduces the reported 0.969/0.353/0.920/0.659 bottleneck. The full-dictionary anchor top-one
capture is 0.9960/0.9810/0.9937/0.9883, confirming that accurate ranking is available after a useful
atom reaches the scored pool. B2 increases near-contact coverage by 0.1650 and layered coverage by
0.0654, but near-contact 0.5178 and asymmetric 0.6751 remain below 0.70 and 0.85. Its equal-object
mean 0.7875 also misses 0.85.
B3’s mean 0.6259 is a substantive negative result. The public receiver-direction addition response
does not capture the changed material GN step or the deletion-dependent retained core. Observation
diversity cannot repair that mismatch by itself. This supports targeting incoming utility directly
rather than assuming a signed addition surrogate ranks material swaps.
The strongest single frozen cheap comparison has mean 0.8463 (B1, seed04), leaving 0.1537 to the
one-swap teacher. That seed still fails the near-contact gate. Another seed improves near-contact
but collapses layered coverage. Choosing a different seed after observing each object or query would
create an unearned oracle. None of the fixed cheap opponents meets all three elimination gates.
The remaining gap justifies a bounded candidate pilot; it does not show a neural model can learn
the gap.
F1 components are measured separately. Their joint warm median is 4.60 ms/query, using already
cached projected maps. Child endpoint scoring remains a post-shortlist cost. These timings do
not include acquiring a physical workspace or establishing a full-wave reference, and are not an
end-to-end acceleration result.


5 Compensation: lawful labels before architecture

For the same legal material difference d, we compare

                                            ∆zU = (JF − JU )d = (JA − JU )d + (JF − JA )d.                         (11)

The known inner correction is not learned. The richer anchor tail is the remaining target. A and
U need not be nested; a smaller empirical tail is not a monotonicity theorem.
Gaussian fullJ gives legal measurement labels for 101 smooth and 103 near-contact matched direc-
tions. Voxel V /fullJV contains only sixteen material directions. The tested voxel differences have
relative projection residual 0.9948–0.9992, far above 10−9 . We also inspect every stored endpoint
step and full Js. Only the asymmetric late k = 8 receiver can be recovered in the lawful saved

                                                                       4
NN-native current design: Stage 0                                                       3 October 2026



Table 1: Recall and summed gain regret. All objects have nine queries and zero no-positive queries.
Complete B1 rows and every query are in the delivered CSVs.
                          Object         Method   Top1    Top4    Regret sum
                          smooth         B0       0.222   0.111      4.79e-05
                          smooth         B2       0.222   0.167      4.34e-05
                          smooth         B3       0.000   0.000      4.18e-04
                          smooth         B4       1.000   1.000     0.00e+00
                          near-contact   B0       0.000   0.000      1.94e-04
                          near-contact   B2       0.111   0.083      1.45e-04
                          near-contact   B3       0.000   0.028      1.76e-04
                          near-contact   B4       1.000   1.000     0.00e+00
                          layered        B0       0.222   0.250      2.08e-04
                          layered        B2       0.444   0.278      3.85e-05
                          layered        B3       0.000   0.028      6.11e-04
                          layered        B4       1.000   1.000     0.00e+00
                          asymmetric     B0       0.000   0.056      1.77e-04
                          asymmetric     B2       0.000   0.111      1.69e-04
                          asymmetric     B3       0.000   0.000      2.09e-04
                          asymmetric     B4       1.000   1.000     0.00e+00


snapshot span, giving one matched difference vector. The other 194 voxel matched vectors remain
missing. A scalar gain does not determine JF d.
There are zero full-current or unresolved-current labels. Measurement outputs cannot identify
current components in the observation nullspace; a full solve in public Q coordinates would still be
a reduced model. The one voxel vector is an inventory finding, excluded from object-level B fitting
and claims.
Median anchor-tail/direct-defect norm ratios are 0.3729 (smooth) and 0.4356 (near-contact). Re-
moving the inner correction substantially reduces target magnitude. Accuracy for a decision, how-
ever, depends on residual orientation and margin, not merely target size.


5.1   Train-only POD, fixed ridge and source oracle

POD (proper orthogonal decomposition) is the left singular basis of uncentered training defect
snapshots. We freeze output ranks 8/16/32, preserving a zero-input linear correction. Ridge uses
a fixed training-RMS scaler, no intercept and α = 10−6 σmax (H)2 . It is a shared state-blind linear
baseline, not an implemented state-conditioned Maxwell-native network.
The prescribed object cycle has 18 train, nine validation and nine test queries per fold. In Fold1,
smooth provides 101 training directions and near-contact 103 test directions; layered validation lacks
complete vectors. Fold0 has no complete Gaussian training object. In Fold3, near-contact trains
and smooth is validation, not a second test. We do not move validation into training to manufacture
a two-way test. Thus B is a two-object diagnostic with one available prescribed primary test fold,
not a four-object result.
The source encoder uses a fixed random physical-row sketch of the projected anchor injection, shared
across the verified common Gaussian physical grid. At 8/16/32 complex rows per illumination it
has 96/192/384 real features. For the saved 54-dimensional Gaussian gauge, its numerical rank is 54
in all cases. Using the evaluation-side full Jacobian, the state-specific unobservable-tail residual is
at most 4.47 × 10−15 relative. This tests ker H ⊆ ker(JF − JA ) on that gauge. It does not furnish a
public correction operator or resolve off-Q current labels. Voxel source suﬀiciency cannot be inferred
from this Gaussian result.




                                                  5
NN-native current design: Stage 0                                                                                  3 October 2026


6 Gain margins and initialized verification cost

For a fixed material difference d and exact archived evaluator output u = r + JF x,

                                               Q = −uT z − (ℓ + Λx)T d − 21 (∥z∥2 + dT Λd),            z = JF d.             (12)

If e = z − ẑ, the difference is −(u + ẑ)T e − ∥e∥2 /2. Hence |Q − Q̂| ≤ vϵ + ϵ2 /2 for v = ∥u + ẑ∥ and
∥e∥ ≤ ϵ. For positive predicted margin m = Q̂ − τ ,
                                                                                  2m
                                                                  ϵcrit = √                .                                 (13)
                                                                              v 2 + 2m + v

Here Q̂ substitutes ẑ in the displayed gain while retaining exact evaluator u. It differs from the
anchor score used for A, whose residual is r + JA x. The stable expression avoids subtracting two
nearly equal terms. We label a check counterfactually skippable only when the actual saved response
error is below this threshold. All 18 primary B4 finalists have true gain above their floor.

                                                                Tail POD oracle                   Tail ridge
                                                               Direct POD oracle               Opportunity gate
                                                            Matched RHS-halving floor




      Counterfactual skip fraction
                                      1

                                     0.8

                                     0.6

                                     0.4

                                     0.2

                                      0
                                           0                8                      16                                   32
                                                                POD output rank (0: uncorrected)

Figure 2: Primary near-contact test, B4’s 18 matched finalists. Tail POD projection improves
residuals but adds no skipped checks over the uncorrected anchor. Direct-defect POD yields zero.
The 60% opportunity gate and 75% initialized RHS-halving floor are different requirements.

Direct-defect POD gives 0/18 at every rank. Anchor-tail POD gives 10/18 at every rank, as does
the uncorrected anchor. Tail ridge gives 10/18 at rank8 and 9/18 at ranks16/32. For B2’s matched
finalists, rank32 tail POD gives 7/18. The rank32 B4 median tail-relative error is 0.8868; its median
error-to-critical-radius ratio is 0.5413. A median below one does not imply that enough finalists
cross the gate. The reversed Gaussian cohort is Fold3 validation; its tail POD skip fraction is 8/18
at all ranks.
This projection oracle uses optimal coeﬀicients for Euclidean response approximation in the fixed
training basis. It is not a universal maximizer of the number of margin-certified decisions. The
negative result applies to the tested POD/ridge pipeline and does not exclude every possible state-
conditioned representation.
Let initialization cost be I full RHS and matched finalist checks cost R RHS, with skip fraction ρ.
Before model overhead,

                                                   Cnew = I + (1 − ρ)R + Crefresh + Caudit + Cfailed .                       (14)

If the additional terms vanish, total RHS halving needs ρ ≥ (1 + I/R)/2. Historical 1164 finalist
RHS and 216 initialization RHS give 59.28%. Our single-decision test has I = 54, R = 108,
requiring 75%. The observed 55.6% oracle opportunity gives an ideal initialized RHS ratio 0.6296.

                                                                               6
NN-native current design: Stage 0                                                      3 October 2026


Cold setup, model/source computation, refresh and audits increase costs. These fullJ/table reuses
create no actual skipped physical solve or measured speedup. Archived high-fidelity labels and
this exact-error diagnostic are not deterministic physical error certificates; certified ROM methods
require their own error controls [6].


7 Verification, reproducibility and retained failures

We wrote 128 fresh non-Hermitian complex algebra cases for the two-sided defect, Schur tail, non-
nested anchor decomposition, phase covariance, original-ID permutation and gain-error bound. The
largest identity residual is 9.69 × 10−16 ; no tested gain bound is violated. On one early k = 8 query
per object, phase/permutation actions, public forbidden-field canaries and real linearity checks pass.
Ridge zero/scaling/additivity and prediction canaries are separate checks. No neural architecture
has been implemented or tested.
The supplied folder lacks the original validation script, original configuration/results and separate
reference file. References were restored from the merged reading version. The new checks are equiv-
alent audits, not reproduction of the original reported 128 PASS result. Parser, SVD-gauge tester,
adapter and serialization failures are retained with their costs. Sixteen upstream Woodbury units
crossed the unchanged 10−10 normal-residual gate; one fixed residual correction of the same normal
equation validates them below the gate. Vendor code, physical operators and λ are unchanged.
This is a numerical validation repair, not a proposed solver or speed experiment.
The reproduction package contains source hashes, frozen configuration, query/move identities, all
seed comparisons, matched vector provenance, fitted bases, source oracles, failure/claim/reference
ledgers and replot scripts. All new artifacts reside under A18_NN_NATIVE. Single-process resource
receipts include failed attempts; peak memory remains below 4 GiB and numerical CPU below one
hour.


8 Independent decisions and stopping point

A: GO to a bounded tiny pilot, as a recommendation only. No fixed cheap opponent closes
the diﬀicult-object gap. A future student must attain near-contact coverage 0.70, asymmetric 0.85
and mean 0.85, then exceed the strongest cheap mean by 0.05 or halve cost at quality within 0.01.
Shared MLP, Set Transformer, ordinary GAT and phase-consistent physics-edge GNN configurations
share inputs, twelve candidates, at most 200k parameters, 60 epochs, four object folds and two seeds.
They remain disabled.
B: globally data-incomplete; no training recommendation from the tested Gaussian
oracle. Complete Gaussian labels support the smaller-tail mechanism and source-rank diagnostic,
but the tested output bases do not add margin/cost opportunity. Four-object B assessment first
needs legal matched voxel vectors and receiver outputs. A new representation would need train-only
evidence that it changes this margin ceiling, with a fixed-state map linear in d and an explicit cache
refresh budget.
Candidate Generator first remains the useful next-stage ordering. A and B have independent gates;
A’s opportunity does not authorize B training. This study ends after the feasibility artifacts and
manuscripts. Training epochs, new full-wave RHS, new nonlinear trajectories, statistical safety
calibration, remote execution and publication all remain unperformed.




                                                  7
NN-native current design: Stage 0                                                      3 October 2026


A Audit entry points

The nine mandatory files are the inventory, query manifest, object splits, feature costs, analytic
baselines, defect-label availability, defect oracle, cost ledger and Stage0 decision. CLAIM_LEDGER.csv
is the shared evidence index. FUTURE_TRAINING_CONFIG.json is disabled. START_HERE.md gives
commands and rendering provenance.


References

[1] Z. Wei and X. Chen, “Physics-Inspired Convolutional Neural Network for Solving Full-Wave
    Inverse Scattering Problems,” IEEE TAP, 2019. doi:10.1109/TAP.2019.2922779. Abstract-level
    evidence in this study.

[2] Y. Liu et al., “SOM-Net: Unrolling the Subspace-based Optimization for Solving Full-wave
    Inverse Scattering Problems,” 2022. arXiv:2209.03567. Method and Algorithm 1 inspected.

[3] S. Lunz et al., “On Learned Operator Correction in Inverse Problems,” 2020 preprint.
    arXiv:2005.07069; journal DOI 10.1137/20M1338460. Sections 3–4 inspected.

[4] T. O’Leary-Roseberry et al., “Derivative-Informed Neural Operator: An Eﬀicient Frame-
    work for High-Dimensional Parametric Derivative Learning,” 2022 preprint, revised 2023.
    arXiv:2206.10745. Abstract-level evidence.

[5] L. Cao et al., “Residual-based error correction for neural operator accelerated infinite-
    dimensional Bayesian inverse problems,” 2022 preprint. arXiv:2210.03008. Abstract-level evi-
    dence.

[6] B. Haasdonk et al., “A new certified hierarchical and adaptive RB-ML-ROM surrogate model
    for parametrized PDEs,” 2022. arXiv:2204.13454. Abstract-level evidence.




                                                  8


## 给陈教授的中文汇报 — extracted PDF text

         关于 Maxwell Current Design 中 Learned Proposal
             与 Defect Compensation 的阶段性研究汇报

                                   给陈教授的研究汇报

                                    2026 年 10 月 3 日



   陈教授：

   我这一轮希望先回答一个实际问题：已有 current design 的瓶颈，到底值得交给 NN（neural
network）来学习哪一部分？我把问题拆成两个独立分支。A 是 learned candidate generator：在昂
贵 endpoint scoring 前提出十二个 incoming currents。B 是 directional defect compensation：预
测 reduced model 在同一 material direction 上遗漏的 observation response。我完成的是 Stage0
feasibility study，全部使用固定 archive；NN epochs、new full-wave RHS（right-hand sides，物理
求解右端项）和 new nonlinear trajectories 都为零。

   我的结论是：Candidate Generator first 仍然成立，可以建议一个 bounded tiny pilot；B
暂不建议直接进入训练。后一个判断包含两件不同的事：Gaussian 有数据部分的表示/margin 结果
不支持当前表示带来额外价值；voxel 的完整向量标签不足，四对象 B 仍是 data-incomplete。



                    1 我如何把研究问题收缩到实际决策

   以前的结果已经说明 full-dictionary analytical anchor 很擅长排序，但在线 shortlist 往往没有把
有用的 incoming atom 放进去。这里 anchor 指已经构建的更丰富 reduced current model，shortlist
指先筛出的十二个 incoming IDs。排序正确与候选进入池子，是两个不同问题。我先复核这个区分，
再引入有实际物理意义的廉价筛选，而不是先假定 NN 必须有用。

   我保留了原始 material gauge：每个 frozen state 的六个 illumination 共用一个 real material
step。Gaussian 是 54 维实材料坐标，voxel 是 3456 维。当前 reduced current embedding 是 128
维，而 observation 向量是 1536 维实数。保存的 L/S/P B 是这个 embedding 的投影映射，不能把
它的 inverse 当作 full Maxwell truth。Step 与 gain 都使用保存的 total regularization，包含 prior
precision 和 Levenberg–Marquardt（LM）damping；没有把它替换成仅有先验的系数。

   对 selected current basis U ，我用

                   RU = U (U ∗ LU )−1 U ∗ ,   JU = realify{SRU Bp }6p=1 .

一个 query 包括物理状态、selected original atom IDs、公共 receiver Gauss–Newton（GN）step x、
anchor、prior 和 receiver layout；一个 move 是删除一个物理 ID 并加入另一个物理 ID。QR/SVD


                                              1
NN-native current design: Stage 0                                                                                       3 October 2026


只改变 span 的表示，不能改变动作身份。L 与 Schur block（消去 retained core 后的块）不假设
Hermitian/SPD，原来的 10−10 stability gate 保留。

       完整字典重新盘点得到 36 queries、47,712 unique moves、43,108 feasible moves、5208 incoming
labels，恰好覆盖四对象、三 phase、三 k。这些不是几万个独立物理场景。公共和原始 manifest 各
自核对 hash，固定 source commit 与公开入口记录在复现包；历史文件保持只读。



                                   2     A：解析筛选有所改善，却没有消除 NN 的机会

       我比较了六类对手。B0 是原 family-prefix；B1 是三个固定 seed 的 family-matched random；B2
是 observation/coupling/material-injection 三轴固定轮转 union；B3 是 signed transfer 加 observa-
tion diversity，惩罚固定 0.25；B4 是已经付费的完整字典 anchor；B5 是包含 no-op 的 one-swap
teacher。

       Cov12 衡量 shortlist 捕获了多少真正有益的 one-swap gain，而不是数有多少原子入选：先对
每个 query 取池内最大的正 Qtrue ，再与完整字典的对应总量相除。四对象均值等权，避免某个对象
的 objective scale 主导判断。top1/top4 recall 衡量最佳 incoming 及前四 incoming 是否进池，gain
regret 是错失的 gain。


                                       Method        smooth      near-contact   layered     asymmetric       Mean

                                       B0             0.9689          0.3528    0.9198              0.6589   0.7251
                                       B1 (s = 02)    0.9317          0.5809    0.9010              0.7647   0.7946
                                       B1 (s = 03)    0.9658          0.8326    0.3139              0.8541   0.7416
                                       B1 (s = 04)    0.9760          0.5426    0.9379              0.9287   0.8463
                                       B2             0.9718          0.5178    0.9852              0.6751   0.7875
                                       B3             0.7283          0.4142    0.7644              0.5969   0.6259
                                       B4             1.0000          1.0000    1.0000              1.0000   1.0000
                                       B5             1.0000          1.0000    1.0000              1.0000   1.0000


随机 seed 的 02/03/04 分别是 20261002/03/04；没有逐对象事后选择最好的 seed。

                                                     B0 prefix   B1 random mean     B2 O/P/M           B3 signed




    Gain-weighted coverage
                              1




                             0.5




                              0
                                        smooth               near-contact                 layered                  asymmetric


图 1: 四对象 Cov12。random 柱是三个固定 seed 的均值，各 seed 全部保留在表中。图表共用同一
evidence ledger。


                                                                            2
NN-native current design: Stage 0                                          3 October 2026


    我首先恢复了原来 smooth/near-contact/layered/asymmetric 的 0.969/0.353/0.920/0.659 覆盖，
full-dictionary anchor 的 top1 capture 仍有 0.981–0.996。因此这轮观察到的困难，确实主要在 in-
coming pool，而不是完整 anchor 排序突然失效。

    B2 对 near-contact 从 0.3528 提高到 0.5178，对 layered 提高到 0.9852。这说明 dressed obser-
vation、双向 retained coupling 和公共 material-step injection 能提供一些真实任务信息。但 near-
contact 还没到 0.70，asymmetric0.6751 还没到 0.85，mean0.7875 也没到 0.85。这个正结果让我认
为便宜物理特征值得保留，同时不能据此判解析方法已经解决问题。

    B3 的 mean 只有 0.6259，我完整保留了这个负结果。它用公共 receiver step x 做一个 addition
transfer proxy；真正的 one-swap 会改变 retained core 和 child material GN step。这个 mismatch 可
以解释为什么有符号的信息没有自动变成 material utility。这里是一个需要约束 future model target
的发现：学习应该面对 incoming utility，而不是把 addition response 直接包装成实际 swap gain。

    最强单个固定廉价对手的 mean0.8463，距离 one-swap teacher 仍有 0.1537。三 random seeds
的表现并不稳定：一个 seed 改善 near-contact，但 layered 很差；另一个 seed 改善 asymmetric，却
仍丢失 near-contact。我的 A GO 只是认为保留了值得检验的机会，还不是 NN superiority 的实
测结果。下一轮还须在相同输入与成本预算下满足两困难对象与四对象均值门槛，并比最强廉价基
线多 0.05 或同质量成本减半。

    我把信息按取得费用分成四级：F0 是已有缓存，F1 是廉价的投影乘法、coupling 和 injection；
F2 是实际 child GN step 与完整 anchor score；F3 是 full Jacobian 和 truth。Proposal 只能用
F0/F1。F1 每项单独计费；joint warm median 为 4.60 ms/query。actual child step 与完整 Qhat 只
在 shortlist 后使用。这个时间来自已缓存的 projected workspace，不能被说成 full-wave end-to-end
acceleration。



          3 B：为什么先扣掉已知部分，仍然没有得到足够 margin

    这里的 defect 是同一 material direction 的 full response 与 reduced response 的差。设 d 为
child 与 receiver 的材料步差，

                          ∆zU = (JF − JU )d = (JA − JU )d + (JF − JA )d.

第一项是已知的 inner correction，我不让模型重新学习它；第二项是 anchor tail。A 和 U 不一定
嵌套，所以不能事先保证 tail 在每个方向都更小。

    实测 tail/direct norm 的中位数为 smooth0.3729、near-contact0.4356。这个正结果说明 target
magnitude 的确显著降低。但我们需要的是满足 gain margin 的 response error，而不只是降低平均
target norm。

    我能合法构造的 matched measurement labels 是 smooth101、near-contact103，以及一个 asym-
metric voxel 向量。voxel 的 V /fullJV 只支持 16 维，当前差分方向的投影残差 0.9948–0.9992，不
能拿它评价任意 d。我也逐条检查 saved step/Js，只在 asymmetric late k = 8 恢复到一个合法
matched difference。另 194 条 voxel 向量仍缺失。unresolved-current teacher 为零；scalar Q 不够
恢复 response vector，更不够恢复 full current。


                                                3
NN-native current design: Stage 0                                                                                        3 October 2026


    我用 train-only POD（proper orthogonal decomposition，training snapshots 的正交主成分）
比较 rank8/16/32，并运行固定 linear ridge。四折规则没有改变：Fold1 以 smooth101 个方向训练
basis，对 near-contact103 个方向做 test；Fold3 中 reversed smooth 只是 validation。Fold0 的指定
train objects 没有完整 Gaussian 向量，不能把 validation 改作 train 来凑出双向 test。

    另一个诊断是 source compression oracle。同一物理 current-row sketch 在 Gaussian 保存的 54
维 real material gauge 上仍有 rank54，未观测 tail 的 relative residual 最多 4.47 × 10−15 。这说明
在这个小 gauge 的 state-specific 线性信息检验里，source sketch 不是主要瓶颈；并不能据此宣称它
对 3456 维 voxel 或 unseen state 的 NN 足够。



                                           4 我把表示误差与实际 acceptance margin 连起来

    原 gain 对固定方向是

                                           Q = −uT z − (ℓ + Λx)T d − 21 (∥z∥2 + dT Λd),        u = r + JF x, z = JF d.

如果 predictor 输出 ẑ，评价侧有精确的 saved-reference u，则 response error ϵ 会使 gain 最多偏差

                                                                        ∥u + ẑ∥ ϵ + 12 ϵ2 .

因此正 margin m = Q̂ − τ 对应
                                                                                     2m
                                                            ϵcrit = √                               .
                                                                        ∥u + ẑ∥2 + 2m + ∥u + ẑ∥

这里 Q̂ 使用精确的评价侧 u，只把 z 换成 ẑ；它不同于 A 中采用 r + JA x 的 anchor score。我用
actual response error 检查这个条件，得到的 oracle can skip 只是反事实机会：在线并没有这样精确
的 error envelope。

                                                                  Tail POD oracle                 Tail ridge
                                                                 Direct POD oracle             Opportunity gate
                                                              Matched RHS-halving floor




      Counterfactual skip fraction
                                      1

                                     0.8

                                     0.6

                                     0.4

                                     0.2

                                      0
                                           0                  8                      16                                       32
                                                                  POD output rank (0: uncorrected)


图 2: primary near-contact test 的 18 个 B4 finalists。所有 tail POD ranks 都是 10/18，与未补偿
anchor 相同；direct POD 全为 0/18。

    结果比单看 tail 变小更严格：direct-defect POD 的 8/16/32 均为 0/18；anchor-tail 均为 10/18，
即 55.6%，没有增加未补偿 anchor 的可跳过数量。rank8 ridge 也是 10/18，rank16/32 降至 9/18；

                                                                                 4
NN-native current design: Stage 0                                         3 October 2026


B2 的 rank32 tail oracle 为 7/18。POD 的 train-only output orientation 在未见对象上仍很难对齐
真正需要的 margin。这个结果限定于当前 projection/ridge pipeline；我没有把它解释成所有 state-
conditioned compensation 都不可能。

    成本同样改变结论。历史三轮 workload 的 1164 finalist RHS 和 216 initialization RHS 给出
59.28% 的理想 floor；这轮每 query 检查两个 finalists 的 single-decision workload，是 54 初始化
RHS 加 108finalistRHS，需要 75% skip 才能让总 RHS 减半。55.6% 只能得到理想 ratio0.6296，还
没算 refresh/audit、失败和模型成本。continuous acceptance 还会累积 cached Jx error，需要 cache
radius/refresh；exact Jd 不自动修复旧缓存。



                           5 我的分支裁决与希望讨论的问题

    我建议 A 下一阶段只做一个 bounded tiny pilot，保持 MLP、Set Transformer、GAT 和 phase-
consistent physics-edge GNN 的共同输入/预算，先检验能不能兑现 coverage gap，再谈 native ar-
chitecture 的额外价值。当前只生成 disabled config，没有训练。

    对 B，我把全范围状态列为 data-incomplete；在有完整标签的 Gaussian primary test 中，当前
rank8/16/32 pipeline 没有足够额外 margin/cost headroom，因此不建议直接训练。若继续 B，最
少的新证据应是合法 matched voxel vectors/exact receiver output，以及一个能在 train-only 条件
下改变当前 margin ceiling 的 state-conditioned linear target representation。

    我希望与您讨论三个具体问题：第一，incoming utility 是否应成为 NN 的主要监督目标，而不
是 observation/current-energy ranking；第二，phase-consistent physical-edge architecture 的优势
应如何与普通等预算模型分开验证；第三，B 的下一步应先改 representation 还是先补 provenance-
complete labels，哪一项能最快否定错误方向。这些讨论不会自动启动 full-wave label acquisition。

    这一轮完成了 source/scope/hash、完整字典、廉价对手、合法标签、POD/ridge/source/margin、
成本与结构审计。128 个新写非 Hermitian algebra cases 和四对象公共接口测试通过；原验证脚本
缺失，未冒称原版复现。数值 residual failure 与 parser/adapter/serialization 修复都保留；16 个
Woodbury units 用同一 normal equation 的一次固定 residual correction 验证到原门槛以下，无
jitter 或新物理算子。

    Stage0 到此结束。 NN training、new full-wave RHS、new nonlinear trajectories、95% safety
calibration、真实在线 skip 和端到端加速均未运行。三份文稿、图表、claim/reference/failure ledger
与复现入口共享同一账本。



                                    与已有文献的关系

    我没有把“小 current correction”当作新方向；ICLM、SOM-Net、learned operator correction
与 derivative compression 都给出了重要先例 [1, 2, 3, 4]。本轮更具体地把问题放在 physical-ID
proposal 和固定 material-direction 的 acceptance 决策上。现有文献还提醒我，measurement-space
correction 不能自动保证 gradient/adjoint consistency [3]。三个检索 formulation 部分 HTTP429，
PEC graph 全文仍缺，当前 novelty 判断保留 OPEN；错误访问或缺标签不等于科学否定。



                                           5
NN-native current design: Stage 0                                                3 October 2026


                                          参考文献

[1] Z. Wei and X. Chen, “Physics-Inspired Convolutional Neural Network for Solving Full-Wave
   Inverse Scattering Problems,” IEEE TAP, 2019. doi:10.1109/TAP.2019.2922779. Abstract-level
   evidence in this study.

[2] Y. Liu et al., “SOM-Net: Unrolling the Subspace-based Optimization for Solving Full-wave
   Inverse Scattering Problems,” 2022. arXiv:2209.03567. Method and Algorithm 1 inspected.

[3] S. Lunz et al., “On Learned Operator Correction in Inverse Problems,” 2020 preprint.
   arXiv:2005.07069; journal DOI 10.1137/20M1338460. Sections 3–4 inspected.

[4] T. O’Leary-Roseberry et al., “Derivative-Informed Neural Operator: An Eﬀicient Frame-
   work for High-Dimensional Parametric Derivative Learning,” 2022 preprint, revised 2023.
   arXiv:2206.10745. Abstract-level evidence.

[5] L. Cao et al., “Residual-based error correction for neural operator accelerated infinite-
   dimensional Bayesian inverse problems,” 2022 preprint. arXiv:2210.03008. Abstract-level evi-
   dence.

[6] B. Haasdonk et al., “A new certified hierarchical and adaptive RB-ML-ROM surrogate model
   for parametrized PDEs,” 2022. arXiv:2204.13454. Abstract-level evidence.




                                                6


## 中文完整讲解 — extracted PDF text

                      NN-native Maxwell Current Design
                   从问题、推导到本轮证据的完整讲解

                                中文研究稿理解与阅读指南

                                     2026 年 10 月 3 日


                                              摘要

         本讲解按英文研究稿的同一论证顺序，完整解释问题、物理接口、解析算法、实验结果和分支
    判断，再补充逐步推导及双语术语。读完后应能区分：候选覆盖与排序精度，measurement defect
    与 full-current defect，representation oracle 与可部署 predictor，约 60% 的历史机会门槛与本次
    75% 的成本门槛。实际结论是 A 建议下一阶段 tiny pilot；B 全范围数据不足，有完整 Gaussian
    标签部分的当前正交主成分与固定 ridge 表示没有额外 margin/cost 空间。神经网络训练与新增
    物理 actions 均未启动。



                                             目录

1 英文研究稿的核心问题：NN 应学习什么                                                                                 3


2 物理变量、共享材料 gauge 与三个接口                                                                               3

  2.1   从 Maxwell tangent 到共享材料步 . . . . . . . . . . . . . . . . . . . . . . . . . . . .              3

  2.2   公共投影并不等于完整物理算子 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                    3


3 核心推导一：两侧 defect 为什么成立                                                                               4


4 核心推导二：one-swap 的 Schur transfer 与物理 ID                                                              5


5 共同 query、teacher 与四折怎么构造                                                                            5


6 A 的固定算法步骤与结果逐项解读                                                                                    6

  6.1   B0/B1：原前缀和公平随机对手 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                  6

  6.2   B2：三个物理轴的 union           . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   6

  6.3   B3：有符号的 addition proxy 与 diversity . . . . . . . . . . . . . . . . . . . . . . . .            6

  6.4   B4/B5：昂贵参照与覆盖指标 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                 7


7 B 的向量标签：什么能够恢复，什么不能                                                                                 8

                                                1
NN-native current design: Stage 0                                                      3 October 2026


8 核心推导三：anchor tail 与 source/latent oracle                                                           9

   8.1   把已知部分从学习目标中拿掉 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                 9

   8.2   POD 与固定 ridge 的实际步骤 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .             9

   8.3   source compression oracle 检查的是信息条件 . . . . . . . . . . . . . . . . . . . . . .             10


9 核心推导四：gain、error radius 与费用门槛                                                                     10

   9.1   从 full quadratic objective 推导 gain . . . . . . . . . . . . . . . . . . . . . . . . . . .   10

   9.2   为什么 response error 必须服从实际 margin . . . . . . . . . . . . . . . . . . . . . . .             10

   9.3   约 60% 与 75% 为什么并不矛盾 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .            11

   9.4   缓存误差与统计边界 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          12


10 验证、失败、文献与复现边界                                                                                    12


11 最终理解与下一阶段所需最少证据                                                                                  13


A 中英文术语与容易混淆的量                                                                                      13




                                                  2
NN-native current design: Stage 0                                                     3 October 2026


                    1 英文研究稿的核心问题：NN 应学习什么

      Reduced current model（降阶电流模型）用少量电流方向替代高维 Maxwell current solve。Current
dictionary（电流字典）是可以选取的原子集合；一个原子有固定的 physical ID。Incoming shortlist
（候选短名单）先挑十二个未选原子，让后续昂贵端点评分仅在这十二个上进行。

      这种设计的目标最终是 inverse problem 的 material update（材料更新）。观测很强的 current，
不一定能改善材料步；改动 current span，也不等于改动 material direction。英文稿第一个问题因
此是：强的廉价物理筛选之后，learned candidate generator 还有多大可兑现的机会？第二个问题是：
已存的 directional defect 能否被压缩到足够精度，从而有机会减少 gain verification？

      此前 full-dictionary anchor 的排序很准，但 near-contact/asymmetric 的十二候选覆盖差。这
给 A 提供了明确的切入点。B 是另一条独立方向，不能因为 A 有机会就自动训练 B。Stage0 的作
用是先排除错误输入、非法 teacher、低秩误判和成本错觉。



                     2 物理变量、共享材料 gauge 与三个接口

2.1    从 Maxwell tangent 到共享材料步

      设当前物理状态冻结。L 是 Maxwell current operator，S 将电流映射到测量，Bp 是第 p 个
illumination 下材料扰动到 current source 的 tangent map。六个 illumination 都作用在同一个材料
扰动 d 上。我们不是对每个 illumination 独立求一套材料未知量。

      保存的 material gauge（材料坐标规约）是 exact physical orthonormal tangent。Gaussian 有 27
个 complex coordinates，实化后 54 维；voxel 有 1728 个 complex coordinates，实化后 3456 维。ℓ、
steps 与 basis fingerprint 必须一致，不能把同维度但不同 basis 的系数直接混用。Λ = λtotal I 是保
存的 total quadratic regularization（总二次正则）
                                       ，包含 prior precision（先验精度）与 Levenberg–
Marquardt（LM）damping。以 smooth early 为例，λtotal = 10−5 + 10−2 = 0.01001。Step 与 frozen
gain 均保留这个总量；ℓ 是另一个保存的线性项。

      材料 packing 为                       "         #
                                             ℜdc
                                    d=                 ,   dc = d1:q + idq+1:2q .
                                             ℑdc
数据则对每个 illumination 先存 [ℜyp ; ℑyp ]，再依次堆六个块。这给出 6 × 2 × 128 = 1536 real
measurements。材料实化与数据堆叠是两件不同的事。


2.2    公共投影并不等于完整物理算子

      完整 current 有 5184 行，保存的正交 embedding Q 有 128 列。公共数据是

                               e = Q∗ LQ,
                               L                       Se = SQ,     P B p = Q∗ Bp .

                                                         e −1 ，也只是
它们可以给已有字典中的 reduced endpoint 算 response 和 GN step。即使能精确求 L
在 128 维公共模型中求解，不能凭它生成 full-wave JF d。尤其是 Q 外的 current 或 injection 分量
没有因此被观测到。

                                                             3
NN-native current design: Stage 0                                             3 October 2026


    Gauss–Newton（GN）在冻结状态中最小化 reduced quadratic。令 JU 为材料到 observation 的
实 Jacobian（导数矩阵），公共 receiver step 为

                                      x = −(JUT JU + Λ)−1 (JUT r + ℓ).

信息按费用分为 F0 已有缓存、F1 廉价投影特征、F2 实际 child GN/完整 anchor score、F3 full-
wave teacher/reference。三类接口据此分开：public/proposal 只接收 F0/F1；teacher 保存 incoming
utility 或向量 target；evaluator 拥有完整 Jacobian、Qtrue 、truth 与 reference step。F2 的 child
material step/完整 anchor score 在 shortlist 之后才使用。Object ID、路径、几何类别和 seed 可
以记录 provenance/split，却不能进入网络作为捷径。current-family 是原子物理生成来源的元数据，
可以合法使用。



                       3 核心推导一：两侧 defect 为什么成立

    假设完整物理 L 可逆，且 compressed block 通过稳定性门槛。对当前 selected span 的正交
basis U ，定义
                                    AU = U ∗ LU,         RU = U A−1 ∗
                                                                 U U .

RU 是 Galerkin current response，而不是 Euclidean projector。只要 AU 稳定，RU LRU = RU ：

                            U A−1 ∗    −1 ∗     −1    −1 ∗
                               U U LU AU U = U AU AU AU U = RU .


由此，右侧 injection defect 和左侧 observation defect 为

                            IU,p = (I − LRU )Bp ,         OU = S(I − RU L).

展开双侧乘积：

                           OU L−1 IU,p = S(I − RU L)L−1 (I − LRU )Bp
                                         = S(L−1 − RU − RU + RU LRU )Bp
                                         = S(L−1 − RU )Bp .

右边就是 full 与 reduced observation tangent 的差。因此

                                    JF − JU = realify{OU L−1 IU,p }6p=1 .

这个推导使用一般 complex L，没有把它假设成 Hermitian/SPD。它告诉我们遗漏误差同时受 source
injection、full current feedback 和 observation dressing 控制。仅看原子的观测能量，可能漏掉材料
扰动的注入方向和 current feedback。

    这是原有 Galerkin/Schur 代数的解释，不被写成新增数学原创性。实际公共计算只拥有投影
maps；等式中的 full L−1 留在物理解释或 evaluator 概念层。




                                                     4
NN-native current design: Stage 0                                              3 October 2026


           4 核心推导二：one-swap 的 Schur transfer 与物理 ID

    删除一个原子后，剩余 span 记为 V ，其 compressed operator 为 AV = V ∗ LV 。把 incoming
bank 记为 C，原被删 bank 记为 A（这里 A 表示一个 bank，和 richer anchor 的下标应按上下文
区分）。在 V 外表述 bank Z，定义

                                     Z̄ = Z − V A−1 ∗
                                                 V V LZ,

                                    ΓZ = Z ∗ LZ̄,
                               NZ,p = Z ∗ (Bp − LRV Bp ),      YZ = S Z̄.

消去 retained block 后，新增 transfer 是 YZ Γ−1
                                      Z NZ,p 。因此交换造成


                        JV +C − JV +A = realify{YC Γ−1           −1
                                                    C NC,p − YA ΓA NA,p }p .


这里能共用 retained factor，但 child step 仍要解同一 real material normal problem；不能把 current
transfer 直接当作材料 gain。

    动作身份始终是 original dictionary atom ID。SVD/QR 会旋转 basis，因此 “删除第几列正交
基” 不等于 “删除哪个 physical atom”。实际数据接口保存原始 selected IDs、drop/add IDs，置换
存储后映射回它们。

    若 bank 相位变换为 Z ′ = ZE，E 是对角 unitary，则

                               N ′ = E ∗ N,     Γ′ = E ∗ ΓE,   Y ′ = Y E.

因此 Y ′ Γ′−1 N ′ = Y Γ−1 N 。这些是 phase covariance（相位协变）的具体规则，最终物理动作应 phase
invariant（相位不变）
              。普通 complex feature concatenation 不会自动满足它；future network 必须
单独通过这些测试。

    scalar γ 的 usual condition number 总是 1，不能用它辨别近奇异。我们记录 normalized |γ|、re-
tained core 的 reciprocal condition 与实际 child/core gate；block Γ 才有有意义的 condition number。
稳定性阈值保持 10−10 ，不能用 arbitrary jitter 换取 “PASS”。



                      5 共同 query、teacher 与四折怎么构造

    一个 query 的身份包括 object/state/pool、selected physical IDs、公共 x hash、anchor hash、
ℓ/Λ、gauge 和 receiver/operator layout。early 的不同对象可能都从零 contrast 出发，如果只按 state
hash 去重，会错误合并物理上下文。

    每个 query 的完整 one-swap 字典是 k(154 − k)。四对象各三 phase、k = 4, 8, 16，给出

                        12[4(154 − 4) + 8(154 − 8) + 16(154 − 16)] = 47712.

实测 unique moves=47,712、feasible=43,108、unique incoming utilities=5208、duplicate=0。数值
floor、actual rank、core/child feasibility、最优 drop 的并列集合都保留。No-op 的 utility 是 0；本
轮 36 queries 都有超过 floor 的正收益，所以 no-positive query 数为 0。


                                                    5
NN-native current design: Stage 0                                                              3 October 2026


      incoming teacher 是所有合法 drop 中的最大正收益：

                                    ti (q) = max(0,         max          Qq,a,i ).
                                                       a∈Uq , feasible

它不是 “每个 query 恰好一个正标签”；不同 incoming 可以同时有正 utility，最优 drop 也可以并
列。


                       Fold   Train objects                 Validation          Test

                         0    layered, asymmetric           near-contact        smooth
                         1    smooth, asymmetric            layered             near-contact
                         2    smooth, near-contact          asymmetric          layered
                         3    near-contact, layered         smooth              asymmetric


每折有 18 train/9 validation/9 test queries。所有 phases、k、moves、augmentations 都随对象进
入同一 partition。任何 RMS、POD 或超参数选择不能偷看 test。这个 benchmark 是 retrospective
object holdout，并非全新盲采集的数据。



                         6    A 的固定算法步骤与结果逐项解读

6.1    B0/B1：原前缀和公平随机对手

      B0 在 current-family 之间 round-robin，每个 family 按原始顺序取前缀，直到十二个。因此
它不是强物理筛选。B1 先固定 B0 的 family counts，再在各 family 中随机选同样数量，seed 为
20261002/03/04。全部 seed 都报告，不能事后按对象挑一个最优 seed。


6.2    B2：三个物理轴的 union

      用整个 receiver U 构造 c̄i 与 dressed observation Yi 。O 轴衡量归一化 dressed observation；P
轴保留两个方向的 retained coupling；M 轴用公共 receiver material step x 检查 projected injection：

                              Oi = ∥Yi ∥2 / ∥ci ∥2 ,
                              Pi = (∥U ∗ Lci ∥2 + ∥c∗i LU ∥2 )/ ∥ci ∥2 ,
                                   X
                              Mi =    |c∗i (I − LRU )Bp x|2 /(∥x∥2 ∥ci ∥2 ).
                                      p

三个列表分别降序排序。按 O/P/M 固定轮转，每次取列表中下一个尚未出现的原子；重复就跳过，
直到十二个。score relative 10−12 范围作为数值并列，按 original ID 定序。这套参数在首次 outcomes
前冻结，没有依据结果调权重。


6.3    B3：有符号的 addition proxy 与 diversity

      记 wi = realify{Yi γi−1 np,i }p ，uA = r + JA x。proxy 是

                                          si = −uTA wi − 21 ∥wi ∥2 .

                                                        6
NN-native current design: Stage 0                                                                                        3 October 2026


先在 query 内做 percentile rank，再贪心选最大
                                                                                      |Yi∗ Yj |2
                                                   ranknorm(si ) − 0.25 max                         .
                                                                              j∈P   ∥Yi ∥2 ∥Yj ∥2

这个 overlap penalty 避免候选观察方向过度相似。注意 wi 来自 receiver 的 x 和 addition，不是 child
的 d，也没有删除一个 atom 后的真实 retained core。因此它是廉价 proxy，不是 actual one-swap
gain。


6.4                     B4/B5：昂贵参照与覆盖指标

         B4 用已有完整 dictionary endpoint/anchor score 排序；B5 用已有 full-objective one-swap
teacher，包含 no-op。它们本轮都用 archive 查表，不能把查表速度算成真实 deploy speed。

         Cov12（增益加权十二候选覆盖）是每个 query 池内最佳正 gain 的总和，除以完整字典最佳正
gain 总和。Top1 recall 检查是否捕获任何并列最优 incoming；top4 recall 是捕获的 tied top-four 集
合比例；gain regret 是遗漏的 gain。零分母应 NA，不能装成 0 或 1。四对象 mean 等权，不让较
大 objective scale 支配结论。


                                     Method        smooth      near-contact    layered      asymmetric        Mean

                                     B0             0.9689          0.3528      0.9198              0.6589   0.7251
                                     B1 (s = 02)    0.9317          0.5809      0.9010              0.7647   0.7946
                                     B1 (s = 03)    0.9658          0.8326      0.3139              0.8541   0.7416
                                     B1 (s = 04)    0.9760          0.5426      0.9379              0.9287   0.8463
                                     B2             0.9718          0.5178      0.9852              0.6751   0.7875
                                     B3             0.7283          0.4142      0.7644              0.5969   0.6259
                                     B4             1.0000          1.0000      1.0000              1.0000   1.0000
                                     B5             1.0000          1.0000      1.0000              1.0000   1.0000



                                                   B0 prefix   B1 random mean         B2 O/P/M          B3 signed




      Gain-weighted coverage
                                1




                               0.5




                                0
                                      smooth               near-contact                  layered                    asymmetric


图 1: 覆盖图。B0 复核历史瓶颈；B2 对 near-contact/layered 有改善；B3 没有把有符号 addition
transfer 变成更好的材料动作。random 均值不掩盖三个 seed 的完整表。

         第一个发现：完整 anchor 排序够好，incoming pool 才是缺口。 B0 分别为 smooth 0.9689、
near-contact 0.3528、layered 0.9198、asymmetric 0.6589；B4 anchor top1 capture 依次为 0.9960、
0.9810、0.9937、0.9883。仅提高排序并不能把未进入池子的 atom 变出来。

                                                                          7
NN-native current design: Stage 0                                                3 October 2026


    第二个发现：廉价物理特征能改善，但没有关闭 opportunity。 B2 把 near-contact 从 0.3528
提高到 0.5178，把 layered 提高到 0.9852，却仍失败于 near-contact0.70、asymmetric0.85、mean0.85
三门槛。signed B3 的 mean0.6259 是需要保留的失败：更复杂的物理 proxy 不保证更好材料 utility。

    第三个发现：随机 seed 的对象间不稳定，不能用事后 routing 虚构解析胜利。seed03 使 near-
contact 达到 0.8326、asymmetric0.8541，但 layered 只有 0.3139；seed04mean0.8463，near-contact
只有 0.5426。单个固定 cheap baseline 到 teacher 的 mean gap 为 0.1537。这提供 A 的试验价值，
不保证 NN 能达到 teacher。

    下面表列出并列最优相关的 recall 和原始 objective 中的 regret；完整三随机 seed 与每 query
记录在 CSV。

                          Object           Method   Top1    Top4    Regret sum

                          smooth           B0       0.222   0.111     4.79e-05
                          smooth           B2       0.222   0.167     4.34e-05
                          smooth           B3       0.000   0.000     4.18e-04
                          smooth           B4       1.000   1.000     0.00e+00
                          near-contact     B0       0.000   0.000     1.94e-04
                          near-contact     B2       0.111   0.083     1.45e-04
                          near-contact     B3       0.000   0.028     1.76e-04
                          near-contact     B4       1.000   1.000     0.00e+00
                          layered          B0       0.222   0.250     2.08e-04
                          layered          B2       0.444   0.278     3.85e-05
                          layered          B3       0.000   0.028     6.11e-04
                          layered          B4       1.000   1.000     0.00e+00
                          asymmetric       B0       0.000   0.056     1.77e-04
                          asymmetric       B2       0.000   0.111     1.69e-04
                          asymmetric       B3       0.000   0.000     2.09e-04
                          asymmetric       B4       1.000   1.000     0.00e+00

    每项 F1 都单独计费，joint warm median 4.60 ms/query；共用 receiver step 是公共 reduced
GN，child GN/Qhat 是 shortlist 之后的 F2。Gamma normalization、source encoder 和失败尝试
也进入账本。这些 warm 投影时间不含 cold 物理 workspace acquisition。



                    7    B 的向量标签：什么能够恢复，什么不能

    Measurement label（测量响应标签）是 zF = JF d。Unresolved-current label（未解析电流标签）
则需要完整 current response，例如 L−1 IU Bd。测量算子有很大的 nullspace，所以即便存了 zF ，也
不等于存了 full current。

    Gaussian 有 saved fullJ，材料 gauge 和 d 一致时可以直接计算 JF d，没有新增 full-wave RHS。
voxel 只有 V ∈ R3456×16 与 fullJV；先算

                                         ηV = d − V V + d / ∥d∥ .

只有 ηV ≤ 10−9 才能用 fullJV 的同一组合恢复真响应。当前 voxel directions 的 ηV 在 0.9948–0.9992，

                                                    8
NN-native current design: Stage 0                                                       3 October 2026


几乎全部在探针 span 外；不能把 projection 的响应当 full response。

      另外一个合法渠道是保存的 step/Js：若 receiver x 与 child s 都有同一 frozen state 的 full
response，JF (s − x) = JF s − JF x。也可以检验 x 是否落在多个合法 saved steps 的 span 内，并只
在支撑容差通过时作线性组合。全 snapshot 审计中只有 asymmetric late k = 8 receiver 支持这种
恢复，得到一个 matched 向量。


       Object         Legal matched vectors    Missing matched vectors       Full-current labels

       smooth                            101                             0                    0
       near-contact                      103                             0                    0
       layered                             0                          96                      0
       asymmetric                          1                          98                      0


一条 voxel 向量不能形成对象级 POD/margin benchmark；它作为单独 inventory 证据保留。scalar
Qtrue 、step hash、norm、information RHS 都不能补成不存在的 JF d 或 full-current teacher。



             8 核心推导三：anchor tail 与 source/latent oracle

8.1    把已知部分从学习目标中拿掉


                              (JF − JU )d = (JA − JU )d + (JF − JA )d.
                                                   dA 。直接预测 defect 则
第一项是内层已知校正，第二项是 anchor tail。预测 tail 后用 ẑ = JA d + ∆z
              dU 。
是 ẑ = JU d + ∆z

      tail/direct norm 的中位数 smooth0.3729、near-contact0.4356，说明这个拆解能降低目标幅度。
但 A 和 U 不嵌套；小幅度不保证所有方向小，也不保证其跨对象 orientation 容易学。


8.2    POD 与固定 ridge 的实际步骤

      POD（proper orthogonal decomposition，正交主成分分解）把训练 snapshots 的主要变化写
成正交方向。每个合法 train 对象提供匹配向量，组成 T = [∆z1 , . . . , ∆zn ]，不减均值；SVD 为
T = UT ΣVTT 。取前 q = 8, 16, 32 列 Uq 。held-out 最理想的 L2 表示是 Uq UqT ∆z，它是 POD exact-
latent projection oracle。

      固定 ridge 使用 public source sketches H，用 train-only RMS 做行尺度归一化 Hn ，不设
intercept。latent training target Z = UqT T ，

                        W = ZHnT (Hn HnT + αI)−1 ,      α = 10−6 σmax (Hn )2 .

实现用等价较小的 dual normal solve。预测 Uq W hn 。固定 state 的 zero/scaling/additivity 成立，因
为没有 bias 或 direction-dependent 非线性。这个 baseline 的 W 不随 state 变化；若它失败，不能
单独归罪于 source sketch，也不能据此否定 state-conditioned NN。


                                                 9
NN-native current design: Stage 0                                                3 October 2026


      Fold1 只用 smooth101 方向拟合 POD/scaler/ridge，对 near-contact103 方向 test；Fold3 由
near-contact 训练，对 smooth101 方向 validation。Fold0 没有完整 Gaussian train 对象，Fold2 没
有 layered 完整 test，Fold3 没有完整 asymmetric test。完整四折 split 是保留的，B 可用 test 只有
规定的一个 pair；不能改 split 后称完成四对象。


8.3    source compression oracle 检查的是信息条件

      本轮的 source 是 projected anchor injection 的 physical-row random sketch，固定 seed 对同
一物理网格使用同一 encoder，而不把材料坐标编号当跨对象语义。每 illumination 的 complex row
数为 8/16/32，即总 real features96/192/384。两个 Gaussian 的 physical rows 经核对一致。

      对某个固定 state，记 full 材料坐标到 source sketch 的线性 map 为 H，真 tail operator 为
TA = JF − JA 。存在某个线性 W 满足 W H = TA 的必要信息条件是

                                            ker H ⊆ ker TA .

如果 H 有 full column rank，就可以在这个固定 gauge 上取 W = TA H + ；这只是 evaluator oracle，
需要 fullJ，不是 student 可获得的 public 模型。我们计算 TA (I − H + H)d 检查 source 的不可恢复
部分。

      在 Gaussian 保存的 54 维 real material gauge 里，所有 encoder ranks 的数值 rank 都是 54，
null-tail error 最大 relative 4.47 × 10−15 。这说明对已存 gauge 的线性 source 诊断没有看到信息损
失。它没有检查完整 off-Q injection/current，也没有证明小 latent output basis 能泛化。3456 维
voxel 的任意 material directions 更不能借这个 54 维结果下结论。



                 9 核心推导四：gain、error radius 与费用门槛

9.1    从 full quadratic objective 推导 gain

      固定 state 的真实二次目标可写为

                                ΦF (s) = 21 ∥r + JF s∥2 + ℓT s + 12 sT Λs.

从 receiver x 移到 child x + d，gain 为 Q = ΦF (x) − ΦF (x + d)。令 u = r + JF x、z = JF d，展开
得到
                             Q = −uT z − (ℓ + Λx)T d − 21 (∥z∥2 + dT Λd).

这里所有 inner products 在保存的 physical material/data gauge 中。它是 frozen-state gain，不是
新跑的 nonlinear reconstruction 质量。


9.2    为什么 response error 必须服从实际 margin

      预测 ẑ，误差 e = z − ẑ。在评价侧精确 reference u 下，

                              Q − Q̂ = −(u + ẑ)T e − 12 ∥e∥2 ,
                            |Q − Q̂| ≤ vϵ + 21 ϵ2 ,    v = ∥u + ẑ∥ , ∥e∥ ≤ ϵ.


                                                      10
NN-native current design: Stage 0                                                                              3 October 2026


要保证 gain 下界超过阈值 τ ，需要 m = Q̂ − τ > 0 且 vϵ + ϵ2 /2 < m。解二次不等式的正根，得到
                                                              p                         2m
                                                    ϵcrit =       v 2 + 2m − v = √                  .
                                                                                     v 2 + 2m + v

这里的 Q̂ 使用精确评价侧 u，只将 z 替换为 ẑ；A 中 anchor score 使用 r +JA x，二者不能混同。后一
个表达式避免 m 很小时两个接近数字相减。在本轮用 actual residual 检查 ∥JF d − ẑ∥ < ϵcrit ，称为
oracle can skip（反事实可跳过）
                      。在线缺少这个精确 residual envelope，所以不是已实现 certificate
或实际跳过。

                                                            Tail POD oracle                  Tail ridge
                                                           Direct POD oracle              Opportunity gate
                                                        Matched RHS-halving floor




      Counterfactual skip fraction
                                      1

                                     0.8

                                     0.6

                                     0.4

                                     0.2

                                      0
                                           0            8                      16                                   32
                                                              POD output rank (0: uncorrected)


图 2: 同一 18 个 B4 finalists 的误差–margin 结果。横轴 0 为未补偿 reduced response；rank8/16/32
训练对象 POD 不能增加 tail 分支的 skip count。所有 primary finalists true gain 均超过 floor。


                                               Primary test B4              rank8      rank16    rank32

                                               Direct POD oracle               0/18      0/18           0/18
                                               Anchor-tail POD oracle       10/18       10/18       10/18
                                               Anchor-tail ridge            10/18        9/18           9/18


未补偿 anchor 也为 10/18。rank32 B4 median tail-relative error 仍是 0.8868，median error/critical-
radius 为 0.5413；中位数通过不代表足够多的动作通过。B2 rank32 tail 只有 7/18。reversed smooth
的结果是 Fold3 validation，其 tail POD 均为 8/18。

      POD projection 在固定 output subspace 中使 L2 response residual 最小，但 ϵcrit 自身依赖 ẑ；它
不必最大化所有可能 latent 系数的 skip count。因此 B 的 NO-GO 必须限于本轮 tested POD/ridge
pipeline，不能声称任意 compensation 机制被数学证明不可能。


9.3       约 60% 与 75% 为什么并不矛盾

      每个 shared direction 需要六个 full RHS。令初始化总 RHS 为 I，finalist 验证为 R，理想 skip
比例为 ρ。若暂不计模型与 refresh，旧费用 I + R，新费用 I + (1 − ρ)R。要求新费用不超过旧费
用的一半：
                                               I + (1 − ρ)R ≤ 21 (I + R)       =⇒      ρ ≥ 21 (1 + I/R).

                                                                          11
NN-native current design: Stage 0                                          3 October 2026


历史 A17 workload 为 I = 36×6 = 216、R = 1164，floor 为 59.28%。本次 primary single-decision test
为 9queries、每 query 两 finalists：I = 54、R = 108，floor 为 75%。用 55.6% 的 oracle opportunity，
新 ratio 为
                                    54 + 108(1 − 10/18)
                                                        = 0.6296.
                                          54 + 108
这既没减半，也还没算额外费用。若 refresh/audit/failed RHS 为 E，同样推得 floor 增加 E/R；模
型/source/cold setup 的 wall 费用还要另加。只说 “少验证一半 finalists” 会遗漏初始化，不能算完
整流程两倍。


9.4    缓存误差与统计边界

      未来连续 learned acceptance 若更新 cached Jx，旧误差会叠加。下一次 exact Jd 只给出新
direction response，不修复旧 Jx。必须跟踪 cache radius、refresh/audit 以及 selected-finalist 的校
准分布。四个对象及其相关 moves 不足以自动构成 95% safety calibration。A17 提供 high-fidelity
numerical labels，没有物理 deterministic error envelopes；本轮所有 skip 都是离线反事实诊断。



                            10      验证、失败、文献与复现边界

      新写 128 个非 Hermitian complex cases 检查了两侧 defect、Schur tail、anchor decompo-
sition、phase covariance、ID permutation、一般 gain-error bound 及 exact-u 强化界；identity
最大 relative residual 9.69 × 10−16 ，gain-bound violation 为 0。四对象 early k = 8 public
接口通过 phase/permutation 动作一致性和 forbidden-field canary；补偿 source/ridge 通过
zero/scaling/additivity。这些检查针对已实现解析接口与 ridge，没有证明尚未实现的 NN 架构合
格。

      原 A18 validation script/config/results 和独立 REFERENCES 原件未提供。我从 merged read-
ing 版恢复 REFERENCES，另写 equivalent algebra tests，没有复现原版 “128PASS”。第一次 phase
tester 错用了 SVD 前后的不同 gauge；parser、adapter 和 serialization 错误均修复并保留日志及
resource cost。最终 16 个 upstream Woodbury units 有 normal residual failure，用同一 normal
equation 的一次固定 residual correction 降到原 10−10 门槛以下；没有改变 vendor、物理 operator、
λ 或 jitter。这是等价数值验证，不是新的 solver 研究方向。

      文献提示“小 current correction”、operator correction、derivative compression 已有基础 [1,
2, 3, 4, 5]。尤其 Lunz 等指出，仅 data-space correction 未必获得 gradient/adjoint 一致性 [3]；本
轮 fixed-direction gain 通过也不自动扩展成完整 inverse trajectory 保证。certified ROM 另需合法
error control [6]。三条 ScholarQA queries 部分 HTTP429，但七个 citation verification 全 resolved；
SOM-Net 方法与 Lunz 第 3–4 节读取 primary full text，其他文献按 abstract/metadata 分级，PEC
graph 全文细节 OPEN。未作 first-ever 或 exhaustive novelty 结论。

      复现入口 START_HERE.md 指向固定 source、config、query/move IDs 和全部 derived vectors。
九项 mandatory 审计结果以 CSV/JSON/Markdown 保存；claim/reference/failure ledger 保证三文
稿共享同一证据。图表有 PNG/SVG 及 CSV，build_report_assets.py 可重画。LaTeX 中的图
是同一数据内嵌 coordinates，三份文件均可独立编译，不依赖额外图文件。


                                                 12
NN-native current design: Stage 0                                          3 October 2026


                        11     最终理解与下一阶段所需最少证据

    A：GO 只意味着建议 tiny pilot。完整字典的 high anchor capture 验证了 candidate bot-
tleneck；B2 没关闭困难对象缺口。未来 MLP/Set Transformer/GAT/phase-consistent physics-edge
GNN 须共同输入预算、最多 200k parameters/60epochs、4fold 两 seed、总 GPUoccupation 两小
时上限；先兑现 near-contact0.70、asymmetric0.85、mean0.85，再检查额外 0.05 或同质量成本减
半。本轮 config enabled=false。

    B：全范围 INCOMPLETE；当前 Gaussian pipeline 没有额外 margin/cost 空间。需
要先有 matched voxel JF d 与精确 receiver output 的 provenance，再有一个能改变 train-only out-
put/margin ceiling 的 state-conditioned 且对 d 线性的 representation。不会因 A 通过而自动训练
B，也不会为了绕过失败转成 Flow/U-Net 或新 solver。

    所有 actual runs 与 not-run 项目都在 decision ledger。Stage0 完成后停止：new full-wave RHS、
NN epochs、new nonlinear trajectories、GPUtraining 均为 0；没有统计安全认证、真实在线 skip、
整流程实测加速、远端发布或自动后续训练。



                              A 中英文术语与容易混淆的量

 English                       中文             本轮含义

 Material gauge                材料坐标规约         保存的 physical orthonormal tangent 及 fin-
                                              gerprint；step/ell 不能换 basis 混用。
 Total regularization          总二次正则          保存的 Λ 包含 prior 与 LM damping；不是只
                                              用 prior precision。
 Current embedding             电流嵌入空间         公共 Q 的 128 维 span；不等于完整 current
                                              空间。
 Shared material step          共享材料步          六 illumination 共用同一 x, d。
 Receiver query                接收基起点决策        完整物理上下文加 selected IDs、公共 step、an-
                                              chor/prior/layout。
 Dictionary atom ID            字典原子身份         原始物理动作 ID，和 QR/SVD 列编号不同。
 Shortlist                     短候选池           本轮固定十二 incoming。
 Anchor                        丰富降阶参照         用于 post-shortlist ranking；full dictionary 昂
                                              贵参照 B4。
 One-swap teacher              单交换监督参照        所有合法 drop/add 中的最大正 gain，含 no-
                                              op；不是全局 optimum。
 No-op                         不改变当前选择        utility0，本轮 36query 中未成为最优。
 Cov12                         增益加权覆盖         池内最佳正 gain 总量相对完整字典总量。
 Gain regret                   增益损失           完整 teacher 最好正 gain 减 shortlist 最好正
                                              gain。
 Dressed observation           修饰观测响应         Y = S C̄，扣除了 retained 耦合。
 Projected injection           投影注入           只从 saved PB/L/U 获得的 current-source 信
                                              息。
 Schur complement              Schur 补        一般 complex block elimination；不假设 Her-
                                              mitian。


                                         13
NN-native current design: Stage 0                                                    3 October 2026



 English                       中文                    本轮含义

 Phase covariance              相位协变                  N ′ = E ∗ N, Γ′ = E ∗ ΓE, Y ′ = Y E。
 Measurement defect            测量响应缺陷                ∆z = (JF − JU )d，不是 full current。
 Inner correction              已知内层校正                (JA − JU )d，本轮无需学习。
 Anchor tail                   参照剩余误差                (JF − JA )d，richer-model 之后剩余 target。
 POD                           正交主成分分解               train-only 未中心化 snapshots 的 left singular
                                                     vectors。
 Source oracle                 输入信息诊断                evaluator 检查 T (I − H + H)，不是 public pre-
                                                     dictor。
 Latent projection oracle      潜在系数表示诊断              fixed training basis 下用真实系数投影 held-
                                                     out response。
 Gain margin                   决策收益余量                m = Q̂ − τ ；决定所需 response accuracy。
 Critical error radius         临界误差半径                ϵcrit ；不是 ensemble sigma 或已有证书。
 Counterfactual skip           反事实跳过机会               使用 actual offline error，不是实测少跑 RHS。
 Initialization / refresh      初始化/刷新                建立或修复 cached Jx，必须进入总费用。
 Data incomplete               数据不完整                 没有合法 labels；与有数据的科学负结果分开。




                                          参考文献

[1] Z. Wei and X. Chen, “Physics-Inspired Convolutional Neural Network for Solving Full-Wave
   Inverse Scattering Problems,” IEEE TAP, 2019. doi:10.1109/TAP.2019.2922779. Abstract-level
   evidence in this study.

[2] Y. Liu et al., “SOM-Net: Unrolling the Subspace-based Optimization for Solving Full-wave
   Inverse Scattering Problems,” 2022. arXiv:2209.03567. Method and Algorithm 1 inspected.

[3] S. Lunz et al., “On Learned Operator Correction in Inverse Problems,” 2020 preprint.
   arXiv:2005.07069; journal DOI 10.1137/20M1338460. Sections 3–4 inspected.

[4] T. O’Leary-Roseberry et al., “Derivative-Informed Neural Operator: An Eﬀicient Frame-
   work for High-Dimensional Parametric Derivative Learning,” 2022 preprint, revised 2023.
   arXiv:2206.10745. Abstract-level evidence.

[5] L. Cao et al., “Residual-based error correction for neural operator accelerated infinite-
   dimensional Bayesian inverse problems,” 2022 preprint. arXiv:2210.03008. Abstract-level evi-
   dence.

[6] B. Haasdonk et al., “A new certified hierarchical and adaptive RB-ML-ROM surrogate model
   for parametrized PDEs,” 2022. arXiv:2204.13454. Abstract-level evidence.




                                                14

