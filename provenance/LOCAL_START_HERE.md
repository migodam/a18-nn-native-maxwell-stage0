# NN Native Stage 0 交付入口

Stage 0 已完成并停止。A 为 `A_GO_TINY_PILOT`，只建议下一阶段；B 四对象范围为 `B_DATA_INCOMPLETE`，有完整 Gaussian 标签部分的当前 POD/ridge pipeline 不显示额外 margin/cost 空间。NN epochs、新 full-wave RHS、新 nonlinear trajectories、GPU training 均为0；后续配置为 disabled。

建议先读 [完整裁决](A18_STAGE0_DECISION.md)，然后读 [英文研究稿](manuscript/STAGE0_RESEARCH_EN.pdf)、[给陈教授的中文汇报](manuscript/REPORT_TO_PROF_CHEN_ZH.pdf) 或 [中文完整讲解](manuscript/COMPLETE_EXPLANATION_ZH.pdf)。三稿分别为8、6、14页，均附同名 `.tex`；独立源文件含内嵌图，无额外 LaTeX project files。

## 主要结果

完整盘点为36 queries、47,712 unique moves、43,108 feasible moves、5208 incoming labels。B0复核原有瓶颈；B2 near-contact覆盖由0.3528提升至0.5178，但未达到0.70。B3 mean0.6259的负结果保留；全部固定廉价对手均未同时通过覆盖门槛，A仍有值得检验的候选生成机会。

B有204条完整 Gaussian matched measurement directions和一条孤立 voxel direction；194条 voxel matched vectors缺失，full-current teachers为零。规定 Fold1 test 的18个 B4 finalists，anchor-tail POD ranks8/16/32均反事实通过10/18，与未补偿 anchor相同；direct POD为0/18。55.6%既未到60%机会门槛，也未到本次含初始化的75% RHS减半要求。没有实际跳过物理求解或实测加速。

保存的 `lambda_total` 包含 prior precision 与 LM damping，step、gain和material gauge原值均保留。公共投影算子不等于 full Maxwell truth；scalar Q不等于方向向量标签。B的negative诊断只限于所测表示，缺数据独立记录。

## 九项指定审计及共同证据

| 文件 | 用途 |
|---|---|
| [A18_DATA_INVENTORY.csv](A18_DATA_INVENTORY.csv) | shape、dtype、gauge、来源/角色与缺失文件 |
| [A18_QUERY_MANIFEST.jsonl](A18_QUERY_MANIFEST.jsonl) | 完整物理 query身份、原始ID、字典、floor和feasibility |
| [A18_SPLITS.json](A18_SPLITS.json) | 四折，每折18 train/9 validation/9 test queries |
| [A18_FEATURE_COSTS.csv](A18_FEATURE_COSTS.csv) | 每项F1独立费用；补偿source另见results |
| [A18_ANALYTIC_SHORTLIST_BASELINES.csv](A18_ANALYTIC_SHORTLIST_BASELINES.csv) | 288条query×method记录；三个random seeds全报 |
| [A18_DEFECT_LABEL_AVAILABILITY.csv](A18_DEFECT_LABEL_AVAILABILITY.csv) | measurement/current标签及合法方向支撑 |
| [A18_DEFECT_ORACLE.csv](A18_DEFECT_ORACLE.csv) | 2856条误差、margin、εcrit与反事实结果 |
| [A18_COST_LEDGER.jsonl](A18_COST_LEDGER.jsonl) | 实际进程、失败、nested费用和反事实成本式；避免双计 |
| [A18_STAGE0_DECISION.md](A18_STAGE0_DECISION.md) | A/B独立裁决、机制、缺口、停止点 |

[CLAIM_LEDGER.csv](CLAIM_LEDGER.csv)、[文献账本](literature/REFERENCE_LEDGER.csv)、[FAILURE_LEDGER.csv](FAILURE_LEDGER.csv)为三稿的共同证据。对象汇总与source oracle在`results/`；四张可重画图及CSV、PNG、SVG在[figures/](figures/README.md)。

## 来源与验证范围

固定上游：[repository](https://github.com/migodam/a17-maxwell-certified-exchange/tree/56097653e6a814399581c3759fc3e3e8f0c2284c)，[固定 raw入口](https://raw.githubusercontent.com/migodam/a17-maxwell-certified-exchange/56097653e6a814399581c3759fc3e3e8f0c2284c/START_HERE.md)。没有发布新的公开研究仓库。

[FINAL_SOURCE_INTEGRITY.json](verification/FINAL_SOURCE_INTEGRITY.json)记录pinned Git manifest字节哈希、全部5556项public manifest、597项consumed inputs及36份raw teacher核验；42个原始module hashes同时匹配原始与pinned公开module registry。原始和整理后的manifest字节不同，分别核验。REFERENCES从合并阅读版恢复；原A18脚本/config/results缺失，新验证是等价重写。

128个非Hermitian algebra cases、公共phase/permutation/ID/canary和ridge线性检查通过。[独立复现](verification/REPRODUCTION_VALIDATION.json)核对全部数值表及295份派生数组，科学数值完全一致；计时不要求相同。它不是新增独立物理场景。

三份当前源文件均已由原生LaTeX compiler成功编译；使用已有Tectonic导出并以Poppler逐页查看28页PDF。[PDF QA](qa/PDF_QA_RESULTS.json)绑定最终source/PDF/render hashes。计算资源见[RESOURCE_SUMMARY.json](results/RESOURCE_SUMMARY.json)：失败进程也入账；数值CPU约49.1秒，所有已计时进程约80.3秒，峰值RSS约0.824GiB。Native UI compiler的CPU不在shell进程仪表中。

## 复现入口

保留现有交付，输出必须是`reproductions/`下一个不存在的新目录。以下从工作区根目录运行，只重用固定本地snapshot/original文件：

```sh
cd "workspace://"
Gaussian/.venv_nn/bin/python A18_NN_NATIVE/code/reproduce.py \
  --output A18_NN_NATIVE/reproductions/another_offline_check --pdf
```

复现顺序为algebra、36-query全字典审计、合法label/POD/ridge/source/margin、F1/source计费、canary、source integrity、图表、三份TeX以及可选Tectonic PDF。所有拟合避开test；没有NN训练或新物理actions。既有独立验证目录为`reproductions/verification_20261003/`，其测试在最终文字/排版修订之前完成；不宣称PDF逐字节相同。要重新进行PDF图像检查，需使用已提供的bundled Python运行`code/pdf_qa.py`，然后实际逐页查看；自动render不构成视觉通过。

[FUTURE_TRAINING_CONFIG.json](FUTURE_TRAINING_CONFIG.json)仅为下一阶段配置，`enabled=false`；A的GO不授权B训练。本轮交付后停止。
