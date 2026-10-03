"""Seal the reviewed Stage 0 delivery; this performs no new experiment."""
import json,csv,hashlib,ast
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while part:=f.read(1024*1024):h.update(part)
    return h.hexdigest()
required=['A18_DATA_INVENTORY.csv','A18_QUERY_MANIFEST.jsonl','A18_SPLITS.json','A18_FEATURE_COSTS.csv','A18_ANALYTIC_SHORTLIST_BASELINES.csv','A18_DEFECT_LABEL_AVAILABILITY.csv','A18_DEFECT_ORACLE.csv','A18_COST_LEDGER.jsonl','A18_STAGE0_DECISION.md']
assert all((ROOT/p).is_file() and (ROOT/p).stat().st_size>0 for p in required)
assert read('verification/REPRODUCTION_VALIDATION.json')['status']=='PASS'
assert read('verification/FINAL_SOURCE_INTEGRITY.json')['status']=='PASS'
assert read('verification/algebra_results.json')['status']=='PASS'
assert read('verification/INTERFACE_CANARY_RESULTS.json')['status']=='PASS'
qa=read('qa/PDF_QA_RESULTS.json');assert len(qa)==3 and sum(r['pages'] for r in qa)==28
for r in qa:
    assert r['visual_review']=='PASS_ALL_PAGES_VIEWED'
    assert all(p['visual_review']=='PASS' for p in r['page_records'])
    assert sha(ROOT/r['source'])==r['source_sha256'] and sha(ROOT/r['pdf'])==r['pdf_sha256']
native=read('qa/NATIVE_COMPILER_FINAL_RESULTS.json')['records'];assert len(native)==3
for r in native:assert r['kind']=='success' and sha(Path(r['path']))==r['source_sha256']
cfg=read('stage0_config.json');assert sha(ROOT/'stage0_config.json')==read('CONFIG_FREEZE_RECEIPT.json')['final_config_sha256']
assert read('FUTURE_TRAINING_CONFIG.json')['enabled'] is False
summary=read('results/STAGE0_SUMMARY.json');resources=read('results/RESOURCE_SUMMARY.json');counts=read('results/COUNTS.json')
assert counts['unique_queries']==36 and counts['unique_moves']==47712 and counts['feasible_source_rows']==43108
assert resources['all_metered_process_cpu_s']<cfg['cpu_budget_s'] and resources['peak_rss_bytes']<=cfg['max_rss_bytes']
for p in [*(ROOT/'code').glob('*.py'),*(ROOT/'verification').glob('*.py')]:ast.parse(p.read_text(),filename=str(p))
status=dict(status='COMPLETE_STAGE0',completed_at_utc=datetime.now(timezone.utc).isoformat(),
            source_commit=cfg['source_commit'],config_sha256=sha(ROOT/'stage0_config.json'),
            A_status=summary['A_status'],B_status=summary['B_status'],B_Gaussian_primary_status=summary['B_Gaussian_primary_status'],
            completed_queries=36,unique_moves=47712,feasible_moves=43108,legal_direction_labels=205,
            required_audits=9,manuscripts=3,pdf_pages_reviewed=28,native_compilation='ALL_CURRENT_SOURCES_SUCCESS',
            independent_numerical_reproduction='PASS_EXACT_SCIENTIFIC_VALUES',resources=resources,
            nn_training_epochs=0,new_full_wave_rhs=0,new_nonlinear_trajectories=0,gpu_training=0,
            future_training_enabled=False,stop_after_stage0=True,scope_mismatch=False,budget_stop=False,
            scientific_limitations=['four physical objects; correlated moves','B globally data-incomplete','tested Gaussian POD/ridge result only','no statistical safety calibration or measured speedup','novelty search partial'])
(ROOT/'STATUS.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
(ROOT/'PROGRESS.md').write_text('''# Stage 0 完成与恢复入口

1. 重读 inputs/A18_CODEX_PROMPT.md、START_HERE.md 和 STATUS.json。
2. 本轮已完成并停止：九项审计、A/B独立裁决、三份TeX/PDF、28页视觉检查、全来源哈希和独立数值复现。
3. A_GO_TINY_PILOT仅为下一阶段建议；B_DATA_INCOMPLETE与Gaussian所测表示的negative结果分列。不得因恢复/compact自动开始训练或补标签。
4. 新full-wave RHS、NN epochs、new nonlinear trajectories、GPU training均为零。后续FUTURE_TRAINING_CONFIG enabled=false。
5. 主线程为scientific owner，provider/model未变；所有失败与费用保留。改变科学输入或文稿后须重新绑定相应验证与最终哈希，不能沿用旧QA。
''')
notes='''# Codex 主线程最终科学复核

本记录表示Codex对这一轮Stage 0证据与结论的复核，不是陈教授、用户、审稿人或TAP的acceptance。

- 来源：pinned commit、公开manifest全5556项、原始42模块及36teacher hashes各自通过；所有consumed inputs未改变。REFERENCES恢复与原验证文件缺失明确区分。
- 合同：保存的physical material gauge、六illumination、原始ID、realification、lambda_total与ell均保持；Lambda包含prior和LM damping。Public projected full solve不冒认完整物理truth。
- A：完整字典未截断；B0瓶颈复核；B2的局部改善和B3的负结果并列，随机seeds全报；机会空间支持tiny pilot建议，不支持NN superiority或TAP novelty宣称。
- B：Gaussian合法fullJ与voxel方向支撑逐条核验，saved-step恢复只有一条voxel方向；measurement/current labels不混用。Fold1是规定primary test，Fold3是validation；没有迁移split。
- Oracle：POD/RMS/ridge仅用train对象，source oracle使用评价侧fullJ并明确状态限制。L2 projection不保证全空间最高skip-cardinality；负裁决限定tested pipeline。
- Margin/cost：精确评价侧u与A anchor residual区别明确；10/18 counterfactual skips没有增加anchor机会，且未过60%/75%两个不同gate。初始化、refresh/audit、cache accumulation、模型/source/cold成本和失败均记录。
- 验证：新等价non-Hermitian algebra与真实接口测试通过，未冒称原版复现或未实现NN architecture已合格。独立复现的科学数值完全一致，计时与PDF字节不作相同保证。
- 交付：三份当前TeX原生编译成功，已有Tectonic导出，Poppler全28页实际查看；图表负例保留。后续训练配置disabled，Stage0结束。

仍开放：194条matched voxel vectors、full-current teachers、statistical safety calibration、native architecture实测优势、全流程加速、closest PEC graph全文细节及global novelty。
'''
(ROOT/'verification/CODEX_FINAL_SCIENTIFIC_REVIEW.md').write_text(notes)
decision=ROOT/'A18_STAGE0_DECISION.md'
extra='''
## 最终交付封存

三份正式TeX/PDF为8/6/14页，原生compiler成功、28页逐页查看；独立目录复现的全部科学数值一致。最终source/PDF/render、来源与交付文件hash见qa、verification和DELIVERY_MANIFEST。

Lambda的文字定义已统一：原lambda_total包含prior与LM damping，数值实现始终使用保存值。全public manifest5556项、42个raw/pinned module hashes、36份raw teacher核验通过，597个consumed inputs保持原hash。

费用主账按进程envelope加总，nested component/replay只作为视图；累计数值CPU49.113秒，所有已计时进程CPU80.294秒，失败进程CPU19.095秒，峰值RSS884965376bytes。Native UI compiler CPU未在shell仪表中。新增物理actions与NN训练均为0；没有把离线warm计时当作实测加速。

Status为COMPLETE_STAGE0；本轮到此停止，后续配置enabled=false。
'''
if '## 最终交付封存' not in decision.read_text():decision.write_text(decision.read_text()+extra)
entries=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or p.name in ['DELIVERY_MANIFEST.json','.DS_Store'] or '__pycache__' in p.parts:continue
    entries.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
manifest=dict(schema='a18.stage0.delivery.v1',created_utc=status['completed_at_utc'],
              source_commit=cfg['source_commit'],configuration_sha256=status['config_sha256'],
              excludes=['self','OS metadata','Python bytecode'],files=entries)
(ROOT/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(status=status['status'],audits=9,manuscripts=3,pages=28,files_hashed=len(entries),numerical_cpu_s=resources['numerical_cpu_s'],peak_rss_bytes=resources['peak_rss_bytes'])))
