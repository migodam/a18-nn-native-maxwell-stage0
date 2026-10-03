"""Single evidence ledger, durable decisions and replot-capable scientific figures."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[n]='1'
import csv,json,time,resource,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def readcsv(p):return list(csv.DictReader(Path(p).open()))
def jw(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def wc(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
cfg=json.loads((ROOT/'stage0_config.json').read_text());summary=json.loads((ROOT/'results/STAGE0_SUMMARY.json').read_text())
counts=json.loads((ROOT/'results/COUNTS.json').read_text());A=readcsv(ROOT/'results/A18_BASELINE_OBJECT_SUMMARY.csv');B=readcsv(ROOT/'results/A18_B_OBJECT_SUMMARY.csv')
qrows=[json.loads(s) for s in (ROOT/'A18_QUERY_MANIFEST.jsonl').read_text().splitlines()]
dirs=[json.loads(s) for s in (ROOT/'results/A18_DIRECTION_MANIFEST.jsonl').read_text().splitlines()]
def ar(method,oid,seed=''):return next(r for r in A if r['method']==method and r['object_id']==str(oid) and r['seed']==str(seed))
def br(base,model,rank,method='B4',fold=1,partition='test'):
    return next(r for r in B if r['fold']==str(fold) and r['partition']==partition and r['target']==base and r['model']==model and r['rank']==str(rank) and r['method']==method)
summary['B_Gaussian_primary_status']='B_NO_MARGIN_OR_COST_HEADROOM_FOR_TESTED_POD_BASES'
summary['B_scope']='PRESCRIBED FOLD 1 TEST + SEPARATE FOLD 3 VALIDATION COHORT; NOT FOUR-OBJECT B RESULT'
summary['B_recommendation']='DO_NOT_TRAIN_B_FROM_THIS_STAGE0; MISSING_VOXEL_LABELS_SEPARATE'
summary['A_recommendation']='CONFIG_ONLY; NEXT_PILOT_REQUIRES_EXPLICIT_NEXT_STAGE'
jw(ROOT/'results/STAGE0_SUMMARY.json',summary)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':220})
colors={'B0':'#6a7b8c','B1':'#a494b8','B2':'#267e9a','B3':'#cd715c','direct':'#cd715c','tail':'#267e9a'}
def savefig(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.png',bbox_inches='tight');fig.savefig(ROOT/'figures'/f'{name}.svg',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8.4,4.1));ids=cfg['object_order'];x=np.arange(4);width=.18
plot_rows=[]
for z,m in enumerate(['B0','B1','B2','B3']):
    if m=='B1':
        values=np.array([[float(ar(m,oid,s)['Cov12']) for oid in ids] for s in cfg['random_seeds']]);v=values.mean(axis=0)
        lo=v-values.min(axis=0);hi=values.max(axis=0)-v
        ax.bar(x+(z-1.5)*width,v,width,label='B1 random (3-seed mean/range)',color=colors[m],yerr=[lo,hi],capsize=3)
    else:
        v=[float(ar(m,oid)['Cov12']) for oid in ids];ax.bar(x+(z-1.5)*width,v,width,label={'B0':'B0 family prefix','B2':'B2 O/P/M union','B3':'B3 signed + diversity'}[m],color=colors[m])
    plot_rows.extend(dict(method=m,object=cfg['object_names'][str(oid)],coverage=float(v[i])) for i,oid in enumerate(ids))
ax.axhline(1,color='#404040',ls=':',label='B5 one-swap teacher');ax.scatter([1,3],[.70,.85],marker='_',s=700,c='#101010',label='Object-specific future gates')
ax.set_xticks(x,[cfg['object_names'][str(i)] for i in ids]);ax.set_ylim(0,1.07);ax.set_ylabel('Gain-weighted Cov12');ax.legend(ncol=2,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.27));fig.subplots_adjust(top=.77)
savefig(fig,'01_shortlist_coverage');wc(ROOT/'figures/01_shortlist_coverage.csv',plot_rows)

raw=readcsv(ROOT/'A18_ANALYTIC_SHORTLIST_BASELINES.csv');fig,axes=plt.subplots(1,3,figsize=(12,3.9),sharey=True)
regret_rows=[]
for ax,m in zip(axes,['B0','B2','B3']):
    arr=np.zeros((4,9))
    for i,oid in enumerate(ids):
        rr=[r for r in raw if r['method']==m and int(r['object_id'])==oid]
        for j,r in enumerate(rr):
            arr[i,j]=float(r['gain_regret'])/float(r['full_best_positive_gain'])
            regret_rows.append(dict(method=m,object_id=oid,phase=r['phase'],k=r['k'],normalized_gain_regret=arr[i,j]))
    im=ax.imshow(arr,vmin=0,vmax=1,cmap='magma_r',aspect='auto');ax.set_title(m)
    ax.set_xticks(range(9),['E4','E8','E16','M4','M8','M16','L4','L8','L16'],rotation=45)
    ax.set_yticks(range(4),[cfg['object_names'][str(i)] for i in ids])
fig.colorbar(im,ax=axes,label='Lost fraction of positive one-swap gain',shrink=.8,pad=.02);savefig(fig,'02_query_regret');wc(ROOT/'figures/02_query_regret.csv',regret_rows)

fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True);skip_plot=[]
for ax,(fold,part,oid) in zip(axes,[(1,'test',2007),(3,'validation',2001)]):
    for target,c in [('direct_defect',colors['direct']),('anchor_tail',colors['tail'])]:
        for model,ls in [('POD_EXACT_LATENT_ORACLE','-'),('FIXED_LINEAR_RIDGE','--')]:
            ranks=[8,16,32];values=[float(br(target,model,q,fold=fold,partition=part)['skip_fraction']) for q in ranks]
            ax.plot(ranks,values,marker='o',ls=ls,c=c,label=f'{target.replace("_"," ")} / '+('POD oracle' if model.startswith('POD') else 'ridge'))
            skip_plot.extend(dict(fold=fold,partition=part,object_id=oid,target=target,model=model,rank=q,skip_fraction=v) for q,v in zip(ranks,values))
    ax.axhline(.60,c='#888888',ls=':',label='Stage 0 opportunity gate');ax.axhline(.75,c='#222222',ls='-.',label='Matched RHS-halving floor')
    ax.set_ylim(-.03,1.02);ax.set_xticks([8,16,32]);ax.set_xlabel('POD output rank');ax.set_title(cfg['object_names'][str(oid)]+f' ({part}, fold {fold})')
axes[0].set_ylabel('Counterfactual accept-check skips');axes[1].legend(fontsize=6.8,loc='upper right');savefig(fig,'03_defect_margin_oracle');wc(ROOT/'figures/03_defect_margin_oracle.csv',skip_plot)

orows=readcsv(ROOT/'A18_DEFECT_ORACLE.csv');fig,axes=plt.subplots(1,2,figsize=(8.7,3.5))
norms=[]
for i,oid in enumerate([2001,2007]):
    ratios=[s['defectA_norm']/s['defectU_norm'] for s in dirs if s['object_id']==oid]
    axes[0].scatter(np.full(len(ratios),i)+np.linspace(-.12,.12,len(ratios)),ratios,s=8,alpha=.35,c=colors['tail'])
    axes[0].scatter([i],[np.median(ratios)],marker='_',s=1000,c='#121212')
    norms.extend(dict(object_id=oid,direction_index=s['index'],tail_over_direct=s['defectA_norm']/s['defectU_norm']) for s in dirs if s['object_id']==oid)
axes[0].set_xticks([0,1],['smooth','near-contact']);axes[0].set_ylabel('||anchor tail|| / ||direct defect||');axes[0].set_title('Known correction reduces target size')
margin_rows=[]
for rank in [8,16,32]:
    ss=[r for r in orows if r['fold']=='1' and r['partition']=='test' and r['target']=='anchor_tail' and r['model']=='POD_EXACT_LATENT_ORACLE' and r['requested_rank']==str(rank) and 'B4' in json.loads(r['methods'])]
    ratios=sorted(float(r['margin_ratio']) for r in ss if r['margin_ratio']!='')
    axes[1].step(ratios,np.arange(1,len(ratios)+1)/18,where='post',label=f'rank {rank}')
    margin_rows.extend(dict(rank=rank,direction_index=r['direction_index'],margin_ratio=r['margin_ratio'],positive_predicted_margin=float(r['actual_margin'])>0,oracle_can_skip=r['oracle_can_skip']) for r in ss)
axes[1].axvline(1,c='#222222',ls=':');axes[1].set_xscale('log');axes[1].set_xlabel('Response error / epsilon_crit');axes[1].set_ylabel('Fraction of all 18 finalists');axes[1].legend(fontsize=8);axes[1].set_title('near-contact test, B4 finalists')
axes[1].set_xticks([.2,.5,1.,2.],['0.2','0.5','1','2']);axes[1].minorticks_off();axes[1].set_ylim(0,1)
savefig(fig,'04_tail_size_and_margin');wc(ROOT/'figures/04_tail_norms.csv',norms);wc(ROOT/'figures/04_margin_ratios.csv',margin_rows)

feature_rows=readcsv(ROOT/'A18_FEATURE_COSTS.csv');times={}
for r in feature_rows:times.setdefault(r['query_id'],0.);times[r['query_id']]+=float(r['wall_s'])
med_ms=statistics.median(times.values())*1000;max_ms=max(times.values())*1000
cost_detail=[]
for q in qrows:
    rr=[r for r in feature_rows if r['query_id']==q['query_id']]
    total=sum(float(r['wall_s']) for r in rr)
    b2excluded={'Schur_gamma','signed_transfer_vectors','public_anchor_response_receiver_step','signed_transfer_alignment'}
    b3excluded={'O_normalization','P_normalization','M_normalization'}
    cost_detail.append(dict(query_id=q['query_id'],joint_F1_wall_s=total,
        B2_charged_feature_components_wall_s=sum(float(r['wall_s']) for r in rr if r['feature'] not in b2excluded),
        B3_charged_feature_components_wall_s=sum(float(r['wall_s']) for r in rr if r['feature'] not in b3excluded),
        scope='WARM_PROJECTED_COMPONENT_TIMINGS; common Python overhead and source acquisition separate'))
wc(ROOT/'results/A18_METHOD_FEATURE_COSTS.csv',cost_detail)

claims=[
 ('C01','VERIFIED_DATA','36 complete queries, 47712 unique moves, 43108 feasible moves, 5208 incoming labels','results/COUNTS.json','Four objects, twelve frozen states; moves are correlated'),
 ('C02','REANALYZED','B0 Cov12=0.9689/0.3528/0.9198/0.6589; original short-pool bottleneck reproduced','results/A18_BASELINE_OBJECT_SUMMARY.csv','One-swap dictionary scope'),
 ('C03','REANALYZED','B4 anchor top1 capture=0.9960/0.9810/0.9937/0.9883','results/A18_BASELINE_OBJECT_SUMMARY.csv','Ranking quality distinct from pool coverage'),
 ('C04','REANALYZED','B2 Cov12=0.9718/0.5178/0.9852/0.6751, mean0.7875','results/A18_BASELINE_OBJECT_SUMMARY.csv','Fails simultaneous opportunity-elimination gates'),
 ('C05','NEGATIVE_RESULT','B3 mean Cov12=0.6259, below B0 and B2','results/A18_BASELINE_OBJECT_SUMMARY.csv','Fixed receiver-direction addition proxy; no retuning'),
 ('C06','OPPORTUNITY_NOT_SUCCESS','Best single frozen cheap baseline mean0.8463; teacher-relative gap0.1537','results/STAGE0_SUMMARY.json','Best random seed is descriptive; no object-wise or query-wise seed oracle'),
 ('C07','VERIFIED_LABEL_SCOPE','204 Gaussian measurement vectors plus 1 voxel vector; zero full-current teachers','A18_DEFECT_LABEL_AVAILABILITY.csv','194 voxel matched vectors missing; scalar gains are not vector labels'),
 ('C08','REANALYZED','Median tail/direct norm0.3729 smooth and0.4356 near-contact','figures/04_tail_norms.csv','A and U need not be nested; not a monotonic theorem'),
 ('C09','EVALUATION_ORACLE','Source sketch rank54 on saved54-dimensional Gaussian gauge; null-tail error<=4.47e-15 relative','results/A18_SOURCE_ORACLE.csv','State-specific fullJ oracle, not a learned transfer result or full-current label'),
 ('C10','SPLIT_AUDIT','Fold1 train smooth, test near-contact; reversed smooth results are Fold3 validation','verification/FIT_RECEIPTS.json','Fold0 lacks complete Gaussian training object; no four-object B test'),
 ('C11','NEGATIVE_RESULT','B4 held-out POD anchor-tail skip10/18 for ranks8/16/32; uncorrected anchor also10/18; direct defect0/18','results/A18_B_OBJECT_SUMMARY.csv','Projection oracle under tested bases; not a proof excluding all compensation architectures'),
 ('C12','DERIVED_COST','Matched single-decision RHS-halving threshold75%; historical multi-round floor59.28%','results/A18_B_OBJECT_SUMMARY.csv','Excludes refresh, audit and model overhead; no measured acceleration'),
 ('C13','DECISION','A_GO_TINY_PILOT; B_DATA_INCOMPLETE globally; Gaussian primary tested bases have no margin/cost headroom','A18_STAGE0_DECISION.md','Recommendations only; training remains disabled'),
 ('C14','IMPLEMENTATION_VERIFIED','128 fresh non-Hermitian algebra cases PASS, largest identity residual9.69e-16','verification/algebra_results.json','Equivalent newly written tests; original A18 verification source unavailable'),
 ('C15','NUMERICAL_REPAIR','16 upstream Woodbury failed units recovered with one residual correction and unchanged1e-10 gate','FAILURES.json','No jitter, new physical operator, PCG or speed comparison'),
 ('C16','EXECUTION_VERIFIED','New full-wave RHS0, NN epochs0, new nonlinear trajectories0','results/PROCESS_RESOURCE_RECEIPTS.jsonl','Single computation child; failed attempts retained'),
 ('C17','NOVELTY_OPEN','Minor-current learning, operator correction and derivative compression have prior art','literature/REFERENCE_LEDGER.csv','Search rate limited; no first-ever or global novelty claim'),
 ('C18','SOURCE_VERIFIED','Step and frozen gain use saved lambda_total including prior precision plus LM damping; smooth early example0.01001','INPUT_HASHES.json; pinned maxwell_state.py and setup_cost.json','Preserved damped frozen quadratic; not a new undamped or nonlinear objective')]
wc(ROOT/'CLAIM_LEDGER.csv',[dict(claim_id=i,status=s,claim=c,evidence=e,scope_limit=l) for i,s,c,e,l in claims])
refver=json.loads((ROOT/'literature/scholarqa_verify.json').read_text());refmap={r['requested_id']:r['record'] for r in refver['items']}
refspec=[('R5','DOI:10.1109/TAP.2019.2922779','B','https://doi.org/10.1109/TAP.2019.2922779','collect paper_search abstract','Induced-current learning with bypassed major component; detailed layers OPEN'),
 ('R7','DOI:10.1137/20M1338460','A','https://arxiv.org/html/2005.07069v2','Sections3-4; primary full text','Forward correction may fail gradient consistency; this study only tests fixed-direction acceptance'),
 ('R2','ARXIV:2209.03567','A','https://arxiv.org/html/2209.03567v2','Method, Algorithm1; primary full text','Alternating learned current and analytical material update; no claim of direct Stage0 comparator performance'),
 ('R8','ARXIV:2206.10745','B','https://arxiv.org/abs/2206.10745','primary abstract','Derivative compression is established; no empirical extrapolation to this archive'),
 ('R4','DOI:10.1109/TAP.2025.3602467','C','https://yuxiangren.github.io/publications/','author publication list + metadata verification','PEC surface-current graph prior art; full text, edge formula and speed denominator OPEN; S2 title malformed'),
 ('R16','ARXIV:2210.03008','B','https://arxiv.org/abs/2210.03008','primary abstract','Residual-based operator error correction prior art; no imported certificate'),
 ('R10','ARXIV:2204.13454','B','https://arxiv.org/abs/2204.13454','primary abstract','Certified adaptive ROM uses error control; this study lacks physical deterministic envelopes')]
refs=[]
for key,identifier,tier,url,evidence,limit in refspec:
    rec=refmap[identifier];refs.append(dict(reference_id=key,verified_identifier=identifier,title=rec['title'],authors=[a['name'] for a in rec['authors']],
        metadata_year=rec['year'],year_scope='PREPRINT_OR_DATABASE_YEAR; see DOI for journal year',tier=tier,primary_url=url,evidence=evidence,limitation=limit,metadata_resolved=True))
wc(ROOT/'literature/REFERENCE_LEDGER.csv',refs)
(ROOT/'literature/SEARCH_LOG.md').write_text('''# 本轮 scholarly evidence audit\n\n2026-10-03。使用 scholarqa-research 的 collect/verify workflow；三条互补查询：minor induced-current correction、learned operator correction、PEC current graph。collect 的五个 operation errors 均为 HTTP429；仅第一条 paper search 获得五篇候选，snippet 均不可用。verify 的七个指定 citation 全部 resolved。追加读取已经识别的作者原文/arXiv 页面，R2/R7 方法部分达到 TierA，其余按 ledger 分级。没有把继承 REFERENCES 的“已阅读全文”当成本轮新阅读全文。\n\nR4：IEEE DOI 直接访问失败，作者 publication list 和 S2 可核实书目，但 full-text edge/loss/split/cost 仍 OPEN；S2 title 有异常插词，应以作者条目/DOI 为准。R7/DINO/Residual-correction 的 metadata year 常是 preprint year，文稿用 preprint 标识避免混淆 journal year。恢复的 REFERENCES 保留2026-10-02原记录，本轮 ledger 是独立更新审计。\n\n本轮 novelty 判断限定为：minor-current learning、operator correction、derivative compression 均已有先例；决策导向的 physical-ID proposal 是否具有额外价值需要下一阶段实测。检索受限，未作 exhaustive/first-ever 结论。\n''')

repairs=[json.loads(s) for s in (ROOT/'verification/REPAIR_LOG.jsonl').read_text().splitlines()]
failrows=[dict(failure_id=f'IMPLEMENTATION_{i+1}',classification=r['scientific_classification'],failure=r['failure'],resolution=r['resolution'],evidence=r['failed_result'],status='RECORDED_AND_RESOLVED') for i,r in enumerate(repairs)]
for r in failrows:
    if r['failure_id'] in ['IMPLEMENTATION_3','IMPLEMENTATION_4']:
        r['resolution']+='; this records an attempted path; final unchanged-gate resolution is IMPLEMENTATION_5 residual correction'
failrows.extend([
 dict(failure_id='A_NEGATIVE_B3',classification='SCIENTIFIC_NEGATIVE',failure='B3 coverage inferior to both prefix and O/P/M union',resolution='Keep negative result; do not retune on held-out objects',evidence='results/A18_BASELINE_OBJECT_SUMMARY.csv',status='RETAINED'),
 dict(failure_id='B_VOXEL_LABEL_GAP',classification='DATA_INCOMPLETE',failure='194 matched voxel directions have no legal JFd label',resolution='Do not synthesize from scalar Q or unsupported fullJV',evidence='A18_DEFECT_LABEL_AVAILABILITY.csv',status='OPEN_DATA_GAP'),
 dict(failure_id='B_PRIMARY_MARGIN',classification='SCIENTIFIC_NEGATIVE_SCOPED',failure='Rank8/16/32 projection skips55.6%, adds no skips over anchor and misses60%/75% gates',resolution='Do not recommend B training from this representation',evidence='results/A18_B_OBJECT_SUMMARY.csv',status='RETAINED'),
 dict(failure_id='ORIGINAL_VERIFY_ABSENT',classification='SOURCE_INCOMPLETE',failure='Original A18 verification script/config/results not supplied',resolution='Write equivalent new tests, preserve missing inventory, never claim original reproduction',evidence='A18_DATA_INVENTORY.csv',status='EQUIVALENT_AUDIT_COMPLETE_ORIGINAL_OPEN'),
 dict(failure_id='LITERATURE_429',classification='ACCESS_INCOMPLETE',failure='Five collect operations HTTP429; R4 full text unavailable',resolution='Keep partial search, verify citations, inspect identified primary sources',evidence='literature/scholarqa_collect.json',status='NOVELTY_SCOPE_OPEN'),
 dict(failure_id='DOCUMENT_DEFINITION',classification='AUTHORING_CORRECTION_NOT_NUMERICAL_CHANGE',failure='Draft guide called total Lambda only prior precision; source lambda_total also includes LM damping',resolution='Define archived total regularization in all three current manuscripts; numerical code always used saved lambda_total',evidence='manuscript/COMPLETE_EXPLANATION_ZH.tex; pinned maxwell_state.py; INPUT_HASHES.json',status='RECORDED_AND_RESOLVED')])
wc(ROOT/'FAILURE_LEDGER.csv',failrows)

future=dict(enabled=False,status='CONFIG_ONLY_NO_TRAINING_AUTHORIZED_THIS_STAGE',source_commit=cfg['source_commit'],splits='A18_SPLITS.json',
    A_status=summary['A_status'],B_status=summary['B_status'],common_input_levels=['F0','F1'],forbidden_inputs=['object_id','geometry_class','path','seed','fullJ','Q_true','truth','reference_step','pre-shortlist_child_step','pre-shortlist_Qhat'],
    shortlist_size=12,parameter_cap=200000,epoch_cap=60,folds=4,seeds=[20261002,20261003],total_GPU_occupation_cap_s=7200,
    candidate_models=cfg['future_training_only']['models'],main_loss='listwise incoming utility; teacher=max feasible positive one-swap gain',
    auxiliary_loss='top-m recall, m=4; no-positive query maps to no-op',all_scalers_fit_train_only=True,
    invariance_gates=['permutation equivariant','complex phase covariance/invariant actions','original physical atom ID action consistency'],
    comparison_gates=cfg['A_gates'],B_training_enabled=False,B_requirements=['legal matched voxel vectors and exact receiver output','state-conditioned linear-in-direction interface','output basis crosses margin and cost gates','selected-finalist calibration and cache refresh account'])
jw(ROOT/'FUTURE_TRAINING_CONFIG.json',future)

header='| Method | smooth | near-contact | layered | asymmetric | Equal mean |\n|---|---:|---:|---:|---:|---:|\n'
table=header
for m,s in [('B0',''),('B1','20261002'),('B1','20261003'),('B1','20261004'),('B2',''),('B3',''),('B4',''),('B5','')]:
    table+='| '+m+(' seed'+s if s else '')+' | '+' | '.join(f"{float(ar(m,o,s)['Cov12']):.4f}" for o in [*ids,'EQUAL_OBJECT_MEAN'])+' |\n'
decision=f'''# A18 NN Native Stage 0：完成记录与独立分支裁决

Stage 0 已完成；A=`A_GO_TINY_PILOT`（仅建议下一阶段），B 全范围=`B_DATA_INCOMPLETE`。有完整标签的 Gaussian primary test 对当前 rank8/16/32 POD 表示判 `B_NO_MARGIN_OR_COST_HEADROOM_FOR_TESTED_POD_BASES`。Candidate Generator first 仍是较合理的下一阶段顺序。没有进入训练或补标签。

## 来源与本轮边界

固定 commit `{cfg['source_commit']}`。36 个 canonical receiver queries，四对象×三 frozen states×三 k，全部 dictionary 覆盖；47,712 个唯一 move、43,108 个可行 move、5,208 个 incoming utility 标签，duplicate=0。公共 manifest 的输出 hash 与原始 manifest/source-epoch 的 hash 分别核验；36 original teacher hashes 另对 NN dataset manifest 核验。公开 JSON 元数据整理可能改变字节哈希，不能要求它等于 raw teacher 字节哈希。没有 source/scope mismatch。

新增 full-wave RHS=0，NN epochs=0，新增 nonlinear trajectories=0，GPU training=0。所有计算均为单 CPU worker；实际进程/失败费用见 PROCESS_RESOURCE_RECEIPTS。DeepSeek profiles 未配置，实际 routing 为主线程直接实现，不更改 provider/model。

公共接口只使用投影 L/S/PB、dictionary、公共 receiver GN step 与 r/ell/lambda；完整 Jacobian、Q_true、truth、reference step 均不进入 proposal。原始 atom IDs、共享实材料 gauge、六 illumination、data packing 和1e-10稳定性门槛保留。公共 Q-space inverse 不是真实 full L inverse。

## A：实际结果

{table}

B0 恢复 A17 的已报瓶颈。B4 的 anchor top1 capture 分别为0.9960/0.9810/0.9937/0.9883，确认困难主要在 incoming pool。B2 提高 near-contact 的覆盖，却仍低于0.70；asymmetric0.6751仍低于0.85；平均0.7875低于0.85。B3 固定 receiver-step addition proxy 的符号信息并未带来有效 one-swap material utility，平均0.6259的失败保留。三随机 seeds 全报，不能逐对象或逐 query 事后选 seed。

最强单个已冻结廉价比较对象的描述性 mean 为0.8463，离 one-swap teacher ceiling 尚有0.1537；这是机会空间，不是 learned success。下一阶段须同时满足 near-contact>=0.70、asymmetric>=0.85、mean>=0.85，并超过最佳廉价基线0.05或在质量差<=0.01时成本减半。没有本轮 NN 实测值。所有 Cov12、top1/top4 recall、gain regret 和零正收益统计在 object summary 和 query CSV 中；36 queries 的 no-positive 数为0，no-op作为合法选项保留。

F1 joint components 的 warm median={med_ms:.2f}ms/query，max={max_ms:.2f}ms；每项单独计费，child GN/Qhat 只在 shortlist 后。此时间仅是已缓存投影数据上的计算，不是 cold full-wave workflow 或加速实测。

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
'''
(ROOT/'A18_STAGE0_DECISION.md').write_text(decision)
(ROOT/'PROGRESS.md').write_text('''# A18 Stage 0 进度与恢复入口\n\n1. 重新读取 inputs/A18_CODEX_PROMPT.md 和本记录。\n2. 来源、配置、36-query numerical audit、POD/source/ridge/margin analysis 已完成。正式三份 LaTeX/PDF 和逐页 QA 正在交付阶段。\n3. 不启动训练、full-wave labels 或 nonlinear trajectory；future config 为 disabled。\n4. 当前裁决见 A18_STAGE0_DECISION.md，机器摘要见 results/STAGE0_SUMMARY.json。原始 A17 全部只读。\n5. 数值失败、所有 attempts 和 resource receipts 保留；不要删除负结果，不把 partial B 当四对象 complete。\n''')
print(json.dumps(dict(A=summary['A_status'],B=summary['B_status'],Gaussian_B=summary['B_Gaussian_primary_status'],figures=4,claims=len(claims),references=len(refs))),flush=True)
