"""Object-holdout diagnostics using existing labels; no NN and no full RHS."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[n]='1'
import json,csv,time,resource,sys,hashlib
from pathlib import Path
import numpy as np
from scipy import linalg as la
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'code'),str(ROOT/'vendor')]
from a18_public import PublicContext,source_sketch,reduced_response
cfg=json.loads((ROOT/'stage0_config.json').read_text());start=time.process_time()
UP=Path(cfg['source_root']);remaining=float(os.environ.get('A18_CPU_REMAINING',3600))
def guard():
    if time.process_time()-start>=remaining or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>cfg['max_rss_bytes']:
        (ROOT/'STATUS.json').write_text(json.dumps({'status':'BUDGET_STOP_PARTIAL','stage':'ANALYZE'}));raise SystemExit(2)
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def csvrows(p):return list(csv.DictReader(Path(p).open()))
def writecsv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})
def native(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)
def jw(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=native)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
arows=csvrows(ROOT/'A18_ANALYTIC_SHORTLIST_BASELINES.csv');queries=jl(ROOT/'A18_QUERY_MANIFEST.jsonl')
samples=jl(ROOT/'results/A18_DIRECTION_MANIFEST.jsonl');splits=json.loads((ROOT/'A18_SPLITS.json').read_text())['folds']
features=csvrows(ROOT/'A18_FEATURE_COSTS.csv');qmap={q['query_id']:q for q in queries}
agg=[];methods=list(dict.fromkeys((r['method'],r['seed']) for r in arows))
for method,seed in methods:
    for oid in cfg['object_order']:
        rr=[r for r in arows if r['method']==method and r['seed']==seed and int(r['object_id'])==oid]
        den=sum(float(r['full_best_positive_gain']) for r in rr)
        rec=[float(r['top1_recall']) for r in rr if r['top1_recall']!=''];rec4=[float(r['top4_recall']) for r in rr if r['top4_recall']!='']
        agg.append(dict(method=method,seed=seed or None,object_id=oid,object_name=cfg['object_names'][str(oid)],queries=len(rr),
            Cov12=sum(float(r['shortlist_best_positive_gain']) for r in rr)/den if den else None,
            top1_recall=np.mean(rec) if rec else None,top4_recall=np.mean(rec4) if rec4 else None,
            full_gain_sum=den,short_gain_sum=sum(float(r['shortlist_best_positive_gain']) for r in rr),
            gain_regret_sum=sum(float(r['gain_regret']) for r in rr),
            gain_regret_mean=np.mean([float(r['gain_regret']) for r in rr]),
            no_positive_queries=sum(r['no_positive_above_floor']=='True' for r in rr),
            anchor_top1_capture=sum(max(0.,float(r['anchor_top1_true_gain'])) for r in rr)/den if den else None,
            top2_label_replay_capture=sum(float(r['matched_top2_accepted_gain']) for r in rr)/den if den else None))
for method,seed in methods:
    rr=[r for r in agg if r['method']==method and r['seed']==(seed or None)]
    agg.append(dict(method=method,seed=seed or None,object_id='EQUAL_OBJECT_MEAN',object_name='equal object mean',queries=36,
                    Cov12=np.mean([r['Cov12'] for r in rr]),top1_recall=np.mean([r['top1_recall'] for r in rr]),
                    top4_recall=np.mean([r['top4_recall'] for r in rr]),no_positive_queries=sum(r['no_positive_queries'] for r in rr),
                    gain_regret_mean=np.mean([r['gain_regret_mean'] for r in rr]),
                    anchor_top1_capture=np.mean([r['anchor_top1_capture'] for r in rr]),
                    top2_label_replay_capture=np.mean([r['top2_label_replay_capture'] for r in rr])))
writecsv(ROOT/'results/A18_BASELINE_OBJECT_SUMMARY.csv',agg)

# Verify all actual public inputs against output manifest and original numeric
# arrays against recorded source epoch; original teacher metadata against NN manifest.
public_manifest={r['public_path']:r for r in json.loads((UP/'PUBLIC_MANIFEST.json').read_text())}
input_hashes=json.loads((ROOT/'INPUT_HASHES.json').read_text());hash_checks=[]
for name,h in input_hashes.items():
    p=Path(name)
    try:rel=str(p.relative_to(UP))
    except ValueError:continue
    row=public_manifest.get(rel);original=Path(cfg['original_root'])/rel.removeprefix('A17_R1/')
    expected_original=(row or {}).get('source_epoch','').partition(':')[2]
    original_h=sha(original) if original.exists() else None
    hash_checks.append(dict(public_path=rel,public_sha256=h,public_manifest_expected=(row or {}).get('output_sha256'),
        public_match=row is not None and h==row['output_sha256'],original_path=str(original),original_sha256=original_h,
        original_manifest_expected=expected_original or None,
        original_match=original_h==expected_original if original_h and expected_original else None,
        original_hash_role='PUBLIC_SOURCE_EPOCH; original teacher NN-manifest separately checked'))
writecsv(ROOT/'verification/SOURCE_HASH_CHECKS.csv',hash_checks)
source_j=json.loads((Path(cfg['original_root'])/'SOURCE_MANIFEST.json').read_text());source_checks=[]
for module in ['operators','pairtsom_adapter']:
    row=source_j['modules'][module];p=Path(cfg['original_root'])/row['path'];h=sha(p)
    source_checks.append(dict(module=module,path=str(p),sha256=h,expected=row['sha256'],pass_hash=h==row['sha256'],used='READ_ONLY_ORTH_CONVENTION; module not imported'))
jw(ROOT/'verification/ORTH_SOURCE_HASHES.json',source_checks)

# Physical row correspondence is verified before cross-object source/POD use.
with np.load(UP/'A17_R1/inputs/object_2001/common_data.npz') as z:p1=z['points'].copy()
with np.load(UP/'A17_R1/inputs/object_2007/common_data.npz') as z:p2=z['points'].copy()
geometry=dict(gaussian_physical_rows_equal=np.array_equal(p1,p2),physical_points_shape=list(p1.shape),
              common_measurement_real_dimension=1536,illumination_count=6,
              scope='SAME_ARCHIVED_RECEIVER_CONFIGURATION_AND_PACKING; projected S changes with Q')
if not geometry['gaussian_physical_rows_equal']:raise ValueError('SOURCE_PHYSICAL_ROW_MISMATCH')
jw(ROOT/'verification/PHYSICAL_ROW_ALIGNMENT.json',geometry)

# Complete snapshot inventory: test whether any saved step/Js spans receiver x.
endpoint_audit=[];span_audit=[];state_cache={}
for q in queries:
    guard();state=Path(q['teacher_source']).parent.parent
    if state not in state_cache:
        paths=sorted([p for p in state.glob('k*/*.npz') if p.stem.endswith('_endpoint') or '_round_' in p.stem])
        steps=[];js=[];legal_paths=[]
        for p in paths:
            with np.load(p,allow_pickle=False) as z:
                ok={'step','Js','selected_ids'}<=set(z.files)
                if ok:
                    s=z['step'].copy();j=z['Js'].copy();ids=z['selected_ids'].tolist()
                    legal=np.isfinite(s).all() and np.isfinite(j).all() and len(j)==1536
                    h=sha(p);input_hashes[str(p)]=h;rel=str(p.relative_to(UP));pm=public_manifest[rel]
                    orig=Path(cfg['original_root'])/rel.removeprefix('A17_R1/');expected=pm['source_epoch'].partition(':')[2]
                    orig_h=sha(orig) if orig.exists() else None
                    hash_checks.append(dict(public_path=rel,public_sha256=h,public_manifest_expected=pm['output_sha256'],public_match=h==pm['output_sha256'],
                        original_path=str(orig),original_sha256=orig_h,original_manifest_expected=expected,original_match=orig_h==expected if orig_h else None,
                        original_hash_role='PUBLIC_SOURCE_EPOCH_STORED_ENDPOINT'))
                    endpoint_audit.append(dict(path=str(p),sha256=h,object_id=q['object_id'],phase=q['phase'],
                        selected_ids=ids,step_shape=list(s.shape),Js_shape=list(j.shape),valid_snapshot=legal,
                        coordinate_scope='SAME_FROZEN_STATE_FULL_MEASUREMENT_RESPONSE_FOR_THIS_STEP',
                        full_current_label=False,receiver_match_requires='saved step matches receiver; or receiver x is in supported snapshot span'))
                    if legal:steps.append(s);js.append(j);legal_paths.append(str(p))
        G=np.column_stack(steps);J=np.column_stack(js);sv=la.svdvals(G);state_cache[state]=(G,J,legal_paths,int(np.sum(sv>1e-10*sv[0])))
    G,J,paths,rank=state_cache[state]
    with np.load(ROOT/'results'/f"public_query_{q['query_id'][:16]}.npz") as z:x=z['x'].copy()
    co=la.lstsq(G,x,cond=1e-10)[0];rel=float(la.norm(G@co-x)/max(la.norm(x),1e-300))
    span_audit.append(dict(query_id=q['query_id'],object_id=q['object_id'],phase=q['phase'],k=q['k'],
        saved_snapshot_count=G.shape[1],snapshot_rank=rank,receiver_projection_relative_residual=rel,
        receiver_supported=rel<=1e-9,tolerance=1e-9,
        cannot_use_scalar_Q_for_vector_recovery=True))
writecsv(ROOT/'A18_STORED_ENDPOINT_AUDIT.csv',endpoint_audit)
writecsv(ROOT/'results/A18_SNAPSHOT_SPAN_AUDIT.csv',span_audit)
hash_checks=list({r['public_path']:r for r in hash_checks}.values())
writecsv(ROOT/'verification/SOURCE_HASH_CHECKS.csv',hash_checks);jw(ROOT/'INPUT_HASHES.json',input_hashes)

# Complete Gaussian cohort only. One matched voxel vector is retained as an
# ancillary inventory finding, not promoted to an object-level B benchmark.
cohort=[s for s in samples if s['object_id'] in [2001,2007] and s['exact_u_available']]
loaded={s['index']:{k:z.copy() for k,z in np.load(ROOT/s['path']).items()} for s in cohort}
oracle_rows=[];fit_receipts=[];source_rows=[];linear_tests=[]
def H_rank(h,rank):
    t=h.reshape(6,64)
    return np.concatenate([t[:,:rank],t[:,32:32+rank]],axis=1).reshape(-1)
def evaluated_row(s,base,correction,fold,partition,model,rank,train_objects):
    a=loaded[s['index']];target=a['defect'+base];zhat=a['z'+base]+correction;u=a['u']
    error=float(la.norm(a['zF']-zhat));qhat=float(-u@zhat-.5*(zhat@zhat)+s['material_gain_term'])
    margin=qhat-s['tau'];v=float(la.norm(u+zhat))
    crit=2*margin/(np.sqrt(v*v+2*margin)+v) if margin>0 else None
    bound=v*error+.5*error**2
    return dict(fold=fold,partition=partition,object_id=s['object_id'],phase=s['phase'],k=s['k'],query_id=s['query_id'],
        direction_index=s['index'],move_index=s['move_index'],methods=s['methods'],target='direct_defect' if base=='U' else 'anchor_tail',
        model=model,requested_rank=rank,train_objects=train_objects,Q_true=s['Q_true'],tau=s['tau'],
        corrected_Qhat=qhat,actual_margin=margin,exact_u_norm=float(la.norm(u)),v=v,epsilon_crit=crit,
        response_error_norm=error,target_norm=float(la.norm(target)),target_relative_error=error/max(la.norm(target),1e-300),
        zF_relative_error=error/max(la.norm(a['zF']),1e-300),margin_ratio=error/crit if crit else None,
        oracle_can_skip=bool(crit is not None and error<crit),true_positive=s['Q_true']>s['tau'],
        gain_error_absolute=abs(s['Q_true']-qhat),gain_error_bound=bound,
        bound_violation=max(0.,abs(s['Q_true']-qhat)-bound),
        certificate_scope='COUNTERFACTUAL_EXACT_ERROR_ORACLE_NOT_AVAILABLE_ONLINE')
for split in splits:
    guard();train=[s for s in cohort if s['object_id'] in split['train_objects']]
    tr_objects=sorted(set(s['object_id'] for s in train))
    evals=[(part,[s for s in cohort if s['object_id']==split[f'{part}_object']]) for part in ['validation','test']]
    if not train:
        fit_receipts.append(dict(fold=split['fold'],status='NO_COMPLETE_GAUSSIAN_TRAIN_OBJECT',prescribed_train_objects=split['train_objects'],
                                 test_object=split['test_object'],train_directions=0));continue
    H=np.column_stack([loaded[s['index']]['source_sketch'] for s in train])
    rms=np.sqrt(np.mean(H*H,axis=1));floor=max(float(np.max(rms))*1e-12,1e-300);rms=np.maximum(rms,floor)
    hn=H/rms[:,None];alpha=cfg['ridge_relative_alpha']*float(la.svdvals(hn)[0]**2)
    ridge_factor=la.cho_factor(hn.T@hn+alpha*np.eye(hn.shape[1]),lower=True)
    for base in ['U','A']:
        T=np.column_stack([loaded[s['index']]['defect'+base] for s in train]);left,sv,_=la.svd(T,full_matrices=False)
        for rank in cfg['pod_ranks']:
            U=left[:,:min(rank,left.shape[1])];latent=U.T@T
            W=latent@la.cho_solve(ridge_factor,hn.T)
            model_file=ROOT/'results'/f'fit_fold{split["fold"]}_{base}_rank{rank}.npz'
            np.savez_compressed(model_file,basis=U,rms=rms,ridge_weights=W,train_direction_indices=np.array([s['index'] for s in train]),ridge_alpha=alpha)
            fit_receipts.append(dict(fold=split['fold'],status='TRAIN_OBJECTS_ONLY_FIXED_FIT',target=base,requested_rank=rank,effective_rank=U.shape[1],
                train_objects=tr_objects,train_directions=len(train),validation_labels_used_for_fit=False,test_labels_used_for_fit=False,
                ridge_relative_alpha=cfg['ridge_relative_alpha'],ridge_alpha=alpha,model_path=str(model_file.relative_to(ROOT)),sha256=sha(model_file)))
            h0=np.ones(H.shape[0]);h1=np.arange(H.shape[0],dtype=float)/H.shape[0]
            fun=lambda h:U@(W@(h/rms))
            linear_tests.append(dict(fold=split['fold'],target=base,rank=rank,zero_norm=float(la.norm(fun(np.zeros_like(h0)))),
                scale_relative_error=float(la.norm(fun(-2.3*h0)+2.3*fun(h0))/max(la.norm(fun(h0)),1e-300)),
                addition_relative_error=float(la.norm(fun(h0+h1)-fun(h0)-fun(h1))/max(la.norm(fun(h0+h1)),1e-300)),
                scope='FIXED_RIDGE_COMPENSATION_INTERFACE; NO_NN'))
            for part,ss in evals:
                for s in ss:
                    t=loaded[s['index']]['defect'+base];h=loaded[s['index']]['source_sketch']
                    oracle_rows.append(evaluated_row(s,base,U@(U.T@t),split['fold'],part,'POD_EXACT_LATENT_ORACLE',rank,tr_objects))
                    oracle_rows.append(evaluated_row(s,base,U@(W@(h/rms)),split['fold'],part,'FIXED_LINEAR_RIDGE',rank,tr_objects))
                    if rank==8:oracle_rows.append(evaluated_row(s,base,np.zeros_like(t),split['fold'],part,'UNCORRECTED_REDUCED',0,tr_objects))

# Source information oracle on the full saved Gaussian material gauge.
# Exact evaluation operators are used ONLY to diagnose null-space information.
state_source_cache={}
for q in queries:
    if q['object_id'] not in [2001,2007]:continue
    guard();state=Path(q['teacher_source']).parent.parent
    if state not in state_source_cache:
        with np.load(state/'public_workspace.npz') as z:w={k:z[k].copy() for k in z.files}
        with np.load(state/'offline_information_reference.npz') as z:jfull=z['fullJ'].copy()
        with np.load(ROOT/'results'/f"public_query_{q['query_id'][:16]}.npz") as z:x=z['x'].copy()
        ctx=PublicContext(w['L'],w['S'],w['PB'],w['pool'],w['anchor'],w['Q'],w['r'],w['ell'],float(w['lambda_total']),tuple(w['families']),tuple(q['original_atom_ids']),x)
        eye=np.eye(len(x));h32=source_sketch(ctx,eye,32,cfg['source_sketch_seed']);janchor=reduced_response(ctx,ctx.anchor,eye)
        state_source_cache[state]=(h32,jfull-janchor)
    h32,tail=state_source_cache[state]
    for rank in cfg['pod_ranks']:
        hr=np.column_stack([H_rank(h32[:,i],rank) for i in range(h32.shape[1])]);_,sv,vt=la.svd(hr,full_matrices=False)
        rr=int(np.sum(sv>1e-10*sv[0]));P=vt[:rr].T@vt[:rr]
        nulltail=tail@(np.eye(P.shape[0])-P)
        ss=[s for s in cohort if s['query_id']==q['query_id']]
        for s in ss:
            error=float(la.norm(nulltail@loaded[s['index']]['d']))
            source_rows.append(dict(query_id=q['query_id'],object_id=q['object_id'],direction_index=s['index'],source_rank_per_illumination=rank,
                source_real_features=12*rank,physical_real_material_dim=P.shape[0],numerical_source_rank=rr,
                source_nullspace_tail_error=error,tail_norm=float(la.norm(loaded[s['index']]['defectA'])),
                source_nullspace_relative_error=error/max(la.norm(loaded[s['index']]['defectA']),1e-300),
                oracle_scope='STATE_SPECIFIC_EVALUATION_OPERATOR_NULLSPACE_DIAGNOSTIC_NOT_A_TRAINED_PREDICTOR'))
writecsv(ROOT/'A18_DEFECT_ORACLE.csv',oracle_rows);writecsv(ROOT/'results/A18_SOURCE_ORACLE.csv',source_rows)
jw(ROOT/'verification/FIT_RECEIPTS.json',fit_receipts);jw(ROOT/'verification/RIDGE_LINEARITY.json',linear_tests)

# Unique matched finalists are weighted by actual method; correlated copies of
# the same vector are never called independent samples. Report missing labels.
b_summary=[]
for fold,part,base,model,rank in sorted(set((r['fold'],r['partition'],r['target'],r['model'],r['requested_rank']) for r in oracle_rows)):
    rr=[r for r in oracle_rows if (r['fold'],r['partition'],r['target'],r['model'],r['requested_rank'])==(fold,part,base,model,rank)]
    for method in ['ALL_UNIQUE_MATCHED','B0','B1_seed20261002','B1_seed20261003','B1_seed20261004','B2','B3','B4','B5']:
        subset=rr if method=='ALL_UNIQUE_MATCHED' else [r for r in rr if method in r['methods']]
        if not subset:continue
        qids=set(r['query_id'] for r in subset);n=len(subset);init=6*len(qids);R=6*n
        skip=sum(r['oracle_can_skip'] for r in subset);pos=[r for r in subset if r['true_positive']]
        b_summary.append(dict(fold=fold,partition=part,target=base,model=model,rank=rank,method=method,
            object_id=subset[0]['object_id'],queries=len(qids),unique_directions=n,true_positive_directions=len(pos),
            skip_count=skip,skip_fraction=skip/n,positive_skip_fraction=sum(r['oracle_can_skip'] for r in pos)/len(pos) if pos else None,
            mean_target_relative_error=np.mean([r['target_relative_error'] for r in subset]),
            median_target_relative_error=np.median([r['target_relative_error'] for r in subset]),
            median_margin_ratio=np.median([r['margin_ratio'] for r in subset if r['margin_ratio'] is not None]) if any(r['margin_ratio'] is not None for r in subset) else None,
            full_RHS_initialization=init,full_RHS_finalists=R,
            ideal_skip_fraction_for_total_RHS_halving=(1+init/R)/2,
            counterfactual_RHS_ratio_with_init=(init+R*(1-skip/n))/(init+R),
            refresh_RHS=0,audit_RHS=0,scope='COUNTERFACTUAL_LOWER_COST_BOUND; NO_REAL_RHS_SKIPPED'))
writecsv(ROOT/'results/A18_B_OBJECT_SUMMARY.csv',b_summary)
best=max((r for r in agg if r['object_id']=='EQUAL_OBJECT_MEAN' and r['method'] in ['B0','B1','B2','B3']),key=lambda r:r['Cov12'])
analytic_gate={}
for m,s in methods:
    if m not in ['B0','B1','B2','B3']:continue
    vals={r['object_name']:r['Cov12'] for r in agg if r['method']==m and r['seed']==(s or None)}
    analytic_gate[m+(('_seed'+s) if s else '')]=vals['near-contact']>=.70 and vals['asymmetric']>=.85 and vals['equal object mean']>=.85
keyB=[r for r in b_summary if r['fold']==1 and r['partition']=='test' and r['target']=='anchor_tail' and r['model']=='POD_EXACT_LATENT_ORACLE' and r['method']=='B4']
bestB=max(keyB,key=lambda r:r['skip_fraction']) if keyB else None
summary=dict(source_commit=cfg['source_commit'],config_sha256=sha(ROOT/'stage0_config.json'),
    A_status='A_GO_TINY_PILOT' if not any(analytic_gate.values()) else 'A_NO_NEURAL_HEADROOM',
    A_analytic_gate_results=analytic_gate,best_cheap_global_mean=best,absolute_one_swap_teacher_mean=1.,
    mean_coverage_opportunity=1-best['Cov12'],
    B_status='B_DATA_INCOMPLETE',B_complete_Gaussian_objects=[2001,2007],B_primary_test_fold=1,B_primary_test_object=2007,
    B_primary_training_object=2001,B_fold0='NO_COMPLETE_GAUSSIAN_TRAIN_OBJECT',
    B_best_primary_anchor_latent_oracle=bestB,B_ancillary_voxel_labels=sum(s['object_id']>2010 for s in samples),
    B_scope='ONE PRESCRIBED TEST FOLD + ONE SEPARATE VALIDATION DIRECTION; NOT FOUR-OBJECT B RESULT',
    source_nullspace_max_relative_error=max(r['source_nullspace_relative_error'] for r in source_rows),
    gain_bound_max_violation=max(r['bound_violation'] for r in oracle_rows),
    hashes_public_all_pass=all(r['public_match'] for r in hash_checks),
    hashes_original_numeric_all_pass=all(r['original_match'] for r in hash_checks if r['original_match'] is not None),
    public_structure_all_pass=all(r['pass'] for r in json.loads((ROOT/'verification/public_structure_results.json').read_text())),
    algebra_status=json.loads((ROOT/'verification/algebra_results.json').read_text())['status'],
    no_positive_queries=0,nn_epochs=0,new_full_wave_rhs=0,new_nonlinear_trajectories=0,
    cpu_s=time.process_time()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
jw(ROOT/'results/STAGE0_SUMMARY.json',summary)
print(json.dumps(summary,ensure_ascii=False,default=native),flush=True)
