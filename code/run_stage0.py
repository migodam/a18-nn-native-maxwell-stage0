"""Single-process, existing-data Stage 0. Never imports upstream Maxwell runtime."""
import os
for _name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,csv,time,hashlib,resource,zipfile,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy import linalg as la
import numpy.lib.format as nf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from endpoint_batch import EndpointBatch
from exchange_core import orth as original_orth
from a18_public import PublicContext,features,propose,reduced_response,source_sketch,orth,material_complex,pack_data
from cpu_refinement import validated_endpoint
CONFIG=json.loads((ROOT/'stage0_config.json').read_text())
UP=Path(CONFIG['source_root']);R1=UP/'A17_R1';ORIGINAL=Path(CONFIG['original_root'])
START_CPU=time.process_time();START_WALL=time.perf_counter()
INVENTORY=[];COSTS=[];FAILURES=[];FEATURE_COSTS=[];LABEL_AVAIL=[];SAMPLES=[];QUERY_ROWS=[];A_ROWS=[];STRUCT=[]
INPUT_HASHES={};CPU_START_TOTAL=0.


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def jsonwrite(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows,fields=None):
    if fields is None:fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in row.items()})
def jlwrite(p,rows):
    with Path(p).open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def check_budget():
    cpu=time.process_time()-START_CPU
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss # macOS bytes
    if cpu>=float(os.environ.get('A18_CPU_REMAINING',CONFIG['cpu_budget_s'])) or rss>CONFIG['max_rss_bytes']:
        raise RuntimeError('BUDGET_STOP_PARTIAL')
def charge(stage,started_wall,started_cpu,**kw):
    row=dict(stage=stage,wall_s=time.perf_counter()-started_wall,cpu_s=time.process_time()-started_cpu,
             new_full_wave_rhs=0,nn_training_epochs=0,new_nonlinear_trajectories=0,**kw)
    COSTS.append(row);return row
def fail(stage,reason,**kw):FAILURES.append(dict(stage=stage,reason=str(reason),**kw))


def inventory(p,role,oid=None,phase=None,k=None,scope=None):
    p=Path(p)
    if not p.exists():
        INVENTORY.append(dict(source_path=str(p),role=role,object_id=oid,phase=phase,k=k,status='MISSING',missing_reason='NOT_SUPPLIED'));return
    h=sha(p);INPUT_HASHES[str(p)]=h
    basic=dict(source_path=str(p),sha256=h,source_commit=CONFIG['source_commit'],role=role,
               object_id=oid,phase=phase,k=k,coordinate_scope=scope or 'METADATA',status='PRESENT')
    if p.suffix!='.npz':INVENTORY.append(basic);return
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            with z.open(n) as f:
                v=nf.read_magic(f);shape,fort,dtype=nf.read_array_header_1_0(f) if v==(1,0) else nf.read_array_header_2_0(f)
            name=n.removesuffix('.npy')
            row=dict(basic,array_name=name,shape=list(shape),dtype=str(dtype),can_form_full_current_label=False,
                     can_form_JFd=name=='fullJ',can_form_full_injection=False,
                     can_reconstruct_endpoint=name in ['step','selected_ids','U_coeff'])
            if name in ['L','S','PB']:row['coordinate_scope']='PROJECTED_Q_CURRENT_COORDINATES_NOT_FULL_PHYSICS'
            if name in ['V','fullJV']:row['coordinate_scope']='SIXTEEN_MATERIAL_DIRECTIONS_ONLY'
            if name=='fullJ':row['coordinate_scope']='FULL_MEASUREMENT_JACOBIAN_SAVED_MATERIAL_GAUGE_ONLY'
            if name=='Js':row['coordinate_scope']='FULL_DIRECTIONAL_OUTPUT_FOR_SAVED_ENDPOINT_STEP'
            INVENTORY.append(row)


def public_context(work,ids,x):
    # Deliberate allowlist: extra evaluator/canary fields cannot enter this API.
    return PublicContext(L=work['L'],S=work['S'],PB=work['PB'],atoms=work['pool'],anchor=work['anchor'],
        physical_embedding=work['Q'],r=work['r'],ell=work['ell'],lam=float(work['lambda_total']),
        families=tuple(str(v) for v in work['families']),selected_ids=tuple(ids),x=x.copy())


def evaluate_cpu(engine,bases,where):
    out=engine.evaluate(bases)
    # Equivalent fixed normal-equation refinement only after the upstream
    # branch fails. Original gate is unchanged and vendor code stays intact.
    for i,status in enumerate(out['status']):
        if status!='NORMAL_RESIDUAL_FAILED':continue
        fail('CPU_WOODBURY_NORMAL_RESIDUAL',status,location=where,index=i,
             relative_residual=float(out['normal_relative_residual'][i]),recovery='EQUIVALENT_CPU_NORMAL_REFINEMENT_SAME_GATE')
        check_budget();w=time.perf_counter();c=time.process_time()
        recovery=validated_endpoint(engine,bases[i])
        charge('FAILED_UNIT_EQUIVALENT_CPU_RECOVERY',w,c,location=where,index=i,recovered_status=recovery['status'],
               normal_relative_residual=recovery.get('normal_relative_residual'),correction_count=1,new_full_rhs=0)
        if recovery['status']=='OK':
            out['steps'][i]=recovery['step'];out['j_steps'][i]=recovery['j_step']
            out['normal_relative_residual'][i]=recovery['normal_relative_residual'];out['status'][i]='OK'
    return out


def structure_on_public(ctx,f,qid):
    rng=np.random.default_rng(CONFIG['verification_seed'])
    phase=np.exp(1j*rng.uniform(-np.pi,np.pi,ctx.atoms.shape[1]))
    ff=features(replace(ctx,atoms=ctx.atoms*phase[None,:]))
    errors={axis:float(la.norm(ff[axis]-f[axis])/max(la.norm(f[axis]),1e-300)) for axis in ['O','P','M','signed']}
    phase_same={method:propose(ctx,f,method)==propose(ctx,ff,method) for method in ['B2','B3']}
    perm=rng.permutation(ctx.atoms.shape[1]);inv=np.argsort(perm)
    ids=tuple(int(inv[i]) for i in ctx.selected_ids)
    cp=replace(ctx,atoms=ctx.atoms[:,perm],families=tuple(ctx.families[i] for i in perm),selected_ids=ids)
    fp=features(cp)
    perm_errors={axis:float(la.norm(fp[axis][inv]-f[axis])/max(la.norm(f[axis]),1e-300)) for axis in ['O','P','M','signed']}
    # Canonical original-ID interface, never storage-column identities.
    canonical=replace(cp,atoms=cp.atoms[:,inv],families=tuple(cp.families[i] for i in inv),selected_ids=ctx.selected_ids)
    fp_c={**fp,**{axis:fp[axis][inv] for axis in ['O','P','M','signed']},'Y':fp['Y'][:,inv]}
    perm_same={method:propose(ctx,f,method)==propose(canonical,fp_c,method) for method in ['B0','B2','B3']}
    # Inject prohibited fields into the loader source. They are not constructor arguments.
    clean=dict(L=ctx.L,S=ctx.S,PB=ctx.PB,pool=ctx.atoms,anchor=ctx.anchor,Q=ctx.physical_embedding,
               r=ctx.r,ell=ctx.ell,lambda_total=ctx.lam,families=ctx.families)
    dirty={**clean,'fullJ':rng.normal(size=(10,10)),'Q_true':rng.normal(size=20),
           'truth':rng.normal(size=10),'reference_step':rng.normal(size=len(ctx.x)),
           'object_id':-999,'seed':-999,'path':'FORBIDDEN_CANARY'}
    cf=features(public_context(dirty,ctx.selected_ids,ctx.x))
    canary=all(np.array_equal(cf[axis],f[axis]) for axis in ['O','P','M','signed'])
    d=rng.normal(size=len(ctx.x));e=rng.normal(size=len(ctx.x))
    z=lambda v:reduced_response(ctx,ctx.anchor,v)
    sd=lambda v:source_sketch(ctx,v,8,CONFIG['source_sketch_seed'])
    linear_errors={name:float(la.norm(fun(d+e)-fun(d)-fun(e))/max(la.norm(fun(d+e)),1e-300)) for name,fun in [('response',z),('source_sketch',sd)]}
    scale_error=float(la.norm(z(-2.3*d)+2.3*z(d))/max(la.norm(z(d)),1e-300))
    row=dict(query_id=qid,phase_feature_errors=errors,permutation_feature_errors=perm_errors,
             phase_action_consistency=phase_same,permutation_original_ID_consistency=perm_same,
             canary_unchanged=canary,zero_response_norm=float(la.norm(z(np.zeros_like(d)))),
             linear_errors=linear_errors,real_scaling_error=scale_error,
             scope='PUBLIC_ANALYTIC_IMPLEMENTATION; no NN implementation tested')
    row['pass']=canary and all(phase_same.values()) and all(perm_same.values()) and max([*errors.values(),*perm_errors.values(),*linear_errors.values(),scale_error])<1e-9
    STRUCT.append(row)


def teacher_incoming(rows,ctx,tau):
    candidates=[i for i in range(ctx.atoms.shape[1]) if i not in ctx.selected_ids]
    out=[]
    for i in candidates:
        rr=[r for r in rows if r['add']==[i]];valid=[r for r in rr if r['status']=='OK']
        best=max((float(r['Q_true']) for r in valid),default=0.);t=max(0.,best)
        ties=[r['drop'][0] for r in valid if best-float(r['Q_true'])<=tau] if valid else []
        out.append(dict(incoming_id=i,teacher_utility=t,best_drop_ids=ties,
                        best_drop=min(valid,key=lambda r:(-r['Q_true'],r['move_index']))['drop'][0] if valid else None,
                        feasible_drop_count=len(valid),total_drop_count=len(rr)))
    return out


def score_pool(qid,oid,phase,k,method,seed,pool,rows,incoming,tau,feature_wall,proposal_wall):
    valid=[r for r in rows if r['status']=='OK'];subset=[r for r in valid if r['add'][0] in pool]
    fullbest=max(0.,max((r['Q_true'] for r in valid),default=0.));shortbest=max(0.,max((r['Q_true'] for r in subset),default=0.))
    order=sorted(subset,key=lambda r:(-r['Qhat'],r['move_index']))
    final=order[:2];top1=order[0] if order else None
    top1_label=float(top1['Q_true']) if top1 and top1['Qhat']>tau else 0.
    accepted=max([0.]+[float(r['Q_true']) for r in final if r['Q_true']>r['tau']])
    if method=='B5':
        final=sorted(valid,key=lambda r:(-r['Q_true'],r['move_index']))[:1] if fullbest>tau else []
        accepted=fullbest if final else 0.;top1_label=accepted
    positive=[r for r in incoming if r['teacher_utility']>tau]
    ranked=sorted(positive,key=lambda r:(-r['teacher_utility'],r['incoming_id']))
    best_ids=[r['incoming_id'] for r in positive if fullbest-r['teacher_utility']<=tau]
    top4=ranked[:4]
    if top4:
        cutoff=top4[-1]['teacher_utility'];top4_ids=[r['incoming_id'] for r in positive if r['teacher_utility']>=cutoff-tau]
    else:top4_ids=[]
    row=dict(query_id=qid,object_id=oid,object_name=CONFIG['object_names'][str(oid)],phase=phase,k=k,
        method=method,seed=seed,shortlist_size=len(pool),incoming_ids=pool,
        full_best_positive_gain=fullbest,shortlist_best_positive_gain=shortbest,gain_regret=fullbest-shortbest,
        top1_recall=int(bool(set(pool)&set(best_ids))) if best_ids else None,
        top4_recall=len(set(pool)&set(top4_ids))/len(top4_ids) if top4_ids else None,
        anchor_top1_true_gain=top1_label,matched_top2_accepted_gain=accepted,
        finalist_move_indices=[r['move_index'] for r in final],no_positive_above_floor=not positive,
        no_positive_raw_gain=fullbest<=0,feature_wall_s=feature_wall,proposal_wall_s=proposal_wall,
        declared_endpoint_count=k*len(pool),feasible_endpoint_count=len(subset),endpoint_wall_s=None,
        acceptance_scope='SAVED_INITIAL_TEACHER_TAU_HIGH_FIDELITY_LABEL_REPLAY_NO_NEW_ACTION',
        measurement_scope='OFFLINE_REUSE_NOT_MEASURED_PHYSICS_SPEEDUP')
    A_ROWS.append(row)
    return final


def cached_receiver_output(cell,ctx,x):
    for policy in ['directed_verified','adaptive_leq_k','surrogate_exchange','exact_score_teacher']:
        p=cell/f'{policy}_endpoint.npz';meta=cell/f'{policy}.json'
        if not p.exists() or not meta.exists():continue
        j=json.loads(meta.read_text())
        if j.get('accepted'):continue
        with np.load(p,allow_pickle=False) as z:
            if sorted(z['selected_ids'].tolist())==sorted(ctx.selected_ids) and la.norm(z['step']-x)<=1e-9*max(la.norm(x),1e-300):
                return z['Js'].copy(),str(p)
    return None,None


def vector_labels(cell,ctx,engine,query,selected,rows,ref):
    oid=query['object_id'];qid=query['query_id'];x=ctx.x
    gaussian='fullJ' in ref
    jx=ref['fullJ']@x if gaussian else None
    base_source=str(cell.parent/'offline_information_reference.npz')+'#fullJ@receiver_step' if gaussian else None
    if not gaussian:jx,base_source=cached_receiver_output(cell,ctx,x)
    actual_round0={}
    for policy in ['directed_verified','adaptive_leq_k','surrogate_exchange','exact_score_teacher']:
        p=cell/f'{policy}.json';array=cell/f'{policy}_round_0.npz'
        if p.exists() and array.exists():
            meta=json.loads(p.read_text());rr=[z for z in meta.get('rounds',[]) if z['round']==0 and z.get('accepted')]
            if rr:
                move=rr[0]['selected_move']
                with np.load(array,allow_pickle=False) as z:
                    actual_round0[(tuple(move['drop']),tuple(move['add']))]=(z['step'].copy(),z['Js'].copy(),str(array))
                inventory(array,'evaluator',oid,query['phase'],query['k'],'SAVED_ACCEPTED_ENDPOINT')
    ids_to_row={r['move_index']:r for r in rows}
    unique=sorted(set(r['move_index'] for final in selected.values() for r in final))
    wanted=[ids_to_row[i] for i in unique if ids_to_row[i]['status']=='OK']
    w=time.perf_counter();c=time.process_time()
    evaluated=evaluate_cpu(engine,np.stack([orth(ctx.atoms[:,r['ids']]) for r in wanted]),qid+'#finalists') if wanted else None
    charge('F2_RECONSTRUCT_MATCHED_FINALISTS',w,c,query_id=qid,count=len(wanted),scope='REDUCED_ONLY_AFTER_SHORTLIST; not proposal features')
    valid_count=0;supports=[]
    for j,row in enumerate(wanted):
        if evaluated['status'][j]!='OK':
            fail('VECTOR_REDUCED_ENDPOINT',evaluated['status'][j],query_id=qid,move_index=row['move_index']);continue
        step=evaluated['steps'][j];d=step-x
        methods=[method for method,rr in selected.items() if any(v['move_index']==row['move_index'] for v in rr)]
        support=None;source=None;zf=None;reason=None
        if gaussian:zf=ref['fullJ']@d;source=base_source
        else:
            V=ref['V'];proj=V@la.lstsq(V,d)[0]
            support=float(la.norm(d-proj)/max(la.norm(d),1e-300));supports.append(support)
            if support<=1e-9:
                zf=ref['fullJV']@la.lstsq(V,d)[0];source=str(cell.parent/'offline_information_reference.npz')+'#SUPPORTED_V_DIRECTION'
            elif jx is not None and (tuple(row['drop']),tuple(row['add'])) in actual_round0:
                ss,js,ps=actual_round0[(tuple(row['drop']),tuple(row['add']))]
                if la.norm(ss-step)<=1e-9*max(la.norm(step),1e-300):zf=js-jx;source=ps+'#Js-minus-matched-receiver-Js'
                else:reason='SAVED_CHILD_STEP_NOT_MATCHED'
            else:reason='D_OUTSIDE_V_SPAN_AND_NO_MATCHED_STORED_Jd'
        avail=dict(query_id=qid,object_id=oid,phase=query['phase'],k=query['k'],move_index=row['move_index'],
                   drop=row['drop'],add=row['add'],methods=methods,d_hash=ah(d),V_projection_relative_residual=support,
                   vector_available=zf is not None,exact_u_available=jx is not None,
                   full_current_label_available=False,zF_source=source,receiver_Jx_source=base_source,
                   status='LEGAL_VECTOR' if zf is not None else 'MISSING_VECTOR',missing_reason=reason)
        LABEL_AVAIL.append(avail)
        if zf is None:continue
        valid_count+=1
        zu=reduced_response(ctx,ctx.atoms[:,ctx.selected_ids],d);za=reduced_response(ctx,ctx.anchor,d)
        ua=ctx.r+reduced_response(ctx,ctx.anchor,x)
        def gain(u,z):return float(-u@z-(ctx.ell+ctx.lam*x)@d-.5*(z@z+ctx.lam*(d@d)))
        qanchor=gain(ua,za)
        anchor_diff=abs(qanchor-row['Qhat'])
        if anchor_diff>max(1e-9*max(abs(qanchor),abs(row['Qhat']),1e-300),10*row['tau']):
            fail('MATCHED_ANCHOR_REPLAY','Qhat mismatch',query_id=qid,move_index=row['move_index'],absolute_error=anchor_diff)
        if jx is not None:
            u=ctx.r+jx;qfull=gain(u,zf)
            gain_diff=abs(qfull-row['Q_true'])
            if gain_diff>max(1e-9*max(abs(qfull),abs(row['Q_true']),1e-300),10*row['tau']):
                fail('MATCHED_FULL_GAIN_REPLAY','Q_true mismatch',query_id=qid,move_index=row['move_index'],absolute_error=gain_diff)
                LABEL_AVAIL[-1]['status']='VECTOR_GAIN_MISMATCH';continue
        else:u=None;qfull=row['Q_true'];gain_diff=None
        h=source_sketch(ctx,d,max(CONFIG['pod_ranks']),CONFIG['source_sketch_seed'])
        index=len(SAMPLES);p=ROOT/'results'/f'direction_{index:04d}.npz'
        np.savez_compressed(p,d=d,zF=zf,zU=zu,zA=za,defectU=zf-zu,defectA=zf-za,
                            u=u if u is not None else np.empty(0),source_sketch=h)
        SAMPLES.append(dict(index=index,path=str(p.relative_to(ROOT)),query_id=qid,object_id=oid,
            phase=query['phase'],k=query['k'],move_index=row['move_index'],drop=row['drop'],add=row['add'],
            methods=methods,tau=float(row['tau']),Q_true=float(qfull),Qhat=float(qanchor),exact_u_available=u is not None,
            material_gain_term=float(-(ctx.ell+ctx.lam*x)@d-.5*ctx.lam*(d@d)),
            zF_source=source,source_scope='PUBLIC_Q_PROJECTED_PHYSICAL_ROW_RANDOM_SKETCH_NOT_FULL_INJECTION',
            source_rank=32,zF_norm=float(la.norm(zf)),defectU_norm=float(la.norm(zf-zu)),defectA_norm=float(la.norm(zf-za)),
            known_inner_correction_norm=float(la.norm(za-zu)),matched_gain_absolute_error=gain_diff,
            matched_anchor_absolute_error=anchor_diff,d_hash=ah(d),file_sha256=sha(p)))
    LABEL_AVAIL.append(dict(query_id=qid,object_id=oid,phase=query['phase'],k=query['k'],status='QUERY_SUMMARY',
        matched_finalists_attempted=len(wanted),legal_vector_count=valid_count,exact_u_available=jx is not None,
        full_current_label_available=False,V_projection_min=min(supports) if supports else None,
        V_projection_max=max(supports) if supports else None))


def flush(status):
    csvwrite(ROOT/'A18_DATA_INVENTORY.csv',INVENTORY)
    jlwrite(ROOT/'A18_QUERY_MANIFEST.jsonl',QUERY_ROWS)
    csvwrite(ROOT/'A18_FEATURE_COSTS.csv',FEATURE_COSTS)
    csvwrite(ROOT/'A18_ANALYTIC_SHORTLIST_BASELINES.csv',A_ROWS)
    csvwrite(ROOT/'A18_DEFECT_LABEL_AVAILABILITY.csv',LABEL_AVAIL)
    jlwrite(ROOT/'A18_COST_LEDGER.jsonl',COSTS)
    jlwrite(ROOT/'results/A18_DIRECTION_MANIFEST.jsonl',SAMPLES)
    jsonwrite(ROOT/'verification/public_structure_results.json',STRUCT)
    jsonwrite(ROOT/'FAILURES.json',FAILURES)
    jsonwrite(ROOT/'INPUT_HASHES.json',INPUT_HASHES)
    jsonwrite(ROOT/'STATUS.json',dict(status=status,completed_queries=len(QUERY_ROWS),legal_direction_labels=len(SAMPLES),
        cpu_s=time.process_time()-START_CPU,wall_s=time.perf_counter()-START_WALL,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        nn_training_epochs=0,new_full_wave_rhs=0,new_nonlinear_trajectories=0))


def main():
    manifest=json.loads((UP/'PUBLIC_MANIFEST.json').read_text());pub={r['public_path']:r['output_sha256'] for r in manifest}
    ds=json.loads((UP/'analysis/a17_tiny_proposer/DATASET_MANIFEST_FINAL.json').read_text())
    cfgpath=ROOT/'stage0_config.json';inventory(cfgpath,'configuration')
    for p in (ROOT/'inputs').glob('*.md'):inventory(p,'research_contract')
    for n in ['REFERENCES.md','stage0_config.json','verification/verify_a18.py','verification/results.json']:
        inventory(ORIGINAL.parents[1]/'A18'/n,'original_A18_support')
    for n in ['run_exchange.py','maxwell_state.py','exchange_core.py','endpoint_batch.py','removal_core_batch.py']:
        p=R1/'code'/n;inventory(p,'source')
        if sha(p)!=pub['A17_R1/code/'+n]:raise ValueError('SOURCE_HASH_DRIFT '+n)
    if len(ds['checkpoints'])!=36:raise ValueError('SCOPE_MISMATCH_QUERY_COUNT')
    splits=[]
    for i,oid in enumerate(CONFIG['object_order']):
        valid=CONFIG['object_order'][(i+1)%4];train=[v for v in CONFIG['object_order'] if v not in [oid,valid]]
        splits.append(dict(fold=i,test_object=oid,validation_object=valid,train_objects=train,train_queries=18,validation_queries=9,test_queries=9))
    jsonwrite(ROOT/'A18_SPLITS.json',dict(object_order=CONFIG['object_order'],folds=splits,
        scope='RETROSPECTIVE_OBJECT_HOLDOUT_NOT_PRISTINE_BLIND_TEST',all_object_records_share_partition=True))
    source_rows=feasible_rows=unique_moves=unique_incoming=0;seen_queries=set()
    checkpoints=sorted(ds['checkpoints'],key=lambda cp:(CONFIG['object_order'].index(cp['identity']['object_id']),CONFIG['phases'].index(cp['identity']['phase']),cp['identity']['k']))
    work=None;ref=None;loaded_dir=None;initial_import=sys.modules.copy()
    for cp in checkpoints:
        check_budget();rel=cp['path'].split('/Gaussian/A17/EXCHANGE_R1/',1)[1];path=R1/rel
        oid=cp['identity']['object_id'];phase=cp['identity']['phase'];k=cp['identity']['k'];cell=path.parent;state_dir=cell.parent
        if sha(path)!=pub['A17_R1/'+rel] or sha(ORIGINAL/rel)!=cp['sha256']:raise ValueError('TEACHER_SOURCE_HASH_DRIFT '+rel)
        inventory(path,'teacher',oid,phase,k,'COMPLETE_RECEIVER_ONE_SWAP_TABLE')
        if state_dir!=loaded_dir:
            for name,role in [('public_workspace.npz','public'),('offline_information_reference.npz','evaluator'),
                              ('runtime_receipt.json','provenance'),('setup_cost.json','cost'),('result.json','provenance')]:
                inventory(state_dir/name,role,oid,phase,scope='FROZEN_STATE')
            for name in ['common_data.npz','material_gauge.json']:
                inventory(R1/f'inputs/object_{oid}'/name,'evaluator' if name=='common_data.npz' else 'physical_gauge',oid,scope='SAVED_PHYSICAL_MATERIAL_GAUGE')
            with np.load(state_dir/'public_workspace.npz',allow_pickle=False) as z:work={n:z[n] for n in z.files}
            with np.load(state_dir/'offline_information_reference.npz',allow_pickle=False) as z:ref={n:z[n] for n in z.files}
            loaded_dir=state_dir
        receiver=json.loads((cell/'receiver_no_swap.json').read_text());ids=receiver['selected_ids']
        if len(ids)!=k:raise ValueError('requested receiver IDs/rank mismatch')
        engine=EndpointBatch(work['L'],work['S'],work['PB'],work['r'],work['ell'],float(work['lambda_total']),device='cpu',batch_size=16)
        w=time.perf_counter();c=time.process_time();rx=evaluate_cpu(engine,orth(work['pool'][:,ids])[None],f'{oid}/{phase}/k{k}#receiver')
        if rx['status'][0]!='OK':raise ValueError('receiver CPU endpoint '+rx['status'][0])
        x=rx['steps'][0];ctx=public_context(work,ids,x)
        charge('PUBLIC_RECEIVER_REDUCED_ENDPOINT',w,c,object_id=oid,phase=phase,k=k,scope='COMMON_PUBLIC_STATE_ONLY_NO_CHILD_SCREEN')
        receipt=json.loads((state_dir/'runtime_receipt.json').read_text())
        identity=dict(object_id=oid,state_hash=cp['identity']['state_hash'],pool_hash=cp['identity']['pool_hash'],
            original_atom_ids=ids,x_hash=ah(x),anchor_hash=ah(ctx.anchor),ell_hash=ah(ctx.ell),lambda_total=ctx.lam,
            receiver_layout_hash=ah(ctx.S),public_operator_hash=sha(state_dir/'public_workspace.npz'),
            material_gauge_hash=ah(np.asarray(json.dumps(receipt.get('material_gauge',{}),sort_keys=True))),phase=phase,k=k)
        qid=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        if qid in seen_queries:raise ValueError('duplicate canonical query scope');
        seen_queries.add(qid)
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        pairs={(tuple(r['drop']),tuple(r['add'])) for r in rows};required={((i,),(j,)) for i in ids for j in range(ctx.atoms.shape[1]) if j not in ids}
        if pairs!=required or len(rows)!=len(pairs):raise ValueError('SCOPE_MISMATCH_INCOMPLETE_DICTIONARY')
        if any(r['base_U_hash']!=cp['identity']['base_U_hash'] or r['state_hash']!=identity['state_hash'] or r['pool_hash']!=identity['pool_hash'] or r['policy_id']!='receiver_neighborhood_teacher' or r['round']!=0 for r in rows):raise ValueError('teacher query identity mismatch')
        feasible=sum(r['status']=='OK' for r in rows)
        source_rows+=len(rows);feasible_rows+=feasible;unique_moves+=len(pairs)
        tau=max(float(r['tau']) for r in rows);incoming=teacher_incoming(rows,ctx,tau);unique_incoming+=len(incoming)
        query=dict(query_id=qid,**identity,teacher_source=str(path),teacher_public_sha256=sha(path),
            teacher_original_path=str(ORIGINAL/rel),teacher_original_sha256=sha(ORIGINAL/rel),source_rows=len(rows),
            feasible_moves=feasible,unique_incoming=len(incoming),dictionary_atoms=ctx.atoms.shape[1],tau=tau,
            numerical_floor=receiver['risk']['numerical_floor'],actual_rank=receiver['actual_rank'],
            public_coordinate_scope='Q_PROJECTED_NOT_FULL_PHYSICS',material_gauge=receipt.get('material_gauge'),
            shared_material_real_dimension=len(x),illumination_count=ctx.PB.shape[0],measurement_real_dimension=len(ctx.r))
        QUERY_ROWS.append(query)
        np.savez_compressed(ROOT/'results'/f'public_query_{qid[:16]}.npz',x=x,ell=ctx.ell,r=ctx.r,lambda_total=ctx.lam,selected_ids=np.asarray(ids))
        jlwrite(ROOT/'results'/f'incoming_{qid[:16]}.jsonl',incoming)
        w=time.perf_counter();c=time.process_time();f=features(ctx)
        total=charge('F1_PUBLIC_FEATURES',w,c,query_id=qid,scope='EXISTING_PROJECTED_CACHE_CPU_WARM')
        for fc in f['costs']:FEATURE_COSTS.append(dict(query_id=qid,object_id=oid,phase=phase,k=k,**fc))
        np.savez_compressed(ROOT/'results'/f'features_{qid[:16]}.npz',O=f['O'],P=f['P'],M=f['M'],signed_transfer=f['signed'],gamma_normalized=f['gamma_normalized'],proxy_stable=f['proxy_stable'],selected_ids=np.asarray(ids))
        selected={}
        for method,seed in [('B0',None),* [('B1',s) for s in CONFIG['random_seeds']],('B2',None),('B3',None),('B4',None),('B5',None)]:
            w=time.perf_counter();c=time.process_time()
            pool=propose(ctx,f,method,seed,CONFIG['B3_overlap_penalty']) if method in ['B0','B1','B2','B3'] else [i for i in range(ctx.atoms.shape[1]) if i not in ids]
            pc=charge('PROPOSAL',w,c,query_id=qid,method=method,seed=seed,scope='PUBLIC_F0_F1' if method in ['B0','B1','B2','B3'] else 'EXISTING_EXPENSIVE_REFERENCE_ONLY')
            key=method if seed is None else f'{method}_seed{seed}'
            selected[key]=score_pool(qid,oid,phase,k,method,seed,pool,rows,incoming,tau,total['wall_s'] if method in ['B2','B3'] else 0.,pc['wall_s'])
        # Required tests on one k=8 early query per object; all inputs are public.
        if phase=='early' and k==8:
            w=time.perf_counter();c=time.process_time();structure_on_public(ctx,f,qid);charge('PUBLIC_STRUCTURE_TESTS',w,c,query_id=qid)
        w=time.perf_counter();c=time.process_time();vector_labels(cell,ctx,engine,query,selected,rows,ref)
        charge('VECTOR_LABEL_REUSE_TOTAL_INCLUDES_MATCHED_ENDPOINT',w,c,query_id=qid,scope='EXISTING_LABELS_AND_REDUCED_RESPONSES; nested timings not additive')
        flush('RUNNING')
        print(json.dumps(dict(event='query_done',object_id=oid,phase=phase,k=k,queries=len(QUERY_ROWS),labels=len(SAMPLES),cpu_s=time.process_time()-START_CPU)),flush=True)
    if source_rows>47712:raise ValueError('SCOPE_MISMATCH_SOURCE_ROWS')
    summary=dict(source_rows=source_rows,feasible_source_rows=feasible_rows,unique_queries=len(QUERY_ROWS),
        unique_moves=unique_moves,unique_incoming_labels=unique_incoming,physical_objects=4,frozen_states=12,
        row_count_is_not_independent_sample_count=True,source_hash_checks='36_PUBLIC_AND_36_ORIGINAL_PASS',
        public_and_original_metadata_bytes_may_differ=True,legal_vector_directions=len(SAMPLES),
        new_full_wave_rhs=0,nn_training_epochs=0,new_nonlinear_trajectories=0)
    jsonwrite(ROOT/'results/COUNTS.json',summary)
    flush('NUMERICAL_STAGE_COMPLETE')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        fail('RUNNER',str(e),traceback=traceback.format_exc());flush('BUDGET_STOP_PARTIAL' if 'BUDGET_STOP' in str(e) else 'FAILED_PARTIAL')
        print(traceback.format_exc(),file=sys.stderr);sys.exit(1)
