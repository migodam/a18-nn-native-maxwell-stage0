"""Canaries exercise public features and frozen correction prediction paths."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[n]='1'
import sys,json
from pathlib import Path
import numpy as np
from scipy import linalg as la
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'code'),str(ROOT/'vendor')]
from a18_public import source_sketch,features,propose
from run_stage0 import public_context,teacher_incoming,score_pool
cfg=json.loads((ROOT/'stage0_config.json').read_text())
queries=[json.loads(s) for s in (ROOT/'A18_QUERY_MANIFEST.jsonl').read_text().splitlines()]
dirs=[json.loads(s) for s in (ROOT/'results/A18_DIRECTION_MANIFEST.jsonl').read_text().splitlines()]
rng=np.random.default_rng(cfg['verification_seed']);tests=[]
q=next(q for q in queries if q['object_id']==2007 and q['phase']=='early' and q['k']==8)
state=Path(q['teacher_source']).parent.parent
with np.load(state/'public_workspace.npz') as z:w={k:z[k].copy() for k in z.files}
with np.load(ROOT/'results'/f"public_query_{q['query_id'][:16]}.npz") as z:x=z['x'].copy()
sample=next(s for s in dirs if s['query_id']==q['query_id'])
with np.load(ROOT/sample['path']) as z:d=z['d'].copy()
clean=public_context(w,q['original_atom_ids'],x)
dirty={**w,'fullJ':rng.normal(size=(1536,54)),'Q_true':99999.,'truth':rng.normal(size=1728),
       'reference_step':rng.normal(size=54),'object_id':-999,'geometry_class':'CANARY','path':'CANARY','seed':-9}
dirty_ctx=public_context(dirty,q['original_atom_ids'],x)
hc=source_sketch(clean,d,32,cfg['source_sketch_seed']);hd=source_sketch(dirty_ctx,d,32,cfg['source_sketch_seed'])
for p in sorted((ROOT/'results').glob('fit_fold*_rank*.npz')):
    with np.load(p) as z:U=z['basis'];W=z['ridge_weights'];rms=z['rms']
    fun=lambda h:U@(W@(h/rms))
    e=rng.normal(size=hc.shape)
    row=dict(model_path=str(p.relative_to(ROOT)),source_canary_bitwise_unchanged=np.array_equal(hc,hd),
        prediction_canary_bitwise_unchanged=np.array_equal(fun(hc),fun(hd)),nontrivial_prediction_norm=float(la.norm(fun(hc))),
        zero_norm=float(la.norm(fun(np.zeros_like(hc)))),
        scaling_relative_error=float(la.norm(fun(-2.3*hc)+2.3*fun(hc))/max(la.norm(fun(hc)),1e-300)),
        addition_relative_error=float(la.norm(fun(hc+e)-fun(hc)-fun(e))/max(la.norm(fun(hc+e)),1e-300)))
    row['pass']=row['source_canary_bitwise_unchanged'] and row['prediction_canary_bitwise_unchanged'] and row['nontrivial_prediction_norm']>0 and max(row['zero_norm'],row['scaling_relative_error'],row['addition_relative_error'])<1e-9
    tests.append(row)
fc=features(clean);fd=features(dirty_ctx)
proposal_canary={str(s):propose(clean,fc,'B1',s)==propose(dirty_ctx,fd,'B1',s) for s in cfg['random_seeds']}
control=float(la.norm(dirty['fullJ']@d))
# No-positive synthetic query verifies the no-op teacher branch without
# counting it as an observed archived negative query.
negative=[dict(status='OK',add=[i],drop=[clean.selected_ids[0]],Q_true=-1.,Qhat=-1.,move_index=j,tau=1e-10)
          for j,i in enumerate(i for i in range(clean.atoms.shape[1]) if i not in clean.selected_ids)]
t=teacher_incoming(negative,clean,1e-10)
noop=score_pool('SYNTHETIC_NO_POSITIVE',2007,'early',8,'B5',None,[r['add'][0] for r in negative],negative,t,1e-10,0.,0.)
out=dict(status='PASS' if all(t['pass'] for t in tests) and all(proposal_canary.values()) and not noop else 'FAIL',
         correction_models=tests,random_proposal_canary=proposal_canary,evaluator_control_sensitive=control>0,
         synthetic_no_positive_teacher_utility_zero=all(r['teacher_utility']==0 for r in t),synthetic_noop_selected=not noop,
         no_neural_architecture_tested=True,new_full_wave_rhs=0)
(ROOT/'verification/INTERFACE_CANARY_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'frozen_ridge_models':len(tests),'NN_tested':False}))
sys.exit(0 if out['status']=='PASS' else 1)
