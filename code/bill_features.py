"""Complete granular F1 and compensation-source cost measurements; no reranking."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[n]='1'
import json,csv,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from a18_public import PublicContext,features,source_sketch
cfg=json.loads((ROOT/'stage0_config.json').read_text())
queries=[json.loads(s) for s in (ROOT/'A18_QUERY_MANIFEST.jsonl').read_text().splitlines()]
directions=[json.loads(s) for s in (ROOT/'results/A18_DIRECTION_MANIFEST.jsonl').read_text().splitlines()]
newcosts=[];sourcecosts=[];ctxs={};equivalence=[];work=None;loaded=None
for q in queries:
    state=Path(q['teacher_source']).parent.parent
    if state!=loaded:
        with np.load(state/'public_workspace.npz') as z:work={k:z[k].copy() for k in z.files}
        loaded=state
    with np.load(ROOT/'results'/f"public_query_{q['query_id'][:16]}.npz") as z:x=z['x'].copy()
    ctx=PublicContext(work['L'],work['S'],work['PB'],work['pool'],work['anchor'],work['Q'],work['r'],work['ell'],float(work['lambda_total']),tuple(work['families']),tuple(q['original_atom_ids']),x)
    f=features(ctx)
    with np.load(ROOT/'results'/f"features_{q['query_id'][:16]}.npz") as z:
        unchanged=all(np.array_equal(f[a],z['signed_transfer' if a=='signed' else a]) for a in ['O','P','M','signed'])
    equivalence.append(dict(query_id=q['query_id'],public_features_bitwise_unchanged=unchanged))
    for r in f['costs']:newcosts.append(dict(query_id=q['query_id'],object_id=q['object_id'],phase=q['phase'],k=q['k'],feature_scope='A_PROPOSAL_F1',**r))
    for s in (s for s in directions if s['query_id']==q['query_id']):
        with np.load(ROOT/s['path']) as z:d=z['d'].copy();saved=z['source_sketch'].copy()
        w=time.perf_counter();c=time.process_time();h=source_sketch(ctx,d,32,cfg['source_sketch_seed'])
        sourcecosts.append(dict(query_id=q['query_id'],direction_index=s['index'],object_id=q['object_id'],rank_per_illumination=32,
            wall_s=time.perf_counter()-w,cpu_s=time.process_time()-c,new_full_rhs=0,
            unchanged=np.array_equal(h,saved),scope='B_PUBLIC_PROJECTED_SOURCE_ENCODER; matched direction after shortlist'))
for name,rows in [('A18_FEATURE_COSTS.csv',newcosts),('results/A18_COMPENSATION_SOURCE_COSTS.csv',sourcecosts)]:
    p=ROOT/name
    if p.exists():(ROOT/'results'/('initial_'+p.name)).write_bytes(p.read_bytes())
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(ROOT/'verification/COST_AUDIT_EQUIVALENCE.json').write_text(json.dumps(equivalence,indent=2)+'\n')
assert all(r['public_features_bitwise_unchanged'] for r in equivalence)
assert all(r['unchanged'] for r in sourcecosts)
print(json.dumps(dict(feature_queries=len(equivalence),source_directions=len(sourcecosts),all_features_unchanged=True)))
