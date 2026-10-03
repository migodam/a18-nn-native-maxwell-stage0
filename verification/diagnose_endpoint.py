"""Existing CPU Schur validation of one failed endpoint; no gate changes."""
import os
for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[n]='1'
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'code'),str(ROOT/'vendor')]
from a18_public import orth
from removal_core_batch import CoreEndpointBatch
from cpu_refinement import validated_endpoint
cfg=json.loads((ROOT/'stage0_config.json').read_text())
state=Path(cfg['source_root'])/'A17_R1/results/object_2014_late'
with np.load(state/'public_workspace.npz') as z:w={k:z[k] for k in z.files}
for k in [4,8,16]:
    ids=json.loads((state/f'k{k}/receiver_no_swap.json').read_text())['selected_ids']
    engine=CoreEndpointBatch(w['L'],w['S'],w['PB'],w['r'],w['ell'],float(w['lambda_total']),device='cpu',batch_size=1)
    u=orth(w['pool'][:,ids]);a=engine.evaluate(u[None]);core=engine.prepare_removal_core(u[:,:-1]);b=engine.evaluate_incoming(core,u[None,:,-1:])
    refined=validated_endpoint(engine,u)
    row=dict(k=k,direct_status=a['status'],direct_residual=a['normal_relative_residual'].tolist(),
             refined_status=refined['status'],refined_residual=refined['normal_relative_residual'],
             Schur_status=b['status'],Schur_residual=b['normal_relative_residual'].tolist(),core_status=core.status)
    print(json.dumps(row),flush=True)
